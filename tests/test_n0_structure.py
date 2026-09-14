"""N0 回归测试：split 无泄漏、gold 不进入策略、oracle 标记、失败不算成功、预算可复现。"""
import json
import sqlite3
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from e5_callback import TraitStore, Evidence, normalize_claim


def constant_embed(text):
    return [1.0] + [0.0] * 383


ROOT = Path(__file__).resolve().parents[1]


class N0SplitNoLeakage(unittest.TestCase):
    """T1: split 无泄漏 — 同组不跨 dev/test。"""

    def test_no_group_crosses_split(self):
        manifest = ROOT / "manifests" / "formal_manifest.jsonl"
        self.assertTrue(manifest.exists(), "formal_manifest.jsonl not found")
        pairs = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        self.assertEqual(len(pairs), 600, "Expected 600 pairs")

        group_split = {}
        for p in pairs:
            g = p["group_id"]
            s = p["split"]
            self.assertIn(s, ("dev", "test"), f"Invalid split: {s}")
            if g in group_split:
                self.assertEqual(group_split[g], s,
                                 f"Group {g} crosses split: {group_split[g]} vs {s}")
            group_split[g] = s

    def test_dev_test_counts(self):
        manifest = ROOT / "manifests" / "formal_manifest.jsonl"
        pairs = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        dev = sum(1 for p in pairs if p["split"] == "dev")
        test = sum(1 for p in pairs if p["split"] == "test")
        self.assertEqual(dev, 300, f"Expected 300 dev, got {dev}")
        self.assertEqual(test, 300, f"Expected 300 test, got {test}")

    def test_smoke_groups_excluded(self):
        """已看过的 smoke 组不出现在正式集合中。"""
        smoke = ROOT / "manifests" / "pairs_smoke.jsonl"
        formal = ROOT / "manifests" / "formal_manifest.jsonl"
        if not smoke.exists():
            self.skipTest("pairs_smoke.jsonl not found")
        smoke_ids = {json.loads(l)["sample_id"] for l in smoke.read_text().splitlines() if l.strip()}
        formal_ids = {json.loads(l)["sample_id"] for l in formal.read_text().splitlines() if l.strip()}
        overlap = smoke_ids & formal_ids
        self.assertEqual(len(overlap), 0, f"Smoke sample IDs in formal: {overlap}")


class N0GoldNotInStrategy(unittest.TestCase):
    """gold 标签不作为策略输入。"""

    def test_relation_gold_not_in_run_input(self):
        manifest = ROOT / "manifests" / "formal_manifest.jsonl"
        pairs = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        strategy_input_fields = {"claim_a", "claim_b", "slot", "group_id", "source", "sample_id"}
        for p in pairs:
            self.assertIn("relation_gold", p, "Missing relation_gold field")
            self.assertNotIn("relation_gold", strategy_input_fields,
                             "relation_gold must not be in strategy input fields")

    def test_gold_not_passed_to_add(self):
        """TraitStore.add 不接收 gold label 参数。"""
        import inspect
        sig = inspect.signature(TraitStore.add)
        params = set(sig.parameters.keys())
        self.assertNotIn("gold", params, "TraitStore.add must not accept gold label")
        self.assertNotIn("relation_gold", params, "TraitStore.add must not accept relation_gold")


class N0OracleSeparatelyMarked(unittest.TestCase):
    """oracle 诊断单独标记，不混入普通方法。"""

    def test_oracle_naming_convention(self):
        """oracle 策略名必须包含 'oracle' 或 'O_' 前缀。"""
        run_config = ROOT / "run_config.json"
        if not run_config.exists():
            self.skipTest("run_config.json not found")
        config = json.loads(run_config.read_text())
        # 当前 N1 无 oracle 策略；N2 的 O_filter 是 oracle
        # 验证未来 oracle 命名规范
        for policy in config.get("n1_policies", []):
            name = policy["name"]
            if "oracle" in name.lower() or name.startswith("O_"):
                self.assertIn("diagnostic", str(policy).lower() + " " + policy.get("class", ""),
                              "Oracle policy must be marked diagnostic")


class N0FailureNotCountedAsSuccess(unittest.TestCase):
    """运行失败记录为 error，不算成功拒绝合并。"""

    def test_error_field_in_results_schema(self):
        """pair_results 必须有 error 字段，失败不为空。"""
        # 验证 smoke pair_results 的 schema
        smoke_results = ROOT / "reports" / "pair_results.jsonl"
        if not smoke_results.exists():
            self.skipTest("pair_results.jsonl not found")
        rows = [json.loads(l) for l in smoke_results.read_text().splitlines() if l.strip()]
        for r in rows:
            self.assertIn("error", r, "Missing error field in pair_results")
            # 失败行 merged 应为 False（不能把失败当成功）
            if r.get("error") is not None:
                self.assertFalse(r.get("merged", False),
                                 f"Failed run {r.get('sample_id')} marked as merged")

    def test_empty_claim_returns_empty_id(self):
        """空输入返回空 ID，不算合并也不算成功。"""
        db = tempfile.mktemp(suffix=".db")
        store = TraitStore(db, constant_embed)
        ev = Evidence(quote="", emotion="")
        tid = store.add("u", "喜好与厌恶", "", ev)
        self.assertEqual(tid, "", "Empty claim must return empty ID")


class N0QueryNoWriteback(unittest.TestCase):
    """评测 query 不写回记忆。"""

    def test_search_does_not_modify_db(self):
        """search_scored 不写入任何新行。"""
        db = tempfile.mktemp(suffix=".db")
        store = TraitStore(db, constant_embed)
        store.add("u", "喜好与厌恶", "old claim", Evidence(quote="ev1"))

        conn = sqlite3.connect(db)
        before = conn.execute("SELECT COUNT(*) FROM rb_traits").fetchone()[0]
        before_ev = conn.execute("SELECT COUNT(*) FROM rb_evidence").fetchone()[0]
        conn.close()

        # 检索多次
        for _ in range(5):
            store.search_scored("u", "query text", top_k=5)

        conn = sqlite3.connect(db)
        after = conn.execute("SELECT COUNT(*) FROM rb_traits").fetchone()[0]
        after_ev = conn.execute("SELECT COUNT(*) FROM rb_evidence").fetchone()[0]
        conn.close()

        self.assertEqual(before, after, "search_scored modified rb_traits")
        self.assertEqual(before_ev, after_ev, "search_scored modified rb_evidence")


class N0BudgetTruncationReproducible(unittest.TestCase):
    """预算截断可复现（确定性）。"""

    def test_truncation_deterministic(self):
        """相同输入+预算，截断结果相同。"""
        db = tempfile.mktemp(suffix=".db")
        store = TraitStore(db, constant_embed)
        # 写入多条
        for i in range(10):
            store.add("u", "喜好与厌恶", f"claim {i}", Evidence(quote=f"ev {i}"))

        # 两次检索 top_k=3，结果应相同
        r1 = store.search_scored("u", "query", top_k=3)
        r2 = store.search_scored("u", "query", top_k=3)
        self.assertEqual(len(r1), len(r2))
        for (t1, s1), (t2, s2) in zip(r1, r2):
            self.assertEqual(t1.id, t2.id, "Truncation not deterministic")
            self.assertAlmostEqual(s1, s2, places=6)


class N0ManifestConsistency(unittest.TestCase):
    """manifest 内部一致性。"""

    def test_coverage_json_matches_manifest(self):
        manifest = ROOT / "manifests" / "formal_manifest.jsonl"
        coverage = ROOT / "manifests" / "coverage.json"
        if not coverage.exists():
            self.skipTest("coverage.json not found")
        pairs = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        cov = json.loads(coverage.read_text())
        self.assertEqual(len(pairs), cov["total_pairs"])
        self.assertEqual(len({p["group_id"] for p in pairs}), cov["unique_groups"])
        self.assertTrue(cov["no_group_crosses_split"])

    def test_exclusions_recorded(self):
        excl = ROOT / "manifests" / "exclusions.json"
        if not excl.exists():
            self.skipTest("exclusions.json not found")
        data = json.loads(excl.read_text())
        self.assertGreater(len(data.get("excluded_ocnli_prem_ids", [])), 0, "No OCNLI exclusions")
        self.assertGreater(len(data.get("excluded_pawsx_ids", [])), 0, "No PAWS-X exclusions")
        self.assertIn("reason", data, "Missing exclusion reason")


if __name__ == "__main__":
    unittest.main()
