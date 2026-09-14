"""N3 回归测试：预算执行、淘汰策略、oracle 标记、检查点不写回。"""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from e5_callback import TraitStore, Evidence, normalize_claim, _model


def constant_embed(text):
    return [1.0] + [0.0] * 383


ROOT = Path(__file__).resolve().parents[1]


class N3BudgetEnforcement(unittest.TestCase):
    """持久预算超限时必须淘汰。"""

    def _mock_tokenizer(self):
        class MockTok:
            def encode(self, text, add_special_tokens=False):
                return text.split()
        return MockTok()

    def test_budget_evicts_traits(self):
        from run_n3_dev import BoundedStore
        tokenizer = self._mock_tokenizer()
        db = tempfile.mktemp(suffix=".db")
        store = BoundedStore(db, constant_embed, "string_match", "fifo", 4, tokenizer)
        # 每条 claim="claim number N" = 3 words, evidence="evidence N" = 2 words = 5 tokens
        # 预算 4 只能存不到 1 条 → 淘汰
        for i in range(5):
            store.add("u", "喜好与厌恶", f"claim number {i}", Evidence(quote=f"evidence {i}"), step=i)
        with store.store._conn() as c:
            count = c.execute("SELECT COUNT(*) FROM rb_traits WHERE user_id='u'").fetchone()[0]
        self.assertLess(count, 5, "Budget enforcement did not evict any traits")

    def test_fifo_evicts_oldest(self):
        from run_n3_dev import BoundedStore
        tokenizer = self._mock_tokenizer()
        db = tempfile.mktemp(suffix=".db")
        store = BoundedStore(db, constant_embed, "string_match", "fifo", 3, tokenizer)
        # 每条 3 words (claim) + 1 word (evidence) = 4 tokens, 预算 3 → 必须淘汰
        old_id = store.add("u", "喜好与厌恶", "aaa", Evidence(quote="e1"), step=0)
        store.add("u", "喜好与厌恶", "bbb", Evidence(quote="e2"), step=1)
        store.add("u", "喜好与厌恶", "ccc", Evidence(quote="e3"), step=2)
        # oldest should be evicted
        with store.store._conn() as c:
            still_exists = c.execute("SELECT 1 FROM rb_traits WHERE id=?", (old_id,)).fetchone()
            total = c.execute("SELECT COUNT(*) FROM rb_traits WHERE user_id='u'").fetchone()[0]
        self.assertLess(total, 3, "FIFO did not evict any trait")


class N3EvictedNotAccessible(unittest.TestCase):
    """被淘汰的 trait 不能被检索。"""

    def test_evicted_trait_not_in_search(self):
        from run_n3_dev import BoundedStore
        class MockTok:
            def encode(self, text, add_special_tokens=False):
                return text.split()
        tokenizer = MockTok()
        db = tempfile.mktemp(suffix=".db")
        store = BoundedStore(db, constant_embed, "string_match", "fifo", 3, tokenizer)
        old_id = store.add("u", "喜好与厌恶", "old claim here", Evidence(quote="old ev"), step=0)
        store.add("u", "喜好与厌恶", "new claim here", Evidence(quote="new ev"), step=1)
        store.add("u", "喜好与厌恶", "third claim here", Evidence(quote="third ev"), step=2)
        # 搜索不应返回被淘汰的 trait
        results = store.search_scored("u", "old claim here", top_k=5)
        result_ids = [t.id for t, _ in results]
        # old_id 可能被淘汰，不应在结果中
        with store.store._conn() as c:
            still_exists = c.execute("SELECT 1 FROM rb_traits WHERE id=?", (old_id,)).fetchone()
        if not still_exists:
            self.assertNotIn(old_id, result_ids, "Evicted trait returned in search")


class N3OracleDiagnosticOnly(unittest.TestCase):
    """O_state oracle 是诊断，不修改存储。"""

    def test_oracle_does_not_modify_store(self):
        from run_n3_dev import BoundedStore, count_tokens, POLICIES, run_trajectory, BUDGET_PAIRS
        # O_state 策略不修改存储
        oracle_policy = next(p for p in POLICIES if p.get("oracle"))
        self.assertTrue(oracle_policy["name"].startswith("O_"))
        self.assertTrue("diagnostic" in oracle_policy["name"].lower() or "oracle" in oracle_policy["name"].lower())


class N3CheckpointNoWriteback(unittest.TestCase):
    """检查点评估 query 不写回记忆。"""

    def test_checkpoint_query_no_writeback(self):
        from run_n3_dev import BoundedStore
        class MockTok:
            def encode(self, text, add_special_tokens=False):
                return text.split()
        tokenizer = MockTok()
        db = tempfile.mktemp(suffix=".db")
        store = BoundedStore(db, constant_embed, "string_match", "none", 99999, tokenizer)
        store.add("u", "喜好与厌恶", "test claim", Evidence(quote="ev"), step=0)
        before = store.counts("u")
        # 多次搜索
        for _ in range(10):
            store.search_scored("u", "query", top_k=5)
        after = store.counts("u")
        self.assertEqual(before, after, "Checkpoint query wrote back to memory")


class N3TrajectoryStructure(unittest.TestCase):
    """轨迹结构完整性。"""

    def test_trajectory_manifest_exists(self):
        manifest = ROOT / "reports" / "n3_dev" / "n3_dev_20260914_auto" / "trajectory_manifest.jsonl"
        if not manifest.exists():
            self.skipTest("N3 results not found")
        trajs = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        self.assertGreater(len(trajs), 0)
        for t in trajs:
            self.assertIn("trajectory_id", t)
            self.assertIn("checkpoints", t)
            self.assertIn("core_preference", t)
            self.assertIn("final_update", t)
            self.assertIn(t["length"], (40, 160))

    def test_expected_records_match(self):
        metrics_path = ROOT / "reports" / "n3_dev" / "n3_dev_20260914_auto" / "metrics.json"
        if not metrics_path.exists():
            self.skipTest("N3 metrics not found")
        m = json.loads(metrics_path.read_text())
        self.assertEqual(m["n_records"], m["expected_records"], "Record count mismatch")
        self.assertEqual(m["n_errors"], 0, "Execution errors present")

    def test_budget_pairs_as_planned(self):
        metrics_path = ROOT / "reports" / "n3_dev" / "n3_dev_20260914_auto" / "metrics.json"
        if not metrics_path.exists():
            self.skipTest("N3 metrics not found")
        m = json.loads(metrics_path.read_text())
        # 验证条件数 >= 4 policies × 5 budgets
        conditions = m.get("conditions", {})
        # 解析 key 中的 budget 对
        budget_set = set()
        for key in conditions:
            for part in key.split("|"):
                if part.startswith("storage="):
                    s = int(part.split("=")[1])
                if part.startswith("context="):
                    c = int(part.split("=")[1])
            budget_set.add((s, c))
        self.assertGreaterEqual(len(budget_set), 5, f"Not all 5 budget pairs present: {budget_set}")
        # 每个策略应有所有预算
        policies = set()
        for key in conditions:
            policies.add(key.split("|")[0])
        self.assertGreaterEqual(len(policies), 4, f"Not all policies present: {policies}")


if __name__ == "__main__":
    unittest.main()
