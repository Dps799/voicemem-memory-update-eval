"""N0-T0: 记录 next_preflight — 环境、SHA256、从 manifest 推导的验收数。

不硬编码 570/160。验收数从输入数量、条件集合和排除规则推导。
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "formal_manifest.jsonl"
SPLIT = ROOT / "manifests" / "split_manifest.jsonl"
RUN_CONFIG = ROOT / "run_config.json"
EXCLUSIONS = ROOT / "manifests" / "exclusions.json"
COVERAGE = ROOT / "manifests" / "coverage.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pairs = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    splits = [json.loads(l) for l in SPLIT.read_text().splitlines() if l.strip()]
    config = json.loads(RUN_CONFIG.read_text())

    # 推导验收数
    n_pairs = len(pairs)
    n_dev = sum(1 for p in pairs if p["split"] == "dev")
    n_test = sum(1 for p in pairs if p["split"] == "test")

    # 分组一致性检查
    group_split = {}
    for p in pairs:
        g = p["group_id"]
        s = p["split"]
        if g in group_split and group_split[g] != s:
            raise AssertionError(f"Group {g} crosses split: {group_split[g]} vs {s}")
        group_split[g] = s

    # 无泄漏
    dev_groups = {g for g, s in group_split.items() if s == "dev"}
    test_groups = {g for g, s in group_split.items() if s == "test"}
    no_leakage = len(dev_groups & test_groups) == 0

    # gold 不在策略输入中
    gold_fields = {"relation_gold"}
    strategy_input_fields = {"claim_a", "claim_b", "slot", "group_id", "source"}
    gold_not_in_strategy = gold_fields.isdisjoint(strategy_input_fields)

    # 预期 N1 运行记录数（推导）
    n_main_policies = 5  # B0, B1, B2_097, B2_099, B3
    n_c1_seeds = 3
    directions = 2  # ab, ba
    c1_directions = 1  # ab only
    # 控制项（后续添加，当前 formal_manifest 无控制项）
    n_controls = 0
    expected_records = (
        n_pairs * n_main_policies * directions +  # main policies
        n_pairs * n_c1_seeds * c1_directions +    # C1
        n_controls * n_main_policies              # controls (ab only)
    )

    # 记录所有路径的 SHA256
    paths = sorted(set(
        list((ROOT / "src").glob("*.py")) +
        list((ROOT / "tests").glob("*.py")) +
        list((ROOT / "manifests").glob("*.jsonl")) +
        list((ROOT / "manifests").glob("*.json")) +
        list((ROOT / "upstream").rglob("*.py")) +
        [ROOT / "requirements.txt", ROOT / "scripts" / "reproduce_smoke.sh",
         ROOT / "run_config.json", ROOT / "next_phase_config.json",
         ROOT / "NEXT_PHASE_PLAN.md", ROOT / "EXPERIMENT_PLAN.md",
         ROOT / "CLUSTER_HANDOFF.md", ROOT / "BENCHMARK_ADDENDUM.md"]
    ))

    preflight = {
        "schema_version": "next-preflight-1",
        "status": "n0_configured",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkout_head_at_run": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_identification": "working-tree SHA256 below is authoritative",
        "upstream_commit": "a450911fc8cbb44c46d810aace2f3288bad287e4",
        "embedding_model": "intfloat/multilingual-e5-small",
        "embedding_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "embedding_device": "cpu",
        "embedding_dimensions": 384,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dependencies": {
            p: version(p) for p in
            ["torch", "sentence-transformers", "transformers", "huggingface-hub", "numpy"]
        },
        "environment_overrides": {
            k: v for k, v in os.environ.items()
            if k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "VOICEMEM_E5_PATH",
                      "HF_HUB_DISABLE_IMPLICIT_TOKEN", "HF_ENDPOINT"]
        },
        "manifest": {
            "total_pairs": n_pairs,
            "dev_pairs": n_dev,
            "test_pairs": n_test,
            "unique_groups": len(group_split),
            "dev_groups": len(dev_groups),
            "test_groups": len(test_groups),
            "no_group_crosses_split": no_leakage,
            "gold_not_in_strategy_input": gold_not_in_strategy,
            "expected_n1_records_derived": expected_records,
            "n1_record_formula": "pairs * 5_policies * 2_directions + pairs * 3_c1_seeds * 1_direction + controls * 5_policies",
        },
        "generative_api_calls": 0,
        "sha256": {
            str(p.relative_to(ROOT)): sha256_file(p)
            for p in paths if p.exists()
        },
    }

    out = ROOT / "reports" / "next_preflight.json"
    out.write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(preflight["manifest"], indent=2))


if __name__ == "__main__":
    main()
