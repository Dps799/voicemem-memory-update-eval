"""N1 测试集推理：在冻结的 dev 配置上运行 299 对测试集。

不选择新阈值（使用 dev 冻结的 B2=0.95=B0）。
不运行 B2_097/B2_099（dev-only 候选）。
C1 使用 dev 校准的合并率。

产物:
  reports/n1_test/<run_id>/pair_results.jsonl
  reports/n1_test/<run_id>/metrics.json
  reports/n1_test/<run_id>/preflight.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent))
from e5_callback import (
    TraitStore, Evidence, normalize_claim, real_embed,
    precompute_embeddings, cos_sim, E5_MODEL, E5_REVISION,
)
from run_phase_a import B1ExactStore, RandomMergeStore
from run_n1_dev import (
    CheckedTraitStore, CheckedExactStore, CheckedRandomStore,
    strategy_input, execute, cluster_metric, summarize, NON_EQ,
)
import traits_store
from prepare_n1 import read_rows
from aggregate_metrics import wilson_ci

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "run_config.json"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())

    all_pairs = read_rows(ROOT / config["sampling"]["effective_manifest"])
    pairs = [r for r in all_pairs if r["split"] == "test"]
    if not pairs:
        raise ValueError("No test samples")
    print(f"Test pairs: {len(pairs)}")

    # 读取 dev 冻结的选择
    dev_selection_path = ROOT / "reports" / "n1_dev" / "n1_dev_20260914_auto" / "selection.json"
    dev_selection = json.loads(dev_selection_path.read_text())
    c1_rate = dev_selection["c1_rate"]
    frozen_threshold = dev_selection["selected_b2_threshold"]
    print(f"Dev frozen: B2={frozen_threshold}, C1_rate={c1_rate:.4f}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("n1_test_%Y%m%dT%H%M%SZ")
    if Path(run_id).name != run_id:
        raise ValueError("run-id must be a single directory name")
    output = ROOT / "reports" / "n1_test" / run_id
    output.mkdir(parents=True, exist_ok=False)

    # 预计算 embedding
    start = perf_counter()
    texts = [r[k] for r in pairs for k in ("claim_a", "claim_b")]
    precompute_embeddings(texts + [normalize_claim(t) for t in texts])
    precompute_seconds = perf_counter() - start
    print(f"Precompute: {precompute_seconds:.1f}s")

    # 测试集只运行冻结的策略：B0(=B2=0.95), B1, B3, C1×3, raw ablation
    # 不运行 B2_097/B2_099（dev-only 候选）
    test_policies = [
        {"name": "B0_original", "threshold": 0.95},
        {"name": "B1_exact", "threshold": "string_match"},
        {"name": "B3_never", "threshold": 2.0},
    ]

    rows = []
    normalization = []

    for policy in test_policies:
        for pair in pairs:
            for direction in config["n1_directions"]:
                metadata = {k: pair[k] for k in ("sample_id", "group_id", "source", "split", "relation_gold")}
                try:
                    result = execute(strategy_input(pair), policy, direction)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}", "merged": None,
                              "normalization_mode": "upstream"}
                row = dict(result, **metadata, policy=policy["name"], direction=direction,
                           label_provenance="dataset_proxy_no_human_adjudication")
                rows.append(row)
        print(f"Completed {policy['name']}", flush=True)

    # C1 使用 dev 校准率
    for seed in [20260911, 20260912, 20260913]:
        rng = random.Random(seed)
        policy_name = f"C1_random_{seed}"
        for pair in pairs:
            draw = rng.random() < c1_rate
            metadata = {k: pair[k] for k in ("sample_id", "group_id", "source", "split", "relation_gold")}
            try:
                result = execute(strategy_input(pair),
                                 {"name": policy_name, "threshold": "string_match"},
                                 "ab", random_reuse=draw)
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}", "merged": None,
                          "normalization_mode": "upstream"}
            row = dict(result, **metadata, policy=policy_name, direction="ab",
                       label_provenance="dataset_proxy_no_human_adjudication",
                       random_reuse_draw=draw, calibration_rate=c1_rate)
            rows.append(row)
        print(f"Completed {policy_name}", flush=True)

    # Raw ablation
    raw_policy = {"name": "B0_raw_ablation", "threshold": 0.95}
    for pair in pairs:
        for direction in ("ab", "ba"):
            metadata = {k: pair[k] for k in ("sample_id", "group_id", "source", "split", "relation_gold")}
            try:
                result = execute(strategy_input(pair), raw_policy, direction, raw=True)
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}", "merged": None,
                          "normalization_mode": "raw_identity"}
            row = dict(result, **metadata, policy=raw_policy["name"], direction=direction,
                       label_provenance="dataset_proxy_no_human_adjudication")
            rows.append(row)
        a, b = pair["claim_a"], pair["claim_b"]
        na, nb = normalize_claim(a), normalize_claim(b)
        normalization.append({
            "sample_id": pair["sample_id"], "source": pair["source"],
            "text_changed": a != na or b != nb,
            "raw_distinct_normalized_equal": a != b and na == nb,
        })
    print("Completed B0_raw_ablation", flush=True)

    # 写结果
    (output / "pair_results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    (output / "normalization_audit.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in normalization)
    )

    # 指标
    metrics = summarize(rows)
    metrics["n_pairs"] = len(pairs)
    metrics["n_records"] = len(rows)
    metrics["n_errors"] = sum(r["error"] is not None for r in rows if "error" in r)
    metrics["split"] = "test"
    metrics["dev_frozen_config"] = dev_selection
    metrics["label_scope"] = "test inference with dev-frozen config; dataset-proxy labels"
    metrics["test_config_frozen_before_viewing"] = True

    # raw ablation
    raw = [r for r in rows if r["policy"] == "B0_raw_ablation" and r["direction"] == "ab" and r.get("error") is None]
    metrics["raw_ablation_by_source"] = {}
    upstream = {r["sample_id"]: r for r in rows if r["policy"] == "B0_original" and r["direction"] == "ab"}
    for source in sorted({r["source"] for r in raw}):
        rs = [r for r in raw if r["source"] == source]
        metrics["raw_ablation_by_source"][source] = {
            "FMR": cluster_metric(rs, NON_EQ),
            "EMR": cluster_metric(rs, {"equivalent"}),
            "temporal_merge_rate": cluster_metric(rs, {"temporal_update"}),
            "merge_decisions_changed": sum(
                r["merged"] != upstream.get(r["sample_id"], {}).get("merged", False) for r in rs
            ),
        }

    # order state comparison
    order = {}
    for policy in test_policies:
        counts = Counter()
        for pair in pairs:
            a = next((r for r in rows if r["sample_id"] == pair["sample_id"] and
                       r["policy"] == policy["name"] and r["direction"] == "ab"), None)
            b = next((r for r in rows if r["sample_id"] == pair["sample_id"] and
                       r["policy"] == policy["name"] and r["direction"] == "ba"), None)
            if not a or not b or a.get("error") or b.get("error"):
                counts["errors"] += 1
                continue
            counts["n"] += 1
            counts["merge_decision_different"] += a["merged"] != b["merged"]
            counts["stored_state_different"] += a.get("stored_state") != b.get("stored_state")
        order[policy["name"]] = dict(counts)
    metrics["order_state_comparison"] = order

    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    # preflight
    files = list((ROOT / "src").glob("*.py")) + list((ROOT / "tests").glob("*.py")) + \
            list((ROOT / "upstream").rglob("*.py")) + \
            [ROOT / config["sampling"]["effective_manifest"], Path(args.config)] + \
            list(output.glob("*.json*"))
    preflight = {
        "head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "status": "completed" if not metrics["n_errors"] else "completed_with_errors",
        "scope": "N1 test inference; dev-frozen config; dataset-proxy labels",
        "platform": platform.platform(), "python": platform.python_version(),
        "model": E5_MODEL, "revision": E5_REVISION, "device": "cpu",
        "embedding_precompute_seconds": precompute_seconds,
        "total_wall_seconds": perf_counter() - start,
        "test_inference_count": len(pairs), "generative_api_calls": 0,
        "n_records": len(rows), "n_errors": metrics["n_errors"],
        "dev_frozen_b2_threshold": frozen_threshold, "dev_c1_rate": c1_rate,
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    (output / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n")

    print(f"\nOutput: {output}")
    print(f"Records: {len(rows)}, Errors: {metrics['n_errors']}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("N1 TEST SET SUMMARY")
    print("=" * 60)
    ab = [r for r in rows if r["direction"] == "ab" and r["normalization_mode"] == "upstream"]
    for policy_name in sorted({r["policy"] for r in ab}):
        pol = [r for r in ab if r["policy"] == policy_name and r.get("error") is None]
        if not pol:
            continue
        fmr = cluster_metric(pol, NON_EQ)
        emr = cluster_metric(pol, {"equivalent"})
        temp = cluster_metric(pol, {"temporal_update"})
        print(f"\n{policy_name} (n={len(pol)}):")
        print(f"  FMR={fmr['rate']:.4f} ({fmr['k']}/{fmr['n']})")
        print(f"  EMR={emr['rate']:.4f} ({emr['k']}/{emr['n']})" if emr["rate"] is not None else "  EMR=N/A")
        print(f"  temporal={temp['rate']:.4f} ({temp['k']}/{temp['n']})" if temp["rate"] is not None else "  temporal=N/A")
        if fmr.get("group_macro_rate") is not None:
            print(f"  group_macro_FMR={fmr['group_macro_rate']:.4f}")
            print(f"  bootstrap_95={fmr['group_macro_bootstrap_95']}")


if __name__ == "__main__":
    main()
