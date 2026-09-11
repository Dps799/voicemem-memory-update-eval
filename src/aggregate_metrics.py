"""阶段 A 指标聚合：从 pair_results.jsonl 计算 metrics.json。

指标（计划 §7）:
  FMR  误合并率 — 非等价且标签确定的输入中被复用为同一判断的比例
  EMR  同义合并召回率 — 等价输入中正确复用已有判断的比例
  已合并条目的错误比例 — 错误合并数 / 全部合并数
  Wilson 95% 区间
  按来源/关系/语言分层
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PAIR_RESULTS = Path(__file__).parent.parent / "reports" / "pair_results.jsonl"
OUT = Path(__file__).parent.parent / "reports" / "metrics.json"


def wilson_ci(k: int, n: int, conf: float = 0.95) -> list[float]:
    """Wilson score 95% 区间。"""
    if n == 0:
        return [0.0, 1.0]
    z = 1.959963985  # 95%
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def main():
    rows = [json.loads(l) for l in open(PAIR_RESULTS) if l.strip()]
    print(f"Loaded {len(rows)} pair results")

    # 只看 ab 方向（ba 用于顺序一致性验证，不重复计入主指标）
    ab = [r for r in rows if r.get("direction") == "ab" and r.get("error") is None]
    # 排除控制维度（不同用户/不同类别/空输入）— 这些是压力测试，不混入主指标
    eligible = [r for r in ab
                if not r.get("cross_user") and not r.get("cross_slot")
                and r.get("normalized_claim_a") and r.get("normalized_claim_b")]

    results = {}

    for pol_name in sorted(set(r.get("policy") for r in eligible)):
        pol = [r for r in eligible if r.get("policy") == pol_name]
        if not pol:
            continue

        # 非等价且标签确定
        nonequiv = [r for r in pol
                    if r["relation_gold"] in (
                        "contradiction_same_scope",
                        "nonparaphrase_high_overlap",
                        "contextual_difference",
                        "unrelated",
                    )]
        # 等价
        equiv = [r for r in pol if r["relation_gold"] == "equivalent"]
        # temporal_update 单独（偏好变化 — 不应合并，应保留历史）
        temporal = [r for r in pol if r["relation_gold"] == "temporal_update"]

        nonequiv_merged = [r for r in nonequiv if r["merged"]]
        equiv_merged = [r for r in equiv if r["merged"]]
        temporal_merged = [r for r in temporal if r["merged"]]
        all_merged = [r for r in pol if r["merged"]]

        fmr = len(nonequiv_merged) / len(nonequiv) if nonequiv else None
        emr = len(equiv_merged) / len(equiv) if equiv else None
        temporal_merge_rate = len(temporal_merged) / len(temporal) if temporal else None
        merged_error_ratio = len(nonequiv_merged) / len(all_merged) if all_merged else None

        results[pol_name] = {
            "n_eligible": len(pol),
            "n_nonequiv": len(nonequiv),
            "n_equiv": len(equiv),
            "n_temporal": len(temporal),
            "n_merged_total": len(all_merged),
            "n_nonequiv_merged_FMR": len(nonequiv_merged),
            "n_equiv_merged_EMR": len(equiv_merged),
            "n_temporal_merged": len(temporal_merged),
            "FMR": round(fmr, 4) if fmr is not None else None,
            "FMR_wilson_95": wilson_ci(len(nonequiv_merged), len(nonequiv)) if nonequiv else None,
            "EMR": round(emr, 4) if emr is not None else None,
            "EMR_wilson_95": wilson_ci(len(equiv_merged), len(equiv)) if equiv else None,
            "temporal_merge_rate": round(temporal_merge_rate, 4) if temporal_merge_rate is not None else None,
            "merged_error_ratio": round(merged_error_ratio, 4) if merged_error_ratio is not None else "N/A (no merges)",
        }

    # 顺序一致性（ab vs ba）：同一对在两种顺序下 merged 是否一致
    order_check = []
    for pol_name in ["B0_original", "B1_exact", "B2_097", "B2_099", "B3_never"]:
        ab_pol = {r["sample_id"]: r["merged"] for r in eligible if r.get("policy") == pol_name}
        ba_pol = {r["sample_id"]: r["merged"] for r in rows
                  if r.get("policy") == pol_name and r.get("direction") == "ba" and r.get("error") is None}
        consistent = 0
        total = 0
        for sid, m_ab in ab_pol.items():
            if sid in ba_pol:
                total += 1
                if m_ab == ba_pol[sid]:
                    consistent += 1
        if total:
            order_check.append({
                "policy": pol_name,
                "n_checked": total,
                "n_consistent": consistent,
                "order_inconsistency_rate": round(1 - consistent / total, 4),
            })

    # 控制维度验证
    controls = [r for r in ab if r.get("cross_user") or r.get("cross_slot") or not r.get("normalized_claim_a")]
    control_check = []
    for r in controls:
        control_check.append({
            "sample_id": r["sample_id"],
            "policy": r.get("policy"),
            "merged": r["merged"],
            "should_merge": False,
            "passed": not r["merged"],
            "note": "cross_user" if r.get("cross_user") else ("cross_slot" if r.get("cross_slot") else "empty_input"),
        })

    # 边界对
    boundaries = [r for r in ab if r.get("source") == "synthetic_boundary"]

    out = {
        "n_total_results": len(rows),
        "n_ab_eligible": len(eligible),
        "policies": results,
        "order_consistency": order_check,
        "control_dimension_checks": control_check,
        "boundary_pairs": [{
            "sample_id": r["sample_id"],
            "policy": r.get("policy"),
            "similarity": round(r["similarity"], 4) if r.get("similarity") else None,
            "merged": r["merged"],
            "relation_gold": r["relation_gold"],
        } for r in boundaries],
        "by_source_b0": by_source([r for r in eligible if r.get("policy") == "B0_original"]),
        "by_relation_b0": by_relation([r for r in eligible if r.get("policy") == "B0_original"]),
    }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Written: {OUT}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("METRICS SUMMARY (smoke test, n small — intervals are wide)")
    print("=" * 60)
    for pol, m in results.items():
        print(f"\n{pol}:")
        print(f"  FMR={m['FMR']} (n_nonequiv={m['n_nonequiv']}, merged={m['n_nonequiv_merged_FMR']})")
        print(f"  EMR={m['EMR']} (n_equiv={m['n_equiv']}, merged={m['n_equiv_merged_EMR']})")
        print(f"  temporal_merge={m['temporal_merge_rate']} (n={m['n_temporal']}, merged={m['n_temporal_merged']})")
        print(f"  merged_error_ratio={m['merged_error_ratio']}")
    print("\nORDER CONSISTENCY:")
    for o in order_check:
        print(f"  {o['policy']}: {o['n_consistent']}/{o['n_checked']} consistent (inconsistency={o['order_inconsistency_rate']})")
    print("\nCONTROL CHECKS (all should be merged=False):")
    for c in control_check:
        print(f"  {c['sample_id']} [{c['note']}]: merged={c['merged']} passed={c['passed']}")


def by_source(rows):
    out = {}
    for r in rows:
        s = r["source"]
        if s not in out:
            out[s] = {"n": 0, "merged": 0, "nonequiv_merged": 0, "equiv_merged": 0}
        out[s]["n"] += 1
        if r["merged"]:
            out[s]["merged"] += 1
        if r["relation_gold"] != "equivalent" and r["merged"]:
            out[s]["nonequiv_merged"] += 1
        if r["relation_gold"] == "equivalent" and r["merged"]:
            out[s]["equiv_merged"] += 1
    return out


def by_relation(rows):
    out = {}
    for r in rows:
        rel = r["relation_gold"]
        if rel not in out:
            out[rel] = {"n": 0, "merged": 0}
        out[rel]["n"] += 1
        if r["merged"]:
            out[rel]["merged"] += 1
    return out


if __name__ == "__main__":
    main()
