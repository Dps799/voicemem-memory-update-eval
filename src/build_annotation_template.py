"""N0-T2: 构建标注模板。

从 dev 集 300 对中抽 30 对作为开发校准集，其余 270 对为正式开发标注集。
每对预填规范化文本（系统计算），标注者填关系、操作、对象、时间、条件。

产物:
  manifests/annotation_template.jsonl  — 300 dev 对 + 空标注字段
  manifests/annotation_calibration.jsonl — 30 对校准子集
  docs/annotation_handbook.md           — 标注手册
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from e5_callback import normalize_claim  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "formal_manifest.jsonl"
SEED = 20260914

RELATIONS = [
    "equivalent",
    "contradiction_same_scope",
    "temporal_update",
    "contextual_difference",
    "unrelated",
    "ambiguous",
]
OPERATIONS = [
    "merge_support",
    "add_separate",
    "supersede_current_keep_history",
    "retain_contextual_exception",
    "undetermined",
]


def build_template():
    pairs = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    dev_pairs = [p for p in pairs if p["split"] == "dev"]

    rng = random.Random(SEED)
    rng.shuffle(dev_pairs)

    # 30 calibration, 270 formal dev
    calibration = dev_pairs[:30]
    formal_dev = dev_pairs[30:]

    def make_row(p: dict, subset: str) -> dict:
        norm_a = normalize_claim(p["claim_a"])
        norm_b = normalize_claim(p["claim_b"])
        return {
            # 预填字段（标注者可见）
            "source_id": p["sample_id"],
            "group_id": p["group_id"],
            "source": p["source"],
            "language": p["language"],
            "split": p["split"],
            "subset": subset,
            "raw_claim_a": p["claim_a"],
            "raw_claim_b": p["claim_b"],
            "normalized_claim_a": norm_a,
            "normalized_claim_b": norm_b,
            "slot": p["slot"],
            "dataset_relation_gold": p["relation_gold"],
            # 标注者填写字段（初始为空）
            "object": "",
            "time_range": "",
            "condition": "",
            "relation": "",
            "expected_operation": "",
            "confidence": "",
            "annotator_id": "",
            "adjudication_reason": "",
            # 元数据
            "annotation_schema_version": "n0-1",
            "relation_options": RELATIONS,
            "operation_options": OPERATIONS,
        }

    # 写 annotation_template (全部 300 dev)
    template = [make_row(p, "formal_dev") for p in formal_dev] + [make_row(p, "calibration") for p in calibration]
    out = ROOT / "manifests" / "annotation_template.jsonl"
    with open(out, "w") as f:
        for r in template:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written: {out} ({len(template)} rows: {len(formal_dev)} formal + {len(calibration)} calibration)")

    # 写 calibration 子集
    cal_rows = [make_row(p, "calibration") for p in calibration]
    cal_path = ROOT / "manifests" / "annotation_calibration.jsonl"
    with open(cal_path, "w") as f:
        for r in cal_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written: {cal_path} ({len(cal_rows)} calibration pairs)")

    # 统计
    by_source = {}
    by_rel = {}
    for r in template:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        by_rel[r["dataset_relation_gold"]] = by_rel.get(r["dataset_relation_gold"], 0) + 1
    print(f"\nBy source: {by_source}")
    print(f"By dataset gold: {by_rel}")


if __name__ == "__main__":
    build_template()
