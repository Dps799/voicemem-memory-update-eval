"""从已审计数据构建阶段 A 句对。

每来源先取 10 条检查 schema（不进入正式集合），用固定种子 20260911。
覆盖测试维度：同义、相反、不同用户、不同类别、空输入、阈值边界。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

SEED = 20260911
random.seed(SEED)

OCNLI_DEV = "/quark_speech_nas_zjk/users/wangyuhan/experiments/voicemem-memory-update-eval/data/OCNLI/data/ocnli/dev.json"
PAWSX_ZH = (
    "/root/.cache/huggingface/hub/"
    "datasets--google-research-datasets--paws-x/snapshots/"
    "4cd8187c404bda33cb1f62b49b001115862acf37/zh/validation-00000-of-00001.parquet"
)
PERSONAMEM_DIR = Path(
    "/root/.cache/huggingface/hub/"
    "datasets--bowen-upenn--PersonaMem-v2/snapshots/"
    "ed956dea41521fc4499acbc63f966e0fd3c053ba/data/raw_data"
)

SLOTS = ["情绪", "应对方式", "表达风格", "思维模式", "喜好与厌恶"]


def load_ocnli_contradiction(n: int = 10) -> list[dict]:
    rows = [json.loads(l) for l in open(OCNLI_DEV) if l.strip()]
    contra = [r for r in rows if r.get("label") == "contradiction"]
    random.shuffle(contra)
    out = []
    for r in contra[:n]:
        out.append({
            "sample_id": f"ocnli_{r['id']}",
            "group_id": r["prem_id"],
            "source": "OCNLI",
            "split": "dev_schema_check",
            "language": "zh",
            "relation_gold": "contradiction_same_scope",
            "claim_a": r["sentence1"],
            "claim_b": r["sentence2"],
            "slot": "喜好与厌恶",
        })
    return out


def load_pawsx_paraphrase(n: int = 10) -> list[dict]:
    df = pd.read_parquet(PAWSX_ZH)
    para = df[df["label"] == 1].to_dict("records")
    random.shuffle(para)
    out = []
    for r in para[:n]:
        out.append({
            "sample_id": f"pawsx_eq_{r['id']}",
            "group_id": f"pawsx_{r['id']}",
            "source": "PAWS-X-zh",
            "split": "dev_schema_check",
            "language": "zh",
            "relation_gold": "equivalent",
            "claim_a": r["sentence1"],
            "claim_b": r["sentence2"],
            "slot": "喜好与厌恶",
        })
    return out


def load_pawsx_nonparaphrase(n: int = 10) -> list[dict]:
    df = pd.read_parquet(PAWSX_ZH)
    nonpara = df[df["label"] == 0].to_dict("records")
    random.shuffle(nonpara)
    out = []
    for r in nonpara[:n]:
        out.append({
            "sample_id": f"pawsx_ne_{r['id']}",
            "group_id": f"pawsx_{r['id']}",
            "source": "PAWS-X-zh",
            "split": "dev_schema_check",
            "language": "zh",
            "relation_gold": "nonparaphrase_high_overlap",
            "claim_a": r["sentence1"],
            "claim_b": r["sentence2"],
            "slot": "喜好与厌恶",
        })
    return out


def load_personamem_pairs(n: int = 10) -> list[dict]:
    """从 PersonaMem-v2 preference_updates 构建偏好更新对。

    preference_updates 是 dict{pref_desc: update_detail}。
    取旧/新偏好作为 claim_a/claim_b，relation=temporal_update。
    """
    files = sorted(PERSONAMEM_DIR.glob("*.json"))
    random.shuffle(files)
    out = []
    for f in files:
        if len(out) >= n:
            break
        try:
            data = json.load(open(f))
            persona_key = list(data.keys())[0]
            persona = data[persona_key]
            updates = persona.get("preference_updates", {})
            name = persona.get("name", f.name)
            for pref, detail in updates.items():
                if len(out) >= n:
                    break
                # detail 可能是 dict{old,new} 或 str
                if isinstance(detail, dict):
                    old = detail.get("old_preference") or detail.get("old") or pref
                    new = detail.get("new_preference") or detail.get("new") or pref
                elif isinstance(detail, str):
                    old, new = pref, detail
                else:
                    continue
                if not old or not new or old == new:
                    continue
                out.append({
                    "sample_id": f"pmem_{f.stem}_{len(out)}",
                    "group_id": name,
                    "source": "PersonaMem-v2",
                    "split": "dev_schema_check",
                    "language": "en",
                    "relation_gold": "temporal_update",
                    "claim_a": old,
                    "claim_b": new,
                    "slot": "喜好与厌恶",
                })
        except Exception:
            continue
    return out


def build_boundary_pairs() -> list[dict]:
    """阈值边界对：相似度应在 0.95 附近的细微差异。

    VoiceMem 代码注释提到：
    「喜欢手冲咖啡」↔「偏好手冲咖啡」是 0.964（该合并）
    「讨厌吃饭吧唧嘴」↔「讨厌被打断」是 0.934（不该合并）
    复现这两组作为边界。
    """
    return [
        {
            "sample_id": "boundary_merge_0",
            "group_id": "boundary",
            "source": "synthetic_boundary",
            "split": "dev_boundary",
            "language": "zh",
            "relation_gold": "equivalent",
            "claim_a": "喜欢手冲咖啡",
            "claim_b": "偏好手冲咖啡",
            "slot": "喜好与厌恶",
            "note": "VoiceMem comments: sim=0.964, should merge at 0.95",
        },
        {
            "sample_id": "boundary_nomerge_0",
            "group_id": "boundary",
            "source": "synthetic_boundary",
            "split": "dev_boundary",
            "language": "zh",
            "relation_gold": "contextual_difference",
            "claim_a": "讨厌吃饭吧唧嘴",
            "claim_b": "讨厌被打断",
            "slot": "喜好与厌恶",
            "note": "VoiceMem comments: sim=0.934, should NOT merge at 0.95",
        },
    ]


def build_control_pairs() -> list[dict]:
    """控制维度：不同用户、不同类别、空输入。"""
    return [
        {
            "sample_id": "control_diffuser_0",
            "group_id": "control_diffuser",
            "source": "synthetic_control",
            "split": "dev_control",
            "language": "zh",
            "relation_gold": "unrelated",
            "claim_a": "喜欢手冲咖啡",
            "claim_b": "喜欢手冲咖啡",
            "slot": "喜好与厌恶",
            "user_a": "user_diff_A",
            "user_b": "user_diff_B",
            "note": "Same claim, different user_id — must NOT merge across users",
        },
        {
            "sample_id": "control_diffslot_0",
            "group_id": "control_diffslot",
            "source": "synthetic_control",
            "split": "dev_control",
            "language": "zh",
            "relation_gold": "contextual_difference",
            "claim_a": "喜欢手冲咖啡",
            "claim_b": "喜欢手冲咖啡",
            "slot_a": "喜好与厌恶",
            "slot_b": "情绪",
            "note": "Same claim, different slot — must NOT merge across slots",
        },
        {
            "sample_id": "control_empty_0",
            "group_id": "control_empty",
            "source": "synthetic_control",
            "split": "dev_control",
            "language": "zh",
            "relation_gold": "ambiguous",
            "claim_a": "",
            "claim_b": "喜欢手冲咖啡",
            "slot": "喜好与厌恶",
            "note": "Empty input A — normalize_claim returns '', add() should return ''",
        },
    ]


def build_all() -> list[dict]:
    pairs = []
    pairs += load_ocnli_contradiction(10)
    pairs += load_pawsx_paraphrase(10)
    pairs += load_pawsx_nonparaphrase(10)
    pairs += load_personamem_pairs(10)
    pairs += build_boundary_pairs()
    pairs += build_control_pairs()
    return pairs


if __name__ == "__main__":
    pairs = build_all()
    print(f"Total pairs: {len(pairs)}")
    by_source = {}
    for p in pairs:
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
    print("By source:", by_source)
    by_rel = {}
    for p in pairs:
        by_rel[p["relation_gold"]] = by_rel.get(p["relation_gold"], 0) + 1
    print("By relation:", by_rel)
    out = Path("/quark_speech_nas_zjk/users/wangyuhan/experiments/voicemem-memory-update-eval/manifests/pairs_smoke.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Written: {out}")
