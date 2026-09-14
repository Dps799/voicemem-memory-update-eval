"""N0-T1: 正式 600 对采样与分组。

seed 20260914。排除已看过的 smoke 组（persona604、phase B 的 5 persona、
smoke 用过的 OCNLI prem_id 和 PAWS-X id）。按组 ~50:50 划分 dev/test。
不按 embedding 分数筛选主测试集。

产物:
  manifests/formal_manifest.jsonl   — 600 对，含 split
  manifests/split_manifest.jsonl     — 按组划分
  manifests/exclusions.json          — 排除记录
  manifests/coverage.json            — 实际覆盖
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

SEED = 20260914
DEV_FRACTION = 0.5

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
SMOKE_PAIRS = Path(__file__).parent.parent / "manifests" / "pairs_smoke.jsonl"
REPLAY_SMOKE = Path(__file__).parent.parent / "manifests" / "replay_smoke.jsonl"
OUT_DIR = Path(__file__).parent.parent / "manifests"

# 排除的 smoke persona 文件名前缀
EXCLUDED_PERSONA_FILES = {
    "raw_data_250820_224501_persona604",  # phase A temporal 全部来源
    "raw_data_250815_163935_persona126",  # phase B
    "raw_data_250815_163935_persona313",
    "raw_data_250818_011555_persona436",
    "raw_data_250819_225921_persona481",
    "raw_data_250826_131038_persona988",
}


def load_excluded_groups() -> dict:
    """从 smoke pairs 提取已看过的组。"""
    excluded = {"ocnli_prem_ids": set(), "pawsx_ids": set(), "persona_files": set(EXCLUDED_PERSONA_FILES)}
    if SMOKE_PAIRS.exists():
        for p in [json.loads(l) for l in SMOKE_PAIRS.read_text().splitlines() if l.strip()]:
            gid = p.get("group_id", "")
            if p.get("source") == "OCNLI":
                excluded["ocnli_prem_ids"].add(gid)
            elif p.get("source") == "PAWS-X-zh":
                # group_id is "pawsx_<id>"
                excluded["pawsx_ids"].add(gid.replace("pawsx_", ""))
            elif p.get("source") == "PersonaMem-v2":
                # sample_id like pmem_raw_data_..._persona604_0
                sid = p.get("sample_id", "")
                if "raw_data_" in sid:
                    # extract file prefix
                    parts = sid.split("_")
                    # reconstruct file name
                    idx = sid.index("raw_data_")
                    file_part = sid[idx:]
                    # remove trailing _N
                    last_us = file_part.rfind("_")
                    file_part = file_part[:last_us]
                    # this is tricky; use persona name instead
                    excluded["persona_files"].add(file_part)
    return excluded


def sample_ocnli(rng: random.Random, excluded: dict, target: int = 200) -> list[dict]:
    """采样 OCNLI contradiction 对，按 prem_id 分组。"""
    rows = [json.loads(l) for l in open(OCNLI_DEV) if l.strip()]
    # 排除 '-' 和非 contradiction
    candidates = [r for r in rows
                  if r.get("label") == "contradiction"
                  and r.get("prem_id") not in excluded["ocnli_prem_ids"]]
    # 按 prem_id 分组
    by_prem = defaultdict(list)
    for r in candidates:
        by_prem[r["prem_id"]].append(r)

    prem_ids = sorted(by_prem.keys())
    rng.shuffle(prem_ids)

    pairs = []
    for prem_id in prem_ids:
        if len(pairs) >= target:
            break
        group_rows = by_prem[prem_id]
        # 每个 prem_id 取 1 对（同 prem 的多对不独立）
        r = group_rows[0]
        pairs.append({
            "sample_id": f"ocnli_{r['id']}",
            "group_id": prem_id,
            "source": "OCNLI",
            "language": "zh",
            "relation_gold": "contradiction_same_scope",
            "claim_a": r["sentence1"],
            "claim_b": r["sentence2"],
            "slot": "喜好与厌恶",
            "prem_id": prem_id,
            "original_id": str(r["id"]),
        })
    return pairs[:target]


def sample_pawsx(rng: random.Random, excluded: dict, target_eq: int = 100, target_ne: int = 100) -> list[dict]:
    """采样 PAWS-X zh，按 id 分组。"""
    df = pd.read_parquet(PAWSX_ZH)
    excluded_ids = excluded["pawsx_ids"]

    eq_candidates = df[(df["label"] == 1) & ~df["id"].astype(str).isin(excluded_ids)]
    ne_candidates = df[(df["label"] == 0) & ~df["id"].astype(str).isin(excluded_ids)]

    eq_rows = eq_candidates.to_dict("records")
    ne_rows = ne_candidates.to_dict("records")
    rng.shuffle(eq_rows)
    rng.shuffle(ne_rows)

    pairs = []
    for r in eq_rows[:target_eq]:
        pairs.append({
            "sample_id": f"pawsx_eq_{r['id']}",
            "group_id": f"pawsx_{r['id']}",
            "source": "PAWS-X-zh",
            "language": "zh",
            "relation_gold": "equivalent",
            "claim_a": r["sentence1"],
            "claim_b": r["sentence2"],
            "slot": "喜好与厌恶",
            "original_id": str(r["id"]),
        })
    for r in ne_rows[:target_ne]:
        pairs.append({
            "sample_id": f"pawsx_ne_{r['id']}",
            "group_id": f"pawsx_{r['id']}",
            "source": "PAWS-X-zh",
            "language": "zh",
            "relation_gold": "nonparaphrase_high_overlap",
            "claim_a": r["sentence1"],
            "claim_b": r["sentence2"],
            "slot": "喜好与厌恶",
            "original_id": str(r["id"]),
        })
    return pairs


def sample_personamem(rng: random.Random, excluded: dict, target: int = 200,
                      min_personas: int = 100, max_per_persona: int = 2) -> list[dict]:
    """采样 PersonaMem-v2 preference_updates，按 persona 分组。"""
    files = sorted(PERSONAMEM_DIR.glob("*.json"))
    # 排除已用 persona 文件
    eligible_files = []
    for f in files:
        stem = f.stem  # e.g. raw_data_250820_224501_persona604
        is_excluded = any(stem.startswith(prefix) or prefix in stem for prefix in excluded["persona_files"])
        if not is_excluded:
            eligible_files.append(f)
    rng.shuffle(eligible_files)

    pairs = []
    persona_counts = defaultdict(int)
    for f in eligible_files:
        if len(pairs) >= target:
            break
        try:
            data = json.load(open(f))
            persona_key = list(data.keys())[0]
            persona = data[persona_key]
            name = persona.get("name", f.stem)
            updates = persona.get("preference_updates", {})

            if persona_counts[name] >= max_per_persona:
                continue

            for pref, detail in updates.items():
                if persona_counts[name] >= max_per_persona:
                    break
                if len(pairs) >= target:
                    break
                if isinstance(detail, dict):
                    old = detail.get("old_preference") or detail.get("old") or pref
                    new = detail.get("new_preference") or detail.get("new") or pref
                elif isinstance(detail, str):
                    old, new = pref, detail
                else:
                    continue
                if not old or not new or old == new:
                    continue
                pairs.append({
                    "sample_id": f"pmem_{f.stem}_{persona_counts[name]}",
                    "group_id": name,
                    "source": "PersonaMem-v2",
                    "language": "en",
                    "relation_gold": "temporal_update",
                    "claim_a": old,
                    "claim_b": new,
                    "slot": "喜好与厌恶",
                    "persona_id": name,
                    "persona_file": f.stem,
                })
                persona_counts[name] += 1
        except Exception:
            continue

    return pairs[:target]


def split_by_group(pairs: list[dict], dev_fraction: float, rng: random.Random) -> list[dict]:
    """按组 ~50:50 划分 dev/test。同组不跨 split。"""
    by_group = defaultdict(list)
    for p in pairs:
        by_group[p["group_id"]].append(p)

    groups = sorted(by_group.keys())
    rng.shuffle(groups)

    n_dev = int(len(groups) * dev_fraction)
    dev_groups = set(groups[:n_dev])

    for p in pairs:
        p["split"] = "dev" if p["group_id"] in dev_groups else "test"
    return pairs


def main():
    rng = random.Random(SEED)
    excluded = load_excluded_groups()

    print(f"Excluded OCNLI prem_ids: {len(excluded['ocnli_prem_ids'])}")
    print(f"Excluded PAWS-X ids: {len(excluded['pawsx_ids'])}")
    print(f"Excluded persona files: {len(excluded['persona_files'])}")

    ocnli = sample_ocnli(rng, excluded, 200)
    pawsx = sample_pawsx(rng, excluded, 100, 100)
    personamem = sample_personamem(rng, excluded, 200, 100, 2)

    all_pairs = ocnli + pawsx + personamem
    print(f"\nSampled: OCNLI={len(ocnli)}, PAWS-X={len(pawsx)}, PersonaMem={len(personamem)}, Total={len(all_pairs)}")

    # 按 source 分 dev/test
    # 计划要求各来源分别约 50:50
    by_source = defaultdict(list)
    for p in all_pairs:
        by_source[p["source"]].append(p)

    final_pairs = []
    for source, sps in by_source.items():
        rng.shuffle(sps)
        sps = split_by_group(sps, DEV_FRACTION, rng)
        final_pairs.extend(sps)

    rng.shuffle(final_pairs)

    # 统计
    by_source_split = defaultdict(lambda: defaultdict(int))
    by_relation = defaultdict(int)
    by_group_split = defaultdict(set)
    personas = set()
    for p in final_pairs:
        by_source_split[p["source"]][p["split"]] += 1
        by_relation[p["relation_gold"]] += 1
        by_group_split[p["split"]].add(p["group_id"])
        if p.get("persona_id"):
            personas.add(p["persona_id"])

    print(f"\n=== COVERAGE ===")
    print(f"Total pairs: {len(final_pairs)}")
    print(f"Unique groups: {len(set(p['group_id'] for p in final_pairs))}")
    print(f"Unique personas: {len(personas)}")
    for src, splits in sorted(by_source_split.items()):
        print(f"  {src}: dev={splits['dev']}, test={splits['test']}")
    print(f"Dev groups: {len(by_group_split['dev'])}, Test groups: {len(by_group_split['test'])}")
    print(f"By relation: {dict(by_relation)}")

    # 写产物
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    formal_path = OUT_DIR / "formal_manifest.jsonl"
    with open(formal_path, "w") as f:
        for p in final_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nWritten: {formal_path} ({len(final_pairs)} pairs)")

    # split_manifest: 按组列出
    split_data = []
    for split in ["dev", "test"]:
        groups = sorted(by_group_split[split])
        for g in groups:
            gpairs = [p for p in final_pairs if p["group_id"] == g and p["split"] == split]
            split_data.append({
                "split": split,
                "group_id": g,
                "source": gpairs[0]["source"],
                "n_pairs": len(gpairs),
                "sample_ids": [p["sample_id"] for p in gpairs],
            })
    split_path = OUT_DIR / "split_manifest.jsonl"
    with open(split_path, "w") as f:
        for s in split_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Written: {split_path} ({len(split_data)} groups)")

    # exclusions
    excl = {
        "seed": SEED,
        "excluded_ocnli_prem_ids": sorted(excluded["ocnli_prem_ids"]),
        "excluded_pawsx_ids": sorted(excluded["pawsx_ids"]),
        "excluded_persona_files": sorted(excluded["persona_files"]),
        "reason": "Prior smoke test groups excluded per NEXT_PHASE_PLAN §3.1: persona604, phase B personas, smoke OCNLI prem_ids and PAWS-X ids",
        "ocnli_total_contradiction": 900,
        "ocnli_excluded_by_smoke": len(excluded["ocnli_prem_ids"]),
        "pawsx_total_eq": 853,
        "pawsx_total_ne": 1147,
        "personamem_total_personas": 999,
        "personamem_excluded_personas": len(excluded["persona_files"]),
    }
    excl_path = OUT_DIR / "exclusions.json"
    excl_path.write_text(json.dumps(excl, ensure_ascii=False, indent=2))
    print(f"Written: {excl_path}")

    # coverage
    coverage = {
        "seed": SEED,
        "total_pairs": len(final_pairs),
        "target_pairs": 600,
        "target_met": len(final_pairs) >= 600,
        "unique_groups": len(set(p["group_id"] for p in final_pairs)),
        "unique_personas": len(personas),
        "min_personas_target": 100,
        "min_personas_met": len(personas) >= 100,
        "by_source": {src: dict(splits) for src, splits in by_source_split.items()},
        "by_relation": dict(by_relation),
        "dev_groups": len(by_group_split["dev"]),
        "test_groups": len(by_group_split["test"]),
        "dev_fraction_target": DEV_FRACTION,
        "no_group_crosses_split": all(
            g not in by_group_split["dev"] for g in by_group_split["test"]
        ),
    }
    cov_path = OUT_DIR / "coverage.json"
    cov_path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2))
    print(f"Written: {cov_path}")
    print(f"\nTarget met: {coverage['target_met']} (pairs), {coverage['min_personas_met']} (personas>=100)")


if __name__ == "__main__":
    main()
