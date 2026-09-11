"""阶段 A 最小执行器：直接调用固定版本 TraitStore.add() + 真实 E5 编码回调。

每一对、每个策略创建独立临时数据库：写 A，再写 B，记录两次返回 ID、
规范化前后 claim、实际编码字符串、余弦分数和证据所属 ID。清空后交换顺序。

策略:
  B0 原实现   — MERGE_THRESHOLD=0.95（上游默认）
  B1 精确匹配 — MERGE_THRESHOLD=1.0（仅规范化后完全相同才合并）
  B2 阈值调整 — 0.97 / 0.99（开发集比较，这里都跑记录原始数据）
  B3 始终新增 — MERGE_THRESHOLD=2.0（永不合并）
  C1 随机控制 — 以 B0 合并率随机决定合并，种子固定

不自行重写余弦规则，直接 monkeypatch traits_store.MERGE_THRESHOLD。
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
import tempfile
import time
import uuid
from pathlib import Path

# 用本地模块
sys.path.insert(0, str(Path(__file__).parent))
from e5_callback import (  # noqa: E402
    TraitStore, Evidence, SLOTS, normalize_claim,
    real_embed, real_embed_query, cos_sim, embedding_hash, MERGE_THRESHOLD,
    precompute_embeddings,
)
import traits_store  # noqa: E402  for monkeypatching MERGE_THRESHOLD
from traits_store import _now  # noqa: E402

SEED = 20260911
PAIRS_FILE = Path(__file__).parent.parent / "manifests" / "pairs_smoke.jsonl"
OUT_DIR = Path(__file__).parent.parent / "reports"

POLICIES = [
    {"name": "B0_original", "threshold": 0.95},
    {"name": "B1_exact", "threshold": "string_match"},  # 真正的字符串精确匹配
    {"name": "B2_097", "threshold": 0.97},
    {"name": "B2_099", "threshold": 0.99},
    {"name": "B3_never", "threshold": 2.0},
]


class B1ExactStore(TraitStore):
    """B1 精确匹配：规范化后文本完全相同才合并（字符串比较，非余弦阈值=1.0）。

    仍调用原 add()（存储 embedding、evidence 逻辑不变），但 _find_similar
    改为查询 claim 字段是否完全相同，而非余弦相似度是否 >= 阈值。
    """
    _current_claim: str | None = None

    def add(self, user_id, slot, claim, ev):
        self._current_claim = normalize_claim(claim)
        return super().add(user_id, slot, claim, ev)

    def _find_similar(self, user_id, slot, vec):
        """字符串精确匹配：查同 user + 同 slot + claim 完全相同。"""
        claim = self._current_claim
        if not claim:
            return None
        with self._conn() as c:
            row = c.execute(
                "SELECT id FROM rb_traits "
                "WHERE user_id=? AND slot=? AND claim=?",
                (user_id, slot, claim),
            ).fetchone()
        return row["id"] if row else None


def make_store(db_path: str, threshold) -> TraitStore:
    """建独立 DB 的 TraitStore，monkeypatch 阈值。

    B1 用字符串精确匹配（B1ExactStore），其余用余弦阈值。
    """
    if threshold == "string_match":
        store = B1ExactStore(db_path, real_embed)
    else:
        traits_store.MERGE_THRESHOLD = threshold
        store = TraitStore(db_path, real_embed)
    store._effective_threshold = threshold
    return store


def run_pair(store: TraitStore, pair: dict, direction: str) -> dict:
    """写 A 再写 B（或 B 再写 A），记录两次 ID 和相似度。

    direction: "ab" = 先 A 后 B; "ba" = 先 B 后 A
    """
    user_id_a = pair.get("user_a", "user_test")
    user_id_b = pair.get("user_b", user_id_a)  # 跨用户控制：默认同 user
    # 如果是控制维度的跨用户对，第二次用不同 user_id
    cross_user = pair.get("user_b") is not None and pair.get("user_b") != pair.get("user_a")
    slot = pair.get("slot", pair.get("slot_a", "喜好与厌恶"))
    slot_a = pair.get("slot_a", pair.get("slot", "喜好与厌恶"))
    slot_b = pair.get("slot_b", pair.get("slot", "喜好与厌恶"))
    claim_a = pair["claim_a"]
    claim_b = pair["claim_b"]

    norm_a = normalize_claim(claim_a)
    norm_b = normalize_claim(claim_b)

    ev_a = Evidence(quote=claim_a, emotion="test", at="2026-09-11T00:00:00+00:00")
    ev_b = Evidence(quote=claim_b, emotion="test", at="2026-09-11T00:00:01+00:00")

    first_claim, first_slot, first_ev = claim_a, slot_a, ev_a
    second_claim, second_slot, second_ev = claim_b, slot_b, ev_b
    first_user, second_user = user_id_a, user_id_b
    if direction == "ba":
        first_claim, second_claim = second_claim, first_claim
        first_slot, second_slot = second_slot, first_slot
        first_ev, second_ev = second_ev, first_ev
        first_user, second_user = second_user, first_user

    t0 = time.time()
    id1 = store.add(first_user, first_slot, first_claim, first_ev)
    id2 = store.add(second_user, second_slot, second_claim, second_ev)
    latency_ms = (time.time() - t0) * 1000

    # 两个 claim 的余弦相似度
    if norm_a and norm_b:
        vec_a = real_embed(norm_a)
        vec_b = real_embed(norm_b)
        sim = cos_sim(vec_a, vec_b)
    else:
        sim = None

    merged = (id1 != "" and id2 != "" and id1 == id2)
    # 如果用了不同 user 或不同 slot，merged 应为 False
    cross_slot = slot_a != slot_b

    return {
        "sample_id": pair["sample_id"],
        "group_id": pair["group_id"],
        "source": pair["source"],
        "split": pair["split"],
        "language": pair["language"],
        "relation_gold": pair["relation_gold"],
        "policy": store._effective_threshold,
        "direction": direction,
        "raw_claim_a": claim_a,
        "raw_claim_b": claim_b,
        "normalized_claim_a": norm_a,
        "normalized_claim_b": norm_b,
        "slot": slot,
        "cross_user": cross_user,
        "cross_slot": cross_slot,
        "embedding_hash_a": embedding_hash(claim_a) if norm_a else None,
        "embedding_hash_b": embedding_hash(claim_b) if norm_b else None,
        "similarity": sim,
        "id_a": id1 if direction == "ab" else id2,
        "id_b": id2 if direction == "ab" else id1,
        "merged": merged,
        "merge_threshold": store._effective_threshold,
        "latency_ms": round(latency_ms, 2),
        "error": None,
    }


def run_c1_random_control(pairs: list[dict], b0_results: list[dict]) -> list[dict]:
    """C1 随机控制：同类别候选内，以开发集 B0 的合并率随机决定合并。

    种子固定。不改变合并数量以外的语义。
    """
    rng = random.Random(SEED)
    # B0 的合并率
    b0_merged = [r for r in b0_results if r["merged"]]
    b0_total = len(b0_results)
    merge_rate = len(b0_merged) / b0_total if b0_total else 0.0

    results = []
    for pair in pairs:
        user_id = pair.get("user_a", "user_test")
        slot = pair.get("slot", "喜好与厌恶")
        norm_a = normalize_claim(pair["claim_a"])
        norm_b = normalize_claim(pair["claim_b"])
        # 随机决定是否合并
        will_merge = rng.random() < merge_rate
        if will_merge and norm_a and norm_b:
            # 模拟合并：写 A，第二次也用 A 的 claim（强制合并）
            # 但这不公平 — C1 应该用相同 claim，随机决定是否复用
            # 正确做法：写 A，然后以概率 merge_rate 把 B 也写成 A（复用 ID）
            db = tempfile.mktemp(suffix=".db")
            traits_store.MERGE_THRESHOLD = 2.0  # 先不合并
            store = TraitStore(db, real_embed)
            store._effective_threshold = "C1_random"
            ev_a = Evidence(quote=pair["claim_a"], emotion="test")
            ev_b = Evidence(quote=pair["claim_b"], emotion="test")
            id1 = store.add(user_id, slot, pair["claim_a"], ev_a)
            if will_merge:
                # 强制复用：用 A 的 claim 再写一次
                id2 = store.add(user_id, slot, pair["claim_a"], ev_b)
            else:
                id2 = store.add(user_id, slot, pair["claim_b"], ev_b)
            sim = cos_sim(real_embed(norm_a), real_embed(norm_b)) if norm_a and norm_b else None
            results.append({
                "sample_id": pair["sample_id"],
                "group_id": pair["group_id"],
                "source": pair["source"],
                "split": pair["split"],
                "language": pair["language"],
                "relation_gold": pair["relation_gold"],
                "policy": "C1_random",
                "direction": "ab",
                "normalized_claim_a": norm_a,
                "normalized_claim_b": norm_b,
                "similarity": sim,
                "id_a": id1,
                "id_b": id2,
                "merged": id1 == id2 and id1 != "",
                "merge_rate_b0": round(merge_rate, 4),
                "error": None,
            })
    return results


def main():
    pairs = [json.loads(l) for l in open(PAIRS_FILE) if l.strip()]
    print(f"Loaded {len(pairs)} pairs")

    # 预计算所有 claim 的 embedding（批量 encode，避免逐条调用）
    all_claims = []
    for p in pairs:
        all_claims.append(p["claim_a"])
        all_claims.append(p["claim_b"])
    # 也预计算 normalized 版本（TraitStore.add 内部会 normalize）
    for p in pairs:
        all_claims.append(normalize_claim(p["claim_a"]))
        all_claims.append(normalize_claim(p["claim_b"]))
    print(f"Precomputing embeddings for {len(set(c for c in all_claims if c))} unique claims...")
    precompute_embeddings(all_claims)
    print("Embeddings cached.")

    all_results = []

    for pol in POLICIES:
        print(f"\n>>> Policy {pol['name']} (threshold={pol['threshold']})")
        for pair in pairs:
            for direction in ["ab", "ba"]:
                # 不同用户/不同类别/空输入只跑 ab（控制维度不需要交换顺序）
                if pair["source"] == "synthetic_control" and direction == "ba":
                    continue
                db = tempfile.mktemp(suffix=".db")
                store = make_store(db, pol["threshold"])
                try:
                    res = run_pair(store, pair, direction)
                    res["policy"] = pol["name"]
                    all_results.append(res)
                except Exception as e:
                    all_results.append({
                        "sample_id": pair["sample_id"],
                        "policy": pol["name"],
                        "direction": direction,
                        "error": str(e),
                        "merged": False,
                    })

    # C1 随机控制
    b0_results = [r for r in all_results if r.get("policy") == "B0_original" and r.get("direction") == "ab"]
    print(f"\n>>> Policy C1_random (B0 merge_rate from {len(b0_results)} ab results)")
    c1_results = run_c1_random_control(pairs, b0_results)
    all_results.extend(c1_results)

    # 写 pair_results.jsonl
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pair_results.jsonl"
    with open(out, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWritten {len(all_results)} results to {out}")

    # 打印摘要
    summarize(all_results)


def summarize(results: list[dict]):
    """计算并打印阶段 A 核心指标。"""
    print("\n" + "=" * 60)
    print("PHASE A SMOKE TEST SUMMARY")
    print("=" * 60)

    for pol_name in ["B0_original", "B1_exact", "B2_097", "B2_099", "B3_never", "C1_random"]:
        pol_res = [r for r in results if r.get("policy") == pol_name and r.get("direction") == "ab" and r.get("error") is None]
        if not pol_res:
            continue
        # 只看同 user、同 slot、非空的对（排除控制维度）
        eligible = [r for r in pol_res
                    if not r.get("cross_user") and not r.get("cross_slot")
                    and r.get("normalized_claim_a") and r.get("normalized_claim_b")]
        if not eligible:
            continue
        merged = [r for r in eligible if r["merged"]]
        # 误合并：非等价且标签确定的输入中被复用为同一判断
        nonequiv_merged = [r for r in merged
                          if r["relation_gold"] in ("contradiction_same_scope", "nonparaphrase_high_overlap", "contextual_difference", "unrelated")]
        # 同义合并召回
        equiv_total = [r for r in eligible if r["relation_gold"] == "equivalent"]
        equiv_merged = [r for r in equiv_total if r["merged"]]

        fmr = len(nonequiv_merged) / len(eligible) if eligible else None
        emr = len(equiv_merged) / len(equiv_total) if equiv_total else None

        print(f"\n{pol_name} (n={len(eligible)} eligible same-user same-slot non-empty):")
        print(f"  merged={len(merged)}  FMR={fmr}  EMR={emr if emr is not None else 'N/A'} (equiv n={len(equiv_total)})")
        if nonequiv_merged:
            print(f"  FALSE MERGES: {[(r['sample_id'], r['relation_gold'], round(r['similarity'],4)) for r in nonequiv_merged]}")

    # 边界对
    print("\nBOUNDARY PAIRS (B0):")
    b0_boundary = [r for r in results if r.get("policy") == "B0_original"
                   and r.get("source") == "synthetic_boundary" and r.get("direction") == "ab"]
    for r in b0_boundary:
        print(f"  {r['sample_id']}: sim={round(r['similarity'],4) if r.get('similarity') else None} "
              f"merged={r['merged']} gold={r['relation_gold']}")


if __name__ == "__main__":
    main()
