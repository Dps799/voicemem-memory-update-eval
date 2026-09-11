"""阶段 B：固定抽取结果的存储与检索检查。

跳过真实抽取器，把审核过的新旧判断按时间写入，检查：
  1. 新偏好是否被正确保存（作为新 trait 还是合并到旧 trait）
  2. 旧判断是否仍作为现状返回（检索旧 claim 时是否返回旧 trait）
  3. 同一条证据是否被错误挂到旧判断下

使用 B0 原实现（0.95）和 B1 精确匹配对照。
CPU 即可，不需要生成模型 API。

RQ2 诊断：即使没有误合并，旧判断是否仍被检索并影响回答？
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e5_callback import (  # noqa: E402
    TraitStore, Evidence, SLOTS, normalize_claim,
    real_embed, real_embed_query, cos_sim, precompute_embeddings,
)
import traits_store  # noqa: E402
from run_phase_a import B1ExactStore  # noqa: E402

SEED = 20260911
PERSONAMEM_DIR = Path(
    "/root/.cache/huggingface/hub/"
    "datasets--bowen-upenn--PersonaMem-v2/snapshots/"
    "ed956dea41521fc4499acbc63f966e0fd3c053ba/data/raw_data"
)
OUT_DIR = Path(__file__).parent.parent / "reports"


def load_preference_updates(n_personas: int = 10) -> list[dict]:
    """从 PersonaMem-v2 加载偏好更新对（old → new）。

    每个 persona 取 1-2 个 preference_update，确保跨 persona 独立。
    """
    files = sorted(PERSONAMEM_DIR.glob("*.json"))
    random.seed(SEED)
    random.shuffle(files)

    episodes = []
    for f in files:
        if len(episodes) >= n_personas:
            break
        try:
            data = json.load(open(f))
            persona_key = list(data.keys())[0]
            persona = data[persona_key]
            updates = persona.get("preference_updates", {})
            name = persona.get("name", f.stem)
            convs = persona.get("conversations", {})

            for pref, detail in updates.items():
                if len([e for e in episodes if e["persona"] == name]) >= 2:
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
                # 找一段对话作为 evidence
                conv_text = ""
                for conv_type in ["chat_message", "personal_email", "trouble_consult"]:
                    cl = convs.get(conv_type, [])
                    if isinstance(cl, list) and cl:
                        conv_text = str(cl[0])[:200] if isinstance(cl[0], str) else str(cl[0].get("content", ""))[:200]
                        break
                episodes.append({
                    "episode_id": f"b_{f.stem}_{len(episodes)}",
                    "persona": name,
                    "old_preference": old,
                    "new_preference": new,
                    "slot": "喜好与厌恶",
                    "evidence_old": f"用户说：{old}",
                    "evidence_new": f"用户改口说：{new}",
                    "conv_excerpt": conv_text,
                })
        except Exception:
            continue
    return episodes


def run_episode(store: TraitStore, ep: dict) -> dict:
    """按时间写入 old → new，然后检索检查。

    检查项：
    1. new 是否保存为新 trait（未合并到 old）
    2. 检索 new claim 时是否返回 new trait
    3. 检索 old claim 时是否仍返回 old trait（旧判断残留）
    4. new 的 evidence 是否挂到 old trait 下（证据错挂）
    """
    user_id = "user_b"
    slot = ep["slot"]
    old = ep["old_preference"]
    new = ep["new_preference"]

    norm_old = normalize_claim(old)
    norm_new = normalize_claim(new)

    # T1: 写旧偏好
    ev_old = Evidence(quote=ep["evidence_old"], emotion="calm", at="2026-09-01T10:00:00+00:00")
    id_old = store.add(user_id, slot, old, ev_old)

    # T2: 写新偏好（偏好变化）
    ev_new = Evidence(quote=ep["evidence_new"], emotion="decisive", at="2026-09-11T14:00:00+00:00")
    id_new = store.add(user_id, slot, new, ev_new)

    # 检查 1: 新偏好是否保存为新 trait
    merged = (id_old != "" and id_new != "" and id_old == id_new)
    new_saved = (id_new != "")

    # 检查 2: 检索 new claim（query 前缀）
    search_results_new = store.search_scored(user_id, new, top_k=5)
    top_result_new = search_results_new[0] if search_results_new else None
    top_trait_new = top_result_new[0] if top_result_new else None
    top_sim_new = top_result_new[1] if top_result_new else None

    # 检查 3: 检索 old claim
    search_results_old = store.search_scored(user_id, old, top_k=5)
    top_result_old = search_results_old[0] if search_results_old else None
    top_trait_old = top_result_old[0] if top_result_old else None
    top_sim_old = top_result_old[1] if top_result_old else None

    # old trait 是否仍出现在检索结果中（旧判断残留）
    old_trait_still_retrievable = any(
        t.id == id_old for t, _ in search_results_new
    ) if id_old else False

    # old trait 在检索 new 时的排名
    old_rank_when_search_new = None
    for i, (t, s) in enumerate(search_results_new):
        if t.id == id_old:
            old_rank_when_search_new = i + 1
            break

    # 检查 4: evidence 归属
    # 如果 merged，new 的 evidence 挂到 old trait 下（同一 ID）
    # 如果 not merged，new 的 evidence 应挂到 new trait 下
    evidence_misattributed = False
    if not merged and id_new:
        # 检查 new trait 的 evidence 是否包含 ev_new
        new_trait = None
        with store._conn() as c:
            evs = c.execute("SELECT * FROM rb_evidence WHERE trait_id=?", (id_new,)).fetchall()
            new_trait_evs = [e["quote"] for e in evs]
        # old trait 的 evidence 是否错误包含了 new 的 evidence
        if id_old:
            with store._conn() as c:
                old_evs = c.execute("SELECT * FROM rb_evidence WHERE trait_id=?", (id_old,)).fetchall()
                old_trait_evs = [e["quote"] for e in old_evs]
            # 如果 new evidence 出现在 old trait 下，是错挂
            evidence_misattributed = ep["evidence_new"] in old_trait_evs

    # sim between old and new
    if norm_old and norm_new:
        sim = cos_sim(real_embed(norm_old), real_embed(norm_new))
    else:
        sim = None

    return {
        "episode_id": ep["episode_id"],
        "persona": ep["persona"],
        "old_preference": old,
        "new_preference": new,
        "normalized_old": norm_old,
        "normalized_new": norm_new,
        "similarity_old_new": round(sim, 4) if sim else None,
        "id_old": id_old,
        "id_new": id_new,
        "merged": merged,
        "new_saved_as_new_trait": (not merged) and new_saved,
        "new_saved": new_saved,
        # 检索 new 时的结果
        "search_new_top_claim": top_trait_new.claim if top_trait_new else None,
        "search_new_top_sim": round(top_sim_new, 4) if top_sim_new else None,
        "search_new_top_is_new": (top_trait_new.id == id_new) if (top_trait_new and id_new) else None,
        "search_new_top_is_old": (top_trait_new.id == id_old) if (top_trait_new and id_old) else None,
        # 旧判断残留
        "old_trait_still_retrievable_when_search_new": old_trait_still_retrievable,
        "old_rank_when_search_new": old_rank_when_search_new,
        # 检索 old 时的结果
        "search_old_top_claim": top_trait_old.claim if top_trait_old else None,
        "search_old_top_is_old": (top_trait_old.id == id_old) if (top_trait_old and id_old) else None,
        # 证据错挂
        "evidence_misattributed": evidence_misattributed,
        # 风险判定
        "stale_risk": old_trait_still_retrievable and (top_trait_new and top_trait_new.id == id_old),
        "error": None,
    }


def main():
    episodes = load_preference_updates(n_personas=10)
    print(f"Loaded {len(episodes)} preference-update episodes from {len(set(e['persona'] for e in episodes))} personas")

    # 预计算 embedding
    all_texts = []
    for ep in episodes:
        all_texts.extend([ep["old_preference"], ep["new_preference"],
                          normalize_claim(ep["old_preference"]), normalize_claim(ep["new_preference"])])
    print(f"Precomputing {len(set(t for t in all_texts if t))} unique embeddings...")
    precompute_embeddings(all_texts)
    print("Cached.")

    results = []

    for policy_name, make_store_fn in [
        ("B0_original_095", lambda db: (setattr(traits_store, 'MERGE_THRESHOLD', 0.95) or TraitStore(db, real_embed))),
        ("B1_exact_string", lambda db: B1ExactStore(db, real_embed)),
    ]:
        print(f"\n>>> Policy {policy_name}")
        store_cls = B1ExactStore if "B1" in policy_name else TraitStore
        if "B0" in policy_name:
            traits_store.MERGE_THRESHOLD = 0.95

        for ep in episodes:
            db = tempfile.mktemp(suffix=".db")
            if "B1" in policy_name:
                store = B1ExactStore(db, real_embed)
            else:
                traits_store.MERGE_THRESHOLD = 0.95
                store = TraitStore(db, real_embed)
            store._policy_name = policy_name
            try:
                res = run_episode(store, ep)
                res["policy"] = policy_name
                results.append(res)
            except Exception as e:
                results.append({
                    "episode_id": ep["episode_id"],
                    "policy": policy_name,
                    "error": str(e),
                })

    # 写结果
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "replay_results.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWritten {len(results)} results to {out}")

    summarize(results)


def summarize(results):
    print("\n" + "=" * 60)
    print("PHASE B: STORAGE & RETRIEVAL CHECK (fixed extraction)")
    print("=" * 60)

    for pol in ["B0_original_095", "B1_exact_string"]:
        pol_res = [r for r in results if r.get("policy") == pol and r.get("error") is None]
        if not pol_res:
            continue
        n = len(pol_res)
        merged = [r for r in pol_res if r["merged"]]
        new_saved = [r for r in pol_res if r.get("new_saved_as_new_trait")]
        old_still_retrievable = [r for r in pol_res if r.get("old_trait_still_retrievable_when_search_new")]
        stale_risk = [r for r in pol_res if r.get("stale_risk")]
        ev_misattr = [r for r in pol_res if r.get("evidence_misattributed")]
        search_new_returns_old = [r for r in pol_res if r.get("search_new_top_is_old")]

        print(f"\n{pol} (n={n} episodes):")
        print(f"  merged (old==new): {len(merged)}/{n} ({len(merged)/n*100:.1f}%)")
        print(f"  new saved as new trait: {len(new_saved)}/{n}")
        print(f"  old trait still retrievable when search new: {len(old_still_retrievable)}/{n}")
        print(f"  search new returns OLD as top: {len(search_new_returns_old)}/{n}")
        print(f"  stale_risk (old top + old retrievable): {len(stale_risk)}/{n}")
        print(f"  evidence misattributed: {len(ev_misattr)}/{n}")

        # 逐条显示合并的
        if merged:
            print(f"\n  MERGED EPISODES (preference update lost):")
            for r in merged:
                print(f"    {r['episode_id']}: old='{r['normalized_old']}' new='{r['normalized_new']}' sim={r['similarity_old_new']}")

        # 逐条显示旧判断残留
        if old_still_retrievable:
            print(f"\n  OLD TRAIT STILL RETRIEVABLE (search new returns old in top-5):")
            for r in old_still_retrievable:
                rank = r.get("old_rank_when_search_new")
                print(f"    {r['episode_id']}: old rank={rank} top_claim='{r.get('search_new_top_claim')}'")

    print("\n" + "=" * 60)
    print("RQ2 DIAGNOSIS:")
    print("  If old trait is still retrievable when searching for the new preference,")
    print("  the system may present stale (old) judgment as current — even without merge.")
    print("  This is the storage-layer mechanism RQ2 targets.")
    print("  Answer-layer impact (does the model use old?) requires Phase C (blocked).")
    print("=" * 60)


if __name__ == "__main__":
    main()
