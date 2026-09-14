"""CPU development replay: fixed claims, synthetic evidence/background, natural queries.

Measures DB association, claim identity and profile-directive exposure separately.
Does not claim full end-to-end retrieval or answer correctness.
"""
from __future__ import annotations
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from e5_callback import TraitStore, Evidence, normalize_claim, real_embed, precompute_embeddings
from run_phase_a import B1ExactStore
import traits_store
from upstream_retrieval import retrieve_profile, provenance

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'manifests/replay_smoke.jsonl'


def snapshot(store, user):
    with store._conn() as c:
        traits = [dict(r) for r in c.execute(
            'SELECT id,user_id,slot,claim,created_at,updated_at FROM rb_traits WHERE user_id=? ORDER BY id', (user,))]
        evidence = [dict(r) for r in c.execute(
            'SELECT trait_id,quote,created_at FROM rb_evidence WHERE user_id=? ORDER BY created_at,trait_id', (user,))]
    return {'traits': traits, 'evidence': evidence}


def run_episode(store, ep, background_count):
    user, slot = 'user_b', '喜好与厌恶'
    old, new = ep['old_preference'], ep['new_preference']
    # Only claims and evidence enter writes; queries/gold labels never enter write policy.
    for claim in ep['background_claims'][:background_count]:
        store.add(user, slot, claim, Evidence(quote=claim, at='2026-08-01T00:00:00+00:00'))
    old_quote, new_quote = f'用户说：{old}', f'用户改口说：{new}'
    old_id = store.add(user, slot, old, Evidence(quote=old_quote, at='2026-09-01T10:00:00+00:00'))
    new_id = store.add(user, slot, new, Evidence(quote=new_quote, at='2026-09-11T14:00:00+00:00'))
    if not old_id or not new_id:
        raise RuntimeError('Empty target write ID')
    state = snapshot(store, user)
    target_claim = next(t['claim'] for t in state['traits'] if t['id'] == new_id)
    links = [e for e in state['evidence'] if e['quote'] == new_quote]
    associated = any(e['trait_id'] == new_id for e in links)
    attached_old = any(e['trait_id'] == old_id for e in links)
    merged = old_id == new_id
    base = dict(episode_id=ep['episode_id'], persona_id=ep['persona_id'],
                split=ep['split'], old_preference=old, new_preference=new,
                background_requested=background_count, stored_trait_count=len(state['traits']),
                merged=merged, old_id=old_id, new_id=new_id,
                stored_new_claim=target_claim,
                new_claim_represented=target_claim == normalize_claim(new),
                new_evidence_present=bool(links),
                evidence_link_integrity=associated and len(links) == 1,
                new_evidence_attached_to_old_id=attached_old,
                semantic_conflict_candidate=attached_old and target_claim != normalize_claim(new),
                semantic_conflict_status='dataset-derived update; no independent human adjudication',
                db_snapshot=state)
    rows = []
    for query_kind, query in [('natural', ep['query']), ('claim_probe', new)]:
        raw = store.search_scored(user, query, top_k=5)
        for budget in (1, 3):
            hits, directive = retrieve_profile(store, user, query, budget)
            row = dict(base, query_kind=query_kind, query=query, profile_budget=budget,
                       raw_top5=[{'id': t.id, 'claim': t.claim, 'similarity': float(sim)} for t, sim in raw],
                       old_id_in_raw_top5=any(t.id == old_id for t, _ in raw),
                       profile_hits=[asdict(h) for h in hits], profile_directive=directive,
                       directive_characters=len(directive),
                       old_claim_in_directive=any(h.metadata['claim'] == normalize_claim(old) for h in hits),
                       new_claim_in_directive=any(h.metadata['claim'] == normalize_claim(new) for h in hits),
                       old_claim_top1=bool(hits) and hits[0].metadata['claim'] == normalize_claim(old),
                       new_claim_top1=bool(hits) and hits[0].metadata['claim'] == normalize_claim(new),
                       error=None)
            rows.append(row)
    return rows


def summarize(rows):
    groups = {}
    for r in rows:
        key = f"{r['policy']}|background={r['background_requested']}|query={r['query_kind']}|budget={r['profile_budget']}"
        groups.setdefault(key, []).append(r)
    metrics = {}
    for key, group in groups.items():
        fields = ['merged', 'new_claim_represented', 'evidence_link_integrity',
                  'new_evidence_attached_to_old_id', 'semantic_conflict_candidate',
                  'old_id_in_raw_top5', 'old_claim_in_directive', 'new_claim_in_directive',
                  'old_claim_top1', 'new_claim_top1']
        metrics[key] = {'n_episodes': len(group), 'n_personas': len({r['persona_id'] for r in group}),
                        'counts': {f: sum(bool(r[f]) for r in group) for f in fields},
                        'mean_directive_characters': sum(r['directive_characters'] for r in group)/len(group)}
    return {'scope': 'development diagnostic; correlated conditions, not independent samples',
            'n_rows': len(rows), 'n_episodes': len({r['episode_id'] for r in rows}),
            'n_personas': len({r['persona_id'] for r in rows}),
            'answer_quality_measured': False, 'retrieval': provenance(), 'conditions': metrics}


def main():
    episodes = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    texts = []
    for ep in episodes:
        texts += [ep['old_preference'], ep['new_preference'], ep['query']] + ep['background_claims']
    precompute_embeddings(texts + [normalize_claim(t) for t in texts])
    rows = []
    start = perf_counter()
    for policy, cls in [('B0_original_095', TraitStore), ('B1_exact_string', B1ExactStore)]:
        traits_store.MERGE_THRESHOLD = 0.95
        for ep in episodes:
            for background_count in (0, 20):
                with tempfile.TemporaryDirectory() as directory:
                    store = cls(str(Path(directory)/'traits.db'), real_embed)
                    for row in run_episode(store, ep, background_count):
                        rows.append(dict(row, policy=policy))
    output = ROOT/'reports'
    (output/'replay_results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    metrics = summarize(rows)
    metrics['wall_seconds_after_embedding_precompute'] = perf_counter() - start
    (output/'replay_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k,v in metrics.items() if k != 'conditions'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
