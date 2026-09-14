"""Automatic controlled N2 development probes. Not natural-dialogue or answer evaluation.

Topics are mechanically extracted from OLD text; backgrounds are synthetic discussion
interests, never presented as observed facts about real dataset personas. Dataset gold
is used only for the explicitly named oracle/evaluator, not ordinary write policies.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
import subprocess
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
import tempfile
from e5_callback import real_embed,normalize_claim,precompute_embeddings,Evidence,_model,E5_REVISION
from prepare_n1 import read_rows
from run_n1_dev import CheckedTraitStore,CheckedExactStore
from upstream_retrieval import HELPERS,provenance
import traits_store
ROOT=Path(__file__).resolve().parents[1]


def topic_from_old(text):
    words=re.findall(r"[A-Za-z]+",text.lower())
    stop=set('i a an the my her his their for and or to of in on over rather than not does enjoys enjoy likes like dislikes dislikes prefers prefer favors values supports admires keeps up with avidly strong high'.split())
    return ' '.join(w for w in words if w not in stop)[:100] or 'personal interests'


def pack_hits(hits,budget,tokenizer):
    """Greedy atomic upstream-rendered records; token count includes newline separators."""
    chosen=[];dropped=[]
    for hit in hits:
        proposed='\n'.join(h.content for h in chosen+[hit])
        if len(tokenizer.encode(proposed,add_special_tokens=False))<=budget:chosen.append(hit)
        else:dropped.append(hit.metadata['trait_id'])
    text='\n'.join(h.content for h in chosen)
    return chosen,text,len(tokenizer.encode(text,add_special_tokens=False)),dropped


def filter_oracle_claim(hits,kind,old_claim):
    """Mask the actual old claim, not an aliased ID that may represent a background."""
    return [h for h in hits if not(kind=='current' and h.metadata['claim']==normalize_claim(old_claim))]


def make_episodes(pairs):
    dev=[p for p in pairs if p['split']=='dev' and p['source']=='PersonaMem-v2']
    groups=sorted({p['group_id'] for p in dev});random.Random(20260914).shuffle(groups);selected=set(groups[:12])
    topics=sorted({topic_from_old(p['claim_a']) for p in dev})
    episodes=[]
    for p in sorted(dev,key=lambda p:p['sample_id']):
        if p['group_id'] not in selected:continue
        topic=topic_from_old(p['claim_a'])
        # Same-topic discussion interests compete lexically without asserting a new target preference.
        near=[f'{verb} {topic} {context}' for verb in ['Enjoys discussing','Reads articles about','Collects notes about','Asks friends about']
              for context in ['at a reading group','during weekend meetings','in online forums','at local workshops','on long train journeys']]
        other=[f'Enjoys discussing {t} at a reading group' for t in topics if t!=topic]
        other += [f'Reads articles about {t} during weekend meetings' for t in topics if t!=topic]
        random.Random(20260914).shuffle(other)
        background=list(dict.fromkeys(near+other))[:100]
        if len(background)<100:raise ValueError('Insufficient distinct synthetic backgrounds')
        episodes.append({'episode_id':p['sample_id'],'group_id':p['group_id'],'persona_file':p['persona_file'],
                         'split':'dev','old':p['claim_a'],'new':p['claim_b'],
                         'queries':{'current':f'What are my current preferences regarding {topic}?',
                                    'historical':f'What did I previously prefer regarding {topic}?'},
                         'background':background,'query_mode':'mechanical_old_topic_probe',
                         'background_mode':'synthetic_discussion_interests_not_observed_persona_facts',
                         'gold_mode':'dataset_update_proxy_not_semantically_adjudicated'})
    return episodes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default=None);args=parser.parse_args()
    name=args.run_id or datetime.now(timezone.utc).strftime('n2_dev_%Y%m%dT%H%M%SZ')
    if Path(name).name!=name:raise ValueError('single directory name required')
    output=ROOT/'reports/n2_dev'/name;output.mkdir(parents=True,exist_ok=False)
    episodes=make_episodes(read_rows(ROOT/'manifests/n1_effective_manifest.jsonl'))
    (output/'episode_manifest.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in episodes))
    texts=[t for e in episodes for t in [e['old'],e['new'],*e['queries'].values(),*e['background']]]
    start=perf_counter();precompute_embeddings(texts+[normalize_claim(t) for t in texts]);tokenizer=_model().tokenizer
    rows=[];states=[]
    for e in episodes:
        for background_n in (20,100):
            for policy,cls in [('B0_original',CheckedTraitStore),('B1_exact_diagnostic',CheckedExactStore)]:
                with tempfile.TemporaryDirectory() as directory:
                    traits_store.MERGE_THRESHOLD=.95;store=cls(str(Path(directory)/'db'),real_embed)
                    for text in e['background'][:background_n]:
                        store.add('user', '喜好与厌恶',text,Evidence(quote=text,at='2026-08-01T00:00:00+00:00'))
                    old_id=store.add('user','喜好与厌恶',e['old'],Evidence(quote='Previously: '+e['old'],at='2026-09-01T00:00:00+00:00'))
                    new_id=store.add('user','喜好与厌恶',e['new'],Evidence(quote='Now: '+e['new'],at='2026-09-14T00:00:00+00:00'))
                    with store._conn() as c:
                        db_state=[dict(r) for r in c.execute('SELECT id,claim FROM rb_traits')]
                    before=store.counts('user')
                    states.append({'episode_id':e['episode_id'],'policy':policy,'background_requested':background_n,
                                   'old_id':old_id,'new_id':new_id,'traits':db_state,'counts':before})
                    for kind,query in e['queries'].items():
                        raw=store.search_scored('user',query,top_k=4)
                        filtered=HELPERS['_rb_trait_hits'](store,'user',query)
                        filtered.sort(key=lambda h:h.priority,reverse=True)
                        for variant in ([policy,'O_filter_diagnostic'] if policy=='B0_original' else [policy]):
                            # Gold-only oracle: current state probes exclude old-ID; history never blindly masks it.
                            oracle_hits=filter_oracle_claim(filtered,kind,e['old']) if variant.startswith('O_') else filtered
                            quota_hits=HELPERS['_apply_source_quota'](oracle_hits)
                            for budget in (128,256,512):
                                hits,text,used,dropped=pack_hits(quota_hits,budget,tokenizer)
                                claims=[h.metadata['claim'] for h in hits]
                                rows.append({'episode_id':e['episode_id'],'group_id':e['group_id'],'split':'dev',
                                    'policy':variant,'query_kind':kind,'query':query,'background_requested':background_n,
                                    'stored_traits':before[0],'merged':old_id==new_id,
                                    'old_id_stored_claim':next(t['claim'] for t in db_state if t['id']==old_id),
                                    'new_id_stored_claim':next(t['claim'] for t in db_state if t['id']==new_id),
                                    'context_token_budget':budget,'context_tokens':used,
                                    'raw_top4':[{'claim':t.claim,'similarity':float(s)} for t,s in raw],
                                    'filtered_claims':[h.metadata['claim'] for h in filtered],
                                    'quota_claims':[h.metadata['claim'] for h in quota_hits],
                                    'presented_claims':claims,'directive':text,'budget_dropped_ids':dropped,
                                    'old_claim_visible':normalize_claim(e['old']) in claims,
                                    'new_claim_visible':normalize_claim(e['new']) in claims,
                                    'new_text_visible_verbatim':e['new'] in text,
                                    'old_claim_with_new_text_visible':normalize_claim(e['old']) in claims and e['new'] in text,
                                    'oracle_diagnostic':variant.startswith('O_'),'error':None})
                        if store.counts('user')!=before:raise RuntimeError('Probe wrote back to memory')
        print('Completed episode',e['episode_id'],flush=True)
    (output/'replay_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (output/'state_snapshots.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in states))
    buckets=defaultdict(list)
    for r in rows:buckets[f"{r['policy']}|{r['query_kind']}|background={r['background_requested']}|budget={r['context_token_budget']}"].append(r)
    summary={}
    for k,rs in buckets.items():
        summary[k]={'n':len(rs),'n_groups':len({r['group_id'] for r in rs}),
                    'counts':{f:sum(r[f] for r in rs) for f in ['old_claim_visible','new_claim_visible','new_text_visible_verbatim','old_claim_with_new_text_visible']},
                    'min_stored_traits':min(r['stored_traits'] for r in rs),'max_stored_traits':max(r['stored_traits'] for r in rs),
                    'mean_context_tokens':sum(r['context_tokens'] for r in rs)/len(rs),
                    'n_with_budget_drops':sum(bool(r['budget_dropped_ids']) for r in rs)}
    metrics={'scope':'automatic controlled development probes; claim visibility is not semantic answer correctness',
             'n_episodes':len(episodes),'n_personas':len({e['group_id'] for e in episodes}),'n_records':len(rows),
             'expected_records':len(episodes)*2*3*2*3,'conditions':summary,
             'answer_quality_measured':False,'human_annotation_used':False,'retrieval':provenance(),
             'tokenizer_revision':E5_REVISION,'generative_api_calls':0,'test_inference_count':0,
             'total_wall_seconds':perf_counter()-start}
    (output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    files=[ROOT/'src/run_n2_dev.py',ROOT/'src/run_n1_dev.py',ROOT/'src/upstream_retrieval.py',ROOT/'manifests/n1_effective_manifest.jsonl']+list(output.glob('*.json*'))
    (output/'preflight.json').write_text(json.dumps({'head_at_run':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
       'status':'completed','supersedes':'local n2_dev_20260914_auto used ID-based oracle; current version filters actual claim text','expected_records':metrics['expected_records'],'actual_records':len(rows)},indent=2)+'\n')
    print('Output:',output,'records',len(rows))

if __name__=='__main__':main()
