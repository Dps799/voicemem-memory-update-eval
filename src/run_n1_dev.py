"""Execute N1 on development only using dataset-proxy labels; no annotation gate.

No test inference is implemented here. Gold is attached only after store execution.
Outputs are immutable run directories; all label-dependent results are conditional on
original dataset labels, not an independently verified relation after normalization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import random
import subprocess
import tempfile
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
from e5_callback import TraitStore, Evidence, normalize_claim, real_embed, precompute_embeddings, cos_sim, E5_MODEL, E5_REVISION
from run_phase_a import B1ExactStore, RandomMergeStore
import traits_store
from aggregate_metrics import wilson_ci
from prepare_n1 import read_rows

ROOT=Path(__file__).resolve().parents[1]
NON_EQ={'contradiction_same_scope','nonparaphrase_high_overlap','contextual_difference','unrelated'}

class CheckedVectors:
    def _vec(self,text):
        vec=super()._vec(text)
        if vec is None or vec.shape!=(384,) or not np.isfinite(vec).all():
            raise RuntimeError('Embedding failure or incompatible vector; cannot count as non-merge')
        return vec

class CheckedTraitStore(CheckedVectors,TraitStore): pass
class CheckedExactStore(CheckedVectors,B1ExactStore): pass
class CheckedRandomStore(CheckedVectors,RandomMergeStore): pass

def strategy_input(pair):
    return {key:pair[key] for key in ('claim_a','claim_b','slot')}

def execute(inputs,policy,direction,embed=real_embed,raw=False,random_reuse=False):
    """Strategy input contains no gold or split metadata. Snapshot uses A/B evidence labels."""
    if set(inputs)!={'claim_a','claim_b','slot'}:raise ValueError('Unexpected strategy input fields')
    normalizer=(lambda s:s) if raw else normalize_claim
    claims={k:inputs[f'claim_{k.lower()}'] for k in ('A','B')}
    ev={k:Evidence(quote=claims[k],emotion='test',at=f'2026-09-14T00:00:0{i}+00:00') for i,k in enumerate(('A','B'))}
    context=patch.object(traits_store,'normalize_claim',normalizer) if raw else nullcontext()
    with context,tempfile.TemporaryDirectory() as directory:
        db=str(Path(directory)/'traits.db')
        if policy['name']=='B1_exact':store=CheckedExactStore(db,embed)
        elif policy['name'].startswith('C1_random'):
            store=CheckedRandomStore(db,embed,random_reuse)
        else:
            traits_store.MERGE_THRESHOLD=policy['threshold']
            store=CheckedTraitStore(db,embed)
        ids={};start=perf_counter()
        for key in (('A','B') if direction=='ab' else ('B','A')):
            ids[key]=store.add('user_n1',inputs['slot'],claims[key],ev[key])
            if not ids[key]:raise RuntimeError('Empty write ID; invalid sample or input')
        elapsed=(perf_counter()-start)*1000
        with store._conn() as c:
            records=c.execute('SELECT id,claim FROM rb_traits').fetchall()
            states=[]
            integrity=True
            for record in records:
                evidence=c.execute('SELECT quote,created_at FROM rb_evidence WHERE trait_id=? ORDER BY created_at',(record['id'],)).fetchall()
                sources=[]
                for e in evidence:
                    matches=[k for k in ('A','B') if e['quote']==ev[k].quote and e['created_at']==ev[k].at]
                    if len(matches)!=1:integrity=False
                    sources.extend(matches)
                    for k in matches:integrity &= ids[k]==record['id']
                states.append({'claim':record['claim'],'evidence_sources':sources})
            integrity &= sorted(k for s in states for k in s['evidence_sources'])==['A','B']
        na,nb=normalizer(claims['A']),normalizer(claims['B'])
        sim=cos_sim(embed(na),embed(nb))
        return {'merged':ids['A']==ids['B'],'id_a':ids['A'],'id_b':ids['B'],
                'encoded_claim_a':na,'encoded_claim_b':nb,'similarity':sim,
                'stored_state':sorted(states,key=lambda s:(s['claim'],s['evidence_sources'])),
                'evidence_link_integrity':bool(integrity),'write_latency_ms_cached_embedding':elapsed,
                'normalization_mode':'raw_identity' if raw else 'upstream',
                'error':None}

def cluster_metric(rows,relation_set,replicates=2000):
    chosen=[r for r in rows if r['relation_gold'] in relation_set and r['error'] is None]
    groups=defaultdict(list)
    for r in chosen:groups[r['group_id']].append(float(r['merged']))
    n=len(chosen);k=sum(r['merged'] for r in chosen)
    if not n:return {'n':0,'k':0,'rate':None,'group_macro_rate':None,'group_macro_bootstrap_95':None,'n_groups':0}
    means=np.array([np.mean(v) for _,v in sorted(groups.items())]);rng=np.random.default_rng(20260914)
    boot=means[rng.integers(0,len(means),size=(replicates,len(means)))].mean(axis=1)
    return {'n':n,'k':k,'rate':k/n,'wilson_95_descriptive_only':wilson_ci(k,n),'n_groups':len(groups),
            'group_macro_rate':float(means.mean()),'group_macro_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}

def summarize(rows):
    ab=[r for r in rows if r['direction']=='ab' and r['normalization_mode']=='upstream']
    out={}
    for policy in sorted({r['policy'] for r in ab}):
        pol=[r for r in ab if r['policy']==policy]
        strata={}
        for source in sorted({r['source'] for r in pol})+['ALL_DIAGNOSTIC_MIX_ONLY']:
            rs=[r for r in pol if r['source']==source] if source!='ALL_DIAGNOSTIC_MIX_ONLY' else pol
            valid=[r for r in rs if r['error'] is None]
            total_merges=sum(r['merged'] for r in valid)
            bad=sum(r['merged'] for r in valid if r['relation_gold'] in NON_EQ|{'temporal_update'})
            strata[source]={'n_attempted':len(rs),'n_errors':len(rs)-len(valid),
                'FMR':cluster_metric(valid,NON_EQ),'EMR':cluster_metric(valid,{'equivalent'}),
                'temporal_merge_rate':cluster_metric(valid,{'temporal_update'}),
                'merged_error_ratio_dataset_proxy':{'k':bad,'n':total_merges,'rate':bad/total_merges if total_merges else None}}
        out[policy]=strata
    paired={}
    baseline={r['sample_id']:r for r in ab if r['policy']=='B0_original' and r['error'] is None}
    for policy in out:
        if policy=='B0_original':continue
        contrasts={}
        for source in sorted({r['source'] for r in ab}):
            group_diffs=defaultdict(list)
            for r in ab:
                if r['policy']!=policy or r['source']!=source or r['error'] is not None:continue
                base=baseline.get(r['sample_id'])
                if base is None or r['relation_gold']=='equivalent':continue
                group_diffs[r['group_id']].append(float(r['merged'])-float(base['merged']))
            if group_diffs:
                arr=np.array([np.mean(v) for _,v in sorted(group_diffs.items())]);rng=np.random.default_rng(20260914)
                boot=arr[rng.integers(0,len(arr),size=(2000,len(arr)))].mean(axis=1)
                contrasts[source]={'outcome':'non-equivalent/temporal merge, lower is better','n_groups':len(arr),
                                   'macro_difference_pp':100*float(arr.mean()),'paired_group_bootstrap_95_pp':(100*np.quantile(boot,[.025,.975])).tolist()}
        paired[policy]=contrasts
    return {'label_scope':'original_dataset_proxy_no_human_adjudication','policies_by_source':out,'paired_vs_b0':paired,
            'interval_scope':'development descriptive; grouped resampling; not confirmatory significance tests'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(ROOT/'run_config.json'));parser.add_argument('--run-id',default=None)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    all_pairs=read_rows(ROOT/config['sampling']['effective_manifest'])
    pairs=[r for r in all_pairs if r['split']=='dev']
    if not pairs:raise ValueError('No development samples')
    run_id=args.run_id or datetime.now(timezone.utc).strftime('n1_dev_%Y%m%dT%H%M%SZ')
    if Path(run_id).name!=run_id:raise ValueError('run-id must be a single directory name')
    output=ROOT/'reports/n1_dev'/run_id;output.mkdir(parents=True,exist_ok=False)
    (output/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    start=perf_counter();texts=[r[k] for r in pairs for k in ('claim_a','claim_b')]
    precompute_embeddings(texts+[normalize_claim(t) for t in texts]);precompute_seconds=perf_counter()-start
    rows=[];normalization=[]
    deterministic=[p for p in config['n1_policies'] if not p['name'].startswith('C1_')]
    def run(pair,policy,direction,raw=False,draw=False):
        metadata={k:pair[k] for k in ('sample_id','group_id','source','split','relation_gold')}
        try:result=execute(strategy_input(pair),policy,direction,raw=raw,random_reuse=draw)
        except Exception as exc:
            result={'error':f'{type(exc).__name__}: {exc}','merged':None,'normalization_mode':'raw_identity' if raw else 'upstream'}
        row=dict(result,**metadata,policy=policy['name'],direction=direction,label_provenance='dataset_proxy_no_human_adjudication')
        rows.append(row);return row
    for policy in deterministic:
        for pair in pairs:
            for direction in config['n1_directions']:run(pair,policy,direction)
        print('Completed',policy['name'],flush=True)
    b0=[r for r in rows if r['policy']=='B0_original' and r['direction']=='ab' and r['error'] is None]
    if len(b0)!=len(pairs):raise RuntimeError('B0 errors; refuse biased C1 calibration')
    rate=sum(r['merged'] for r in b0)/len(b0)
    for policy in [p for p in config['n1_policies'] if p['name'].startswith('C1_')]:
        rng=random.Random(policy['seed'])
        for pair in pairs:
            draw=rng.random()<rate
            for direction in config['n1_c1_directions']:
                row=run(pair,policy,direction,draw=draw);row.update(random_reuse_draw=draw,calibration_rate=rate)
    raw_policy={'name':'B0_raw_ablation','threshold':.95}
    for pair in pairs:
        for direction in ('ab','ba'):run(pair,raw_policy,direction,raw=True)
        a,b=pair['claim_a'],pair['claim_b'];na,nb=normalize_claim(a),normalize_claim(b)
        normalization.append({'sample_id':pair['sample_id'],'source':pair['source'],
            'text_changed':a!=na or b!=nb,'raw_distinct_normalized_equal':a!=b and na==nb,
            'a_length_ratio':len(na)/max(1,len(a)),'b_length_ratio':len(nb)/max(1,len(b)),
            'semantic_damage_status':'not_inferred_from_text_difference'})
    (output/'pair_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (output/'normalization_audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in normalization))
    metrics=summarize(rows)
    upstream={(r['sample_id'],r['policy'],r['direction']):r for r in rows if r['normalization_mode']=='upstream'}
    order={}
    for policy in deterministic:
        counts=Counter()
        for pair in pairs:
            a=upstream[pair['sample_id'],policy['name'],'ab'];b=upstream[pair['sample_id'],policy['name'],'ba']
            if a['error'] or b['error']:counts['errors']+=1;continue
            counts['n']+=1;counts['merge_decision_different']+=a['merged']!=b['merged'];counts['stored_state_different']+=a['stored_state']!=b['stored_state']
        order[policy['name']]=dict(counts)
    raw=[r for r in rows if r['policy']=='B0_raw_ablation' and r['direction']=='ab' and r['error'] is None]
    metrics['raw_ablation_by_source']={}
    for source in sorted({r['source'] for r in raw}):
        rs=[r for r in raw if r['source']==source]
        metrics['raw_ablation_by_source'][source]={'FMR':cluster_metric(rs,NON_EQ),'EMR':cluster_metric(rs,{'equivalent'}),
            'temporal_merge_rate':cluster_metric(rs,{'temporal_update'}),
            'merge_decisions_changed':sum(r['merged']!=upstream[r['sample_id'],'B0_original','ab']['merged'] for r in rs)}
    metrics['order_state_comparison']=order
    metrics['normalization_text_changed']=sum(r['text_changed'] for r in normalization)
    metrics['normalization_collisions']=sum(r['raw_distinct_normalized_equal'] for r in normalization)
    metrics['n_pairs']=len(pairs);metrics['n_records']=len(rows);metrics['n_errors']=sum(r['error'] is not None for r in rows)
    metrics['n_evidence_link_errors']=sum(not r.get('evidence_link_integrity',False) for r in rows if r['error'] is None)
    pooled=metrics['policies_by_source'];b0metric=pooled['B0_original']['ALL_DIAGNOSTIC_MIX_ONLY']
    candidates=[]
    for name,threshold in [('B0_original',.95),('B2_097',.97),('B2_099',.99)]:
        m=pooled[name]['ALL_DIAGNOSTIC_MIX_ONLY']
        if m['EMR']['rate']>=b0metric['EMR']['rate']-.05:candidates.append((m['FMR']['rate'],threshold,name))
    chosen=min(candidates)
    frozen={'status':'dev_selection_only_test_not_run','selected_b2_threshold':chosen[1],'selected_b2_policy':chosen[2],
            'criterion':'original pre-specified pooled dev non-temporal FMR with EMR >= B0 - 5pp; lower threshold tie break',
            'c1_rate':rate,'c1_calibration_n':len(b0),'manifest_sha256':hashlib.sha256((ROOT/config['sampling']['effective_manifest']).read_bytes()).hexdigest(),
            'all_later_test_configs_must_be_frozen_before_test_execution':True}
    (output/'selection.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n')
    (output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    files=list((ROOT/'src').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+list((ROOT/'upstream').rglob('*.py'))+[ROOT/config['sampling']['effective_manifest'],Path(args.config)]+list(output.glob('*.json*'))
    provenance={'status':'completed' if not metrics['n_errors'] else 'completed_with_errors','scope':'N1 development only; dataset-proxy labels',
        'head_at_run':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'timestamp':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),'python':platform.python_version(),
        'dependencies':{p:version(p) for p in ('torch','sentence-transformers','transformers','numpy')},
        'model':E5_MODEL,'revision':E5_REVISION,'device':'cpu','write_and_query_prefix':'passage: ',
        'embedding_precompute_seconds':precompute_seconds,'total_wall_seconds':perf_counter()-start,
        'test_inference_count':0,'generative_api_calls':0,'input_counts':dict(Counter(r['split'] for r in all_pairs)),
        'expected_records':len(pairs)*(len(deterministic)*len(config['n1_directions'])+3*len(config['n1_c1_directions'])+2),
        'actual_records':len(rows),'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (output/'preflight.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print('Output:',output);print(json.dumps(frozen,ensure_ascii=False));print('rows',len(rows),'errors',metrics['n_errors'])
    if metrics['n_errors']:raise SystemExit(1)

if __name__=='__main__':main()
