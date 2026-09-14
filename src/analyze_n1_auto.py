"""Post-hoc structural diagnostics, explicitly not semantic relabelling or new selection."""
import argparse
import json
from pathlib import Path
from prepare_n1 import read_rows
from run_n1_dev import NON_EQ,cluster_metric
ROOT=Path(__file__).resolve().parents[1]

def analyze(output):
    pairs={r['sample_id']:r for r in read_rows(ROOT/'manifests/n1_effective_manifest.jsonl') if r['split']=='dev'}
    rows=read_rows(output/'pair_results.jsonl')
    b0={r['sample_id']:r for r in rows if r['policy']=='B0_original' and r['direction']=='ab' and r['error'] is None}
    cases=[]
    for sid,r in b0.items():
        p=pairs[sid]
        if r['relation_gold'] not in NON_EQ:continue
        if p['claim_a']==p['claim_b']:
            kind='raw_identical_but_non_equivalent_dataset_label'
        elif r['encoded_claim_a']==r['encoded_claim_b']:
            kind='raw_distinct_collapsed_to_identical_normalized_text'
        else:continue
        cases.append({'sample_id':sid,'source':r['source'],'kind':kind,
                      'raw_a':p['claim_a'],'raw_b':p['claim_b'],'encoded':r['encoded_claim_a']})
    excluded={c['sample_id'] for c in cases}
    sensitivity={}
    for policy in sorted({r['policy'] for r in rows if r['normalization_mode']=='upstream'}):
        pol=[r for r in rows if r['policy']==policy and r['direction']=='ab' and r['sample_id'] not in excluded]
        sensitivity[policy]={source:cluster_metric([r for r in pol if r['source']==source],NON_EQ)
                             for source in ('OCNLI','PAWS-X-zh')}
    out={'scope':'post-hoc structural sensitivity only; primary metrics and selection unchanged',
         'human_annotation_used':False,'semantic_relabelling_performed':False,'cases':cases,
         'n_raw_identical_label_conflicts':sum(c['kind'].startswith('raw_identical') for c in cases),
         'n_distinct_raw_normalization_collisions':sum(c['kind'].startswith('raw_distinct') for c in cases),
         'sensitivity_excluding_identical_normalized_non_equivalent_pairs':sensitivity,
         'b0_temporal_merged_groups':len({r['group_id'] for r in b0.values() if r['relation_gold']=='temporal_update' and r['merged']})}
    (output/'automatic_label_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);args=p.parse_args();analyze(args.run_dir)
