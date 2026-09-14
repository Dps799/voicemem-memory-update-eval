"""Record local provenance and verify artifact consistency after successful replay."""
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from e5_callback import E5_MODEL, E5_REVISION, E5_SNAPSHOT
ROOT=Path(__file__).resolve().parents[1]

def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def main():
    a=read_rows(ROOT/'reports/pair_results.jsonl')
    b=read_rows(ROOT/'reports/replay_results.jsonl')
    legacy=read_rows(ROOT/'reports/archive/3302a96/pair_results.jsonl')
    index={(r['policy'],r['sample_id'],r['direction']):r for r in legacy if r['policy']!='C1_random'}
    compared=[r for r in a if (r['policy'],r['sample_id'],r['direction']) in index]
    changed=sum(r['merged']!=index[r['policy'],r['sample_id'],r['direction']]['merged'] for r in compared)
    assert all(r.get('error') is None for r in a+b), 'Execution errors present'
    assert len(a)==570 and len(b)==160, 'Incomplete fixed smoke replay'
    assert all(not(r['old_claim_top1'] and r['new_claim_top1']) for r in b), 'Aliased content labels'
    assert all(r['evidence_link_integrity'] for r in b), 'Missing or invalid evidence link'
    for seed in [20260911,20260912,20260913]:
        rows=[r for r in a if r['policy']==f'C1_random_{seed}']
        assert len(rows)==45
        eligible=[r for r in rows if not r['cross_user'] and not r['cross_slot'] and r['normalized_claim_a'] and r['normalized_claim_b']]
        assert all(r['merged']==r['random_reuse_draw'] for r in eligible)
    paths=sorted(set(list((ROOT/'src').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+
                     list((ROOT/'manifests').glob('*.jsonl'))+list((ROOT/'upstream').rglob('*.py'))+
                     [ROOT/'requirements.txt',ROOT/'scripts/reproduce_smoke.sh']+
                     [ROOT/'reports'/p for p in ['REPORT.md','metrics.json','replay_metrics.json','pair_results.jsonl','replay_results.jsonl']]))
    out={'schema_version':'0.3','status':'cpu_smoke_a_b_completed',
         'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
         'base_commit':'3302a96','checkout_head_at_run':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'source_identification':'working-tree SHA256 below is authoritative for this pre-commit run',
         'upstream_commit':'a450911fc8cbb44c46d810aace2f3288bad287e4',
         'embedding_model':E5_MODEL,'embedding_revision':E5_REVISION,'embedding_path':E5_SNAPSHOT,
         'embedding_device':'cpu','embedding_dimensions':384,
         'actual_trait_write_prefix':'passage: ','actual_trait_query_prefix':'passage: ',
         'query_prefix_alternative_used':False,'embedding_normalize':True,
         'retrieval_scope':'profile-only unchanged upstream AST helpers; no full VoiceMem pipeline',
         'platform':platform.platform(),'python':platform.python_version(),
         'dependencies':{p:version(p) for p in ['torch','sentence-transformers','transformers','huggingface-hub','numpy']},
         'environment_overrides':{k:v for k,v in os.environ.items() if k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','VOICEMEM_RB_TRAIT_MIN_SIM','VOICEMEM_RB_PROFILE_MAX','VOICEMEM_E5_PATH']},
         'generative_api_calls':0,'dataset_audit_rerun':False,
         'source_manifests':'committed smoke inputs; replay claims recovered from archived results',
         'artifact_validation':{'phase_a_rows':len(a),'phase_b_rows':len(b),'legacy_core_rows_compared':len(compared),'legacy_core_merge_decisions_changed':changed,
                                'phase_b_unique_episodes':len({r['episode_id'] for r in b}),'phase_b_unique_personas':len({r['persona_id'] for r in b}),
                                'execution_errors':0,'all_evidence_links_valid':True},
         'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT/'reports/preflight.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out['artifact_validation'],indent=2))

if __name__=='__main__':main()
