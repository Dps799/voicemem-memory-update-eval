"""Derive leakage-safe groups from committed N0 inputs; do not require annotation."""
import hashlib
import json
import re
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def text_key(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text)).strip().casefold()

def derive(pairs, smoke):
    parent=list(range(len(pairs)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j): parent[find(j)]=find(i)
    seen={}
    for i,r in enumerate(pairs):
        keys=[('group',r['source'],str(r['group_id']))]
        if r['source']=='PersonaMem-v2':
            keys.append(('persona',r['source'],r.get('persona_file',str(r['group_id']))))
        else:
            keys += [('text',r['source'],text_key(r[k])) for k in ('claim_a','claim_b')]
        for key in keys:
            if key in seen:union(i,seen[key])
            else:seen[key]=i
    components=defaultdict(list)
    for i,r in enumerate(pairs):components[find(i)].append(r)
    smoke_text={(r['source'],text_key(r[k])) for r in smoke if r['source']!='PersonaMem-v2' for k in ('claim_a','claim_b')}
    out=[];cross=[];excluded=[]
    for component in components.values():
        ids=sorted(r['sample_id'] for r in component)
        group='component_'+hashlib.sha256('\n'.join(ids).encode()).hexdigest()[:16]
        if any((r['source'],text_key(r[k])) in smoke_text for r in component for k in ('claim_a','claim_b')):
            excluded.append({'sample_ids':ids,'reason':'shared source text with smoke'});continue
        splits={r['split'] for r in component}
        # Any component touching development stays entirely in development.
        split='dev' if 'dev' in splits else 'test'
        if len(splits)>1:cross.append({'group_id':group,'sample_ids':ids,'assigned_split':split})
        for r in component:
            out.append(dict(r,original_group_id=r['group_id'],original_split=r['split'],group_id=group,split=split,
                            label_provenance='dataset_proxy_no_human_adjudication',
                            normalized_relation_status='not_semantically_relabelled'))
    out.sort(key=lambda r:r['sample_id'])
    return out,{'cross_split_components_repaired':cross,'excluded_components':excluded,
                'n_pairs':len(out),'by_split':dict(Counter(r['split'] for r in out)),
                'n_groups':len({r['group_id'] for r in out}),
                'policy':'group + NFKC/whitespace/case-normalized shared original text; dev wins; no score-based selection',
                'scope':'components within committed sample pool; full source-pool duplicates not yet audited'}

def main():
    pairs=read_rows(ROOT/'manifests/formal_manifest.jsonl')
    out,audit=derive(pairs,read_rows(ROOT/'manifests/pairs_smoke.jsonl'))
    manifest=ROOT/'manifests/n1_effective_manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
    audit['input_sha256']=hashlib.sha256((ROOT/'manifests/formal_manifest.jsonl').read_bytes()).hexdigest()
    audit['effective_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    (ROOT/'manifests/n1_split_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
