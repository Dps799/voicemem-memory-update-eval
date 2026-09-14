import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_n1 import derive,read_rows
from run_n1_dev import execute,strategy_input,cluster_metric

def embed(text):return [1.]+[0.]*383

class N1AutoTests(unittest.TestCase):
    def test_shared_text_component_cannot_cross_splits(self):
        rows=[dict(sample_id='a',source='PAWS-X-zh',group_id='g1',split='dev',claim_a='Ａ  B',claim_b='c'),
              dict(sample_id='b',source='PAWS-X-zh',group_id='g2',split='test',claim_a='a b',claim_b='d')]
        out,audit=derive(rows,[])
        self.assertEqual({r['split'] for r in out},{'dev'})
        self.assertEqual(len({r['group_id'] for r in out}),1)
        self.assertEqual(len(audit['cross_split_components_repaired']),1)

    def test_smoke_shared_text_excludes_whole_component(self):
        rows=[dict(sample_id='a',source='OCNLI',group_id='g1',split='dev',claim_a='seen',claim_b='middle'),
              dict(sample_id='b',source='OCNLI',group_id='g2',split='test',claim_a='middle',claim_b='other')]
        out,audit=derive(rows,[dict(source='OCNLI',claim_a='seen',claim_b='x')])
        self.assertEqual(out,[])
        self.assertEqual(len(audit['excluded_components'][0]['sample_ids']),2)

    def test_label_flip_cannot_affect_strategy_input_or_decision(self):
        p=dict(claim_a='likes tea',claim_b='dislikes tea',slot='喜好与厌恶',relation_gold='equivalent')
        first=strategy_input(p);p['relation_gold']='contradiction_same_scope'
        self.assertEqual(first,strategy_input(p))
        a=execute(first,{'name':'B0_original','threshold':.95},'ab',embed)
        b=execute(strategy_input(p),{'name':'B0_original','threshold':.95},'ab',embed)
        self.assertEqual(a['stored_state'],b['stored_state'])
        with self.assertRaises(ValueError):execute(p,{'name':'B0_original','threshold':.95},'ab',embed)

    def test_encoder_failure_is_not_successful_rejection(self):
        def broken(text):raise RuntimeError('encoder failure')
        with self.assertRaises(RuntimeError):
            execute(dict(claim_a='a',claim_b='b',slot='喜好与厌恶'),{'name':'B0_original','threshold':.95},'ab',broken)

    def test_raw_ablation_preserves_original_and_restores_normalizer(self):
        p=dict(claim_a='用户喜欢散步。',claim_b='喜欢散步',slot='喜好与厌恶')
        raw=execute(p,{'name':'B0_raw_ablation','threshold':.95},'ab',embed,raw=True)
        regular=execute(p,{'name':'B0_original','threshold':.95},'ab',embed)
        self.assertEqual(raw['encoded_claim_a'],p['claim_a'])
        self.assertEqual(regular['encoded_claim_a'],'喜欢散步')

    def test_same_merge_decision_can_hide_order_state_difference(self):
        p=dict(claim_a='old state',claim_b='new state',slot='喜好与厌恶')
        a=execute(p,{'name':'B0_original','threshold':.95},'ab',embed)
        b=execute(p,{'name':'B0_original','threshold':.95},'ba',embed)
        self.assertEqual(a['merged'],b['merged'])
        self.assertNotEqual(a['stored_state'],b['stored_state'])
        self.assertTrue(a['evidence_link_integrity'] and b['evidence_link_integrity'])

    def test_group_macro_does_not_weight_large_groups_more(self):
        rows=[dict(group_id='g1',merged=True,relation_gold='temporal_update',error=None)]*9
        rows += [dict(group_id='g2',merged=False,relation_gold='temporal_update',error=None)]
        m=cluster_metric(rows,{'temporal_update'},100)
        self.assertEqual(m['rate'],.9)
        self.assertEqual(m['group_macro_rate'],.5)

    def test_effective_manifest_has_no_shared_generic_text_cross_split(self):
        from prepare_n1 import text_key
        root=Path(__file__).resolve().parents[1]
        rows=read_rows(root/'manifests/n1_effective_manifest.jsonl');seen={}
        for r in rows:
            if r['source']=='PersonaMem-v2':continue
            for k in ('claim_a','claim_b'):
                key=(r['source'],text_key(r[k]))
                self.assertEqual(seen.get(key,r['split']),r['split'])
                seen[key]=r['split']

if __name__=='__main__':unittest.main()
