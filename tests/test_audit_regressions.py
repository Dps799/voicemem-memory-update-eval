"""Regression tests for the audited measurement failures; embeddings are test fixtures."""
import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import run_phase_a as a
import run_phase_b as b
import aggregate_metrics as aggregate
from upstream_retrieval import retrieve_profile, HELPERS
from e5_callback import TraitStore, Evidence


def constant_embed(text):
    return [1.] + [0.]*383

class Regressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name)/'db')
    def tearDown(self):
        self.tmp.cleanup()

    def test_random_reuse_preserves_incoming_evidence_and_user_slot_isolation(self):
        store = a.RandomMergeStore(self.db, constant_embed, True)
        old = store.add('u', '喜好与厌恶', 'old', Evidence(quote='old evidence'))
        new = store.add('u', '喜好与厌恶', 'new', Evidence(quote='new evidence'))
        self.assertEqual(old, new)
        self.assertNotEqual(old, store.add('other', '喜好与厌恶', 'new', Evidence(quote='other')))
        self.assertNotEqual(old, store.add('u', '情绪', 'new', Evidence(quote='other slot')))
        state = b.snapshot(store, 'u')
        self.assertTrue(any(e['trait_id']==old and e['quote']=='new evidence' for e in state['evidence']))
        store.merge = False
        self.assertNotEqual(old, store.add('u', '喜好与厌恶', 'old', Evidence(quote='not reused')))

    def test_c1_keeps_all_draws_and_controls(self):
        pairs=[dict(sample_id=str(i), group_id=str(i), source='fixture', split='dev_fixture',
                    language='en', relation_gold='equivalent',claim_a='old',claim_b='new') for i in range(20)]
        calibration=[dict(normalized_claim_a='a',normalized_claim_b='b',merged=i<10,split='dev_fixture') for i in range(20)]
        with patch.object(a,'real_embed',constant_embed):
            rows=a.run_c1_random_control(pairs,calibration)
        self.assertEqual(len(rows),60)
        self.assertTrue(any(r['merged'] for r in rows))
        self.assertTrue(any(not r['merged'] for r in rows))
        self.assertTrue(all(r['merged']==r['random_reuse_draw'] for r in rows))
        self.assertEqual({r['raw_claim_b'] for r in rows},{'new'})

    def test_exact_string_does_not_merge_identical_vectors(self):
        store=a.B1ExactStore(self.db,constant_embed)
        old=store.add('u','喜好与厌恶','old',Evidence(quote='a'))
        self.assertNotEqual(old,store.add('u','喜好与厌恶','new',Evidence(quote='b')))
        self.assertEqual(old,store.add('u','喜好与厌恶','old。',Evidence(quote='c')))

    def test_merged_old_claim_is_not_new_claim_and_evidence_is_checked(self):
        ep=dict(episode_id='e',persona_id='p',split='dev_fixture',old_preference='old preference',
                new_preference='new preference',query='what suits me?',background_claims=[])
        rows=b.run_episode(TraitStore(self.db,constant_embed),ep,0)
        for r in rows:
            self.assertTrue(r['merged'])
            self.assertTrue(r['evidence_link_integrity'])
            self.assertTrue(r['new_evidence_attached_to_old_id'])
            self.assertTrue(r['semantic_conflict_candidate'])
            self.assertTrue(r['old_claim_top1'])
            self.assertFalse(r['new_claim_top1'])
            self.assertIn('new preference',r['profile_directive'])

    def test_merged_error_ratio_includes_temporal_errors(self):
        rows=[]
        for i, rel in enumerate(['equivalent','contradiction_same_scope','temporal_update']):
            rows.append(dict(sample_id=str(i),policy='B0_original',direction='ab',
                             source='fixture',relation_gold=rel,normalized_claim_a='a',
                             normalized_claim_b='b',merged=True,error=None))
        source=Path(self.tmp.name)/'pairs.jsonl'
        out=Path(self.tmp.name)/'metrics.json'
        source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with patch.object(aggregate,'PAIR_RESULTS',source), patch.object(aggregate,'OUT',out):
            with contextlib.redirect_stdout(io.StringIO()): aggregate.main()
        m=json.loads(out.read_text())['policies']['B0_original']
        self.assertEqual(m['merged_error_ratio'],0.6667)
        self.assertEqual(m['FMR'],1.0)
        self.assertEqual(m['n_error_merged_including_temporal'],2)

    def test_upstream_filter_quota_render_and_budget(self):
        def embed(text):
            v=np.zeros(384);v[0 if text=='query' else 1]=1
            return v
        store=a.B1ExactStore(self.db,embed)
        store.add('u','喜好与厌恶','irrelevant',Evidence(quote='no'))
        self.assertEqual(retrieve_profile(store,'u','query')[0],[])
        # Identical vectors, distinct strings: exercise real upstream profile quota.
        store=a.B1ExactStore(str(Path(self.tmp.name)/'quota'),constant_embed)
        for i in range(6):store.add('u','喜好与厌恶',f'claim {i}',Evidence(quote=f'evidence {i}'))
        hits,directive=retrieve_profile(store,'u','query',budget=10)
        self.assertEqual(len(hits),3)
        self.assertEqual(len(retrieve_profile(store,'u','query',budget=1)[0]),1)
        self.assertIn('evidence',directive)

if __name__=='__main__': unittest.main()
