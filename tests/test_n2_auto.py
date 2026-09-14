import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_n2_dev import pack_hits,filter_oracle_claim,make_episodes

class Tokenizer:
    def encode(self,text,add_special_tokens=False):return list(text)

def hit(text,identity='id'):
    return SimpleNamespace(content=text,metadata={'trait_id':identity,'claim':text})

class N2AutomaticTests(unittest.TestCase):
    def test_budget_counts_separator_and_never_splits_claim(self):
        a,b,c=hit('否定句'),hit('保留时间'),hit('短')
        selected,text,tokens,dropped=pack_hits([a,b,c],5,Tokenizer())
        self.assertEqual(text,'否定句\n短')
        self.assertEqual(tokens,5)
        self.assertEqual(selected,[a,c])
        self.assertEqual(len(dropped),1)

    def test_zero_budget_keeps_no_nonempty_claim(self):
        selected,text,tokens,dropped=pack_hits([hit('abc')],0,Tokenizer())
        self.assertEqual((selected,text,tokens),([], '',0))
        self.assertEqual(len(dropped),1)

    def test_oracle_cannot_delete_background_just_because_id_is_aliased(self):
        background=hit('talks about music','old_id')
        actual_old=hit('likes old music','another_id')
        selected=filter_oracle_claim([background,actual_old],'current','likes old music')
        self.assertEqual(selected,[background])
        self.assertEqual(filter_oracle_claim([background,actual_old],'historical','likes old music'),[background,actual_old])

    def test_probes_and_background_do_not_depend_on_future_new_label(self):
        pairs=[dict(sample_id=str(i),group_id=str(i),persona_file=str(i),split='dev',source='PersonaMem-v2',
                    claim_a=f'Likes music topic number {i} alpha'+chr(65+i//26)+chr(65+i%26),claim_b='new') for i in range(60)]
        # Add held-out sentinel: it must not affect constructed development episodes.
        a=make_episodes(pairs)
        for p in pairs:p['claim_b']='a different future state'
        pairs.append(dict(sample_id='secret',group_id='secret',persona_file='secret',split='test',source='PersonaMem-v2',claim_a='hidden',claim_b='hidden'))
        b=make_episodes(pairs)
        self.assertEqual([e['queries'] for e in a],[e['queries'] for e in b])
        self.assertEqual([e['background'] for e in a],[e['background'] for e in b])
        self.assertNotIn('secret',{e['episode_id'] for e in b})

if __name__=='__main__':unittest.main()
