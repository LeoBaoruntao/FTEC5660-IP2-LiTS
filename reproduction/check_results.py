"""Offline checks on the complete experimental evidence package."""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[1]


def check():
    archived=json.loads((ROOT/'demos/demo_results/terminal_nodes/terminal_nodes_0.json').read_text(encoding='utf-8'))
    dataset=json.loads((ROOT/'demos/crosswords/data/mini0505.json').read_text(encoding='utf-8'))
    assert ''.join(archived['query'].splitlines()[:5])==''.join(dataset[2][1])
    hashes=set()
    for seed in (41,42,43):
        pair=[]
        for it in (30,10):
            p=ROOT/'reproduction/runs'/f'iter{it}_seed{seed}'
            m=json.loads((p/'run_manifest.json').read_text(encoding='utf-8'))
            e=json.loads((p/'eval_results.json').read_text(encoding='utf-8'))
            c=json.loads((p/'config.json').read_text(encoding='utf-8'))
            assert e['dataset_name']=='crosswords' and len(e['examples'])==1
            assert e['examples'][0]['idx']==2
            assert m['iterations']==it and m['local_seed']==seed
            assert m['exit_code']==0 and m['puzzle_indices']==[2]
            assert c['search_args']['n_iters']==it
            assert c['policy_model_name']=='openai/deepseek-v4-flash'
            hashes.add(m['dataset_sha256'])
            params=dict(c['search_args']); params.pop('n_iters')
            pair.append(params)
            log=(p/'execution.log').read_text(encoding='utf-8')
            assert 'Traceback (most recent call last)' not in log
            assert 0<=e['soft_accuracy']<=1
        assert pair[0]==pair[1], 'More than one search parameter changed'
    assert len(hashes)==1
    print('PASS: six evaluable runs, identical dataset, one changed search parameter.')


if __name__=='__main__': check()
