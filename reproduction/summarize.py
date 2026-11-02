"""Derive every reported number from immutable raw run files; no API calls."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['word_accuracy', 'exact_accuracy', 'calls', 'input_tokens', 'output_tokens', 'elapsed_seconds']


def summarize():
    rows, provenance = [], {}
    for seed in (41, 42, 43):
        for iterations in (30, 10):
            d = ROOT/'reproduction/runs'/f'iter{iterations}_seed{seed}'
            manifest = json.loads((d/'run_manifest.json').read_text(encoding='utf-8'))
            evaluation = json.loads((d/'eval_results.json').read_text(encoding='utf-8'))
            usage = [json.loads(x) for x in (d/'inferencelogger.log').read_text(encoding='utf-8').splitlines() if x.strip()]
            assert manifest['exit_code'] == 0
            assert manifest['puzzle_indices'] == [2]
            assert len(evaluation['examples']) == 1
            assert 0 <= evaluation['soft_accuracy'] <= 1
            rows.append(dict(condition='baseline' if iterations==30 else 'reduced_budget',
                             iterations=iterations, seed=seed,
                             word_accuracy=evaluation['soft_accuracy'],
                             exact_accuracy=evaluation['accuracy'],
                             calls=len(usage),
                             input_tokens=sum(x['input_tokens'] for x in usage),
                             output_tokens=sum(x['output_tokens'] for x in usage),
                             elapsed_seconds=manifest['elapsed_seconds']))
            for name in ['config.json', 'run_manifest.json', 'eval_results.json', 'inferencelogger.log']:
                path=d/name
                provenance[str(path.relative_to(ROOT)).replace('\\','/')]=hashlib.sha256(path.read_bytes()).hexdigest()
    groups={}
    for condition in ['baseline', 'reduced_budget']:
        subset=[r for r in rows if r['condition']==condition]
        groups[condition]={field:{'mean':statistics.mean(r[field] for r in subset),
                                 'sample_variance':statistics.variance(r[field] for r in subset),
                                 'sample_sd':statistics.stdev(r[field] for r in subset)} for field in FIELDS}
    out=ROOT/'reproduction/analysis'
    out.mkdir(exist_ok=True)
    with (out/'trials.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    data={'archived_reference':{'word_accuracy':.6,'exact_accuracy':0,'calls':133,
                               'input_tokens':55630,'output_tokens':590},
          'trials':rows,'groups':groups,'sha256':provenance}
    (out/'summary.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(groups,indent=2))


if __name__=='__main__': summarize()
