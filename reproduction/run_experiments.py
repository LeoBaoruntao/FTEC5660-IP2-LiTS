"""Run all six prespecified trials and the upstream evaluator. No result filtering."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_job(iterations, seed):
    name = f'iter{iterations}_seed{seed}'
    dest = ROOT / 'reproduction/runs' / name
    if dest.exists():
        raise RuntimeError(f'{dest} exists; refusing to mix or overwrite trials')
    console = ROOT / 'reproduction/runs' / (name + '.log')
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
               PYTHONPATH=str(ROOT / 'demos') + os.pathsep + str(ROOT),
               PYTHONHASHSEED=str(seed))
    with console.open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, 'reproduction/run_one.py', '--iterations', str(iterations),
                        '--seed', str(seed), '--output', str(dest)], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
        subprocess.run([sys.executable, '-m', 'lits.cli.eval_search', '--result_dir', str(dest)],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=300)
    if not (dest/'eval_results.json').is_file():
        raise RuntimeError('Missing evaluation artifact despite upstream exit code zero')
    return name


if __name__ == '__main__':
    (ROOT/'reproduction/runs').mkdir(exist_ok=True)
    # Interleave conditions; at most three independent API clients are active.
    jobs = [(it, seed) for seed in (41, 42, 43) for it in (30, 10)]
    failed = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_job, *job): job for job in jobs}
        for future in as_completed(futures):
            try:
                print('Completed', future.result(), flush=True)
            except Exception as exc:
                failed.append(futures[future])
                print('FAILED', futures[future], type(exc).__name__, flush=True)
    if failed:
        raise SystemExit(f'Failed trials: {failed}; inspect logs rather than silently dropping them.')
