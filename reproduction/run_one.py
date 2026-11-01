"""Launch the original LiTS CLI with a fixed, recorded experiment configuration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'demos'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iterations', type=int, choices=[2, 10, 30], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    os.chdir(ROOT)
    from dotenv import load_dotenv
    import numpy as np
    load_dotenv(ROOT / '.env')
    if not args.dry_run and not os.getenv('DEEPSEEK_API_KEY'):
        raise SystemExit('Set DEEPSEEK_API_KEY in .env or the environment.')
    os.environ['OPENAI_API_KEY'] = os.getenv('DEEPSEEK_API_KEY', 'dry-run')
    os.environ['OPENAI_API_BASE'] = 'https://api.deepseek.com'
    os.environ['LITS_NETWORK_RECOVERY_TIMEOUT'] = '60'
    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = ROOT / 'demos/crosswords/data/mini0505.json'
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Output directory already contains files; choose a new one.')
    argv = ['lits-search', '--include', 'lits_benchmark.crosswords',
            '--dataset', 'crosswords', '--transition', 'crosswords',
            '--policy-model', 'openai/deepseek-v4-flash',
            '--search-arg', f'n_iters={args.iterations}', 'n_actions=3',
            'max_steps=10', 'roll_out_steps=10', 'n_action_for_simulate=3',
            'n_confidence=3', 'force_terminating_on_depth_limit=false',
            'terminate_on_terminal_node=true', 'terminate_on_first_solution=false',
            'add_continuation=false', '--component-arg', 'max_new_tokens=1024',
            '--dataset-arg', f'data_file={dataset}',
            '--var', 'offset=0', 'limit=1', '-o', str(out)]
    if args.dry_run:
        argv.append('--dry-run')
    sys.argv = argv
    start = time.monotonic()
    from lits.cli.search import main as search_main
    code = search_main()
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(iterations=args.iterations, local_seed=args.seed,
                    elapsed_seconds=time.monotonic()-start, exit_code=code,
                    model='deepseek-v4-flash', provider='DeepSeek',
                    base_url='https://api.deepseek.com', thinking='disabled',
                    dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    puzzle_indices=[0], argv=argv, python=sys.version)
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
