"""Generate independent complete CoT responses per registered dataset example.

Run ``python -m lits.cli.cot --dataset math500 --include lits_benchmark.math_qa
--limit 100 --model openai/local-instruct --output-dir NEW_DIRECTORY``.
Uses the same dataset registry and --dataset-arg KEY=VALUE parser as tree search.
Examples require a question field; offset/limit retain original dataset indices.
Only the question is sent to the model, never the reference or worked solution.

The fresh output directory contains config.json, cohort.json, responses.jsonl,
and inference logs. Responses preserve raw text, original IDs, time, and token
usage; failures are recorded separately in the same stream. Generation uses no
answer extraction, correctness judging, policy objects, or search components.

Evaluate separately with ``python -m lits.cli.eval_cot --result-dir DIRECTORY``.
Use ``--samples 6`` for six independent solutions with the same prompt and no
shared history. Each sample is saved immediately with its zero-based sample_index.
Evaluation votes across samples; --samples 1 retains the single-CoT baseline.
Alternatively, --sample-counts counts.json accepts {"0": 7, "2": 5}: generate
seven solutions for dataset index 0 and five for index 2, excluding unlisted
examples. Counts must be positive integers; offset/limit still refer to original
dataset positions. Resolved counts are saved in config.json for later evaluation.

For raw-completion generation, use ``python -m lits.cli.cot_completion``.
"""

import argparse

from lits import get_lm
from lits.benchmarks.cot import load_sample_counts, run_cot
from lits.cli.args import parse_dataset_kwargs
from lits.components.utils import create_role
from lits.constants import DEFAULT_REASONING_TEMPERATURE
from lits.registry import import_custom_modules


PROMPT = (
    "Given a science problem, your task is to answer the question step-by-step "
    "in a clear and specific manner.\n"
    'The format of the solution is limited to: "Solution: ...\\nSummary: The final answer is $...$"\n'
    "Please complete the answer step-by-step, and finally outline the final answer.\n"
    "Problem: {question}\nSolution:"
)


def run(args):
    """Generate from a registered dataset; refuse existing directories and continue on errors.

    Args is the namespace from main(). Model endpoint/authentication also accept
    the standard OPENAI_API_BASE and OPENAI_API_KEY environment variables.
    """
    import_custom_modules(args.import_modules)
    model = get_lm(args.model, base_url=args.base_url, enable_thinking=args.thinking)

    def generate(question, index, repeat):
        """Generate one independent chat response without extracting its answer."""
        response = model(PROMPT.format(question=question), role=create_role("policy", index),
                         temperature=args.temperature, max_new_tokens=args.max_new_tokens)
        return {"raw_response": response.text}

    run_cot({**vars(args), "prompt": PROMPT}, model, generate,
            dataset_kwargs=parse_dataset_kwargs(args))


def main():
    """Parse generation settings; evaluation is a separate operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--include", dest="import_modules", nargs="+")
    parser.add_argument("--dataset-arg", dest="dataset_args", nargs="+", action="extend")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--samples", type=int, default=1,
                        help="Independent complete solutions per example (default: 1).")
    sampling.add_argument("--sample-counts", help="JSON map of dataset indices to positive sample counts; excludes unlisted examples.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_REASONING_TEMPERATURE)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature < 0:
        parser.error("max-new-tokens must be positive and temperature nonnegative")
    if args.samples <= 0:
        parser.error("samples must be positive")
    if args.offset < 0 or (args.limit is not None and args.limit < 0):
        parser.error("offset and limit must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
