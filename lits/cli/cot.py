"""Generate one complete CoT response per registered dataset example.

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
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from lits import get_lm
from lits.benchmarks.registry import load_dataset
from lits.benchmarks.runner import _slice_dataset, log_example_error, run_benchmark
from lits.cli.args import parse_dataset_kwargs
from lits.components.utils import create_role
from lits.lm import setup_inference_logging, subtract_inference_metrics
from lits.log import setup_logging
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
    dataset = load_dataset(args.dataset, **parse_dataset_kwargs(args))
    indices = _slice_dataset(range(len(dataset)), args.offset, args.limit)
    cohort = [{"query_idx": index, "example": dataset[index]} for index in indices]
    if any(not isinstance(row["example"]["question"], str) or
           not row["example"]["question"].strip() for row in cohort):
        raise ValueError("Each cohort example requires a nonempty question")
    model = get_lm(args.model, base_url=args.base_url, enable_thinking=args.thinking)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(
        json.dumps({**vars(args), "prompt": PROMPT}, indent=2), encoding="utf-8")
    (directory / "cohort.json").write_text(json.dumps(cohort, indent=2), encoding="utf-8")
    setup_logging(run_id="execution", result_dir=directory, add_console_handler=True)
    usage = setup_inference_logging(model, root_dir=directory)

    with (directory / "responses.jsonl").open("x", encoding="utf-8") as stream:
        def run_example(example, index, repeat):
            """Save the unmodified response and generation-only costs immediately."""
            before = usage.get_metrics_by_prefix("policy")
            started = perf_counter()
            response = model(PROMPT.format(question=example["question"]),
                             role=create_role("policy", index),
                             temperature=args.temperature, max_new_tokens=args.max_new_tokens)
            record = {"query_idx": index, "raw_response": response.text,
                      "generation_seconds": perf_counter() - started,
                      "generation_costs": subtract_inference_metrics(
                          usage.get_metrics_by_prefix("policy"), before)}
            stream.write(json.dumps(record) + "\n")
            stream.flush()

        def on_error(example, index, repeat, error):
            """Keep failed examples visible under their original dataset IDs."""
            stream.write(json.dumps({"query_idx": index,
                                     "error": f"{type(error).__name__}: {error}"}) + "\n")
            stream.flush()
            log_example_error(example, index, repeat, error)

        run_benchmark(args.dataset, run_example, dataset=dataset, offset=args.offset,
                      limit=args.limit, on_error=on_error)


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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature < 0:
        parser.error("max-new-tokens must be positive and temperature nonnegative")
    if args.offset < 0 or (args.limit is not None and args.limit < 0):
        parser.error("offset and limit must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
