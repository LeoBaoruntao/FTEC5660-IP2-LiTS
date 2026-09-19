"""Capture CLI indexing, saving, and resume behavior without model inference.

Run: ``PYTHONBREAKPOINT=0 python -m unit_test.cli.test_search_loop``.
Calls ``search.py::main`` with a registered local cohort, real LiTS components,
and the configured OpenAI-compatible model client. Zero MCTS iterations avoid
generation. A recording wrapper delegates to the real benchmark
``search.py::run_tree_search``;
no models or search implementations are replaced. Results remain in a fresh directory.
Registered preparation is counted on fresh and resumed runs. Answer resolution,
memory saving, and nonzero-iteration search are not tested here.
"""

import json
from functools import partial
import os
from pathlib import Path
import sys
import tempfile

from lits.benchmarks.registry import register_dataset, register_resource
from lits.benchmarks import search as benchmark_search
from lits.cli import search


EXAMPLES = [{"question": f"What is {i} + 1?", "answer": str(i + 1)} for i in range(4)]
LOADS = []
PREPARED = []


@register_dataset("cli_loop_check", task_type="language_grounded")
@register_dataset("cli_loop_preparation_check", task_type="language_grounded")
def load_cohort():
    """Return a small local cohort with distinguishable questions and original indices."""
    LOADS.append("load")
    return EXAMPLES


@register_resource("cli_loop_preparation_check")
def load_preparation_resource():
    """Provide an observable preparation callback through the actual registry."""
    def prepare(example):
        """Record the original example supplied by shared per-example execution."""
        PREPARED.append(example)
    return {"prepare_tool_state": prepare}


def run_cli(output_dir, overrides, dataset_name="cli_loop_check"):
    """Record real CLI dispatch; restore argv and the delegating wrapper afterward."""
    original_run, original_argv = benchmark_search.run_tree_search, sys.argv
    original_benchmark = search.run_benchmark
    loads_before = len(LOADS)
    prepared_before = len(PREPARED)
    benchmark_calls = []
    observed = []

    def record_benchmark(*args, **kwargs):
        """Verify delegation uses the full loaded dataset and keeps CLI resume handling."""
        if kwargs.get("dataset") is not EXAMPLES or kwargs.get("is_complete") is not None:
            raise RuntimeError("CLI must reuse its loaded dataset and internal resume handling")
        execute = args[1]
        if not isinstance(execute, partial) or execute.func is not benchmark_search.run_example:
            raise RuntimeError("CLI must bind the shared run_example directly")
        benchmark_calls.append(args[0])
        return original_benchmark(*args, **kwargs)

    def record_search(*args, **kwargs):
        """Observe dispatch and delegate unchanged, rejecting nonzero iteration settings."""
        if kwargs["search_config"].n_iters != 0:
            raise RuntimeError("This check must not run model inference")
        index = kwargs["query_idx"]
        if kwargs["init_state_kwargs"] is not EXAMPLES[index]:
            raise RuntimeError("CLI changed the example passed for initialization")
        observed.append(index)
        return original_run(*args, **kwargs)

    benchmark_search.run_tree_search = record_search
    search.run_benchmark = record_benchmark
    sys.argv = [
        "lits-search", "--dataset", dataset_name,
        "--policy-model", "openai/local-instruct", "--output-dir", str(output_dir),
        "--search-arg", "n_iters=0", "add_continuation=false",
        "terminate_constraints=[step_signal]", "--cfg", *overrides,
    ]
    try:
        if search.main() != 0:
            raise RuntimeError("CLI returned a nonzero exit code")
        if benchmark_calls != [dataset_name] or LOADS[loads_before:] != ["load"]:
            raise RuntimeError("CLI did not delegate once or loaded the dataset more than once")
        prepared = PREPARED[prepared_before:]
        if dataset_name == "cli_loop_preparation_check":
            if len(prepared) != len(observed) or any(
                example is not EXAMPLES[idx] for example, idx in zip(prepared, observed)
            ):
                raise RuntimeError("Preparation must run once per dispatched example, including resume")
        elif prepared:
            raise RuntimeError("Preparation ran without a configured resource")
    finally:
        benchmark_search.run_tree_search, sys.argv = original_run, original_argv
        search.run_benchmark = original_benchmark
    return observed


def check_resource_preparation(root):
    """Run registered preparation once per example on both fresh and resumed runs."""
    output_dir = root / "preparation"
    overrides = ["eval_idx=[2,0]"]
    for _ in range(2):
        observed = run_cli(output_dir, overrides, "cli_loop_preparation_check")
        if observed != [2, 0]:
            raise RuntimeError(f"Unexpected preparation cohort: {observed}")
    print("PASS registered preparation: once per example on fresh and resumed runs")


def check_dataset_loop(root):
    """Verify actual dispatch and saved query IDs, then repeat one run to test resume."""
    cases = (
        ("offset_limit", ["offset=1", "limit=2"], [1, 2]),
        ("explicit_order", ["eval_idx=[3,1]"], [3, 1]),
        ("explicit_slice", ["eval_idx=[3,1,2]", "offset=1", "limit=1"], [1]),
        ("empty_slice", ["limit=0"], []),
    )
    report = {}
    for name, overrides, expected in cases:
        output_dir = root / name
        observed = run_cli(output_dir, overrides)
        paths = sorted((output_dir / "terminal_nodes").glob("*.json"))
        rows = [json.loads(path.read_text()) for path in paths]
        if observed != expected or sorted(row["query_idx"] for row in rows) != sorted(expected):
            raise RuntimeError(f"{name}: expected {expected}, dispatched {observed}, saved {rows}")
        if any(row["query"] != EXAMPLES[row["query_idx"]]["question"] for row in rows):
            raise RuntimeError(f"{name}: saved query differs from the original example")
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
        if name == "offset_limit":
            run_cli(output_dir, overrides)
            if any((path.read_bytes(), path.stat().st_mtime_ns) != value for path, value in before.items()):
                raise RuntimeError("Resume rewrote completed terminal files")
        usage = output_dir / "inferencelogger.log"
        if usage.exists() and usage.read_text().strip():
            raise RuntimeError("Unexpected inference usage during zero-iteration search")
        report[name] = {"indices": observed, "saved_queries": [row["query"] for row in rows]}
        print(f"PASS {name}: {observed}")
    (root / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    breakpoint()  # inspect: report, root


def main():
    """Configure the real client without HF login and retain fresh CLI baseline artifacts."""
    settings = {
        "OPENAI_API_BASE": "https://xinzhes-macbook-pro-2.tailde6fe4.ts.net/v1",
        "OPENAI_API_KEY": "no-key", "HF_TOKEN": "",
    }
    previous = {key: os.environ.get(key) for key in settings}
    root = Path(tempfile.mkdtemp(prefix="lits-cli-loop-"))
    print(f"Retained baseline: {root}")
    os.environ.update(settings)
    try:
        check_dataset_loop(root)
        check_resource_preparation(root)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
