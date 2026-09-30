"""Shared dataset execution and persistence for independent CoT generation."""

import json
from pathlib import Path
from time import perf_counter

from .registry import load_dataset
from .runner import _slice_dataset, log_example_error, run_benchmark
from ..lm import setup_inference_logging, subtract_inference_metrics
from ..log import setup_logging


def load_sample_counts(path, dataset_size):
    """Read an explicit example-to-count JSON map, rejecting invalid IDs/counts.

    For a dataset of size three, {"0": 7, "2": 5} selects examples 0 and 2.
    Unlisted examples are excluded, not assigned a fallback count. JSON keys
    must be canonical nonnegative dataset indices; counts must be positive ints.
    Returns an integer-keyed dictionary independent of the source file.
    """
    counts = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(counts, dict) or not counts:
        raise ValueError("sample-counts must be a nonempty JSON object")
    for index, count in counts.items():
        if (not index.isascii() or not index.isdecimal() or str(int(index)) != index
                or int(index) >= dataset_size):
            raise ValueError(f"Invalid dataset index in sample-counts: {index!r}")
        if type(count) is not int or count <= 0:
            raise ValueError(f"Sample count for example {index} must be a positive integer")
    return {int(index): count for index, count in counts.items()}


def run_cot(config: dict, model, generate, *, dataset_kwargs: dict | None = None):
    """Select examples, execute independent generations and save each result immediately.

    config supplies dataset, output_dir, offset/limit and optional samples or
    sample_counts path; remaining settings are saved verbatim. Dataset modules
    must already be imported. model is the actual generator used for cost logging.
    generate(question, index, repeat) returns a dict containing raw_response and
    optional evidence (prompt, seed, finish_reason). It receives no reference answer.
    The shared loop adds identity, elapsed time and generation-only costs.

    Refuse existing directories. Save config.json, cohort.json, responses.jsonl
    and inference logs in the format consumed by eval_cot. Failures are recorded
    and remaining examples continue. No prompt construction or evaluation occurs.
    """
    dataset = load_dataset(config["dataset"], **(dataset_kwargs or {}))
    indices = _slice_dataset(range(len(dataset)), config.get("offset", 0), config.get("limit"))
    counts_path = config.get("sample_counts")
    counts = (load_sample_counts(counts_path, len(dataset)) if counts_path
              else {index: config.get("samples", 1) for index in indices})
    counts = {index: counts[index] for index in indices if index in counts}
    indices = list(counts)
    if counts_path and not counts:
        raise ValueError("sample-counts and offset/limit select no examples")
    cohort = [{"query_idx": index, "example": dataset[index]} for index in indices]
    if any(not isinstance(row["example"]["question"], str) or
           not row["example"]["question"].strip() for row in cohort):
        raise ValueError("Each cohort example requires a nonempty question")
    directory = Path(config["output_dir"])
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(
        json.dumps({**config, "sample_counts": counts}, indent=2), encoding="utf-8")
    (directory / "cohort.json").write_text(json.dumps(cohort, indent=2), encoding="utf-8")
    setup_logging(run_id="execution", result_dir=directory, add_console_handler=True)
    usage = setup_inference_logging(model, root_dir=directory)

    with (directory / "responses.jsonl").open("x", encoding="utf-8") as stream:
        def run_example(example, index, repeat):
            """Save the unmodified response and generation-only costs immediately."""
            before = usage.get_metrics_by_prefix("policy")
            started = perf_counter()
            with usage.log_context(query_idx=index, sample_index=repeat):
                record = generate(example["question"], index, repeat)
            record.update(query_idx=index, sample_index=repeat,
                          generation_seconds=perf_counter() - started,
                          generation_costs=subtract_inference_metrics(
                              usage.get_metrics_by_prefix("policy"), before))
            stream.write(json.dumps(record) + "\n")
            stream.flush()

        def on_error(example, index, repeat, error):
            """Keep failed examples visible under their original dataset IDs."""
            stream.write(json.dumps({"query_idx": index, "sample_index": repeat,
                                     "error": f"{type(error).__name__}: {error}"}) + "\n")
            stream.flush()
            log_example_error(example, index, repeat, error)

        if indices:
            run_benchmark(config["dataset"], run_example, dataset=dataset, eval_idx=indices,
                          repetitions=max(counts.values()),
                          is_complete=lambda example, index, repeat: repeat >= counts[index],
                          on_error=on_error)
