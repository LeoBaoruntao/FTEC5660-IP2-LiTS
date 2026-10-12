"""Shared dataset execution and persistence for independent CoT generation."""

import json
import os
import shutil
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
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


def load_resume_records(directory, config, cohort):
    """Validate a saved run and return records keyed by (example ID, sample ID).

    Compare resolved sample counts, generation settings and the entire cohort.
    Ignore output location, resume flag and volatile server model-list metadata;
    callers must keep the actual served model unchanged. Reject corrupt or
    duplicate records before making any changes. Missing/failed samples may run
    again; successful responses retain their original text, costs and timing.
    """
    saved = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    ignored = {"output_dir", "resume", "model_metadata"}
    changed = [key for key in saved.keys() | config.keys()
               if key not in ignored and saved.get(key) != config.get(key)]
    if changed:
        raise ValueError(f"Resume settings differ: {', '.join(sorted(changed))}")
    if json.loads((directory / "cohort.json").read_text(encoding="utf-8")) != cohort:
        raise ValueError("Resume dataset/cohort differs from the saved run")
    path = directory / "responses.jsonl"
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        row = json.loads(line)
        index, sample = row.get("query_idx"), row.get("sample_index", 0)
        count = config["sample_counts"].get(str(index), 0)
        if (type(index) is not int or type(sample) is not int
                or not 0 <= sample < count or (index, sample) in records):
            raise ValueError(f"Invalid/duplicate saved sample: {index}/{sample}")
        if "error" not in row and not isinstance(row.get("raw_response"), str):
            raise ValueError(f"Malformed saved response: {index}/{sample}")
        records[index, sample] = row
    return records


def save_resumed_records(path, records):
    """Atomically checkpoint one current record per sample, including failures.

    Write and fsync a temporary file in the same directory before replacement;
    an interruption before replacement leaves the previous checkpoint intact.
    """
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                            prefix=".responses-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for key in sorted(records):
                stream.write(json.dumps(records[key]) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def run_cot(config: dict, model, generate, *, dataset_kwargs: dict | None = None):
    """Select examples, execute independent generations and save each result immediately.

    config supplies dataset, output_dir, offset/limit and optional samples or
    sample_counts path; remaining settings are saved verbatim. Dataset modules
    must already be imported. model is the actual generator used for cost logging.
    generate(question, index, repeat) returns a dict containing raw_response and
    optional evidence (prompt, seed, finish_reason). It receives no reference answer.
    The shared loop adds identity, elapsed time and generation-only costs.

    Refuse existing directories unless config["resume"] is true. Resume validates
    saved settings/cohort, backs up responses.jsonl and regenerates only missing
    or failed samples under their original IDs. Each replacement is atomic;
    existing metadata and successful records are unchanged, logs are appended.
    Only one process may write a run directory at a time.
    Save config.json, cohort.json, responses.jsonl
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
    config = {**config, "sample_counts": {str(index): count for index, count in counts.items()}}
    resume = config.get("resume", False)
    path = directory / "responses.jsonl"
    records = load_resume_records(directory, config, cohort) if resume else {}
    if resume:
        if path.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            shutil.copy2(path, directory / f"responses.before-resume-{stamp}.jsonl")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (directory / "cohort.json").write_text(json.dumps(cohort, indent=2), encoding="utf-8")
    setup_logging(run_id="execution", result_dir=directory, add_console_handler=True,
                  override=not resume)
    usage = setup_inference_logging(model, root_dir=directory, override=not resume)

    with (nullcontext() if resume else path.open("x", encoding="utf-8")) as stream:
        def save_record(record):
            """Replace a resumed sample or append a fresh one, retaining its usage evidence."""
            records[record["query_idx"], record["sample_index"]] = record
            if resume:
                save_resumed_records(path, records)
            else:
                stream.write(json.dumps(record) + "\n")
                stream.flush()

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
            save_record(record)

        def on_error(example, index, repeat, error):
            """Keep failed examples visible under their original dataset IDs."""
            save_record({"query_idx": index, "sample_index": repeat,
                         "error": f"{type(error).__name__}: {error}"})
            log_example_error(example, index, repeat, error)

        if indices:
            run_benchmark(config["dataset"], run_example, dataset=dataset, eval_idx=indices,
                          repetitions=max(counts.values()),
                          is_complete=lambda example, index, repeat: repeat >= counts[index]
                          or ((index, repeat) in records and "error" not in records[index, repeat]),
                          on_error=on_error)
