"""Checks for eval_cot.py::evaluate using a real local-endpoint model.

Run: PYTHONBREAKPOINT=0 python -m unit_test.eval.test_cot_evaluation
Blank/malformed responses exercise existing extraction/checking without API calls.
Add --live for cot.py::run generating two independent samples of one MATH example,
then evaluating their joint vote. Live artifacts remain in a fresh temporary directory.
Use --saved-run RAW_RUN --indices 14 --output-dir NEW_DIRECTORY to check saved
cot_completion outputs, including cutoffs, through the unchanged evaluator.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from lits import get_lm
from lits.cli.eval_cot import evaluate


def check_saved_cot_evaluation():
    """Verify errors, coverage, separate costs, and preservation of saved generation."""
    model = get_lm("openai/local-instruct", base_url="http://127.0.0.1:1/v1",
                   api_key="no-key", enable_thinking=False)
    voter = get_lm("openai/local-instruct", base_url="http://127.0.0.1:1/v1",
                   api_key="no-key", enable_thinking=True)
    with TemporaryDirectory(prefix="lits-cot-eval-") as temporary:
        directory = Path(temporary)
        (directory / "config.json").write_text(json.dumps({
            "dataset": "math500", "import_modules": ["lits_benchmark.math_qa_eval"],
            "model": "openai/local-instruct",
        }))
        (directory / "cohort.json").write_text(json.dumps([
            {"query_idx": i, "example": {"question": "2+2?", "answer": "4"}}
            for i in (3, 5, 7, 9)]))
        responses = [{"query_idx": 3, "raw_response": "", "generation_costs": {"num_calls": 1}},
                     {"query_idx": 5, "raw_response": None},
                     {"query_idx": 7, "error": "Generation failed"}]
        (directory / "responses.jsonl").write_text("\n".join(map(json.dumps, responses)))
        original = {p: p.read_bytes() for p in directory.iterdir()}
        for mode in ("binary", "none"):
            summary = evaluate(directory, model, llm_eval_mode=mode, voting_model=voter)
            if summary["total"] != 1 or summary["correct_count"] != 0:
                raise RuntimeError("Missing/failed cases were counted as evaluated")
        records_files = sorted((directory / "evaluation").glob("*/records.jsonl"))
        if len(records_files) != 2 or any(p.read_bytes() != data for p, data in original.items()):
            raise RuntimeError("Reevaluation overwrote existing artifacts")
        for path in records_files:
            settings = json.loads((path.parent / "config.json").read_text())
            if (settings["enable_thinking"] or not settings["voting_enable_thinking"]
                    or settings["voting_max_new_tokens"] != 8192
                    or voter.inference_logger is not model.inference_logger):
                raise RuntimeError("Voter thinking, output budget, or shared cost logging is incorrect")
            records = [json.loads(line) for line in path.read_text().splitlines()]
            if [r["query_idx"] for r in records] != [3, 5]:
                raise RuntimeError("Original query IDs or generation-error filtering changed")
            if records[0]["prediction"] != "" or "error" not in records[1]:
                raise RuntimeError("Blank and malformed responses were not distinguished")
            if any(r["evaluation_costs"]["num_calls"] != 0 for r in records):
                raise RuntimeError("Evaluation costs contain generation calls")
        breakpoint()  # inspect: summary, records_files, records, original
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        config["dataset"] = "scibench_calculus"
        config_path.write_text(json.dumps(config))
        summary = evaluate(directory, model)
        if summary["total"] != 1 or summary["correct_count"] != 0:
            raise RuntimeError("Checker-free binary evaluation changed missing-answer handling")
        try:
            evaluate(directory, model, llm_eval_mode="none")
        except ValueError as error:
            if "has no registered checker" not in str(error):
                raise
        else:
            raise RuntimeError("Checker-free evaluation accepted disabled judging")
        breakpoint()  # inspect: summary; SciBench supports binary judging, not checker-free none mode
        config.update(dataset="math500", samples=2)
        config_path.write_text(json.dumps(config))
        responses = [
            {"query_idx": 3, "sample_index": 1, "raw_response": ""},
            {"query_idx": 3, "sample_index": 0, "raw_response": ""},
            {"query_idx": 5, "sample_index": 0, "raw_response": ""},
            {"query_idx": 7, "sample_index": 0, "raw_response": ""},
            {"query_idx": 7, "sample_index": 1, "error": "Generation failed"},
        ]
        responses_path = directory / "responses.jsonl"
        responses_path.write_text("\n".join(map(json.dumps, responses)))
        summary = evaluate(directory, model, llm_eval_mode="none")
        latest = sorted((directory / "evaluation").glob("*/records.jsonl"))[-1]
        records = [json.loads(line) for line in latest.read_text().splitlines()]
        if summary["total"] != 1 or records[0].get("sample_count") != 2:
            raise RuntimeError("Multiple samples were not evaluated as one example")
        if any("incomplete" not in row.get("error", "") for row in records[1:]):
            raise RuntimeError("Partial sample sets were silently evaluated")
        responses.append(responses[0])
        responses_path.write_text("\n".join(map(json.dumps, responses)))
        try:
            evaluate(directory, model, llm_eval_mode="none")
        except ValueError as error:
            if "Duplicate response" not in str(error):
                raise
        else:
            raise RuntimeError("Duplicate sample IDs were accepted")
        breakpoint()  # inspect: records; one joint result, two incomplete sample-set errors
        config["sample_counts"] = {"3": 2, "5": 1, "7": 2, "9": 3}
        config_path.write_text(json.dumps(config))
        responses_path.write_text("\n".join(map(json.dumps, responses[:-1])))
        summary = evaluate(directory, model, llm_eval_mode="none")
        latest = sorted((directory / "evaluation").glob("*/records.jsonl"))[-1]
        records = [json.loads(line) for line in latest.read_text().splitlines()]
        if summary["total"] != 2 or [r.get("sample_count") for r in records[:2]] != [2, 1]:
            raise RuntimeError("Evaluation ignored saved per-example sample counts")
    print("CoT evaluation checks passed; no inference")


def check_sample_counts():
    """Verify cot.py::load_sample_counts and the shared benchmark sampling loop without inference."""
    from lits.cli.cot import load_sample_counts
    from lits.benchmarks.runner import run_benchmark

    with TemporaryDirectory(prefix="lits-cot-counts-") as temporary:
        path = Path(temporary) / "counts.json"
        path.write_text(json.dumps({"0": 2, "2": 1}))
        counts = load_sample_counts(path, 3)
        calls = []
        run_benchmark(
            "unused", lambda example, index, repeat: calls.append((index, repeat)),
            dataset=[0, 1, 2], eval_idx=list(counts), repetitions=max(counts.values()),
            is_complete=lambda example, index, repeat: repeat >= counts[index],
            show_progress=False,
        )
        if calls != [(0, 0), (0, 1), (2, 0)]:
            raise RuntimeError("Per-example counts changed sample IDs or included unlisted examples")
        for invalid in ({}, [], {"3": 1}, {"-1": 1}, {"01": 1},
                        {"0": 0}, {"0": True}, {"0": 1.5}):
            path.write_text(json.dumps(invalid))
            try:
                load_sample_counts(path, 3)
            except ValueError:
                pass
            else:
                raise RuntimeError(f"Invalid sample counts accepted: {invalid}")
        breakpoint()  # inspect: counts, calls; unlisted index 1 is excluded
    print("Per-example sample-count checks passed; no inference")


def check_live_cot_sampling():
    """Verify saved sample identities, per-call costs, shared temperature, and joint voting."""
    from argparse import Namespace
    from tempfile import mkdtemp
    from lits.cli.cot import run
    from lits.constants import DEFAULT_REASONING_TEMPERATURE

    directory = Path(mkdtemp(prefix="lits-cot-sampling-")) / "run"
    args = Namespace(
        dataset="math500", import_modules=["lits_benchmark.math_qa"],
        dataset_args=None, offset=0, limit=1, samples=2,
        model="openai/local-instruct",
        base_url="https://xinzhes-macbook-pro-2.tailde6fe4.ts.net/v1",
        thinking=False, temperature=DEFAULT_REASONING_TEMPERATURE,
        max_new_tokens=512, output_dir=str(directory),
    )
    run(args)
    rows = [json.loads(line) for line in (directory / "responses.jsonl").read_text().splitlines()]
    if [(row["query_idx"], row["sample_index"]) for row in rows] != [(0, 0), (0, 1)]:
        raise RuntimeError("Generation did not save two independently identified samples")
    if any("error" in row or row["generation_costs"]["num_calls"] != 1 for row in rows):
        raise RuntimeError("Sample generation failed or costs were accumulated across samples")
    model = get_lm(args.model, base_url=args.base_url, enable_thinking=False)
    summary = evaluate(directory, model)
    records = [json.loads(line) for line in next((directory / "evaluation").glob("*/records.jsonl")).read_text().splitlines()]
    if summary["total"] != 1 or records[0].get("sample_count") != 2:
        raise RuntimeError("Independent generation did not reach joint evaluation")
    print(f"Live two-sample CoT check passed: {directory}")
    breakpoint()  # inspect: rows, records; separate generations and one joint vote


def check_saved_cot_voting(source, indices, trace_enabled=False, output_dir=None):
    """Replay selected saved samples through eval_cot.py::evaluate with thinking.

    Copy generation inputs into a fresh temporary directory; preserve source
    evaluations and retain replay artifacts for inspecting votes and costs.
    No new solutions are generated. The judge remains non-thinking.
    trace_enabled captures SDK requests and responses in a local trace archive.
    output_dir optionally retains evidence outside temporary storage and must be
    new. Verify coverage, vote population and byte-preservation of generation
    files, including raw-completion prompts, seeds and cutoff metadata.
    """
    from tempfile import mkdtemp

    if output_dir is None:
        directory = Path(mkdtemp(prefix="lits-cot-voting-"))
    else:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
    original = {path: path.read_bytes() for path in source.iterdir() if path.is_file()}
    config = json.loads((source / "config.json").read_text())
    cohort = [row for row in json.loads((source / "cohort.json").read_text())
              if row["query_idx"] in indices]
    if {row["query_idx"] for row in cohort} != set(indices):
        raise ValueError("Requested examples are absent from the saved cohort")
    responses = [json.loads(line) for line in (source / "responses.jsonl").read_text().splitlines()]
    responses = [row for row in responses if row["query_idx"] in indices]
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "cohort.json").write_text(json.dumps(cohort))
    (directory / "responses.jsonl").write_text("\n".join(map(json.dumps, responses)))
    copied = {path: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    model = get_lm(config["model"], base_url=config.get("base_url"), enable_thinking=False)
    voter = get_lm(config["model"], base_url=config.get("base_url"), enable_thinking=True)
    from lits import observability as tracing
    if trace_enabled:
        tracing.configure(enabled=True, send_to_logfire=False, console=False,
                          archive_dir=directory / "telemetry", run_id="saved-cot-voting",
                          clients=(model.client, voter.client))
    try:
        summary = evaluate(directory, model, voting_model=voter)
    finally:
        if trace_enabled and not tracing.close():
            raise RuntimeError(f"Trace export incomplete: {directory / 'telemetry'}")
    records = [json.loads(line) for line in next(
        (directory / "evaluation").glob("*/records.jsonl")).read_text().splitlines()]
    if any(path.read_bytes() != content for path, content in {**original, **copied}.items()):
        raise RuntimeError("Evaluation modified saved generation files")
    if summary["total"] != len(cohort) or {r["query_idx"] for r in records} != set(indices):
        raise RuntimeError(f"Incomplete evaluation; inspect {directory}")
    for record in records:
        count = sum(row["query_idx"] == record["query_idx"] for row in responses)
        if (record.get("sample_count") != count or sum(record["votes"].values()) > count
                or not isinstance(record["task_success"], bool)):
            raise RuntimeError("Saved samples did not retain their voting population or judgment")
    print(records)
    print(f"Passed saved-output evaluation and generation preservation: {directory}")
    breakpoint()  # inspect: records, directory; semantic groups and retained replay evidence


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--saved-run", type=Path)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--trace-enabled", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.saved_run and not args.indices:
        parser.error("--saved-run requires --indices")
    check_sample_counts()
    check_saved_cot_evaluation()
    if args.live:
        check_live_cot_sampling()
    if args.saved_run:
        check_saved_cot_voting(args.saved_run, args.indices, args.trace_enabled, args.output_dir)
