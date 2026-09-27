"""Evaluate saved CoT responses without rerunning generation.

Run ``python -m lits.cli.eval_cot --result-dir DIRECTORY``. Model and endpoint
default to the saved config. Evaluation uses shared answer voting, registered
checking, and an optional binary judge (--llm-eval-mode none disables only judging).
Each run saves separate records/costs under evaluation/TIMESTAMP; it never
regenerates responses or overwrites earlier results.
The CLI enables thinking for voting with an 8192-token output allowance, while
keeping thinking disabled for the separate correctness judge.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from lits import get_lm
from lits.lm import setup_inference_logging, subtract_inference_metrics
from lits.registry import import_custom_modules


def evaluate(directory, model, *, llm_eval_mode="binary", voting_model=None):
    """Evaluate saved raw responses with the shared LiTS extraction/checking pipeline.

    One vote_answers call receives all independent samples for an example.
    A singleton input performs extraction, not self-consistency. Legacy records
    without sample_index are sample zero. Incomplete sample sets are evaluation
    errors; failed samples never silently reduce the requested voting population.
    Per-example counts come from saved config.sample_counts when present;
    otherwise config.samples (or the legacy default one) applies uniformly.
    Reference answers reach only check_answer, never extraction. Generation errors
    are skipped; evaluation exceptions remain errors, not incorrect answers.
    Returns aggregate_results' summary. Printed accuracy excludes missing/errors,
    with coverage shown explicitly. Repeated evaluations keep separate artifacts.
    The supplied model gets a new evaluation-only inference logger.
    voting_model optionally supplies a separate voter; otherwise model is reused.
    Both share the evaluation logger so all voting and judging costs are retained.
    Without a registered checker, exact matches pass directly and other answers
    use the context-aware binary judge; disabling that judge requires a checker.
    """
    from datetime import datetime, timezone
    from tqdm import tqdm
    from lits.benchmarks.registry import get_evaluator
    from lits.eval import GeneralEvaluator
    from lits.eval.aggregation import aggregate_results
    from lits.eval.answer_voting import vote_answers
    from lits.eval.search import check_answer

    directory = Path(directory)
    voting_model = voting_model if voting_model is not None else model
    config = json.loads((directory / "config.json").read_text())
    import_custom_modules(config.get("import_modules"))
    checker = get_evaluator(config["dataset"])
    if llm_eval_mode not in ("binary", "none"):
        raise ValueError("CoT evaluation requires binary/none judge mode")
    if checker is None and llm_eval_mode == "none":
        raise ValueError(f"Dataset {config['dataset']!r} has no registered checker; use --llm-eval-mode binary")
    cohort = {row["query_idx"]: row["example"]
              for row in json.loads((directory / "cohort.json").read_text())}
    responses = [json.loads(line) for line in (directory / "responses.jsonl").read_text().splitlines()]
    sample_counts = {index: config["sample_counts"][str(index)] if "sample_counts" in config
                     else config.get("samples", 1) for index in cohort}
    grouped = {}
    for row in responses:
        index, sample = row["query_idx"], row.get("sample_index", 0)
        if index not in cohort or type(sample) is not int or not 0 <= sample < sample_counts[index]:
            raise ValueError("Responses contain out-of-cohort query IDs or invalid sample indices")
        group = grouped.setdefault(index, {})
        if sample in group:
            raise ValueError(f"Duplicate response for example {index}, sample {sample}")
        group[sample] = row
    destination = directory / "evaluation" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    destination.mkdir(parents=True)
    (destination / "config.json").write_text(json.dumps({
        "model": model.model_name, "llm_eval_mode": llm_eval_mode,
        "enable_thinking": model.enable_thinking,
        "voting_model": voting_model.model_name,
        "voting_enable_thinking": voting_model.enable_thinking,
        "voting_max_new_tokens": 8192,
    }, indent=2))
    usage = setup_inference_logging(model, voting_model, root_dir=destination)
    judge = GeneralEvaluator(base_model=model, eval_perspectives=[{
        "eval_id": "correct", "description": "Does the predicted answer contain the correct value? Ignore formatting differences and compare the core answer value.",
        "options": ["yes", "no"],
    }]) if llm_eval_mode == "binary" else None
    evaluated = []
    with (destination / "records.jsonl").open("x") as stream:
        for index, group in tqdm(grouped.items(), desc="Evaluate CoT", unit="example"):
            samples = sample_counts[index]
            if all("error" in row for row in group.values()):
                continue
            record = {"query_idx": index}
            before, started = usage.get_metrics_by_role(), perf_counter()
            try:
                if len(group) != samples or any("error" in row for row in group.values()):
                    raise ValueError(f"Example {index} requires {samples} successful samples; generation is incomplete")
                raw_answers = {f"n{sample}": group[sample]["raw_response"] for sample in range(samples)}
                with usage.log_context(query_idx=index):
                    votes = vote_answers(cohort[index]["question"], raw_answers,
                                         model=voting_model, max_new_tokens=8192)
                    prediction = max(votes, key=votes.get) if votes else ""
                    correct, score = check_answer(prediction, cohort[index]["answer"],
                                                  context=cohort[index]["question"],
                                                  custom_evaluator=checker, llm_evaluator=judge)
                record.update(prediction=prediction, task_success=correct, score=score,
                              votes=votes, sample_count=samples)
                evaluated.append({"idx": index, "correct": correct})
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            record["evaluation_seconds"] = perf_counter() - started
            record["evaluation_costs"] = subtract_inference_metrics(usage.get_metrics_by_role(), before)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
    summary = aggregate_results(evaluated)
    accuracy = f"{summary['accuracy']:.1%}" if evaluated else "N/A"
    print(f"Evaluated: {summary['total']}/{len(cohort)}; correct: {summary['correct_count']}; "
          f"accuracy (evaluated): {accuracy}")
    print(f"Evaluation records and costs: {destination}")
    return summary


def main():
    """Load saved generation settings and evaluate with optional model overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--llm-eval-mode", choices=("binary", "none"), default="binary")
    args = parser.parse_args()
    config = json.loads((args.result_dir / "config.json").read_text())
    model = get_lm(args.model or config["model"],
                   base_url=args.base_url or config.get("base_url"), enable_thinking=False)
    voting_model = get_lm(args.model or config["model"],
                         base_url=args.base_url or config.get("base_url"), enable_thinking=True)
    evaluate(args.result_dir, model, llm_eval_mode=args.llm_eval_mode, voting_model=voting_model)


if __name__ == "__main__":
    main()
