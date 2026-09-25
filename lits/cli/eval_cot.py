"""Evaluate saved CoT responses without rerunning generation.

Run ``python -m lits.cli.eval_cot --result-dir DIRECTORY``. Model and endpoint
default to the saved config. Evaluation uses shared answer extraction, registered
checking, and an optional binary judge (--llm-eval-mode none disables only judging).
Each run saves separate records/costs under evaluation/TIMESTAMP; it never
regenerates responses or overwrites earlier results.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from lits import get_lm
from lits.lm import setup_inference_logging, subtract_inference_metrics
from lits.registry import import_custom_modules


def evaluate(directory, model, *, llm_eval_mode="binary"):
    """Evaluate saved raw responses with the shared LiTS extraction/checking pipeline.

    A singleton vote_answers input performs extraction, not self-consistency.
    Reference answers reach only check_answer, never extraction. Generation errors
    are skipped; evaluation exceptions remain errors, not incorrect answers.
    Returns aggregate_results' summary. Printed accuracy excludes missing/errors,
    with coverage shown explicitly. Repeated evaluations keep separate artifacts.
    The supplied model gets a new evaluation-only inference logger.
    """
    from datetime import datetime, timezone
    from tqdm import tqdm
    from lits.benchmarks.registry import get_evaluator
    from lits.eval import GeneralEvaluator
    from lits.eval.aggregation import aggregate_results
    from lits.eval.answer_voting import vote_answers
    from lits.eval.search import check_answer

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    import_custom_modules(config.get("import_modules"))
    checker = get_evaluator(config["dataset"])
    if checker is None or llm_eval_mode not in ("binary", "none"):
        raise ValueError("CoT evaluation requires a registered checker and binary/none judge mode")
    cohort = {row["query_idx"]: row["example"]
              for row in json.loads((directory / "cohort.json").read_text())}
    responses = [json.loads(line) for line in (directory / "responses.jsonl").read_text().splitlines()]
    ids = [row["query_idx"] for row in responses]
    if len(set(ids)) != len(ids) or set(ids) - cohort.keys():
        raise ValueError("Responses contain duplicate or out-of-cohort query IDs")
    destination = directory / "evaluation" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    destination.mkdir(parents=True)
    (destination / "config.json").write_text(json.dumps({
        "model": model.model_name, "llm_eval_mode": llm_eval_mode,
        "enable_thinking": model.enable_thinking,
    }, indent=2))
    usage = setup_inference_logging(model, root_dir=destination)
    judge = GeneralEvaluator(base_model=model, eval_perspectives=[{
        "eval_id": "correct", "description": "Does the predicted answer contain the correct value? Ignore formatting differences and compare the core answer value.",
        "options": ["yes", "no"],
    }]) if llm_eval_mode == "binary" else None
    evaluated = []
    with (destination / "records.jsonl").open("x") as stream:
        for row in tqdm([r for r in responses if "error" not in r], desc="Evaluate CoT", unit="example"):
            index = row["query_idx"]
            record = {"query_idx": index}
            before, started = usage.get_metrics_by_role(), perf_counter()
            try:
                with usage.log_context(query_idx=index):
                    votes = vote_answers(cohort[index]["question"], {"n0": row["raw_response"]}, model=model)
                    prediction = max(votes, key=votes.get) if votes else ""
                    correct, score = check_answer(prediction, cohort[index]["answer"],
                                                  context=cohort[index]["question"],
                                                  custom_evaluator=checker, llm_evaluator=judge)
                record.update(prediction=prediction, task_success=correct, score=score)
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
    evaluate(args.result_dir, model, llm_eval_mode=args.llm_eval_mode)


if __name__ == "__main__":
    main()
