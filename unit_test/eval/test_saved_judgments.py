"""Recheck saved MATH judgments without rerunning voting or changing run artifacts.

Run with demos/ on PYTHONPATH and OPENAI_API_BASE/OPENAI_API_KEY configured:
    python -m unit_test.eval.test_saved_judgments RUN_DIR --indices 68 83
"""

import argparse
import json
from pathlib import Path
import tempfile

from lits.eval import GeneralEvaluator
from lits.lm import get_lm, setup_inference_logging
from lits_benchmark.math_qa_eval import evaluate_math_qa


def check_saved_judgments(directory, indices, output_dir):
    """Replay general_eval.py::GeneralEvaluator.evaluate with raw responses retained.

    Read pointwise predictions and their saved judge settings. Use the same
    parsed-value comparison as general_eval.py::GeneralEvaluator.check_correct,
    without a second inference call. Save prompts, responses, and usage separately;
    these are fresh observations, not recovered historical model responses.
    """
    records = {row["query_idx"]: row for row in map(
        json.loads, (directory / "beam_pointwise/records.jsonl").read_text().splitlines())}
    cohort = {row["query_idx"]: row["example"] for row in
              json.loads((directory / "cohort.json").read_text())}
    model = get_lm("openai/local-instruct", enable_thinking=False, max_new_tokens=1024)
    model.client = model.client.with_options(timeout=120, max_retries=0)
    setup_inference_logging(model, root_dir=str(output_dir))
    for index in indices:
        record = records[index]
        settings = record["evaluation_setup"]["judge"]
        judge = GeneralEvaluator(
            base_model=model, eval_perspectives=settings["perspectives"],
            prompt_template=settings["prompt_template"],
            default_temperature=settings["temperature"],
            default_max_new_tokens=settings["max_new_tokens"],
            max_retries=settings["validation_retries"],
        )
        pred, truth = record["prediction"], cohort[index]["answer"]
        prompt = judge.build_prompt(solution=pred, truth=str(truth))
        (output_dir / f"prompt_{index}.txt").write_text(prompt)
        result = judge.evaluate(solution=pred, truth=str(truth),
                                include_input=False, return_raw_output=True)
        evidence = {
            "source": str(directory), "query_idx": index,
            "prediction": pred, "truth": truth,
            "saved_correct": record["task_success"],
            "deterministic_correct": evaluate_math_qa(pred, truth),
            "judge_response": result, "fresh_correct": result.get("correct") == "yes",
        }
        (output_dir / f"result_{index}.json").write_text(json.dumps(evidence, indent=2))
        print(json.dumps(evidence), flush=True)
        breakpoint()  # inspect: result, prompt; compare raw response with parsed correctness


def main():
    """Run bounded live checks, saving evidence in a fresh temporary directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    output_dir = Path(tempfile.mkdtemp(prefix="lits-saved-judgments-"))
    print(f"Artifacts: {output_dir}", flush=True)
    check_saved_judgments(args.directory, args.indices, output_dir)


if __name__ == "__main__":
    main()
