"""No-inference checks for eval_cot.py::evaluate using a real local-endpoint model.

Run: PYTHONBREAKPOINT=0 python -m unit_test.eval.test_cot_evaluation
Blank/malformed responses exercise existing extraction/checking without API calls.
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
            summary = evaluate(directory, model, llm_eval_mode=mode)
            if summary["total"] != 1 or summary["correct_count"] != 0:
                raise RuntimeError("Missing/failed cases were counted as evaluated")
        records_files = sorted((directory / "evaluation").glob("*/records.jsonl"))
        if len(records_files) != 2 or any(p.read_bytes() != data for p, data in original.items()):
            raise RuntimeError("Reevaluation overwrote existing artifacts")
        for path in records_files:
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
    print("CoT evaluation checks passed; no inference")


if __name__ == "__main__":
    check_saved_cot_evaluation()
