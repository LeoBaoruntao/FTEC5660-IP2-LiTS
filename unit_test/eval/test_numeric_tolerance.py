"""Live-check context-aware numeric voting and judging with positive/negative controls.

Run with PYTHONPATH=.:demos and OPENAI_API_BASE/OPENAI_API_KEY configured:
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_numeric_tolerance
"""

import json
import logging
from pathlib import Path
import tempfile

from lits import get_lm
from lits.eval import GeneralEvaluator
from lits.eval.answer_voting import vote_answers


def check_numeric_tolerance():
    """Exercise answer_voting.py::vote_answers and general_eval.py::GeneralEvaluator.evaluate.

    Voting sees two candidates without a reference designation; only judging
    receives a reference. Retain the voting response and raw judgment;
    check all cases before reporting mismatches. No search or existing records change.
    """
    output = Path(tempfile.mkdtemp(prefix="lits-numeric-tolerance-"))
    print(f"Artifacts: {output}", flush=True)
    model = get_lm("openai/local-instruct", enable_thinking=False)
    model.client = model.client.with_options(timeout=120, max_retries=0)
    voting_logger = logging.getLogger("lits.eval.answer_voting")
    voting_logger.setLevel(logging.DEBUG)
    voting_logger.addHandler(logging.FileHandler(output / "voting.log"))
    judge = GeneralEvaluator(model, [{"eval_id": "correct", "description":
        "Does the predicted answer have the correct value in the given context?",
        "options": ["yes", "no"]}])
    cases = [
        ("rounding", "Give the volume in cubic feet to two decimal places.",
         "5654.87", "5654.86677646", True),
        ("units", "Give the extension in centimetres.", "0.108 m", "10.8 cm", True),
        ("wrong_value", "Give the volume in cubic feet to two decimal places.",
         "5650.00", "5654.86677646", False),
        ("exact_count", "How many objects are there? Give the exact integer count.",
         "99", "100", False),
    ]
    results = []
    for name, context, pred, truth, equivalent in cases:
        votes = vote_answers(context, {"n0": pred, "n1": truth}, model=model)
        response = judge.evaluate(pred, truth=truth, context=context,
                                  include_input=False, return_raw_output=True)
        grouped = len(votes) == 1 and sum(votes.values()) == 2
        passed = (sum(votes.values()) == 2 and grouped == equivalent
                  and (response.get("correct") == "yes") == equivalent)
        row = dict(case=name, context=context, prediction=pred, reference=truth,
                   expected_equivalent=equivalent, votes=votes, response=response, passed=passed)
        results.append(row)
        print(json.dumps(row), flush=True)
        (output / "results.json").write_text(json.dumps(results, indent=2))
    breakpoint()  # inspect: results; failures distinguish grouping from judging
    if not all(row["passed"] for row in results):
        raise RuntimeError(f"Numeric tolerance checks failed; inspect {output}")


if __name__ == "__main__":
    check_numeric_tolerance()
