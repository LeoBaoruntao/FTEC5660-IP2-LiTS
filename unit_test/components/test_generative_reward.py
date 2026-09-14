"""Sequential score parsing and real-model reward checks.

Run ``python -m unit_test.components.test_generative_reward``. Configure
``LITS_TEST_MODEL_NAME``, ``LITS_TEST_BASE_URL``, and ``LITS_TEST_API_KEY`` for
an OpenAI-compatible endpoint; set ``PYTHONBREAKPOINT=0`` for batch execution.
"""

import json
import os
from pathlib import Path
import tempfile

from lits import get_lm
from lits.components.reward.generative import GenerativePRM
from lits.eval import parse_reasoning_and_label
from lits.lm import setup_inference_logging
from lits.log import setup_logging
from lits.structures import ThoughtStep


def check_score_parsing() -> None:
    """Check ``generative.py::GenerativePRM._parse_score`` without inference."""
    valid = [
        ("0", "correctness", 0.0),
        ("1\n\nThe calculation is correct.", "correctness", 1.0),
        ("0", "usefulness", 0.0),
        ("1.0", "usefulness", 1.0),
        ("0.5\n\nThe step is partially useful.", "usefulness", 0.5),
        ("\n  **0.5**\nAn explanation follows.", "usefulness", 0.5),
        ("```\n0.5\n```", "usefulness", 0.5),
        ('"0.5"', "usefulness", 0.5),
        ("1e-3\nA small contribution.", "usefulness", 0.001),
        (parse_reasoning_and_label("<think>Partly useful.</think>\n0.5")["label"],
         "usefulness", 0.5),
    ]
    for text, score_type, expected in valid:
        actual = GenerativePRM._parse_score(text, score_type)
        if actual != expected:
            raise RuntimeError(f"{text!r}: expected {expected}, received {actual}")
    invalid = [
        ("", "usefulness"), ("   ", "usefulness"),
        ("The answer is 42.\n0.5", "usefulness"),
        ("0.5 because it helps", "usefulness"), ("1/2", "usefulness"),
        ("-0.1", "usefulness"), ("1.1", "usefulness"),
        ("nan", "usefulness"), ("inf", "usefulness"),
        ("0.5", "correctness"), ("1", "unknown"),
    ]
    for text, score_type in invalid:
        try:
            GenerativePRM._parse_score(text, score_type)
        except ValueError:
            continue
        raise RuntimeError(f"Unexpectedly accepted {score_type}: {text!r}")
    print(f"Score parsing: {len(valid) + len(invalid)} checks passed")
    breakpoint()  # inspect: valid and invalid score cases


def check_live_reward(thinking: bool = False) -> None:
    """Check ``base.py::RewardModel.fast_reward`` using two real model calls.

    The default matches the paper's thinking-disabled reward configuration;
    ``thinking=True`` separately checks the tagged-reasoning prompt. Saved raw
    replies must reproduce their scores and the returned reward. Retain logs.
    """
    model_kwargs = {}
    if base_url := os.environ.get("LITS_TEST_BASE_URL"):
        model_kwargs.update(base_url=base_url, api_key=os.environ.get("LITS_TEST_API_KEY", "no-key"))
    model = get_lm(os.environ.get(
        "LITS_TEST_MODEL_NAME", "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    ), **model_kwargs)
    run_dir = Path(tempfile.mkdtemp(prefix="lits-generative-reward-"))
    print(f"Retained run directory: {run_dir}")
    setup_logging("execution", str(run_dir), verbose=True, allowed_namespaces=("lits",))
    inference_logger = setup_inference_logging(model, root_dir=run_dir / "inference")
    query = ("What is 17 + 25? First identify the required operation without evaluating "
             "it; only afterward compute the result.")
    state = [ThoughtStep(action="Identify the required operation, which is addition.")]
    reward = GenerativePRM.from_config(
        base_model=model, search_args={}, task_name="math500", save_dir=str(run_dir),
        component_args={"think_for_correctness": thinking, "think_for_usefulness": thinking,
                        "n_for_correctness": 1, "n_for_usefulness": 1},
    )
    with inference_logger.log_context(thinking=thinking):
        score, _ = reward.fast_reward(
            state, ThoughtStep(action="The answer is 42", terminate=True),
            query, query_idx=0, from_phase="expand",
        )
    saved_scores = {}
    for score_type in ("correctness", "usefulness"):
        saved = json.loads((run_dir / f"{score_type}.jsonl").read_text())
        text = saved["text"]
        if thinking:
            label = parse_reasoning_and_label(text)["label"]
            text = label if label is not None else text
        parsed = GenerativePRM._parse_score(text, score_type)
        if parsed != saved["score"]:
            raise RuntimeError(f"Saved {score_type} differs from its raw response")
        saved_scores[score_type] = parsed
    if saved_scores["correctness"] != 1.0 or score != saved_scores["usefulness"]:
        raise RuntimeError(f"Unexpected returned reward {score}: {saved_scores}")
    print(f"thinking={thinking}: {saved_scores}, returned reward={score}")
    records = [json.loads(line) for line in Path(inference_logger.filepath).read_text().splitlines()]
    if len(records) != 2 or any("logits" in record["role"] for record in records):
        raise RuntimeError("Expected two scoring calls without logits fallback")
    print("Live reward: two scoring calls, no fallback")
    breakpoint()  # inspect: run_dir, saved_scores, score, records


if __name__ == "__main__":
    check_score_parsing()
    check_live_reward()
