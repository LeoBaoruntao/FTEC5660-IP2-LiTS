"""Verify real-model malformed JSON retry without executing a search.

Run with PYTHONBREAKPOINT=0 python -m unit_test.components.test_joint_action_retry.
Uses the configured Tailscale endpoint; at most four inference calls. Thirty-two output
tokens deliberately truncate tool arguments. This tests bounded failure, not recovery
of previously failed searches. A separate normal-limit call checks valid actions.
Add --schema-only to check tool schema and parsing without inference.
"""

import json
import logging
from pathlib import Path
import tempfile
import sys

from lits import get_lm
from lits.components.policy.concat import JointActionPolicy
from lits.lm import setup_inference_logging
from lits.lm.base import ToolCall, ToolCallOutput
from lits.structures import TrajectoryState


def check_reasoning_step_schema():
    """Check concat.py::JointActionPolicy._read_action_call with explicit payload fixtures.

    No model is substituted or called; fixtures exercise schema/parser agreement,
    step termination, score preservation, and invalid-payload rejection.
    """
    tool = JointActionPolicy._action_tool(2)
    schema = tool["input_schema"]
    if (tool["name"] != "submit_reasoning_steps" or schema["required"] != ["candidates"]
            or set(schema["properties"]) != {"candidates"}
            or schema["properties"]["candidates"]["minItems"] != 2
            or schema["properties"]["candidates"]["maxItems"] != 2):
        raise RuntimeError("Unexpected reasoning-step tool schema")
    for variant in schema["properties"]["candidates"]["items"]["anyOf"]:
        field = variant["required"][0]
        if not variant["properties"][field].get("description"):
            raise RuntimeError(f"Missing completion guidance for {field}")
    candidates = [{"step": "17 + 25 = 42.", "reward": 0.8},
                  {"answer": "42", "reward": 0.9}]
    cases = [(tool["name"], {"candidates": candidates}, True),
             ("submit_actions", {"candidates": candidates}, False),
             (tool["name"], {"actions": candidates}, False),
             (tool["name"], {"candidates": candidates[:1]}, False),
             (tool["name"], {"candidates": [candidates[0], {"answer": "42", "reward": True}]}, False)]
    for name, payload, valid in cases:
        response = ToolCallOutput("", [ToolCall("fixture", name, payload)], "tool_use", {})
        try:
            steps = JointActionPolicy._read_action_call(response, 2)
        except ValueError:
            if valid:
                raise
        else:
            if not valid:
                raise RuntimeError(f"Accepted invalid tool payload: {name}, {payload}")
            if (steps[0].action != "17 + 25 = 42." or steps[0].terminate
                    or steps[1].action != "The answer is 42" or not steps[1].terminate
                    or [step.policy_score for step in steps] != [0.8, 0.9]):
                raise RuntimeError("Reasoning-step conversion changed")
    print(f"PASS reasoning-step schema and parser: {len(cases)} cases; no inference")
    breakpoint()  # inspect: tool, cases, steps


def check_duplicate_candidates():
    """Check concat.py::JointActionPolicy._read_action_call deduplication without inference.

    Repeated candidates retain their first score, not additional branches/votes.
    All candidates remain validated, including duplicates with invalid scores.
    """
    candidates = [{"step": "The sum is 284.", "reward": 0.9},
                  {"step": "The sum is 284.", "reward": 0.8},
                  {"answer": "284", "reward": 0.7},
                  {"answer": "The answer is 284", "reward": 0.6},
                  {"step": "The answer is 284", "reward": 0.5}]
    response = ToolCallOutput("", [ToolCall(
        "fixture", "submit_reasoning_steps", {"candidates": candidates}
    )], "tool_use", {})
    steps = JointActionPolicy._read_action_call(response, len(candidates))
    actual = [(step.action, step.terminate, step.policy_score) for step in steps]
    expected = [("The sum is 284.", False, 0.9), ("The answer is 284", True, 0.7),
                ("The answer is 284", False, 0.5)]
    if actual != expected:
        raise RuntimeError(f"Duplicate handling changed: {actual}")
    candidates[1]["reward"] = True
    try:
        JointActionPolicy._read_action_call(response, len(candidates))
    except ValueError:
        pass
    else:
        raise RuntimeError("Invalid duplicate score escaped validation")
    print("PASS duplicate candidates: stable scores/order, termination flags, validation")
    breakpoint()  # inspect: actual, expected, response


def check_joint_action_retry():
    """Exercise concat.py::JointActionPolicy through base.py::Policy.get_actions.

    Both requests and decoding are real. No model/transport substitution is used.
    Failure to reproduce JSON truncation is reported, not treated as a pass.
    """
    directory = Path(tempfile.mkdtemp(prefix="lits-joint-json-retry-"))
    logging.basicConfig(filename=directory / "execution.log", level=logging.DEBUG)
    model = get_lm(
        "openai/local-instruct",
        base_url="https://xinzhes-macbook-pro-2.tailde6fe4.ts.net/v1",
        api_key="no-key", enable_thinking=False,
    )
    model.client = model.client.with_options(timeout=120, max_retries=0)
    setup_inference_logging(model, root_dir=str(directory))
    print(f"Artifacts: {directory}", flush=True)
    policy = JointActionPolicy(base_model=model, task_name="math500", n_actions=3,
                               max_new_tokens=32, temperature=1e-6)
    try:
        policy.get_actions(TrajectoryState([]), query="What is 17 + 25?",
                           query_idx=0, from_phase="expand")
    except json.JSONDecodeError as error:
        calls = model.inference_logger.get_metrics_by_role()["num_calls"]
        if calls != 2:
            raise RuntimeError(f"Expected two attempts, observed {calls}") from error
        print(f"PASS malformed JSON: two attempts, final error propagated: {error}", flush=True)
    else:
        raise RuntimeError("Real model did not reproduce malformed JSON")

    policy.max_new_tokens = 1024
    steps = policy.get_actions(TrajectoryState([]), query="What is 17 + 25?",
                               query_idx=1, from_phase="expand")
    if not 1 <= len(steps) <= 3:
        raise RuntimeError(f"Expected one to three unique actions, got {steps}")
    result = {"malformed_attempts": calls, "valid_actions": [s.to_dict() for s in steps],
              "usage": model.inference_logger.get_metrics_by_role()}
    (directory / "results.json").write_text(json.dumps(result, indent=2))
    print(f"PASS normal-limit actions: {len(steps)}; usage: {result['usage']}", flush=True)
    breakpoint()  # inspect: steps, result, directory


if __name__ == "__main__":
    check_reasoning_step_schema()
    check_duplicate_candidates()
    if "--schema-only" not in sys.argv:
        check_joint_action_retry()
