"""Verify real-model malformed JSON retry without executing a search.

Run with PYTHONBREAKPOINT=0 python -m unit_test.components.test_joint_action_retry.
Uses the configured Tailscale endpoint; at most four inference calls. Thirty-two output
tokens deliberately truncate tool arguments. This tests bounded failure, not recovery
of previously failed searches. A separate normal-limit call checks valid actions.
"""

import json
import logging
from pathlib import Path
import tempfile

from lits import get_lm
from lits.components.policy.concat import JointActionPolicy
from lits.lm import setup_inference_logging
from lits.structures import TrajectoryState


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
    if len(steps) != 3:
        raise RuntimeError(f"Expected three valid actions, got {steps}")
    result = {"malformed_attempts": calls, "valid_actions": [s.to_dict() for s in steps],
              "usage": model.inference_logger.get_metrics_by_role()}
    (directory / "results.json").write_text(json.dumps(result, indent=2))
    print(f"PASS normal-limit actions: {len(steps)}; usage: {result['usage']}", flush=True)
    breakpoint()  # inspect: steps, result, directory


if __name__ == "__main__":
    check_joint_action_retry()
