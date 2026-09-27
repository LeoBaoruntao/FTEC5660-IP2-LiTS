"""Validate native PRM arguments; --live exercises factory scoring on a real model.

Run PYTHONBREAKPOINT=0 python -m unit_test.components.test_generative_json_reward
--live. OPENAI_API_BASE and OPENAI_API_KEY configure openai/local-instruct.
No benchmark results are changed; inference usage logs and sample scores use a
temporary directory. No substitute model is used for validation tests.
"""

import argparse
import json
from pathlib import Path
from tempfile import mkdtemp

from lits.components.reward.generative_json import GenerativeJSONPRM
from lits.lm.base import ToolCall, ToolCallOutput, Output


def check_payloads():
    """Exercise generative_json.py::GenerativeJSONPRM._read_score_call on value objects."""
    cases = [
        ({"score": 0}, "correctness", False, True),
        ({"score": 1}, "correctness", False, True),
        ({"score": 0.5}, "usefulness", False, True),
        ({"reasoning": "Valid deduction", "score": 1}, "correctness", True, True),
        ({"score": 0.5}, "correctness", False, False),
        ({"score": True}, "correctness", False, False),
        ({"score": "1"}, "correctness", False, False),
        ({"score": -0.1}, "usefulness", False, False),
        ({"score": 1.1}, "usefulness", False, False),
        ({"score": float("nan")}, "usefulness", False, False),
        ({"score": float("inf")}, "usefulness", False, False),
        ({}, "correctness", False, False),
        ({"score": 1, "extra": 0}, "correctness", False, False),
        ({"score": 1}, "correctness", True, False),
        ({"score": 1, "reasoning": None}, "correctness", True, False),
    ]
    for payload, kind, thinking, expected in cases:
        response = ToolCallOutput("", [ToolCall("1", "submit_reward", payload)], "tool_use", {})
        try:
            GenerativeJSONPRM._read_score_call(response, kind, thinking)
            accepted = True
        except ValueError:
            accepted = False
        if accepted != expected:
            raise RuntimeError(f"Unexpected validation: {payload}, {kind}, {thinking}")
    for response in (
        Output("1"), ToolCallOutput("", [], "tool_use", {}),
        ToolCallOutput("", [ToolCall("1", "wrong_name", {"score": 1})], "tool_use", {}),
        ToolCallOutput("", [ToolCall(str(i), "submit_reward", {"score": 1}) for i in range(2)], "tool_use", {}),
    ):
        try:
            GenerativeJSONPRM._read_score_call(response, "correctness", False)
        except ValueError:
            continue
        raise RuntimeError("Invalid call envelope accepted")
    print(f"Passed {len(cases) + 4} native PRM validation checks")
    breakpoint()  # inspect: cases; invalid numeric values must not become zero rewards


def check_live(thinking=False):
    """Exercise factory.py::create_components_language_grounded and real fast_reward calls.

    Check binary rejection stops before usefulness, while accepted correctness
    samples lead to the saved usefulness mean. These are semantic checks too;
    valid tool arguments alone do not imply a correct judgment.
    """
    from lits import get_lm
    from lits.components.factory import create_components_language_grounded
    from lits.lm import setup_inference_logging
    from lits.structures import ThoughtStep

    output = Path(mkdtemp(prefix="lits-tool-prm-"))
    print(f"Retained logs: {output}")
    model = get_lm("openai/local-instruct", enable_thinking=thinking)
    model.client = model.client.with_options(timeout=120, max_retries=0)
    setup_inference_logging(model, root_dir=output / "inference")
    _, _, reward = create_components_language_grounded(
        model, model, task_name="scibench_calculus", search_framework="bfs",
        search_args={"n_actions": 1},
        component_args={"n_for_correctness": 2, "n_for_usefulness": 2,
                        "think_for_correctness": thinking, "think_for_usefulness": thinking},
        policy_override="concat_json", transition_override="concat",
        reward_override="generative_json",
    )
    if not isinstance(reward, GenerativeJSONPRM):
        raise RuntimeError("Factory did not select generative_json")
    reward.save_dir = str(output)
    reward.file_path_correctness = str(output / "correctness.jsonl")
    reward.file_path_usefulness = str(output / "usefulness.jsonl")
    for index, action in enumerate(("17 + 25 = 42.", "17 + 25 = 99.")):
        value, _ = reward.fast_reward([], ThoughtStep(action=action), "What is 17 + 25?",
                                      query_idx=index, from_phase="expand")
        samples = {}
        for kind in ("correctness", "usefulness"):
            path = output / f"{kind}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            samples[kind] = [r["score"] for r in rows if r["query_idx"] == index]
        if index == 0:
            if samples["correctness"] != [1.0, 1.0] or len(samples["usefulness"]) != 2:
                raise RuntimeError(f"Unexpected accepted-step samples: {samples}")
            if value != sum(samples["usefulness"]) / 2:
                raise RuntimeError("Reward differs from usefulness mean")
        elif value != 0 or samples != {"correctness": [0.0], "usefulness": []}:
            raise RuntimeError(f"Incorrect-step gate failed: {samples}")
        print(f"example={index}, thinking={thinking}, reward={value}, samples={samples}")
    breakpoint()  # inspect: output, samples, value; usage logs and tool arguments remain available


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    check_payloads()
    if args.live:
        check_live(args.thinking)
