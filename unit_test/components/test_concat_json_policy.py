"""Validate native reasoning calls; add --live for a bounded factory-based check.

Run with PYTHONBREAKPOINT=0 python -m unit_test.components.test_concat_json_policy.
Live calls use OPENAI_API_BASE and OPENAI_API_KEY, with no benchmark files modified.
"""

import argparse
import json
from pathlib import Path

from lits.components.policy.concat_json import ConcatJSONPolicy
from lits.lm.base import ToolCall, ToolCallOutput


def check_tool_payloads():
    """Check concat_json.py::ConcatJSONPolicy._read_action_call on response values."""
    cases = [
        ({"content": "x + 1 = 3", "is_final": False}, True),
        ({"content": "2", "is_final": True}, True),
        ({"content": "x", "is_final": "false"}, False),
        ({"content": None, "is_final": False}, False),
        ({"content": "x"}, False),
        ({"content": "x", "is_final": False, "reward": 0.9}, False),
        ({"content": "", "is_final": False}, False),
        ({"content": "Use \f rac", "is_final": False}, False),
    ]
    for payload, expected in cases:
        response = ToolCallOutput("", [ToolCall("1", "submit_reasoning_step", payload)],
                                  "tool_use", {})
        try:
            result = ConcatJSONPolicy._read_action_call(response)
        except ValueError:
            accepted = False
        else:
            accepted = True
            key = "answer" if payload["is_final"] else "step"
            if result != {key: payload["content"]}:
                raise RuntimeError("Validated content changed")
        if accepted != expected:
            raise RuntimeError(f"Unexpected acceptance: {payload}")
    for calls in ([], [response.tool_calls[0]] * 2):
        try:
            ConcatJSONPolicy._read_action_call(ToolCallOutput("", calls, "tool_use", {}))
        except ValueError:
            continue
        raise RuntimeError("Invalid tool-call count accepted")
    print("Passed 10 native-tool validation checks")
    breakpoint()  # inspect: cases; independent rewards must not enter policy output


def check_live_factory():
    """Exercise factory.py::create_components_language_grounded and inherited sampling."""
    from tempfile import mkdtemp
    from lits import get_lm
    from lits.lm import setup_inference_logging
    from lits.components.factory import create_components_language_grounded
    from lits.components.reward.generative import GenerativePRM
    from lits.structures import ThoughtStep

    output = mkdtemp(prefix="lits-concat-native-tool-")
    model = get_lm("openai/local-instruct", enable_thinking=False)
    model.client = model.client.with_options(timeout=120, max_retries=0)
    setup_inference_logging(model, root_dir=output)
    _, policy, reward = create_components_language_grounded(
        model, model, task_name="scibench_calculus", search_framework="bfs",
        search_args={"n_actions": 1},
        component_args={"max_new_tokens": 1024, "n_for_correctness": 2,
                        "n_for_usefulness": 1},
        policy_override="concat_json", transition_override="concat",
        reward_override="generative",
    )
    if not isinstance(policy, ConcatJSONPolicy) or not isinstance(reward, GenerativePRM):
        raise RuntimeError("Factory selected incorrect components")
    for state in ([], [ThoughtStep(action="Integrating x gives x^2/2. Evaluating from 0 to 2 gives 2.")]):
        steps = policy.get_actions(state=state, n_actions=1,
                                   query="Evaluate the integral of x from 0 to 2.",
                                   query_idx=len(state), from_phase="expand")
        if len(steps) != 1 or steps[0].policy_score is not None:
            raise RuntimeError("Expected one unscored reasoning result")
        print(steps[0].to_dict())
    print(f"Live request/response logs: {output}")
    breakpoint()  # inspect: steps, policy, reward; format validity is not accuracy


def check_saved_request(trace_file):
    """Replay a saved initial policy prompt with native and corrective instructions.

    Exercise concat_json.py::ConcatJSONPolicy._generate_single_output using a
    real model, without changing source artifacts. The corrective case appends
    the production retry instruction explicitly; it does not simulate a failure.
    """
    from tempfile import mkdtemp
    from lits import get_lm
    from lits.lm import setup_inference_logging

    requests = []
    for line in trace_file.read_text().splitlines():
        for resource in json.loads(line)["resourceSpans"]:
            for scope in resource["scopeSpans"]:
                for span in scope["spans"]:
                    for attribute in span.get("attributes", []):
                        if attribute["key"] == "request_data":
                            request = json.loads(attribute["value"]["stringValue"])
                            if request.get("tools"):
                                requests.append(request)
    request = requests[0]
    messages = request["messages"]
    model = get_lm("openai/local-instruct", enable_thinking=False)
    model.client = model.client.with_options(timeout=120, max_retries=0)
    output = mkdtemp(prefix="lits-concat-tool-replay-")
    setup_inference_logging(model, root_dir=output)
    policy = ConcatJSONPolicy.from_config(
        model, {}, {"max_new_tokens": request["max_tokens"]},
        task_name="scibench_calculus",
    )
    model.sys_prompt = next(m["content"] for m in messages if m["role"] == "system")
    prompt = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    correction = policy._structured_retry_instruction(ValueError("Expected exactly one tool call"))
    results = []
    for suffix in ("", correction):
        try:
            raw, _ = policy._generate_single_output(prompt + suffix, request["temperature"])
        except ValueError as error:
            results.append({"error": str(error)})
        else:
            results.append(json.loads(raw))
    (Path(output) / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved-input observations: {output}\n{json.dumps(results, indent=2)}")
    if "error" in results[-1]:
        raise RuntimeError("Corrective request failed validation; inspect retained results")
    breakpoint()  # inspect: results; these are fresh responses, not historical recovery


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--saved-trace", type=Path)
    args = parser.parse_args()
    check_tool_payloads()
    if args.live:
        check_live_factory()
    if args.saved_trace:
        check_saved_request(args.saved_trace)
