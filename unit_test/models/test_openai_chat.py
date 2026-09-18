"""Verify openai_chat.py::OpenAIChatModel.__call__ with real requests.

Run ``python -m unit_test.models.test_openai_chat`` with LITS_TEST_MODEL_NAME,
LITS_TEST_BASE_URL and optional LITS_TEST_API_KEY for an OpenAI-compatible server.
HTTPX observes actual outgoing requests, without mocking the model or transport.
Retain requests, outputs and inference usage under benchmark_contract_results/.
Use --observability for two traced text/tool calls and one untraced text call;
its telemetry and HTTP evidence are retained in a fresh temporary directory.
Set PYTHONBREAKPOINT=0 for batch runs.
"""

import json
import os
from pathlib import Path
import tempfile

import httpx

from lits import get_lm
from lits import observability as tracing
from lits.lm import setup_inference_logging
from lits.lm.base import DETERMINISTIC_TEMPERATURE, Output, ToolCallOutput
from lits.lm.openai_chat import OpenAIChatModel


def check_newline_stop():
    """Check stop merging, default behavior, caller-list preservation and live output."""
    model = get_lm(
        os.environ["LITS_TEST_MODEL_NAME"],
        base_url=os.environ["LITS_TEST_BASE_URL"],
        api_key=os.environ.get("LITS_TEST_API_KEY", "no-key"),
        max_new_tokens=32,
    )
    if not isinstance(model, OpenAIChatModel):
        raise TypeError("This check requires an OpenAIChatModel")
    root = Path("benchmark_contract_results").resolve()
    root.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="openai-newline-", dir=root))
    print("Retained results:", run_dir)
    setup_inference_logging(model, root_dir=run_dir / "inference")
    requests = []

    def record_request(request):
        """Observe the actual HTTP request body without modifying it."""
        requests.append(json.loads(request.content))

    cases = [
        ("default", {}, None),
        ("disabled_list", {"stop": ["END"], "new_line_stop": False}, ["END"]),
        ("newline_only", {"new_line_stop": True}, ["\n"]),
        ("existing_string", {"stop": "END", "new_line_stop": True}, ["END", "\n"]),
        ("existing_list", {"stop": ["END"], "new_line_stop": True}, ["END", "\n"]),
        ("existing_newline", {"stop": ["\n", "END"], "new_line_stop": True}, ["\n", "END"]),
        ("empty_list", {"stop": [], "new_line_stop": True}, ["\n"]),
    ]
    original_client = model.client
    results = []
    try:
        with httpx.Client(event_hooks={"request": [record_request]}, timeout=60) as transport:
            model.client = original_client.with_options(http_client=transport, max_retries=0)
            for name, options, expected_stop in cases:
                before = json.loads(json.dumps(options))
                output = model(
                    "Output exactly two lines, with no explanation or code fence:\nalpha\nbeta",
                    temperature=DETERMINISTIC_TEMPERATURE, **options,
                )
                checks = {
                    "request_stop": requests[-1].get("stop") == expected_stop,
                    "caller_options_unchanged": options == before,
                    "nonempty_output": bool(output.text),
                    "single_line": not options.get("new_line_stop") or "\n" not in output.text,
                }
                result = {"case": name, "request": requests[-1], "output": output.text,
                          "checks": checks, "passed": all(checks.values())}
                results.append(result)
                (run_dir / "results.json").write_text(json.dumps(results, indent=2))
                print(name, checks, repr(output.text))
                breakpoint()  # inspect: result, expected_stop, before, options
    finally:
        model.client = original_client
        original_client.close()
    if not all(result["passed"] for result in results):
        raise RuntimeError("Newline-stop checks failed; inspect results.json")
    print(f"Passed {len(results)} live request cases")


def check_observability():
    """Compare observability/tracing.py::instrument_openai against actual HTTP exchanges.

    Configure before constructing the model, as application callers do. HTTPX
    hooks observe the constructor's real client; no client/transport is replaced.
    Verify text/tool output conversion, role-specific usage, SDK span nesting,
    duplicate-registration protection, and cleanup before reusing an untraced client.
    """
    from unit_test.observability.test_observability_prototype import read_archive, telemetry_environment
    from openai.types.chat import ChatCompletionMessage

    root = Path(tempfile.mkdtemp(prefix="lits-openai-tracing-"))
    directory = root / "telemetry" / "example-0-repeat-0-attempt-0"
    run_id = directory.relative_to(root).as_posix()
    wire = {"requests": [], "responses": []}
    print("Retained results:", root, flush=True)

    def record_request(request):
        """Save only the actual request body, never authentication headers."""
        wire["requests"].append(json.loads(request.content))
        (root / "http.json").write_text(json.dumps(wire, indent=2))

    def record_response(response):
        """Read and retain the non-streaming response without changing its contents."""
        response.read()
        wire["responses"].append(response.json())
        (root / "http.json").write_text(json.dumps(wire, indent=2))

    tool = {
        "name": "record_choice", "description": "Record the requested candidate selection.",
        "input_schema": {"type": "object", "properties": {
            "candidate_id": {"type": "string", "enum": ["node-a", "node-b"]},
        }, "required": ["candidate_id"], "additionalProperties": False},
        "strict": True,
    }
    cases = (
        ("policy_0_expand", [("user", "Reply with exactly: ready")],
         {"stop": ["END"], "new_line_stop": True}),
        ("evaluator_0_selector", "Call record_choice with candidate_id node-b. Do not use prose.",
         {"tools": [tool]}),
    )
    model = None
    with telemetry_environment():
        tracing.configure(enabled=True, console=False, archive_dir=directory, run_id=run_id)
        try:
            model = get_lm(
                os.environ["LITS_TEST_MODEL_NAME"], base_url=os.environ["LITS_TEST_BASE_URL"],
                api_key=os.environ.get("LITS_TEST_API_KEY", "no-key"),
                sys_prompt="Follow the user's instructions precisely.", enable_thinking=False,
            )
            if not isinstance(model, OpenAIChatModel):
                raise TypeError("This check requires an OpenAIChatModel")
            inference_logger = setup_inference_logging(model, root_dir=root / "inference")
            model.client.max_retries = 0
            model.client.timeout = httpx.Timeout(120)
            model.client._client.event_hooks["request"].append(record_request)
            model.client._client.event_hooks["response"].append(record_response)
            instrumented_request = model.client.request
            tracing.instrument_openai(model.client)
            if model.client.request is not instrumented_request:
                raise RuntimeError("Repeated registration wrapped the client again")

            for role, prompt, options in cases:
                output = model(prompt, role=role, temperature=DETERMINISTIC_TEMPERATURE,
                               max_new_tokens=64, **options)
                message = wire["responses"][-1]["choices"][0]["message"]
                if "tools" in options:
                    if not isinstance(output, ToolCallOutput):
                        raise RuntimeError("Model did not return the requested native tool call")
                    expected = message["tool_calls"][0]
                    actual = output.tool_calls[0]
                    if (actual.id, actual.name, actual.input_args) != (
                        expected["id"], expected["function"]["name"],
                        json.loads(expected["function"]["arguments"]),
                    ):
                        raise RuntimeError("Tracing changed tool-call conversion")
                elif type(output) is not Output or output.text != (message.get("content") or "").strip():
                    raise RuntimeError("Tracing changed text output conversion")

            if not tracing.close():
                raise RuntimeError("Model telemetry archive failed to close")
            archived = {path: path.read_bytes() for path in directory.iterdir()}
            if model.client.request is instrumented_request:
                raise RuntimeError("Closing tracing left its client instrumentation installed")
            tracing.instrument_openai(model.client)  # Disabled: leave the reused client unchanged.
            model(cases[0][1], role="policy_0_disabled", temperature=DETERMINISTIC_TEMPERATURE,
                  max_new_tokens=64, **cases[0][2])
            if wire["requests"][0] != wire["requests"][-1]:
                raise RuntimeError("Tracing changed the outgoing request body")
            if any(path.read_bytes() != content for path, content in archived.items()):
                raise RuntimeError("Disabled call wrote to a closed archive")
        finally:
            tracing.close()
            if model is not None:
                model.client.close()

    records = read_archive(directory, run_id)
    spans = records["spans"]
    if len(spans) != 4 or records["logRecords"]:
        raise RuntimeError("Expected one LiTS parent and one SDK child per traced request")
    attrs = {span["spanId"]: {item["key"]: item["value"] for item in span["attributes"]}
             for span in spans}
    for index, (role, _, _) in enumerate(cases):
        parent = next(span for span in spans if span["name"] == "llm.call"
                      and attrs[span["spanId"]]["lits_role"] == {"stringValue": role})
        child = next(span for span in spans if span.get("parentSpanId") == parent["spanId"])
        parent_attrs, child_attrs = attrs[parent["spanId"]], attrs[child["spanId"]]
        request = json.loads(child_attrs["request_data"]["stringValue"])
        extra = json.loads(parent_attrs["request_extra_body"]["stringValue"])
        if {**request, **(extra or {})} != wire["requests"][index]:
            raise RuntimeError("Archived messages/settings differ from the actual HTTP request")
        response = wire["responses"][index]
        saved_response = json.loads(child_attrs["response_data"]["stringValue"])
        message = response["choices"][0]["message"]
        if (saved_response["message"] != ChatCompletionMessage.model_validate(message).model_dump()
                or child_attrs["gen_ai.response.model"] != {"stringValue": response["model"]}
                or child["traceId"] != parent["traceId"]):
            raise RuntimeError("Archived response/model or hierarchy differs from the actual call")
        if not child_attrs.get("gen_ai.input.messages") or not child_attrs.get("gen_ai.output.messages"):
            raise RuntimeError("SDK omitted Gen AI message fields")
        usage = response["usage"]
        metrics = inference_logger.get_metrics_by_role(role)
        if (metrics["num_calls"] != 1 or metrics["input_tokens"] != usage["prompt_tokens"]
                or metrics["output_tokens"] != usage["completion_tokens"]
                or saved_response["usage"]["prompt_tokens"] != usage["prompt_tokens"]
                or saved_response["usage"]["completion_tokens"] != usage["completion_tokens"]):
            raise RuntimeError("Tracing or inference accounting changed reported usage")
    if len(wire["requests"]) != 3 or inference_logger.get_metrics_by_role()["num_calls"] != 3:
        raise RuntimeError("Unexpected inference calls or usage records")
    print("PASS traced text/tool requests, messages/settings, response, role and usage")
    print("PASS duplicate registration, cleanup, and unchanged untraced request:", root)
    breakpoint()  # inspect: root, wire, records, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observability", action="store_true", help="Run the live tracing check")
    args = parser.parse_args()
    if args.observability:
        check_observability()
    else:
        check_newline_stop()
