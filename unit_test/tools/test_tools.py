"""Inspect real tools; the module entry point runs local SQL tracing checks only.

Run ``PYTHONBREAKPOINT=0 python -m unit_test.tools.test_tools`` with the
observability extra to retain a local OTLP archive. No model or AWS calls run.
"""

import json
from pathlib import Path
from tempfile import mkdtemp
import sys
sys.path.append('../')
from lits.clients.als_client import AmazonLocationClient
from dotenv import load_dotenv

def test_gocode():
    load_dotenv()
    REGION = "ap-southeast-2"
    client = AmazonLocationClient(
        region=REGION,
    )
    coords = client.request("Melbourne CBD")
    print(f"📍 Coordinates (longitude, latitude): {coords} (type: {type(coords)})")
    print(client.ping())
    # client.delete_place_index('MyPlaceIndex')
    

def check_observability():
    """Check tools/utils.py::_attempt_tool_call through real SQL tools and transitions.

    Execute only SELECT/list operations on a query-only, in-memory SQLite database.
    Compare tracing off/on; retain one span per retry, original tool results, and
    exceptions. A selected SQL string containing a network-error marker exercises
    the existing string-result classifier, not an actual unavailable server.
    """
    from lits import observability as tracing
    from lits.clients.sql_client import SQLDBClient
    from lits.components.transition.tool_use import ToolUseTransition
    from lits.structures import ToolUseAction, ToolUseState, ToolUseStep
    from lits.tools.sql_tools import ListSQLDatabaseTool, QuerySQLDatabaseTool
    from lits.tools.utils import ToolServerDownError, execute_tool_action
    from lits.utils import PREFIX_FOR_ERROR_OBSERVATION
    from unit_test.agents.test_observability_prototype import read_archive, telemetry_environment

    root = Path(mkdtemp(prefix="lits-tool-tracing-"))
    directory = root / "telemetry" / "example-0-repeat-0-attempt-0"
    run_id = directory.relative_to(root).as_posix()
    print("Retained results:", root, flush=True)
    outputs = {}
    with telemetry_environment():
        for enabled in (False, True):
            options = {"archive_dir": directory, "run_id": run_id} if enabled else {}
            tracing.configure(enabled=enabled, console=False, **options)
            client = SQLDBClient("sqlite:///:memory:")
            query_tool = QuerySQLDatabaseTool(client)
            tools = [query_tool, ListSQLDatabaseTool(client)]
            object.__setattr__(query_tool, "server_down_retry_delays", (0, 0))
            cases = (
                (query_tool.name, {"query": "SELECT 42"}),
                (tools[1].name, "   "),
                (query_tool.name, {}),
                (query_tool.name, {"query": "SELECT * FROM missing_table"}),
                (query_tool.name, {"query": "SELECT 'ConnectionError: connection refused'"}),
            )
            sdk_loaded = "logfire" in sys.modules  # SQL dependencies can import it independently.
            try:
                with client.db._engine.connect() as connection:
                    connection.exec_driver_sql("PRAGMA query_only=ON")
                results = []
                with tracing.span("tool.check"):
                    for tool_name, action_input in cases:
                        action = json.dumps({"action": tool_name, "action_input": action_input})
                        try:
                            results.append(execute_tool_action(action, tools))
                        except ToolServerDownError as error:
                            results.append((type(error).__name__, error.tool_name, str(error)))

                    state = ToolUseState()
                    state.append(ToolUseStep(action=ToolUseAction(json.dumps({
                        "action": query_tool.name, "action_input": {"query": "SELECT 41"},
                    })), observation="[(41,)]"))
                    before = [step.to_dict() for step in state]
                    step = ToolUseStep(action=ToolUseAction(json.dumps({
                        "action": query_tool.name, "action_input": {"query": "SELECT 43"},
                    })))
                    new_state, aux = ToolUseTransition(tools).step(state, step, query_idx=0)
                    if ([item.to_dict() for item in state] != before
                            or [item.to_dict() for item in new_state[:-1]] != before
                            or new_state[-1].observation != "[(43,)]"
                            or aux != {"confidence": 1.0}):
                        raise RuntimeError("Tracing changed the transition's input state or result")
                    results.append(new_state[-1].observation)
                outputs[enabled] = results
            finally:
                client.db._engine.dispose()
                closed = tracing.close()
            if not closed:
                raise RuntimeError("Tool telemetry archive failed to close")
            if not enabled and (directory.exists() or ("logfire" in sys.modules) != sdk_loaded):
                raise RuntimeError("Disabled tool tracing imported the SDK or created an archive")

    results = outputs[True]
    if (outputs[False] != results or results[:2] != ["[(42,)]", ""]
            or not results[2].startswith(PREFIX_FOR_ERROR_OBSERVATION)
            or "no such table" not in results[3]
            or results[4][:2] != ("ToolServerDownError", query_tool.name)):
        raise RuntimeError("Tool results/classification changed between tracing modes")
    from opentelemetry.trace import StatusCode

    records = read_archive(directory, run_id)
    parent = next(span for span in records["spans"] if span["name"] == "tool.check")
    calls = [span for span in records["spans"] if span["name"] == "tool.call"]
    if len(calls) != 8 or len(records["spans"]) != 9 or records["logRecords"]:
        raise RuntimeError("Expected eight tool attempts under one parent, with no native logs")
    expected = list(cases[:4]) + [cases[4]] * 3 + [(query_tool.name, {"query": "SELECT 43"})]
    expected_outputs = results[:4] + ["[('ConnectionError: connection refused',)]"] * 3 + [results[5]]
    for index, (span, (name, action_input)) in enumerate(zip(calls, expected)):
        attrs = {item["key"]: item["value"] for item in span["attributes"]}
        if (span.get("parentSpanId") != parent["spanId"] or span["traceId"] != parent["traceId"]
                or attrs["tool_name"] != {"stringValue": name}
                or json.loads(attrs["tool_input"]["stringValue"]) != ({} if index == 1 else action_input)):
            raise RuntimeError("Archived input/name or parent-child hierarchy is incorrect")
        if index == 2:
            if ("tool_output" in attrs or span.get("status", {}).get("code") != StatusCode.ERROR.value
                    or not any(event["name"] == "exception" for event in span.get("events", []))):
                raise RuntimeError("Raised tool exception was not recorded")
        elif (attrs["tool_output"] != {"stringValue": expected_outputs[index]}
                or span.get("status", {}).get("code") == StatusCode.ERROR.value):
            raise RuntimeError("Returned tool output was changed or mistaken for a raised exception")
    print("PASS real SQL tools: normalized input, output, raised/returned errors, three retry attempts")
    print("PASS enabled/disabled equivalence, transition state, eight nested tool spans")
    print("Retained results:", root)
    breakpoint()  # inspect: outputs, calls, directory


if __name__ == "__main__":
    check_observability()
