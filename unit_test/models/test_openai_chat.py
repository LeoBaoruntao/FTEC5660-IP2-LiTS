"""Verify openai_chat.py::OpenAIChatModel.__call__ with real newline-stop requests.

Run ``python -m unit_test.models.test_openai_chat`` with LITS_TEST_MODEL_NAME,
LITS_TEST_BASE_URL and optional LITS_TEST_API_KEY for an OpenAI-compatible server.
HTTPX observes actual outgoing requests, without mocking the model or transport.
Retain requests, outputs and inference usage under benchmark_contract_results/.
Set PYTHONBREAKPOINT=0 for batch runs.
"""

import json
import os
from pathlib import Path
import tempfile

import httpx

from lits import get_lm
from lits.lm import setup_inference_logging
from lits.lm.base import DETERMINISTIC_TEMPERATURE
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


if __name__ == "__main__":
    check_newline_stop()
