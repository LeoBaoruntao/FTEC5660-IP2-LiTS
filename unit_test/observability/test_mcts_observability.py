"""Verify mcts.py::MCTSSearch.search spans with existing no-inference regressions.

Run: PYTHONBREAKPOINT=0 python -m unit_test.observability.test_mcts_observability
Requires the observability extra. Real components reuse prepared rewards/actions;
terminal completion, revisits, rollout, backup, and injected failures are checked
by the existing search tests rather than duplicated here. Archives are retained.
"""

import os
from pathlib import Path
from tempfile import mkdtemp

from lits import get_lm, observability as tracing
from unit_test.agents.tree.test_selection_hooks import check_selected_terminal_backup
from unit_test.agents.tree.test_search_failures import check_search_failures
from unit_test.observability.test_observability_prototype import read_archive, telemetry_environment


def check_mcts_spans():
    """Compare real-SDK hierarchy with iteration indices; run regressions in both modes."""
    root = Path(mkdtemp(prefix="lits-mcts-tracing-"))
    settings = {
        "LITS_TEST_MODEL_NAME": "openai/local-instruct",
        "LITS_TEST_BASE_URL": "http://localhost:8080/v1",
        "HF_HUB_OFFLINE": "1",
    }
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        model = get_lm("openai/local-instruct", base_url=settings["LITS_TEST_BASE_URL"], api_key="no-key")
        for enabled in (False, True):
            archive = root / str(enabled)
            with telemetry_environment():
                options = {"archive_dir": archive, "run_id": "mcts-check"} if enabled else {}
                tracing.configure(enabled=enabled, console=False, **options)
                try:
                    check_selected_terminal_backup()
                    check_search_failures(model)
                finally:
                    closed = tracing.close()
            if not closed:
                raise RuntimeError("MCTS archive failed to close")
            if not enabled:
                if archive.exists():
                    raise RuntimeError("Disabled tracing created an archive")
                continue
            spans = read_archive(archive, "mcts-check")["spans"]
            runs = [s for s in spans if s["name"] == "search.run" and any(
                a["key"] == "algorithm" and a["value"].get("stringValue") == "mcts"
                for a in s["attributes"]
            )]
            if not runs:
                raise RuntimeError("MCTS run spans are missing")
            counts = set()
            for run in runs:
                iterations = [s for s in spans if s["name"] == "search.iteration"
                              and s.get("parentSpanId") == run["spanId"]]
                indices = sorted(int(a["value"]["intValue"]) for s in iterations
                                 for a in s["attributes"] if a["key"] == "iteration")
                if indices != list(range(len(iterations))):
                    raise RuntimeError(f"Incorrect MCTS iteration indices: {indices}")
                counts.add(len(iterations))
                for span in iterations:
                    if (span["traceId"] != run["traceId"]
                            or not int(run["startTimeUnixNano"]) <= int(span["startTimeUnixNano"])
                            <= int(span["endTimeUnixNano"]) <= int(run["endTimeUnixNano"])):
                        raise RuntimeError("Iteration escaped its run span")
            if not {1, 3} <= counts:
                raise RuntimeError("Missing early-stop or terminal-revisit coverage")
            print(f"PASS MCTS hierarchy: {len(runs)} runs; iteration counts {sorted(counts)}")
            breakpoint()  # inspect: root, runs, counts
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(f"Retained MCTS archives: {root}")


if __name__ == "__main__":
    check_mcts_spans()
