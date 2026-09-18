"""Run a bounded real BFS and verify its end-to-end observability.

Run: PYTHONBREAKPOINT=0 python -m unit_test.observability.test_bfs_observability
Add --mode all to check enabled, disabled, and failed-export runs sequentially.

This script calls the Tailscale model; telemetry stays local. LITS_TEST_MODEL_NAME,
LITS_TEST_BASE_URL, and LITS_TEST_API_KEY can override the configured endpoint.
Each invocation retains a fresh result directory. Lower-level offline checks
remain in unit_test.observability.test_observability_prototype.
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp

from lits import observability as tracing
from lits.observability import tracing as tracing_impl
from unit_test.observability.test_observability_prototype import read_archive, telemetry_environment


def check_bfs_archive(directory, run_id, result, metrics):
    """Compare saved BFS snapshots and model spans with the real tree and usage.

    References: bfs.py::BFSSearch.search, tracing.py::observe,
    openai_chat.py::OpenAIChatModel.__call__. No inference occurs here.
    """
    spans = read_archive(directory, run_id)["spans"]
    attributes = {s["spanId"]: {a["key"]: a["value"] for a in s["attributes"]} for s in spans}

    def value(span, key):
        """Read a scalar OTLP attribute; structured Logfire values remain JSON strings."""
        item = attributes[span["spanId"]][key]
        return int(item["intValue"]) if "intValue" in item else next(iter(item.values()))

    runs = [s for s in spans if s["name"] == "search.run"]
    iterations = [s for s in spans if s["name"] == "search.iteration"]
    if len(runs) != 1 or not 1 <= len(iterations) <= 3:
        raise RuntimeError("Expected one BFS run with at most three depth iterations")
    run = runs[0]
    # Post-search isolation checks have a separate trace, not BFS snapshots.
    spans = [s for s in spans if s["traceId"] == run["traceId"]]
    iteration_ids = {s["spanId"] for s in iterations}
    if any(s.get("parentSpanId") != run["spanId"] for s in iterations):
        raise RuntimeError("BFS depth spans do not belong to the run")

    nodes = {node.id: node for node in result.root.walk()}
    snapshots = [s for s in spans if s["name"] == "search.node"]
    if {value(s, "id") for s in snapshots} != set(nodes):
        raise RuntimeError("Node snapshots do not cover the returned tree")
    for snapshot in snapshots:
        node = nodes[value(snapshot, "id")]
        parent_span = iteration_ids if node.parent else {run["spanId"]}
        if (snapshot.get("parentSpanId") not in parent_span
                or json.loads(value(snapshot, "state")) != node.to_dict()["state"]
                or value(snapshot, "is_terminal") != node.is_terminal):
            raise RuntimeError("Node state, completion flag, or hierarchy differs from the tree")
        if node.parent and (value(snapshot, "score") != node.fast_reward
                            or node.is_terminal != node.state[-1].terminate):
            raise RuntimeError("Node reward or step_signal termination differs")

    scores = [s for s in spans if s["name"] == "search.score"]
    if {value(s, "node_id") for s in scores} != set(nodes) - {result.root.id}:
        raise RuntimeError("Missing child score snapshots")
    for score in scores:
        node = nodes[value(score, "node_id")]
        if (value(score, "score") != node.fast_reward
                or json.loads(value(score, "details")) != node.fast_reward_details
                or score.get("parentSpanId") not in iteration_ids):
            raise RuntimeError("Score snapshot differs from the existing reward")

    frontiers = [s for s in spans if s["name"] == "search.frontier"]
    for iteration in iterations:
        pair = {value(s, "when"): s for s in frontiers if s.get("parentSpanId") == iteration["spanId"]}
        if not pair:  # BFS also visits an empty depth when the previous one closed all branches.
            continue
        before, after = pair["before_selection"], pair["after_selection"]
        candidates = json.loads(value(before, "node_ids"))
        selected = json.loads(value(after, "node_ids"))
        if selected != sorted(candidates, key=lambda i: nodes[i].fast_reward, reverse=True)[:1]:
            raise RuntimeError("Recorded beam does not match default BFS selection")
        for snapshot, ids in ((before, candidates), (after, selected)):
            if json.loads(value(snapshot, "scores")) != [nodes[i].fast_reward for i in ids]:
                raise RuntimeError("Frontier scores differ from the returned tree")
    if not frontiers or len(nodes) > 7 or any(n.depth > 3 for n in nodes.values()):
        raise RuntimeError("Missing frontier observations or exceeded pilot bounds")
    if {n.id for n in result.terminal_nodes_collected} != {i for i, n in nodes.items() if n.is_terminal}:
        raise RuntimeError("BFS terminal collection differs from the completed nodes")
    if any(n.children for n in nodes.values() if n.is_terminal):
        raise RuntimeError("A terminal node was expanded")

    calls = [s for s in spans if s["name"] == "llm.call"]
    # The SDK also traces GET /models; only model responses count as inference.
    sdk = [s for s in spans if "gen_ai.response.model" in attributes[s["spanId"]]]
    if not calls or len(calls) != len(sdk) or len(calls) != metrics["num_calls"]:
        raise RuntimeError("SDK spans and inference-accounting call counts differ")
    totals, models, roles = [0, 0], set(), set()
    for call in calls:
        children = [s for s in sdk if s.get("parentSpanId") == call["spanId"]]
        if len(children) != 1 or call.get("parentSpanId") not in iteration_ids:
            raise RuntimeError("LLM call is not nested as run -> depth -> call -> SDK")
        child = children[0]
        request = json.loads(value(child, "request_data"))
        response = json.loads(value(child, "response_data"))
        if (not request.get("messages") or not response.get("message")
                or "temperature" not in request or "max_tokens" not in request
                or child["traceId"] != run["traceId"]):
            raise RuntimeError("Missing model messages, settings, response, or trace association")
        models.add(value(child, "gen_ai.response.model"))
        roles.add(value(call, "lits_role"))
        totals[0] += response["usage"]["prompt_tokens"]
        totals[1] += response["usage"]["completion_tokens"]
    if totals != [metrics["input_tokens"], metrics["output_tokens"]]:
        raise RuntimeError("Archived usage differs from inference accounting")
    if not any(r.startswith("policy_") for r in roles) or not any(r.startswith("evaluator_") for r in roles):
        raise RuntimeError("Missing policy or reward-model inference")
    return {"nodes": len(nodes), "terminals": len(result.terminal_nodes_collected),
            "expanded_parents": sum(bool(n.children) for n in nodes.values()),
            "response_models": sorted(models), "roles": sorted(roles), "metrics": metrics}


def check_bfs_result(result, result_dir):
    """Check bfs.py::BFSSearch.run output and persisted tree in every tracing mode.

    Check search invariants, not identical model outputs across independent runs.
    Nonempty depth buckets distinguish ordinary completion from the search's
    exception fallback; tree and checkpoints must remain available without telemetry.
    """
    nodes = list(result.root.walk())
    buckets = result.buckets_with_terminal
    if not buckets or {n.id for group in buckets.values() for n in group} != {n.id for n in nodes}:
        raise RuntimeError("BFS returned fallback output or incomplete depth buckets")
    if not 1 < len(nodes) <= 7 or any(n.depth > 3 or n.state is None for n in nodes):
        raise RuntimeError("Missing materialized nodes or exceeded pilot bounds")
    for depth, group in buckets.items():
        expanded = [n for n in group if n.children]
        eligible = [n for n in group if not n.is_terminal and not n.is_terminal_for_repeat]
        if (len(expanded) > 1 or any(len(n.children) > 2 for n in expanded)
                or any(n.depth != depth for n in group)):
            raise RuntimeError("BFS beam width, branching, or depth buckets changed")
        if expanded and expanded != sorted(eligible, key=lambda n: n.fast_reward, reverse=True)[:1]:
            raise RuntimeError("Expanded beam differs from default reward selection")
    if any(n.children or not n.state[-1].terminate for n in nodes if n.is_terminal):
        raise RuntimeError("A terminal was expanded or lacks its completion signal")
    if {n.id for n in result.terminal_nodes_collected} != {n.id for n in nodes if n.is_terminal}:
        raise RuntimeError("Terminal collection differs from completed nodes")

    def encode(node):
        """Use the same recursive layout as search_base.py::BaseTreeSearch.save_full_tree."""
        return {**node.to_dict(), "children": [encode(child) for child in node.children]}

    if json.loads((result_dir / "full_tree/full_tree_0.json").read_text()) != encode(result.root):
        raise RuntimeError("Saved tree differs from the returned result")
    checkpoints = sorted((result_dir / "checkpoints").glob("0_*.json"))
    if not checkpoints or any(not json.loads(path.read_text()) for path in checkpoints):
        raise RuntimeError("Search checkpoints are missing or empty")
    return {"nodes": len(nodes), "terminals": len(result.terminal_nodes_collected),
            "expanded_parents": sum(bool(n.children) for n in nodes), "checkpoints": len(checkpoints)}


def check_observation_isolation(search, result, inference_logger):
    """Re-observe real BFS nodes with full/empty rules; compare data and model usage.

    Calls bfs.py::BFSSearch._observe_node and BFSSearch._observe_frontier without
    expanding or scoring again. These checks emit a separate observation.check
    trace when enabled; they must not change tree fields, rules, or inference metrics.
    """
    nodes = list(result.root.walk())
    settings = search.observations

    def snapshot():
        """Copy node fields, replacing cyclic parent/children references with IDs."""
        records = [{**vars(node), "parent": node.parent.id if node.parent else None,
                    "children": [child.id for child in node.children],
                    "state": node.to_dict()["state"], "step": node.to_dict().get("step")}
                   for node in nodes]
        return deepcopy((records, settings, inference_logger.get_metrics_by_role()))

    before = snapshot()
    try:
        with tracing.span("observation.check"):
            for rules in (settings, {}):
                search.observations = rules
                for node in nodes:
                    search._observe_node(node)
                search._observe_frontier(result.root.children, "before_selection")
    finally:
        search.observations = settings
    if snapshot() != before:
        raise RuntimeError("Observation changed node data, display rules, or inference usage")


def check_bfs_search(mode="enabled"):
    """Run one real factory-built BFS and verify tracing isolation.

    Uses factory.py::create_components and bfs.py::BFSSearch.run with beam width
    one, two actions, depth three, structured ConcatPolicy, step_signal, and one
    GenerativePRM sample per criterion. The arithmetic smoke query uses math500
    prompts; it is not a MATH-500 dataset accuracy evaluation.

    Configure tracing before the shared model; retain server-reported model
    metadata, config, inference logs, checkpoints, tree, and OTLP files beneath
    config.py::ExperimentConfig.setup_directories. No HTTP-body capture file or
    online telemetry export is enabled. Failed checks retain artifacts too.

    ============== =================================== ====================================
    Mode           Expected search behavior            Expected telemetry
    ============== =================================== ====================================
    enabled        BFS completes normally              Complete archive: snapshots/spans
    disabled       BFS completes normally              No telemetry archive
    export_failure BFS completes and saves its tree    Incomplete archive; failure reported
    ============== =================================== ====================================

    export_failure closes the real span exporter after the first model response.
    HTTP hooks count requests without saving bodies or replacing the real model.
    """
    from lits import get_lm
    from lits.agents.tree.bfs import BFSSearch
    from lits.components.factory import create_components
    from lits.config import ExperimentConfig
    from lits.lm import setup_inference_logging
    from lits.log import setup_logging

    if mode not in ("enabled", "disabled", "export_failure"):
        raise ValueError(f"Unknown tracing mode: {mode}")
    enabled = mode != "disabled"
    model_name = os.environ.get("LITS_TEST_MODEL_NAME", "openai/local-instruct")
    endpoint = os.environ.get("LITS_TEST_BASE_URL", "https://xinzhes-macbook-pro-2.tailde6fe4.ts.net/v1")
    results_root = Path(mkdtemp(prefix=f"lits-bfs-{mode}-"))
    config = ExperimentConfig(
        dataset="math500", search_framework="bfs", search_algorithm="bfs",
        policy="concat", transition="concat", reward="generative",
        policy_model_name=model_name, eval_model_name=model_name, root_dir=str(results_root),
        search_args={"beam_size": 1, "n_actions": 2, "max_steps": 3,
                     "add_continuation": False, "force_terminating_on_depth_limit": True,
                     "terminate_constraints": ["step_signal"]},
        component_args={"output_format": "structured", "max_new_tokens": 256,
                        "n_for_correctness": 1, "n_for_usefulness": 1,
                        "think_for_correctness": False, "think_for_usefulness": False},
    )
    _, result_dir = config.setup_directories()
    result_dir = Path(result_dir)
    config.save_config(result_dir)
    directory = result_dir / "telemetry" / "example-0-repeat-0-attempt-0"
    run_id = directory.relative_to(results_root).as_posix()
    query = ("A shop has 24 boxes containing 18 pencils each. It sells 135 pencils, "
             "then packs the remaining pencils into bundles of 9. How many complete bundles remain?")
    observations = {
        "node": {"name": "search.node", "fields": ("id", "parent_id", "state", "action", "depth", "score", "is_terminal")},
        "frontier": {"name": "search.frontier", "fields": ("when", "node_ids", "scores")},
        "score": {"name": "search.score", "fields": ("node_id", "score", "details")},
    }
    original_observations = deepcopy(observations)
    http_calls, failure_after_call = 0, None

    def count_request(request):
        """Count actual completion requests, excluding the model-catalog lookup."""
        nonlocal http_calls
        if request.method == "POST" and request.url.path.endswith("/chat/completions"):
            http_calls += 1

    def close_exporter(response):
        """Fail the real file exporter mid-search without changing the model response."""
        nonlocal failure_after_call
        if (mode == "export_failure" and failure_after_call is None
                and response.status_code == 200
                and response.request.url.path.endswith("/chat/completions")):
            tracing_impl._archive.span_exporter.shutdown()
            failure_after_call = http_calls

    print("Retained BFS results:", result_dir, flush=True)
    model = None
    with telemetry_environment():
        options = {"archive_dir": directory, "run_id": run_id} if enabled else {}
        tracing.configure(enabled=enabled, console=False, **options)
        try:
            setup_logging("execution", result_dir)
            model = get_lm(model_name, base_url=endpoint,
                           api_key=os.environ.get("LITS_TEST_API_KEY", "no-key"),
                           enable_thinking=False, max_new_tokens=256)
            model.client.max_retries = 0
            model.client.timeout = 120
            model.client._client.event_hooks["request"].append(count_request)
            model.client._client.event_hooks["response"].append(close_exporter)
            metadata = {"query": query, "model_name": model_name, "base_url": endpoint,
                        "enable_thinking": False, "observations": observations, "tracing_mode": mode,
                        "server_models": model.client.models.list().model_dump()}
            (result_dir / "run.json").write_text(json.dumps(metadata, indent=2))
            inference_logger = setup_inference_logging(model, root_dir=result_dir / "inference")
            components = create_components("language_grounded", "math500", model, model, model, None, config)
            search = BFSSearch(config.create_search_config(), *components,
                               checkpoint_dir=result_dir, observations=observations)
            result = search.run(query, query_idx=0)
            search.save_full_tree(query_idx=0)
            metrics = inference_logger.get_metrics_by_role()
            report = check_bfs_result(result, result_dir)
            if observations != original_observations or search.observations is not observations:
                raise RuntimeError("Search changed the caller's observation settings")
            calls_before = http_calls
            check_observation_isolation(search, result, inference_logger)
            if http_calls != calls_before or http_calls != metrics["num_calls"]:
                raise RuntimeError("Observation added requests or HTTP/inference counts differ")
        finally:
            complete = tracing.close()
            if model is not None:
                model.client.close()
    if mode == "export_failure":
        status = json.loads((directory / "archive.json").read_text())
        if (failure_after_call != 1 or http_calls <= failure_after_call or complete
                or status["status"] != "incomplete" or not status.get("error")):
            raise RuntimeError("Search did not continue after a visible telemetry export failure")
    elif mode == "disabled":
        if not complete or directory.exists():
            raise RuntimeError("Disabled tracing wrote telemetry or failed shutdown")
    else:
        status = json.loads((directory / "archive.json").read_text())
        if not complete or status["status"] != "complete":
            raise RuntimeError("BFS telemetry archive did not close successfully")
        report.update(check_bfs_archive(directory, run_id, result, metrics))
    report.update(mode=mode, metrics=metrics, http_calls=http_calls,
                  failure_after_call=failure_after_call, observation_extra_calls=http_calls - calls_before,
                  archive_status="absent" if not enabled else status["status"])
    (result_dir / "verification.json").write_text(json.dumps(report, indent=2))
    print("PASS live BFS observations:", json.dumps(report, indent=2))
    print("Telemetry:", directory if enabled else "disabled; no archive")
    breakpoint()  # inspect: result, report, result_dir, directory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("enabled", "disabled", "export_failure", "all"), default="enabled")
    mode = parser.parse_args().mode
    for case in (("enabled", "disabled", "export_failure") if mode == "all" else (mode,)):
        check_bfs_search(case)
