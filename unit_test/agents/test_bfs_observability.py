"""Run a bounded real BFS and verify its end-to-end observability.

Run: PYTHONBREAKPOINT=0 python -m unit_test.agents.test_bfs_observability

This script calls the Tailscale model; telemetry stays local. LITS_TEST_MODEL_NAME,
LITS_TEST_BASE_URL, and LITS_TEST_API_KEY can override the configured endpoint.
Each invocation retains a fresh result directory. Lower-level offline checks
remain in unit_test.agents.test_observability_prototype.
"""

import json
import os
from pathlib import Path
from tempfile import mkdtemp

from lits import observability as tracing
from unit_test.agents.test_observability_prototype import read_archive, telemetry_environment


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


def check_bfs_search():
    """Run one real factory-built BFS and verify its local telemetry archive.

    Uses factory.py::create_components and bfs.py::BFSSearch.run with beam width
    one, two actions, depth three, structured ConcatPolicy, step_signal, and one
    GenerativePRM sample per criterion. The arithmetic smoke query uses math500
    prompts; it is not a MATH-500 dataset accuracy evaluation.

    Configure tracing before the shared model; retain server-reported model
    metadata, config, inference logs, checkpoints, tree, and OTLP files beneath
    config.py::ExperimentConfig.setup_directories. No HTTP-body capture file or
    online telemetry export is enabled. Failed checks retain artifacts too.
    """
    from lits import get_lm
    from lits.agents.tree.bfs import BFSSearch
    from lits.components.factory import create_components
    from lits.config import ExperimentConfig
    from lits.lm import setup_inference_logging
    from lits.log import setup_logging

    model_name = os.environ.get("LITS_TEST_MODEL_NAME", "openai/local-instruct")
    endpoint = os.environ.get("LITS_TEST_BASE_URL", "https://xinzhes-macbook-pro-2.tailde6fe4.ts.net/v1")
    results_root = Path(mkdtemp(prefix="lits-bfs-observability-"))
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
    print("Retained BFS results:", result_dir, flush=True)
    model = None
    with telemetry_environment():
        tracing.configure(enabled=True, console=False, archive_dir=directory, run_id=run_id)
        try:
            setup_logging("execution", result_dir)
            model = get_lm(model_name, base_url=endpoint,
                           api_key=os.environ.get("LITS_TEST_API_KEY", "no-key"),
                           enable_thinking=False, max_new_tokens=256)
            model.client.max_retries = 0
            model.client.timeout = 120
            metadata = {"query": query, "model_name": model_name, "base_url": endpoint,
                        "enable_thinking": False, "observations": observations,
                        "server_models": model.client.models.list().model_dump()}
            (result_dir / "run.json").write_text(json.dumps(metadata, indent=2))
            inference_logger = setup_inference_logging(model, root_dir=result_dir / "inference")
            components = create_components("language_grounded", "math500", model, model, model, None, config)
            search = BFSSearch(config.create_search_config(), *components,
                               checkpoint_dir=result_dir, observations=observations)
            result = search.run(query, query_idx=0)
            search.save_full_tree(query_idx=0)
            metrics = inference_logger.get_metrics_by_role()
        finally:
            complete = tracing.close()
            if model is not None:
                model.client.close()
    if not complete:
        raise RuntimeError("BFS telemetry archive did not close successfully")
    report = check_bfs_archive(directory, run_id, result, metrics)
    (result_dir / "verification.json").write_text(json.dumps(report, indent=2))
    print("PASS live BFS observations:", json.dumps(report, indent=2))
    print("Telemetry:", directory)
    breakpoint()  # inspect: result, report, result_dir, directory


if __name__ == "__main__":
    check_bfs_search()
