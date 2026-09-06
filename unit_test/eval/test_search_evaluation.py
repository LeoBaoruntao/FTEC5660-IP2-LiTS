"""Compare CLI and shared tree evaluation against recorded baseline expectations.

Run from lits_llm/ with demos/ on PYTHONPATH:
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_search_evaluation
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_search_evaluation --llm-model ollama/llama3.2:latest

This checks the saved-result pipeline, not individual benchmark algorithms.
agents/test_extract_answers.py owns extraction/voting checks and the saved RAP
fixture reused here; tools/test_dbbench_loader.py and tools/test_kgqa_registry.py
own the benchmark comparators. Their registered implementations are used directly.

All additional cases use real LiTS nodes and steps with controlled field values.
The CLI constructs a real OpenAI-compatible client but these deterministic cases
need no inference, dataset download, or database connection. Answer-extraction LLM
fallback checks remain in agents/test_extract_answers.py; --llm-model additionally
checks live binary correctness judging here. Each CLI fixture is also evaluated
directly from its original nodes and from its reloaded terminal records.
Logs and all three outcomes remain in a fresh temporary directory.
"""

from __future__ import annotations

import contextlib
import argparse
import json
import logging
import os
from pathlib import Path
import re
import tempfile

from lits.agents.tree.node import MCTSNode, SearchNode
from lits.agents.tree.search_base import SearchResult
from lits.benchmarks.registry import (
    get_evaluator,
    register_dataset,
    register_evaluator,
    register_resource,
)
from lits.cli.eval_search import (
    _create_llm_evaluator,
    evaluate_from_checkpoints,
    load_terminal_nodes_from_file,
)
from lits.cli.search import save_terminal_nodes
from lits.components.registry import ComponentRegistry
from lits.components.utils import get_fn_retrieve_answer
from lits.eval.search import check_answer, evaluate_search_result, select_env_terminal
from lits.lm import get_lm, setup_inference_logging
from lits.structures.base import TrajectoryState
from lits.structures.env_grounded import EnvState, EnvStep
from lits.structures.qa import ThoughtStep
from lits.structures.tool_use import ToolUseStep

import lits_benchmark.blocksworld  # noqa: F401 -- registers the real goal checker
from lits_benchmark.dbbench import load_dbbench_resource
from lits_benchmark.kgqa import load_kgqa_resource
from unit_test.agents.test_extract_answers import (
    REAL_TERMINAL_NODES_JSON,
    check_saved_rap_retrieval_inputs,
)


MODEL_NAME = "openai/local-instruct"
GOALS = (
    "the red block is on top of the blue block and "
    "the green block is on top of the yellow block"
)
PARTIAL_STATE = "the red block is on top of the blue block"
MATH_DATASET = "eval_check_math"
DB_DATASET = "eval_check_dbbench"
FLOAT_DATASET = "eval_check_kgqa"


def load_examples(examples):
    """Return supplied fixture examples through the real dataset registry."""
    return examples


def register_examples():
    """Register fixture datasets using existing benchmark evaluators/resources."""
    register_dataset(MATH_DATASET, task_type="language_grounded")(load_examples)
    for name, benchmark, resource in (
        (DB_DATASET, "dbbench", load_dbbench_resource),
        (FLOAT_DATASET, "kgqa", load_kgqa_resource),
    ):
        register_dataset(name, task_type="tool_use")(load_examples)
        register_evaluator(name)(get_evaluator(benchmark))
        register_resource(name)(resource)


def terminal(step=None, *, returns=(), fast_reward=0.5, node_class=MCTSNode):
    """Build a terminal whose state, step, and returns survive LiTS serialization."""
    if isinstance(step, EnvStep):
        state = EnvState(init_state="the blocks are on the table")
        state.append(step)
    else:
        state = TrajectoryState([step]) if step is not None else None
    node = node_class(
        state=state,
        action=step.get_action() if step is not None else None,
        fast_reward=fast_reward,
        is_terminal=True,
    )
    if step is not None:
        node.step = step
    if isinstance(node, MCTSNode):
        node.cum_rewards = list(returns)
    return node


def check_case(output_dir, name, dataset, nodes, truth, expected, *, query="Test query",
               selection=None, llm_model=None):
    """Compare CLI, original-node, and reloaded-node evaluation on the same fixture.

    References: search.py::save_terminal_nodes,
    eval_search.py::load_terminal_nodes_from_file,
    eval_search.py::evaluate_from_checkpoints, search.py::evaluate_search_result.
    expected is (prediction, correct, optional_score, environment_progress).
    Shared calls receive nodes/reference and caller-configured retrieval/checkers,
    without a result directory, dataset name, or model-creation arguments.
    llm_model optionally enables live binary judging for tool-use fixtures; the
    shared calls retain inference usage records, and the CLI retains routing logs.
    """
    case_dir = output_dir / name
    case_dir.mkdir()
    result = SearchResult(root=SearchNode(state=None, action=query),
                          terminal_nodes_collected=nodes)
    save_terminal_nodes(result, query, 0, case_dir, logging.getLogger("lits.test"))
    options = {"terminal_selection": selection} if selection is not None else {}
    with (case_dir / "console.log").open("w") as console:
        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
            report = evaluate_from_checkpoints(
                str(case_dir), dataset, llm_model or MODEL_NAME,
                dataset_kwargs={"examples": [{"question": query, "answer": truth}]},
                llm_eval_mode="binary" if llm_model else "none", **options,
            )
    log = (case_dir / "eval.log").read_text()
    matches = re.findall(r"\[0\] Pred='(.*?)', Truth=.*?, Correct=(True|False)", log)
    if report is None or len(report["examples"]) != 1 or len(matches) != 1:
        raise RuntimeError(f"{name}: missing evaluation outcome; inspect {case_dir}")
    if json.loads((case_dir / "eval_results.json").read_text()) != report:
        raise RuntimeError(f"{name}: returned and saved reports disagree")
    item = report["examples"][0]
    observed = (
        matches[0][0], item["correct"], report.get("mean_score"),
        report.get("soft_accuracy"),
    )
    if observed != expected or item["score"] != float(item["correct"]):
        raise RuntimeError(f"{name}: {observed!r} != {expected!r}; inspect {case_dir}")
    if matches[0][1] != str(item["correct"]) or report["accuracy"] != float(item["correct"]):
        raise RuntimeError(f"{name}: logged and saved correctness disagree")
    shared_options = dict(options)
    if dataset == MATH_DATASET:
        task_type = "language_grounded"
        shared_options["retrieve_answer"] = get_fn_retrieve_answer(get_lm(MODEL_NAME))
    elif dataset == "blocksworld":
        task_type = "env_grounded"
        shared_options["goal_check"] = ComponentRegistry.get_transition(dataset).goal_check
    else:
        task_type = "tool_use"
        shared_options["custom_evaluator"] = get_evaluator(dataset)

    if llm_model:
        model = get_lm(llm_model)
        inference_logger = setup_inference_logging(model, root_dir=str(case_dir))
        shared_options["llm_evaluator"] = _create_llm_evaluator(
            model, "binary", logging.getLogger("lits.test"),
        )

    saved = load_terminal_nodes_from_file(case_dir / "terminal_nodes" / "terminal_nodes_0.json")
    expected_shared = dict(zip(
        ("prediction", "correct", "score", "environment_progress"), expected,
    ))
    shared_results = {}
    for source, terminals, source_query in (
        ("in_memory", nodes, query),
        ("reloaded", saved["terminal_nodes"], saved["query"]),
    ):
        outcome = evaluate_search_result(
            terminals, source_query, truth, task_type=task_type, **shared_options,
        )
        if outcome != expected_shared:
            raise RuntimeError(f"{name}/{source}: {outcome!r} != {expected_shared!r}")
        shared_results[source] = outcome

    if llm_model:
        registered_result = shared_options["custom_evaluator"](expected[0], truth)
        registered_correct = (
            registered_result == 1.0 if isinstance(registered_result, float)
            else bool(registered_result)
        )
        needs_fallback = not registered_correct
        if ("LLM fallback:" in log) != needs_fallback:
            raise RuntimeError(f"{name}: unexpected CLI fallback routing")
        usage = [json.loads(line) for line in Path(inference_logger.filepath).read_text().splitlines()]
        if needs_fallback:
            if not 2 <= len(usage) <= 6:
                raise RuntimeError(f"{name}: expected one to three calls per shared evaluation")
        elif usage:
            raise RuntimeError(f"{name}: registered success unnecessarily invoked the LLM")
        if any(record["role"] != "evaluator_general" for record in usage):
            raise RuntimeError(f"{name}: unexpected inference role")
        shared_results["registered_result"] = registered_result
        shared_results["llm_model"] = llm_model
        shared_results["shared_inference_calls"] = len(usage)

    print(f"PASS {name}: CLI = in-memory = reloaded: {observed}")
    return {"case": name, "expected": expected, "observed": observed, **shared_results}


def check_language(output_dir):
    """Check CLI voting/numeric scoring using the existing RAP fixture and edge cases."""
    fixture_path = output_dir / "rap_terminals.json"
    fixture_path.write_text(REAL_TERMINAL_NODES_JSON)
    fixture = load_terminal_nodes_from_file(fixture_path)
    # Reuse saved answer text in the paper's ThoughtStep representation. Native
    # RAP retrieval has its own regression check in test_extract_answers.py.
    nodes = [terminal(ThoughtStep(action=node.state[-1].get_answer(), terminate=True),
                      fast_reward=node.fast_reward, node_class=SearchNode)
             for node in fixture["terminal_nodes"]]
    # Saved answers are 18, 3, 12, and 127: a four-way tie selects the first.
    # The fixture has no gold label; 3 is a controlled reference for this check.
    checks = [check_case(
        output_dir, "math_saved_answers_as_thoughts", MATH_DATASET, nodes, "3",
        ("18", False, None, None), query=fixture["query"],
    )]
    cases = (
        ("math_vote_beats_reward", [("42", 0.1), ("43", 0.9), ("42", 0.2)], "42",
         ("42", True, None, None)),
        ("math_tie_preserves_order", [("43", 0.1), ("42", 0.9)], "42",
         ("43", False, None, None)),
        ("math_numeric_normalization", [("42.0", 0.5)], "42",
         ("42.0", True, None, None)),
        ("math_no_terminals", [], "42", ("", False, None, None)),
    )
    for name, answers, truth, expected in cases:
        nodes = [terminal(ThoughtStep(action=f"The answer is {answer}.", terminate=True),
                          fast_reward=score, node_class=SearchNode)
                 for answer, score in answers]
        checks.append(check_case(output_dir, name, MATH_DATASET, nodes, truth, expected))
    checks.append(check_case(output_dir, "math_missing_state", MATH_DATASET,
                             [terminal(node_class=SearchNode)], "42",
                             ("", False, None, None)))
    return checks


def check_tool_use(output_dir):
    """Check CLI terminal ranking, answer retrieval, and registered bool/float scoring."""
    checks = []
    # Mean and maximum returns deliberately disagree with each other and fast_reward.
    ranked = [terminal(ToolUseStep(answer="43"), returns=[0.9, 0.1], fast_reward=0.99),
              terminal(ToolUseStep(answer="42"), returns=[0.7, 0.7], fast_reward=0.1)]
    for selection, answer, correct in ((None, "42", True), ("q-mean", "42", True),
                                       ("q-max", "43", False), ("first", "43", False)):
        checks.append(check_case(output_dir, f"tool_{selection or 'default'}", DB_DATASET,
                                 ranked, ["42"], (answer, correct, None, None),
                                 selection=selection))
    ties = [terminal(ToolUseStep(answer="43"), returns=[0.7]),
            terminal(ToolUseStep(answer="42"), returns=[0.7])]
    empty_returns = [terminal(ToolUseStep(answer="43"), fast_reward=0.1),
                     terminal(ToolUseStep(answer="42"), fast_reward=0.9)]
    missing_returns = [terminal(ToolUseStep(answer="43"), node_class=SearchNode),
                       terminal(ToolUseStep(answer="42"), node_class=SearchNode)]
    for name, nodes, expected in (
        ("tied_returns", ties, ("43", False, None, None)),
        ("empty_returns", empty_returns, ("43", False, None, None)),
        ("missing_returns", missing_returns, ("43", False, None, None)),
        ("visited_beats_empty", [empty_returns[0], ranked[1]], ("42", True, None, None)),
        ("no_terminals", [], ("", False, None, None)),
        ("missing_answer", [terminal()], ("", False, None, None)),
    ):
        checks.append(check_case(output_dir, f"tool_{name}", DB_DATASET, nodes,
                                 ["42"], expected))
    # Characterize the existing zero normalization; this refactor must not silently fix it.
    checks.append(check_case(output_dir, "tool_missing_answer_matches_zero", DB_DATASET,
                             [terminal()], ["0"], ("", True, None, None)))
    dict_state = terminal()
    dict_state.state = [{"answer": "42"}]
    object_state = terminal(ToolUseStep(answer="42"))
    del object_state.step
    for name, node, prediction, correct in (
        ("dict_state_answer", dict_state, "42", True),
        ("object_state_without_node_step", object_state, "", False),
    ):
        checks.append(check_case(output_dir, f"tool_{name}", DB_DATASET, [node], ["42"],
                                 (prediction, correct, None, None)))
    for name, prediction, correct, score in (
        ("full", "Obedient, Intelligent", True, 1.0),
        ("partial", "Obedient", False, 2 / 3),
        ("empty", "", False, 0.0),
    ):
        checks.append(check_case(
            output_dir, f"float_{name}", FLOAT_DATASET,
            [terminal(ToolUseStep(answer=prediction), returns=[0.5])],
            {"names": ["Obedient", "Intelligent"], "ids": []},
            (prediction, correct, score, None),
        ))
    return checks


def check_environment(output_dir):
    """Check CLI ranking/progress and search.py::select_env_terminal input validation."""
    partial = terminal(EnvStep(next_state=PARTIAL_STATE), returns=[0.9, 0.1])
    complete = terminal(EnvStep(next_state=GOALS), returns=[0.7, 0.7])
    checks = []
    for name, nodes, prediction, correct, progress in (
        ("maximum_not_mean", [partial, complete], "incorrect", False, 0.5),
        ("complete", [complete], "correct", True, 1.0),
        ("no_terminals", [], "incorrect", False, 0.0),
        ("missing_state", [terminal()], "incorrect", False, 0.0),
        ("empty_returns", [terminal(EnvStep(next_state=PARTIAL_STATE)),
                           terminal(EnvStep(next_state=GOALS))], "incorrect", False, 0.5),
        ("missing_returns", [terminal(EnvStep(next_state=PARTIAL_STATE), node_class=SearchNode),
                             terminal(EnvStep(next_state=GOALS), node_class=SearchNode)],
         "incorrect", False, 0.5),
        ("tied_returns", [terminal(EnvStep(next_state=PARTIAL_STATE), returns=[0.7]),
                          complete], "incorrect", False, 0.5),
    ):
        checks.append(check_case(output_dir, f"env_{name}", "blocksworld", nodes, None,
                                 (prediction, correct, None, progress), query=GOALS))

    nonterminal = terminal(EnvStep(next_state=PARTIAL_STATE), returns=[0.1])
    nonterminal.is_terminal = False
    for name, nodes, invalid_index in (
        ("nonterminal_only", [nonterminal], 0),
        ("mixed_terminals", [complete, nonterminal], 1),
    ):
        expected = f"terminal_nodes[{invalid_index}] is not terminal"
        try:
            select_env_terminal(nodes)
        except ValueError as error:
            observed = str(error)
            if observed != expected:
                raise RuntimeError(f"{name}: {observed!r} != {expected!r}") from error
        else:
            raise RuntimeError(f"{name}: nonterminal input was accepted")
        print(f"PASS env_{name}: {observed}")
        checks.append({"case": f"env_{name}", "expected": expected, "observed": observed})
        breakpoint()  # inspect: nodes, invalid_index, observed
    return checks


def check_answer_outcomes():
    """Check search.py::check_answer exact matching, precedence, and error handling."""
    kgqa_evaluator = get_evaluator("kgqa")
    checks = []
    for name, pred, truth, evaluator, expected in (
        ("exact_match", "42", 42, None, (True, None)),
        ("exact_mismatch", "42", "43", None, (False, None)),
        ("exact_format", "42.0", "42", None, (False, None)),
        # KGQA expects entity lists; a parsed integer reference yields no gold
        # entities. A custom failure must not fall through to exact matching.
        ("custom_before_exact", "42", "42", kgqa_evaluator, (False, 0.0)),
        # A non-iterable gold-answer field makes the real comparator raise.
        ("custom_exception", "Obedient", {"names": 42, "ids": []},
         kgqa_evaluator, (False, None)),
    ):
        observed = check_answer(pred, truth, custom_evaluator=evaluator)
        if observed != expected:
            raise RuntimeError(f"{name}: {observed!r} != {expected!r}")
        print(f"PASS {name}: {observed}")
        checks.append({"case": name, "expected": expected, "observed": observed})
    breakpoint()  # inspect: checks; no LLM evaluator is supplied
    return checks


def check_live_fallback(output_dir, model_name):
    """Exercise search.py::check_answer binary fallback with a real model and DBBench checker.

    Each fixture runs through the CLI and both shared-input paths. Correctness is
    checked on simple answers; this is a routing check, not a judge-quality estimate.
    """
    checks = []
    for name, prediction, correct in (
        ("registered_success", "42", True),
        ("binary_rescue", "There are forty-two records.", True),
        ("binary_reject", "43", False),
    ):
        checks.append(check_case(
            output_dir, f"live_{name}", DB_DATASET,
            [terminal(ToolUseStep(answer=prediction), returns=[0.5])], ["42"],
            (prediction, correct, None, None), llm_model=model_name,
        ))
    breakpoint()  # inspect: checks; live_* directories contain eval.log and inferencelogger.log
    return checks


def main():
    """Compare evaluation paths, optionally run live judging, and retain logs/outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-model", help="LiTS model name for optional live binary fallback checks")
    args = parser.parse_args()
    os.environ.setdefault("OPENAI_API_KEY", "no-key")
    os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8000/v1")
    register_examples()
    output_dir = Path(tempfile.mkdtemp(prefix="lits-eval-baseline-"))
    print(f"Evaluation artifacts: {output_dir}", flush=True)
    checks = [check_saved_rap_retrieval_inputs()]
    checks.extend(check_language(output_dir))
    checks.extend(check_tool_use(output_dir))
    checks.extend(check_environment(output_dir))
    checks.extend(check_answer_outcomes())
    if args.llm_model:
        checks.extend(check_live_fallback(output_dir, args.llm_model))
    (output_dir / "baseline.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"Passed {len(checks)} checks (29 CLI/in-memory/reloaded comparisons, 1 retrieval, "
          "2 terminal validation, 5 answer checking"
          f"{', 3 live binary fallback cases' if args.llm_model else ''}). "
          f"Baseline: {output_dir / 'baseline.json'}")
    breakpoint()  # inspect: checks, output_dir; each case contains terminal_nodes/ and eval.log


if __name__ == "__main__":
    main()
