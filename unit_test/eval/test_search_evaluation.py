"""Compare CLI and shared tree evaluation against explicit expected outcomes.

Run from lits_llm/ with demos/ on PYTHONPATH:
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_search_evaluation
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_search_evaluation --llm-model ollama/llama3.2:latest
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_search_evaluation --compare-chain-cli-ref 8fdbfb9
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_search_evaluation --math-llm-model openai/local-instruct
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_search_evaluation --voting-model ollama/llama3.2:latest

This checks the saved-result pipeline, not individual benchmark algorithms.
agents/test_extract_answers.py owns legacy extraction checks and the saved RAP
fixture reused here; tools/test_dbbench_loader.py and tools/test_kgqa_registry.py
own the benchmark comparators. Their registered implementations are used directly.
Voting contracts, grouping, and live semantic checks live in test_answer_voting.py.

All additional cases use real LiTS nodes and steps with controlled field values.
Default cases need no inference, dataset download, or database connection. Empty
language results use a real configured client without calling it. --voting-model
checks language-tree CLI routing; --llm-model additionally
checks live binary and F1 judging. Legacy extraction utility checks remain in
agents/test_extract_answers.py. Each CLI fixture is also evaluated
directly from its original nodes and from its reloaded terminal records.
Logs and all three outcomes remain in a fresh temporary directory.
--compare-chain-cli-ref additionally runs the current and specified Git revision's
CLI on matching chain/pass-at-N fixtures without inference. State serialization
itself is covered in test_state_serialization.py; these checks cover evaluation.
"""

from __future__ import annotations

import contextlib
import argparse
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types

from lits.agents.tree.node import MCTSNode, SearchNode
from lits.agents.tree.search_base import SearchResult
from lits.agents.tree.bfs import BFSResult
from lits.agents.tree.mcts import MCTSResult
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
from lits.eval import aggregate_results
from lits.eval.answer_voting import _raw_terminal_answers
from lits.eval.search import (
    check_answer, evaluate_search_result, select_env_terminal, select_tool_use_terminal,
)
from lits.lm import get_lm, setup_inference_logging
from lits.structures.base import TrajectoryState
from lits.structures.env_grounded import EnvState, EnvStep
from lits.structures.qa import ThoughtStep
from lits.structures.tool_use import ToolUseState, ToolUseStep

import lits_benchmark.blocksworld  # noqa: F401 -- registers the real goal checker
from lits_benchmark.math_qa_eval import eval_output  # registers real numeric QA checkers
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
    register_evaluator(MATH_DATASET)(get_evaluator("math500"))
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
               selection=None, voting_model_name=None, llm_model=None, llm_eval_mode="binary",
               enable_think_eval=False):
    """Compare CLI, original-node, and reloaded-node evaluation on the same fixture.

    References: search.py::save_terminal_nodes,
    eval_search.py::load_terminal_nodes_from_file,
    eval_search.py::evaluate_from_checkpoints, search.py::evaluate_search_result.
    expected is (prediction, correct, optional_score, environment_progress).
    Shared calls receive nodes/reference and caller-configured models/checkers,
    without a result directory, dataset name, or model-creation arguments.
    llm_model optionally enables live binary or F1 judging for answer-based fixtures; the
    shared calls retain inference usage records, and the CLI retains routing logs.
    Language fixtures always supply a voter; voting_model_name selects its real model
    without enabling the judge. Both roles reuse one model when judging is enabled.
    enable_think_eval is applied at model construction in both CLI and shared paths.
    """
    model_name = llm_model or voting_model_name or MODEL_NAME
    case_dir = output_dir / name
    case_dir.mkdir()
    result = SearchResult(root=SearchNode(state=None, action=query),
                          terminal_nodes_collected=nodes)
    save_terminal_nodes(result, query, 0, case_dir, logging.getLogger("lits.test"))
    options = {"terminal_selection": selection} if selection is not None else {}
    with (case_dir / "console.log").open("w") as console:
        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
            report = evaluate_from_checkpoints(
                str(case_dir), dataset, model_name,
                dataset_kwargs={"examples": [{"question": query, "answer": truth}]},
                llm_eval_mode=llm_eval_mode if llm_model else "none",
                enable_think_eval=enable_think_eval, **options,
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
    model = None
    if dataset == MATH_DATASET or llm_model:
        model = get_lm(model_name, enable_thinking=enable_think_eval)
        inference_logger = setup_inference_logging(model, root_dir=str(case_dir))
        if f"enable_think_eval={enable_think_eval}" not in log:
            raise RuntimeError(f"{name}: CLI model setting was not recorded")
    if dataset == MATH_DATASET:
        task_type = "language_grounded"
        shared_options["voting_model"] = model
        shared_options["custom_evaluator"] = get_evaluator(dataset)
    elif dataset == "blocksworld":
        task_type = "env_grounded"
        shared_options["goal_check"] = ComponentRegistry.get_transition(dataset).goal_check
    else:
        task_type = "tool_use"
        shared_options["custom_evaluator"] = get_evaluator(dataset)

    if llm_model:
        shared_options["llm_evaluator"] = _create_llm_evaluator(
            model, llm_eval_mode, logging.getLogger("lits.test"),
        )
        shared_options["llm_eval_mode"] = llm_eval_mode

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

    if model is not None:
        usage_path = Path(inference_logger.filepath)
        usage = ([json.loads(line) for line in usage_path.read_text().splitlines()]
                 if usage_path.exists() else [])
        voting_calls = sum(record["role"] == "evaluator_voting" for record in usage)
        judge_calls = sum(record["role"] == "evaluator_general" for record in usage)
        needs_voting = task_type == "language_grounded" and any(
            answer.strip() for answer in _raw_terminal_answers(nodes).values()
        )
        if voting_calls != 2 * int(needs_voting):
            raise RuntimeError(f"{name}: expected one voting call per nonblank shared evaluation")
        if log.count("Answer voting output:") != int(needs_voting):
            raise RuntimeError(f"{name}: unexpected CLI voting routing")
        deterministic_result = check_answer(
            expected[0], truth, custom_evaluator=shared_options.get("custom_evaluator"),
        )
        needs_fallback = bool(llm_model and llm_eval_mode != "none"
                              and expected[0].strip() and not deterministic_result[0])
        if ("LLM fallback:" in log) != needs_fallback:
            raise RuntimeError(f"{name}: unexpected CLI fallback routing")
        if needs_fallback:
            if not 2 <= judge_calls <= 6:
                raise RuntimeError(f"{name}: expected one to three judge calls per shared evaluation")
        elif judge_calls:
            raise RuntimeError(f"{name}: unexpected correctness judge call")
        if voting_calls + judge_calls != len(usage):
            raise RuntimeError(f"{name}: unexpected inference role")
        shared_results["deterministic_result"] = deterministic_result
        shared_results["llm_model"] = llm_model
        shared_results["llm_eval_mode"] = llm_eval_mode if llm_model else "none"
        shared_results["shared_inference_calls"] = len(usage)
        shared_results["voting_calls"] = voting_calls
        shared_results["judge_calls"] = judge_calls

    print(f"PASS {name}: CLI = in-memory = reloaded: {observed}")
    return {"case": name, "expected": expected, "observed": observed, **shared_results}


def check_empty_language(output_dir):
    """Check empty language results make no calls even with voting and judging configured."""
    checks = []
    for name, nodes in (("math_no_terminals", []),
                        ("math_missing_state", [terminal(node_class=SearchNode)]),
                        ("math_blank_answer", [terminal(ThoughtStep(action=" ", terminate=True))])):
        checks.append(check_case(output_dir, name, MATH_DATASET, nodes, "42",
                                 ("", False, None, None), llm_model=MODEL_NAME))
    return checks


def check_language(output_dir, model_name):
    """Check thinking-enabled CLI voting without judging, including saved RAP answer text."""
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
        voting_model_name=model_name, enable_think_eval=True,
    )]
    cases = (
        ("math_vote_beats_reward", [("42", 0.1), ("43", 0.9), ("42", 0.2)], "42",
         ("42", True, None, None)),
        ("math_tie_preserves_order", [("43", 0.1), ("42", 0.9)], "42",
         ("43", False, None, None)),
        ("math_numeric_normalization", [("42.0", 0.5)], "42",
         ("42.0", True, None, None)),
    )
    for name, answers, truth, expected in cases:
        nodes = [terminal(ThoughtStep(action=f"The answer is {answer}.", terminate=True),
                          fast_reward=score, node_class=SearchNode)
                 for answer, score in answers]
        checks.append(check_case(output_dir, name, MATH_DATASET, nodes, truth, expected,
                                 voting_model_name=model_name, enable_think_eval=True))
    return checks


def check_cli_thinking_config(output_dir):
    """Exercise eval_search.py::main flag/config precedence on empty terminals, without inference."""
    checks = []
    for name, search_args, flags, expected in (
        ("default", {}, [], False),
        ("saved_enabled", {"enable_think_eval": True}, [], True),
        ("enable_override", {"enable_think_eval": False}, ["--enable_think_eval"], True),
        ("disable_override", {"enable_think_eval": True}, ["--no-enable_think_eval"], False),
        ("invalid_string", {"enable_think_eval": "false"}, [], None),
    ):
        case_dir = output_dir / f"cli_thinking_{name}"
        case_dir.mkdir()
        result = SearchResult(root=SearchNode(state=None, action="Test query"),
                              terminal_nodes_collected=[])
        save_terminal_nodes(result, "Test query", 0, case_dir, logging.getLogger("lits.test"))
        config = {"dataset": MATH_DATASET, "eval_model_name": MODEL_NAME,
                  "dataset_kwargs": {"examples": [{"question": "Test query", "answer": "42"}]},
                  "search_args": search_args}
        (case_dir / "config.json").write_text(json.dumps(config))
        command = [sys.executable, "-c",
                   "from unit_test.eval.test_search_evaluation import register_examples; "
                   "register_examples(); from lits.cli.eval_search import main; "
                   "raise SystemExit(main())",
                   "--result_dir", str(case_dir), "--llm-eval", "none", *flags]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        console = completed.stdout + completed.stderr
        (case_dir / "console.log").write_text(console)
        if completed.returncode != (1 if expected is None else 0):
            raise RuntimeError(f"{name}: CLI exited {completed.returncode}; inspect {case_dir / 'console.log'}")
        if expected is None:
            if "enable_think_eval must be a boolean" not in console:
                raise RuntimeError(f"{name}: invalid thinking setting was not rejected")
        else:
            log = (case_dir / "eval.log").read_text()
            if f"enable_think_eval={expected}" not in log:
                raise RuntimeError(f"{name}: incorrect CLI thinking setting; inspect {case_dir}")
            if "Answer voting output:" in log or "LLM fallback:" in log:
                raise RuntimeError(f"{name}: empty fixture unexpectedly invoked inference")
        checks.append({"case": f"cli_thinking_{name}", "enable_think_eval": expected})
    breakpoint()  # inspect: checks; each case retains config.json, console.log, and eval.log
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
    missing_returns = [terminal(ToolUseStep(answer="43"), fast_reward=0.1, node_class=SearchNode),
                       terminal(ToolUseStep(answer="42"), fast_reward=0.9, node_class=SearchNode)]
    for name, nodes, expected in (
        ("tied_returns", ties, ("43", False, None, None)),
        ("empty_returns", empty_returns, ("42", True, None, None)),
        ("missing_returns", missing_returns, ("42", True, None, None)),
        ("visited_beats_empty", [empty_returns[0], ranked[1]], ("42", True, None, None)),
        ("no_terminals", [], ("", False, None, None)),
        ("missing_answer", [terminal()], ("", False, None, None)),
    ):
        checks.append(check_case(output_dir, f"tool_{name}", DB_DATASET, nodes,
                                 ["42"], expected))
    fallback_cases = (
        ("missing_returns", missing_returns, "42"),
        ("empty_returns", empty_returns, "42"),
        ("mixed_fallback_wins", [ranked[1], terminal(ToolUseStep(answer="43"), fast_reward=0.9)], "43"),
        ("mixed_tie", [terminal(ToolUseStep(answer="43"), fast_reward=0.7), ranked[1]], "43"),
        ("backed_overrides_fast", [terminal(ToolUseStep(answer="43"), returns=[0.1], fast_reward=0.99),
                                   ranked[1]], "42"),
        ("zero_return", [terminal(ToolUseStep(answer="43"), returns=[0.0], fast_reward=0.99),
                          terminal(ToolUseStep(answer="42"), fast_reward=0.5)], "42"),
        ("zero_fast_reward", [terminal(ToolUseStep(answer="43"), fast_reward=-0.1),
                               terminal(ToolUseStep(answer="42"), fast_reward=0.0)], "42"),
    )
    for strategy in ("q-mean", "q-max"):
        for name, nodes, prediction in fallback_cases:
            if strategy == "q-mean" and name in ("missing_returns", "empty_returns"):
                continue  # Covered by default q-mean cases above.
            checks.append(check_case(
                output_dir, f"tool_{strategy}_{name}", DB_DATASET, nodes, ["42"],
                (prediction, prediction == "42", None, None), selection=strategy,
            ))
    for name, nodes, truth, prediction, correct in (
        ("missing_answer_not_zero", [terminal()], ["0"], "", False),
        ("empty_answer_not_zero", [terminal(ToolUseStep(answer=""))], ["0"], "", False),
        ("blank_answer_not_zero", [terminal(ToolUseStep(answer="   "))], ["0"], "   ", False),
        ("no_terminals_not_zero", [], ["0"], "", False),
        ("zero_answer", [terminal(ToolUseStep(answer="0"))], ["0"], "0", True),
        ("explicit_empty_result", [terminal(ToolUseStep(answer="[]"))], [], "[]", True),
        ("empty_result_not_zero", [terminal(ToolUseStep(answer="[]"))], ["0"], "[]", False),
    ):
        checks.append(check_case(output_dir, f"tool_{name}", DB_DATASET, nodes, truth,
                                 (prediction, correct, None, None),
                                 llm_model=MODEL_NAME if name != "empty_result_not_zero" else None))
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
            llm_model=MODEL_NAME if name == "empty" else None, llm_eval_mode="f1",
        ))
    checks.append(check_case(
        output_dir, "tool_blank_f1", DB_DATASET, [terminal(ToolUseStep(answer="   "))],
        ["0"], ("   ", False, None, None), llm_model=MODEL_NAME, llm_eval_mode="f1",
    ))
    return checks


def check_result_terminals():
    """Check search_base.py::SearchResult.__post_init__ and both result subclasses.

    Real node fixtures need no inference. Invalid collections must raise, not
    silently drop cutoff nodes or mark them terminal; valid order is unchanged.
    """
    checks = []
    for result_class, node_class in (
        (SearchResult, SearchNode), (BFSResult, SearchNode), (MCTSResult, MCTSNode),
    ):
        root = node_class(state=EnvState(init_state=PARTIAL_STATE), action=None)
        complete = terminal(EnvStep(next_state=GOALS), node_class=node_class)
        other = terminal(EnvStep(next_state=PARTIAL_STATE), node_class=node_class)
        cutoff = terminal(EnvStep(next_state=PARTIAL_STATE), node_class=node_class)
        cutoff.is_terminal = False
        for node in (complete, other, cutoff):
            node.parent = root
            root.add_child(node)
        for name, nodes, invalid_index in (
            ("empty", [], None),
            ("terminals_in_order", [other, complete], None),
            ("cutoff_only", [cutoff], 0),
            ("cutoff_first", [cutoff, complete], 0),
            ("cutoff_last", [complete, cutoff], 1),
        ):
            original = list(nodes)
            try:
                result = result_class(root=root, terminal_nodes_collected=nodes)
            except ValueError as error:
                expected = f"terminal_nodes_collected[{invalid_index}] must have is_terminal=True"
                if invalid_index is None or str(error) != expected:
                    raise RuntimeError(f"{result_class.__name__}/{name}: {error}") from error
            else:
                if invalid_index is not None or result.terminal_nodes_collected != original:
                    raise RuntimeError(f"{result_class.__name__}/{name}: invalid result accepted")
            if nodes != original or cutoff.is_terminal or cutoff not in root.children:
                raise RuntimeError("Result validation mutated the input collection or tree")
            checks.append({"case": f"{result_class.__name__}_{name}", "passed": True})
            print(f"PASS {result_class.__name__}/{name}")
    bfs_result = BFSResult(
        root=root, terminal_nodes_collected=[complete, other],
        buckets_with_terminal={1: [other, cutoff, complete], 0: [root]},
    )
    paths = bfs_result.to_paths()
    expected = [[root], [root, other], [root, cutoff], [root, complete]]
    if paths != expected or BFSResult(root=root).to_paths() != []:
        raise RuntimeError("BFS path conversion lost bucket entries or changed their order")
    if cutoff.is_terminal or bfs_result.terminal_nodes_collected != [complete, other]:
        raise RuntimeError("BFS path conversion changed terminal eligibility")
    print("PASS BFS path conversion: depth order, cutoff/internal nodes, empty buckets")
    breakpoint()  # inspect: checks, root, cutoff, paths; cutoff remains in the tree
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
        ("empty_returns", [terminal(EnvStep(next_state=PARTIAL_STATE), fast_reward=0.1),
                           terminal(EnvStep(next_state=GOALS), fast_reward=0.9)], "correct", True, 1.0),
        ("missing_returns", [terminal(EnvStep(next_state=PARTIAL_STATE), fast_reward=0.1, node_class=SearchNode),
                             terminal(EnvStep(next_state=GOALS), fast_reward=0.9, node_class=SearchNode)],
         "correct", True, 1.0),
        ("mixed_fallback_wins", [complete, terminal(EnvStep(next_state=PARTIAL_STATE), fast_reward=0.9)],
         "incorrect", False, 0.5),
        ("mixed_tie", [terminal(EnvStep(next_state=PARTIAL_STATE), fast_reward=0.7), complete],
         "incorrect", False, 0.5),
        ("backed_overrides_fast", [terminal(EnvStep(next_state=PARTIAL_STATE), returns=[0.1], fast_reward=0.99),
                                   complete], "correct", True, 1.0),
        ("zero_return", [terminal(EnvStep(next_state=PARTIAL_STATE), returns=[0.0], fast_reward=0.99),
                          terminal(EnvStep(next_state=GOALS), fast_reward=0.5)], "correct", True, 1.0),
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


def check_scalar_zero_returns():
    """Check search.py selectors preserve scalar zero as an available return.

    Live MCTS uses lists; these selectors also accept scalar returns. Verify this
    directly because the MCTS node serializer expects list-valued cum_rewards.
    """
    zero = terminal(ToolUseStep(answer="43"), fast_reward=-0.9)
    zero.cum_rewards = 0.0
    other = terminal(ToolUseStep(answer="42"), returns=[-0.1], fast_reward=0.95)
    checks = []
    for strategy in ("q-mean", "q-max", "environment"):
        nodes = [other, zero]
        chosen = (select_env_terminal(nodes) if strategy == "environment"
                  else select_tool_use_terminal(nodes, strategy))
        breakpoint()  # inspect: strategy, chosen, zero.cum_rewards
        if chosen is not zero or zero.cum_rewards != 0.0:
            raise RuntimeError(f"{strategy}: scalar zero was treated as missing or mutated")
        print(f"PASS scalar_zero_{strategy}: zero return outranks -0.1")
        checks.append({"case": f"scalar_zero_{strategy}", "selected_id": chosen.id})
    return checks


def check_answer_outcomes():
    """Check search.py::check_answer and math_qa_eval.py::eval_output numeric/yes-no matching."""
    kgqa_evaluator = get_evaluator("kgqa")
    math_evaluator = get_evaluator("math500")
    gsm8k_evaluator = get_evaluator("gsm8k")
    if math_evaluator is None or gsm8k_evaluator is not math_evaluator:
        raise RuntimeError("MATH-500 and GSM8K must share the registered numeric checker")
    checks = []
    for name, pred, truth, evaluator, expected in (
        ("exact_match", "42", 42, None, (True, None)),
        ("exact_mismatch", "42", "43", None, (False, None)),
        ("exact_format_mismatch", "42.0", "42", None, (False, None)),
        ("exact_exponent_mismatch", "1e-3", "0.001", None, (False, None)),
        ("exact_missing", None, "None", None, (False, None)),
        ("exact_empty", "", "", None, (True, None)),
        ("exact_blank", " \t\n", " \t\n", None, (True, None)),
        ("db_empty_list", [], [], get_evaluator("dbbench"), (True, None)),
        ("db_numeric_zero", 0, ["0"], get_evaluator("dbbench"), (True, None)),
        ("numeric_format", "42.0", "42", math_evaluator, (True, None)),
        ("numeric_exponent", "1e-3", "0.001", gsm8k_evaluator, (True, None)),
        ("numeric_fraction_mismatch", "0.5", "1/2", math_evaluator, (False, None)),
        ("numeric_none", None, "0", math_evaluator, (False, None)),
        ("numeric_empty", "", "0", math_evaluator, (False, None)),
        ("numeric_wrong", "43", "42", math_evaluator, (False, None)),
        ("string_match", "Paris", "Paris", None, (True, None)),
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
    for name, answer, output, expected in (
        ("qa_yes_alias", "yes", "TRUE", True),
        ("qa_no_alias", "false", "n", True),
        ("qa_yn_mismatch", "yes", "no", False),
        ("qa_yn_empty", "yes", "", False),
    ):
        observed = eval_output(answer, output, type="yn")
        if observed != expected:
            raise RuntimeError(f"{name}: {observed!r} != {expected!r}")
        print(f"PASS {name}: {observed}")
        checks.append({"case": name, "expected": expected, "observed": observed})
    breakpoint()  # inspect: checks; no LLM evaluator is supplied
    return checks


def check_live_math_fallback(output_dir, model_name):
    """Check search.py::evaluate_search_result and eval_search.py CLI MATH judging.

    Real model calls test equivalent fraction notation and a wrong answer. A
    Numeric matches still vote but must make no judge calls. Single-answer cases
    exercise voting followed by optional correctness judging on the same model.
    """
    checks = []
    for name, prediction, truth, correct in (
        ("numeric_success", "42.0", "42", True),
        ("equivalent_fraction", "0.5", "1/2", True),
        ("wrong_answer", "43", "42", False),
    ):
        checks.append(check_case(
            output_dir, f"live_math_{name}", MATH_DATASET,
            [terminal(ThoughtStep(action=f"The answer is {prediction}", terminate=True))],
            truth, (prediction, correct, None, None), llm_model=model_name,
        ))
    breakpoint()  # inspect: checks; judge_calls is zero for numeric_success
    return checks


def check_live_fallback(output_dir, model_name):
    """Exercise search.py::check_answer with real DBBench/KGQA checkers and a model.

    Each fixture runs through the CLI and both shared-input paths. Correctness is
    checked on simple answers; this is a routing check, not a judge-quality estimate.
    Separate non-JSON prompts check that binary/F1 errors propagate through check_answer.
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
    for name, prediction, truth, expected_score in (
        ("f1_registered_success", "alpha, beta", ["alpha", "beta"], 1.0),
        ("f1_partial", "alpha", ["alpha", "beta", "gamma"], 0.5),
        ("f1_reject", "delta", ["alpha", "beta"], 0.0),
        # KGQA splits on commas/newlines; the model can recognize this conjunction.
        ("f1_rescue", "alpha and beta", ["alpha", "beta"], 1.0),
    ):
        checks.append(check_case(
            output_dir, f"live_{name}", FLOAT_DATASET,
            [terminal(ToolUseStep(answer=prediction), returns=[0.5])], truth,
            (prediction, expected_score == 1.0, expected_score, None),
            llm_model=model_name, llm_eval_mode="f1",
        ))
    for mode in ("binary", "f1"):
        case_dir = output_dir / f"live_malformed_{mode}"
        model = get_lm(model_name)
        inference_logger = setup_inference_logging(model, root_dir=str(case_dir))
        evaluator = _create_llm_evaluator(model, mode, logging.getLogger("lits.test"))
        # Ask a real model for non-JSON to exercise error propagation without canned responses.
        evaluator.prompt_template = "Return exactly INVALID, without JSON or explanation."
        evaluator.max_retries = 0
        evaluator.default_max_new_tokens = 8
        try:
            check_answer("43", "42", llm_evaluator=evaluator, llm_eval_mode=mode)
        except RuntimeError as error:
            if "Evaluator failed to return valid JSON" not in str(error):
                raise
            (case_dir / "error.txt").write_text(str(error))
        else:
            raise RuntimeError(f"{mode}: malformed judge output did not raise")
        usage = [json.loads(line) for line in Path(inference_logger.filepath).read_text().splitlines()]
        if len(usage) != 1 or usage[0]["role"] != "evaluator_general":
            raise RuntimeError(f"{mode}: expected one real judge call")
        print(f"PASS live_malformed_{mode}: evaluation error propagated")
        checks.append({"case": f"live_malformed_{mode}", "judge_calls": 1,
                       "error": (case_dir / "error.txt").read_text()})
    breakpoint()  # inspect: checks; live_* directories contain eval.log and inferencelogger.log
    return checks


def check_aggregation():
    """Check ``aggregation.py::aggregate_results`` using already-evaluated rows.

    Cover numeric means, optional binary scores, input preservation, and empty
    inputs without models. ``check_chain_checkpoints`` owns CLI compatibility.
    These controlled measurements test arithmetic, not benchmark performance.
    """
    rows = [
        {"idx": 1, "attempt": 2, "correct": False, "output_tokens": 0, "seconds": None},
        {"idx": 0, "attempt": 1, "correct": True, "output_tokens": 12, "seconds": 2},
        {"idx": 1, "attempt": 0, "correct": True, "seconds": 4},
        {"idx": 2, "attempt": None, "correct": False, "output_tokens": None, "seconds": 6},
    ]
    original = [dict(row) for row in rows]
    fields = ("output_tokens", "seconds", "unavailable")
    flat = aggregate_results(rows, mean_fields=fields)
    attempts = aggregate_results(rows, pass_at_n=True)
    empty = aggregate_results([], mean_fields=fields)
    empty_attempts = aggregate_results([], pass_at_n=True)
    checks = {
        "aggregate_flat_accuracy": (flat["total"], flat["correct_count"], flat["accuracy"]) == (4, 2, 0.5),
        "aggregate_numeric_means": flat["means"] == {
            "output_tokens": 6.0, "seconds": 4.0, "unavailable": None,
        },
        "aggregate_observed_attempts": attempts == {
            "total": 3, "means": {},
            "pass_at_1_count": 2, "pass_at_n_count": 2,
            "pass_at_1": 2 / 3, "pass_at_n": 2 / 3,
            "examples": [
                {"idx": 0, "attempts": [1.0], "pass_at_1": True, "pass_at_n": True},
                {"idx": 1, "attempts": [1.0, 0.0], "pass_at_1": True, "pass_at_n": True},
                {"idx": 2, "attempts": [0.0], "pass_at_1": False, "pass_at_n": False},
            ],
        },
        "aggregate_empty_flat": empty == {
            "total": 0, "correct_count": 0, "accuracy": 0.0, "examples": [],
            "means": dict.fromkeys(fields),
        },
        "aggregate_empty_attempts": empty_attempts == {
            "total": 0, "examples": [], "means": {},
            "pass_at_1_count": 0, "pass_at_n_count": 0, "pass_at_1": 0.0, "pass_at_n": 0.0,
        },
        "aggregate_input_preserved": rows == original and flat["examples"] == [
            {"idx": row["idx"], "correct": row["correct"], "score": float(row["correct"])}
            for row in original
        ],
        "aggregate_generator_input": aggregate_results(iter(rows), mean_fields=fields) == flat,
    }
    print(f"Aggregation checks: {checks}")
    breakpoint()  # inspect: flat, attempts, empty, empty_attempts, checks
    if not all(checks.values()):
        raise RuntimeError(f"Aggregation checks failed: {checks}")
    return [{"case": name, "passed": passed} for name, passed in checks.items()]


def check_chain_checkpoints(output_dir, legacy_ref):
    """Compare eval_search.py::evaluate_from_checkpoints with a trusted local Git revision.

    Real ToolUseStates are saved through base.py::TrajectoryState.save. String
    fixture values are final answers; dictionaries are deliberately invalid records.
    Each implementation receives identical inputs in a separate temporary directory.
    Compare full returned/saved reports, predictions, and per-file errors, excluding
    only the different result-directory paths. The reference must support the current
    dependency import paths. No models are called.
    """
    repo = Path(__file__).resolve().parents[2]
    source_path = "lits/cli/eval_search.py"
    source = subprocess.check_output(
        ["git", "show", f"{legacy_ref}:{source_path}"], cwd=repo, text=True,
    )
    legacy = types.ModuleType("lits.cli._eval_search_baseline")
    legacy.__file__ = str(repo / source_path)
    exec(compile(source, f"{legacy_ref}:{source_path}", "exec"), legacy.__dict__)

    truths = [["10"], ["42"], ["43"], ["0"]]
    single = {
        "0.json": "10",
        "1.json": ToolUseState([ToolUseStep(assistant_message="<answer>42</answer>")]),
        "2.json": "42",
        "3.json": ToolUseState(),
        "99_a9_incomplete.json": {},
        "pass_at_n_summary.json": {},
    }
    attempts = {
        "0_a0.json": "11", "0_a1.json": "10", "0_a2.json": "11",
        "1_a0.json": "42", "1_a1.json": "43", "2_a1.json": "43",
        "3.json": ToolUseState(),
        "99_a9_incomplete.json": {}, "pass_at_n_summary.json": {},
    }
    # Preserve the first-available-attempt rule when attempt 0 is absent, and
    # the mixed single-attempt file accepted alongside pass-at-N checkpoints.
    # Its missing answer remains incorrect even when the reference is zero.
    attempt_outcomes = [
        (0, [0.0, 1.0, 0.0], False, True),
        (1, [1.0, 0.0], True, True),
        (2, [1.0], True, True), (3, [0.0], False, False),
    ]
    cases = (
        ("chain_full", DB_DATASET, single, truths, {},
         {"n_attempts": 1, "accuracy": 0.5}, [(0, True), (1, True), (2, False), (3, False)]),
        ("chain_filtered", DB_DATASET, single, truths, {"offset": 1, "limit": 2},
         {"n_attempts": 1, "accuracy": 0.5}, [(1, True), (2, False)]),
        ("chain_empty_range", DB_DATASET, single, truths, {"offset": 10},
         {"n_attempts": 1, "accuracy": 0.0}, []),
        ("chain_missing_answer", DB_DATASET,
         {"0.json": ToolUseState(),
          "1.json": ToolUseState([ToolUseStep(answer="42"), ToolUseStep(think="Continue")])},
         [["42"], ["42"]], {}, {"accuracy": 0.0}, [(0, False), (1, False)]),
        ("chain_numeric", MATH_DATASET, {"0.json": "42.0", "1.json": "43", "2.json": ""},
         ["42"] * 3, {}, {"accuracy": 1 / 3}, [(0, True), (1, False), (2, False)]),
        ("chain_float", FLOAT_DATASET,
         {"0.json": "Obedient, Intelligent", "1.json": "Obedient", "2.json": ""},
         [["Obedient", "Intelligent"]] * 3, {},
         {"accuracy": 1 / 3, "mean_score": (1.0 + 2 / 3) / 3},
         [(0, True), (1, False), (2, False)]),
        ("chain_malformed", DB_DATASET, {"0.json": "10", "1.json": {}, "2.json": "43"},
         truths, {}, {"accuracy": 1.0}, [(0, True), (2, True)]),
        ("pass_at_n_full", DB_DATASET, attempts, truths, {},
         {"n_attempts": 3, "pass_at_1": 0.5, "pass_at_n": 0.75}, attempt_outcomes),
        ("pass_at_n_filtered", DB_DATASET, attempts, truths, {"offset": 1, "limit": 2},
         {"n_attempts": 3, "pass_at_1": 1.0, "pass_at_n": 1.0}, attempt_outcomes[1:3]),
        ("pass_at_n_empty_range", DB_DATASET, attempts, truths, {"offset": 10},
         {"n_attempts": 3, "pass_at_1": 0.0, "pass_at_n": 0.0}, []),
    )
    checks = []
    for name, dataset, files, references, options, expected, expected_examples in cases:
        compared = {}
        for version, evaluate in (
            ("legacy", legacy.evaluate_from_checkpoints), ("current", evaluate_from_checkpoints),
        ):
            case_dir = output_dir / name / version
            checkpoint_dir = case_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            for filename, content in files.items():
                path = checkpoint_dir / filename
                if isinstance(content, dict):
                    path.write_text(json.dumps(content))
                else:
                    state = ToolUseState([ToolUseStep(answer=content)]) if isinstance(content, str) else content
                    state.save(str(path), "Test query")
            with (case_dir / "console.log").open("w") as console:
                with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
                    report = evaluate(
                        str(case_dir), dataset, MODEL_NAME, llm_eval_mode="none",
                        dataset_kwargs={"examples": [{"answer": truth} for truth in references]},
                        **options,
                    )
            if report is None or json.loads((case_dir / "eval_results.json").read_text()) != report:
                raise RuntimeError(f"{name}/{version}: missing or inconsistent saved report")
            if report["result_dir"] != str(case_dir):
                raise RuntimeError(f"{name}/{version}: incorrect recorded result directory")
            if any(report.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"{name}/{version}: {report!r} does not match {expected!r}")
            if "pass_at_n" in report:
                examples = [(row["idx"], row["attempts"], row["pass_at_1"], row["pass_at_n"])
                            for row in report["examples"]]
            else:
                examples = [(row["idx"], row["correct"]) for row in report["examples"]]
                if any(row["score"] != float(row["correct"]) for row in report["examples"]):
                    raise RuntimeError(f"{name}/{version}: per-example score is no longer binary")
            if examples != expected_examples:
                raise RuntimeError(f"{name}/{version}: unexpected examples {examples!r}")
            log = (case_dir / "eval.log").read_text()
            compared[version] = {
                "report": {key: value for key, value in report.items() if key != "result_dir"},
                "predictions": re.findall(r"\[\d+(?:_a\d+)?\] Pred=.*", log),
                "errors": re.findall(r"Error processing .*[/]([^/]+\.json): (.*)", log),
            }
        for filename in files:
            case_dir = output_dir / name
            if ((case_dir / "legacy" / "checkpoints" / filename).read_bytes()
                    != (case_dir / "current" / "checkpoints" / filename).read_bytes()):
                raise RuntimeError(f"{name}: input checkpoint differs between implementations")
        if compared["legacy"] != compared["current"]:
            raise RuntimeError(f"{name}: legacy/current mismatch: {compared!r}")
        print(f"PASS {name}: legacy = current: {expected}, examples={expected_examples}")
        checks.append({"case": name, "legacy_ref": legacy_ref, **compared})
    breakpoint()  # inspect: checks; each case has legacy/ and current/ inputs, logs, and reports
    return checks


def main():
    """Compare evaluation paths, optionally run live judging, and retain logs/outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-model", help="LiTS model name for optional live binary/F1 fallback checks")
    parser.add_argument("--math-llm-model", help="LiTS model name for optional live MATH correctness checks")
    parser.add_argument("--compare-chain-cli-ref", help="Trusted local Git revision for chain/pass-at-N comparison")
    parser.add_argument("--voting-model", help="Real model for live language CLI voting comparisons")
    args = parser.parse_args()
    os.environ.setdefault("OPENAI_API_KEY", "no-key")
    os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8000/v1")
    register_examples()
    output_dir = Path(tempfile.mkdtemp(prefix="lits-eval-baseline-"))
    print(f"Evaluation artifacts: {output_dir}", flush=True)
    checks = [check_saved_rap_retrieval_inputs()]
    checks.extend(check_empty_language(output_dir))
    checks.extend(check_cli_thinking_config(output_dir))
    checks.extend(check_tool_use(output_dir))
    checks.extend(check_result_terminals())
    checks.extend(check_environment(output_dir))
    checks.extend(check_scalar_zero_returns())
    checks.extend(check_answer_outcomes())
    checks.extend(check_aggregation())
    if args.compare_chain_cli_ref:
        checks.extend(check_chain_checkpoints(output_dir, args.compare_chain_cli_ref))
    if args.llm_model:
        checks.extend(check_live_fallback(output_dir, args.llm_model))
    if args.math_llm_model:
        checks.extend(check_live_math_fallback(output_dir, args.math_llm_model))
    if args.voting_model:
        checks.extend(check_language(output_dir, args.voting_model))
    (output_dir / "baseline.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"Passed {len(checks)} checks. Baseline: {output_dir / 'baseline.json'}")
    breakpoint()  # inspect: checks, output_dir; each case contains terminal_nodes/ and eval.log


if __name__ == "__main__":
    main()
