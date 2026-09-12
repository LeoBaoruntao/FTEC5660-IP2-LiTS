"""Compare CLI and shared tree evaluation against explicit expected outcomes.

Run from lits_llm/ with demos/ on PYTHONPATH:
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_search_evaluation
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_search_evaluation --llm-model ollama/llama3.2:latest
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_search_evaluation --compare-chain-cli-ref 8fdbfb9
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_search_evaluation --math-llm-model openai/local-instruct
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_search_evaluation --voting-model ollama/llama3.2:latest

This checks the saved-result pipeline, not individual benchmark algorithms.
agents/test_extract_answers.py owns extraction/voting checks and the saved RAP
fixture reused here; tools/test_dbbench_loader.py and tools/test_kgqa_registry.py
own the benchmark comparators. Their registered implementations are used directly.

All additional cases use real LiTS nodes and steps with controlled field values.
The CLI constructs a real OpenAI-compatible client but these deterministic cases
need no inference, dataset download, or database connection. Answer-extraction LLM
fallback checks remain in agents/test_extract_answers.py; --llm-model additionally
checks live binary and F1 judging here. Each CLI fixture is also evaluated
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
import tempfile
import types

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
from lits.eval.answer_voting import vote_answers, _count_answer_votes, _raw_terminal_answers
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
               selection=None, llm_model=None, llm_eval_mode="binary"):
    """Compare CLI, original-node, and reloaded-node evaluation on the same fixture.

    References: search.py::save_terminal_nodes,
    eval_search.py::load_terminal_nodes_from_file,
    eval_search.py::evaluate_from_checkpoints, search.py::evaluate_search_result.
    expected is (prediction, correct, optional_score, environment_progress).
    Shared calls receive nodes/reference and caller-configured retrieval/checkers,
    without a result directory, dataset name, or model-creation arguments.
    llm_model optionally enables live binary or F1 judging for answer-based fixtures; the
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
                llm_eval_mode=llm_eval_mode if llm_model else "none", **options,
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
        shared_options["custom_evaluator"] = get_evaluator(dataset)
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

    if llm_model:
        deterministic_result = check_answer(
            expected[0], truth, custom_evaluator=shared_options.get("custom_evaluator"),
        )
        needs_fallback = not deterministic_result[0]
        if ("LLM fallback:" in log) != needs_fallback:
            raise RuntimeError(f"{name}: unexpected CLI fallback routing")
        usage = [json.loads(line) for line in Path(inference_logger.filepath).read_text().splitlines()]
        if needs_fallback:
            if not 2 <= len(usage) <= 6:
                raise RuntimeError(f"{name}: expected one to three calls per shared evaluation")
        elif usage:
            raise RuntimeError(f"{name}: deterministic success unnecessarily invoked the LLM")
        if any(record["role"] != "evaluator_general" for record in usage):
            raise RuntimeError(f"{name}: unexpected inference role")
        shared_results["deterministic_result"] = deterministic_result
        shared_results["llm_model"] = llm_model
        shared_results["llm_eval_mode"] = llm_eval_mode
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
    numeric match must make no judge calls. Extraction uses existing decimal
    answers, so these cases isolate correctness judging from answer extraction.
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
    breakpoint()  # inspect: checks; shared_inference_calls is zero for numeric_success
    return checks


def check_live_fallback(output_dir, model_name):
    """Exercise search.py::check_answer with real DBBench/KGQA checkers and a model.

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
    breakpoint()  # inspect: checks; live_* directories contain eval.log and inferencelogger.log
    return checks


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
    attempt_outcomes = [
        (0, [0.0, 1.0, 0.0], False, True),
        (1, [1.0, 0.0], True, True),
        (2, [1.0], True, True), (3, [1.0], True, True),
    ]
    cases = (
        ("chain_full", DB_DATASET, single, truths, {},
         {"n_attempts": 1, "accuracy": 0.75}, [(0, True), (1, True), (2, False), (3, True)]),
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
         {"n_attempts": 3, "pass_at_1": 0.75, "pass_at_n": 1.0}, attempt_outcomes),
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


def check_answer_groups():
    """Check answer_voting.py::_count_answer_votes with structured data, without a model.

    SearchNode/ThoughtStep inputs also exercise answer_voting.py::_raw_terminal_answers;
    membership checks do not claim to verify LLM semantic grouping quality.
    """
    raw = {"n1": "The answer is 1/2.", "n2": "The answer is 0.5.",
           "n3": "The answer is 1/3.", "n4": "   "}
    valid = {
        "groups": [{"answer": "1/3", "member_ids": ["n3"]},
                   {"answer": "1/2", "member_ids": ["n1", "n2"]}],
        "unanswered_ids": ["n4"],
    }
    votes = _count_answer_votes(valid, raw)
    if list(votes.items()) != [("1/2", 2), ("1/3", 1)]:
        raise RuntimeError(f"Wrong grouping counts/order: {votes}")
    tie_payload = {"groups": [valid["groups"][0],
                              {"answer": "1/2", "member_ids": ["n1"]}],
                   "unanswered_ids": []}
    tied = _count_answer_votes(tie_payload, {"n1": raw["n1"], "n3": raw["n3"]})
    if max(tied, key=tied.get) != "1/2":
        raise RuntimeError("Tie-breaking followed model order rather than input order")
    nodes = [terminal(ThoughtStep(action=f"The answer is {answer}", terminate=True))
             for answer in ("1/2", "1/3", "1e-3")]
    observed = _raw_terminal_answers(nodes)
    if list(observed.values()) != ["The answer is 1/2", "The answer is 1/3", "The answer is 1e-3"]:
        raise RuntimeError(f"Raw answer text changed: {observed}")
    cases = {
        "unknown_id": {"groups": [{"answer": "1/2", "member_ids": ["unknown"]}], "unanswered_ids": []},
        "repeated_id": {"groups": valid["groups"], "unanswered_ids": ["n1", "n4"]},
        "omitted_id": {"groups": valid["groups"], "unanswered_ids": []},
        "empty_group": {"groups": [{"answer": "1/2", "member_ids": []}], "unanswered_ids": list(raw)},
        "blank_answer": {"groups": [{"answer": " ", "member_ids": ["n1"]}], "unanswered_ids": []},
        "duplicate_label": {"groups": [{"answer": "1/2", "member_ids": ["n1"]},
                                        {"answer": " 1/2 ", "member_ids": ["n2"]}], "unanswered_ids": ["n3", "n4"]},
        "blank_member": {"groups": [{"answer": "0", "member_ids": ["n4"]}], "unanswered_ids": ["n1", "n2", "n3"]},
        "wrong_type": {"groups": {}, "unanswered_ids": []},
        "extra_field": {**valid, "count": 3},
    }
    checks = [{"name": "answer_group_counts_and_order", "votes": votes},
              {"name": "answer_group_tie", "votes": tied},
              {"name": "untruncated_terminal_answers", "answers": observed}]
    for name, payload in cases.items():
        try:
            _count_answer_votes(payload, raw)
        except ValueError as error:
            checks.append({"name": name, "rejected": str(error)})
        else:
            raise RuntimeError(f"Invalid answer groups accepted: {name}")
    breakpoint()  # inspect: votes, tied, observed, checks
    return checks


def check_live_answer_voting(output_dir, model_name):
    """Exercise answer_voting.py::vote_answers and search.py::evaluate_search_result.

    Store input text, native tool arguments, inference usage, and evaluation outcomes.
    A failed semantic or format check is reported, not replaced by a mock or fallback.
    """
    case_dir = output_dir / "live_answer_voting"
    case_dir.mkdir()
    print(f"Live voting artifacts: {case_dir}", flush=True)
    model = get_lm(model_name)
    setup_inference_logging(model, root_dir=str(case_dir))
    if hasattr(getattr(model, "client", None), "with_options"):
        model.client = model.client.with_options(timeout=60, max_retries=0)
    raw = {"n1": "The answer is 1/2.", "n2": "The answer is 0.5.",
           "n3": "The answer is 1/3.", "n4": ""}
    (case_dir / "inputs.json").write_text(json.dumps(raw, indent=2) + "\n")
    voter_logger = logging.getLogger("lits.eval.answer_voting")
    handler = logging.FileHandler(case_dir / "voting.log")
    previous_level = voter_logger.level
    voter_logger.setLevel(logging.DEBUG)
    voter_logger.addHandler(handler)
    try:
        for empty in ({}, {"n1": " "}):
            if vote_answers("Empty input", empty, model=model) != {}:
                raise RuntimeError("Empty/blank input should produce no votes")
        empty_outcome = evaluate_search_result(
            [], "Empty input", "1/2", task_type="language_grounded", voting_model=model,
            custom_evaluator=get_evaluator("math500"),
        )
        if empty_outcome["prediction"] != "" or empty_outcome["correct"]:
            raise RuntimeError(f"Unexpected empty-vote outcome: {empty_outcome}")
        if model.inference_logger.get_metrics_by_role()["num_calls"] != 0:
            raise RuntimeError("Empty/blank answers invoked the model")
        votes = vote_answers("What is one divided by two?", raw, model=model)
        (case_dir / "votes.json").write_text(json.dumps(votes, indent=2) + "\n")
        if sorted(votes.values()) != [1, 2] or max(votes, key=votes.get) != "1/2":
            raise RuntimeError(f"Unexpected fraction grouping: {votes}; inspect {case_dir}")
        nodes = [terminal(ThoughtStep(action=text, terminate=True)) for text in raw.values()]
        outcome = evaluate_search_result(
            nodes, "What is one divided by two?", "1/2",
            task_type="language_grounded", voting_model=model,
            custom_evaluator=get_evaluator("math500"),
        )
        (case_dir / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
        if outcome["prediction"] != "1/2" or not outcome["correct"]:
            raise RuntimeError(f"Unexpected semantic-voting evaluation: {outcome}")
        usage = model.inference_logger.get_metrics_by_role("evaluator_voting")
        if usage["num_calls"] != 2:
            raise RuntimeError(f"Expected exactly two voting calls: {usage}")
    finally:
        voter_logger.removeHandler(handler)
        voter_logger.setLevel(previous_level)
        handler.close()
    breakpoint()  # inspect: votes, outcome, case_dir
    return [{"name": "live_answer_voting", "votes": votes},
            {"name": "live_semantic_voting_evaluation", "outcome": outcome}]


def main():
    """Compare evaluation paths, optionally run live judging, and retain logs/outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-model", help="LiTS model name for optional live binary/F1 fallback checks")
    parser.add_argument("--math-llm-model", help="LiTS model name for optional live MATH correctness checks")
    parser.add_argument("--compare-chain-cli-ref", help="Trusted local Git revision for chain/pass-at-N comparison")
    parser.add_argument("--voting-model", help="LiTS model for optional live semantic answer voting")
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
    checks.extend(check_scalar_zero_returns())
    checks.extend(check_answer_outcomes())
    checks.extend(check_answer_groups())
    if args.compare_chain_cli_ref:
        checks.extend(check_chain_checkpoints(output_dir, args.compare_chain_cli_ref))
    if args.llm_model:
        checks.extend(check_live_fallback(output_dir, args.llm_model))
    if args.math_llm_model:
        checks.extend(check_live_math_fallback(output_dir, args.math_llm_model))
    if args.voting_model:
        checks.extend(check_live_answer_voting(output_dir, args.voting_model))
    (output_dir / "baseline.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"Passed {len(checks)} checks. Baseline: {output_dir / 'baseline.json'}")
    breakpoint()  # inspect: checks, output_dir; each case contains terminal_nodes/ and eval.log


if __name__ == "__main__":
    main()
