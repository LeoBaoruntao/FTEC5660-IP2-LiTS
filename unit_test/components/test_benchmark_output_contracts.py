"""Inspect real DBBench/BlocksWorld output contracts without tree search or lits_select.

Run ``python -m unit_test.components.test_benchmark_output_contracts dbbench``
or ``blocksworld`` with ``LITS_TEST_MODEL_NAME``, optional ``LITS_TEST_BASE_URL``
and ``LITS_TEST_API_KEY``. DBBench needs ``LITS_TEST_MYSQL_URI`` without a database
name and a MySQL driver; only a new isolated schema is initialized. Results stay
under ``benchmark_contract_results/``. Set ``PYTHONBREAKPOINT=0`` for batch runs.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import tempfile

from lits import get_lm
from lits.lm import setup_inference_logging
from lits.log import setup_logging

MODEL_NAME = os.environ.get(
    "LITS_TEST_MODEL_NAME", "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
)
logger = logging.getLogger("lits.output_contract_check")


def check_dbbench(model, run_dir):
    """Exercise factory.py::create_components, ToolUsePolicy.get_actions and the checker.

    Execute real SQL with the registered DBBench prompt and check the submitted
    JSON-array answer using the registered evaluator, without an LLM judge.
    The policy sees only the question and observations, never the reference.
    Existing deterministic comparator cases live in tools/test_dbbench_loader.py.
    """
    from sqlalchemy.engine import make_url
    from lits.benchmarks.registry import get_evaluator, load_dataset, load_resource
    from lits.components.factory import create_components
    from lits.config import ExperimentConfig
    from lits.eval.search import check_answer
    from lits.prompts.registry import PromptRegistry
    from lits_benchmark.dbbench import build_user_prompt, _clean_answer
    from lits_benchmark.dbbench_db_setup import init_database

    uri = os.environ["LITS_TEST_MYSQL_URI"]
    parsed = make_url(uri)
    if parsed.get_backend_name() != "mysql" or parsed.database:
        raise ValueError("Supply a MySQL server URI without a database name")
    example = load_dataset("dbbench")[0]
    schema = "lits_contract_" + run_dir.name.replace("-", "_")
    resource = load_resource(
        "dbbench", mysql_uri=init_database(uri.rstrip("/"), [example], database=schema)
    )
    config = ExperimentConfig(
        dataset="dbbench", policy_model_name=MODEL_NAME, eval_model_name=MODEL_NAME,
        search_framework="rest", native=False,
        search_args={"n_actions": 1, "max_steps": 4},
        component_args={"max_eval_rollout_steps": 0},
    )
    world, policy, _ = create_components(
        task_type="tool_use", task_name="dbbench", base_model=model,
        eval_base_model=model, terminal_model=None, tool_use_spec=resource, config=config,
    )
    calls = []

    def capture_call(prompt, response, **context):
        """Record actual policy requests/responses without replacing either."""
        calls.append({"system": model.sys_prompt, "messages": prompt, "raw": response.text})

    policy.set_llm_call_fn(capture_call)
    query = build_user_prompt(example)
    state = world.init_state(question=query)
    turns = []
    for _ in range(4):
        steps = policy.get_actions(state=state, query=query, n_actions=1, query_idx=0, from_phase="expand")
        if not steps:
            raise RuntimeError("Policy returned no step")
        step = steps[0]
        state, _ = world.step(state, step, query, query_idx=0, from_phase="expand")
        turns.append(step.to_dict())
        if step.answer is not None:
            break
    else:
        (run_dir / "dbbench.json").write_text(json.dumps({"turns": turns, "calls": calls}, indent=2))
        raise RuntimeError("No final answer within four policy turns; see dbbench.json")

    checker = get_evaluator("dbbench")
    correct, score = check_answer(step.answer, example["label"], custom_evaluator=checker)
    outcome = {"prediction": step.answer, "correct": correct,
               "normalized": _clean_answer(step.answer), "score": score}
    registered = PromptRegistry.list_registered("policy")["tool_use"]
    report = {"schema": schema, "query": query, "reference": example["label"],
              "prompt_keys": list(registered), "turns": turns, "calls": calls,
              "outcome": outcome}
    (run_dir / "dbbench.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"prompt_keys": list(registered), "outcome": outcome}, indent=2))
    breakpoint()  # inspect: report, step.answer, registered
    if not isinstance(json.loads(step.answer), list) or not correct:
        raise RuntimeError("Registered DBBench prompt did not produce a correct JSON-array answer")


def check_blocksworld_change_parser(initial):
    """Compare blocksworld.py::apply_change on change-only and change-plus-state text."""
    from lits_benchmark.blocksworld import apply_change

    change = ("The hand was holding the red block and is now empty, the red block was in the "
              "hand and is now on top of the blue block, the blue block is no longer clear, "
              "and the red block is now clear.")
    extra = ("\n[STATE 1] I have that, the red block is clear, the hand is empty, "
             "the red block is on top of the blue block, and the blue block is on the table.")
    outcomes = {}
    for name, raw in (("change_only", change), ("change_plus_state", change + extra)):
        try:
            outcomes[name] = {"state": apply_change(raw, initial)}
        except Exception as error:
            outcomes[name] = {"error": f"{type(error).__name__}: {error}"}
    print("Deterministic parser checks:", outcomes)
    breakpoint()  # inspect: outcomes; adding the state line changes the parser result
    return outcomes


def check_blocksworld(model, run_dir):
    """Call blocksworld.py::BlocksWorldTransition.step, then probe explicit newline stopping.

    Both real model calls use the same transition prompt and starting state. The
    probe changes only the stopping argument, not the production transition/parser.
    A successful model response alone does not establish general format reliability.
    """
    from lits.structures.env_grounded import EnvAction
    from lits.lm.base import DETERMINISTIC_TEMPERATURE
    from lits_benchmark.blocksworld import BlocksWorldTransition, apply_change

    world = BlocksWorldTransition(base_model=model)
    initial = ("the blue block is clear, the red block is in the hand, "
               "the hand is holding the red block, and the blue block is on the table.")
    goal = "the red block is on top of the blue block"
    action = EnvAction("stack the red block on top of the blue block")
    state = world.init_state(init_state_str=initial)
    report = {"initial_state": initial, "action": str(action),
              "parser_checks": check_blocksworld_change_parser(initial)}
    try:
        result, aux = world.step(state, action, goal, query_idx=0, from_phase="expand")
        report["default"] = {"state": result.env_state, "aux": aux}
    except Exception as error:
        logger.exception("Unchanged BlocksWorld transition failed")
        report["default"] = {"error": f"{type(error).__name__}: {error}"}
    prompt = world._get_prompt_tempate(str(action)).format(initial, str(action).capitalize() + ".")
    raw = world._call_model(prompt, stop=["\n"], temperature=DETERMINISTIC_TEMPERATURE).text
    report["explicit_stop_probe"] = {"raw": raw}
    try:
        updated = apply_change(raw.strip(), initial)
        report["explicit_stop_probe"].update(state=updated, goal_reached=world.goal_check(goal, updated))
    except Exception as error:
        report["explicit_stop_probe"]["error"] = f"{type(error).__name__}: {error}"
    (run_dir / "blocksworld.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    breakpoint()  # inspect: report, raw; default raw response is in execution.log


def main():
    """Run one explicit live diagnostic and retain raw outputs plus inference usage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("dbbench", "blocksworld"))
    task = parser.parse_args().task
    root = Path("benchmark_contract_results").resolve()
    root.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=task + "-", dir=root))
    print("Retained results:", run_dir)
    setup_logging("execution", str(run_dir), verbose=True, allowed_namespaces=("lits",))
    kwargs = {"max_new_tokens": 512}
    if base_url := os.environ.get("LITS_TEST_BASE_URL"):
        kwargs.update(base_url=base_url, api_key=os.environ.get("LITS_TEST_API_KEY", "no-key"))
    model = get_lm(MODEL_NAME, **kwargs)
    setup_inference_logging(model, root_dir=run_dir / "inference")
    check = check_dbbench if task == "dbbench" else check_blocksworld
    check(model, run_dir)


if __name__ == "__main__":
    main()
