"""Check answer-voting contracts, grouping, and optional live semantic evaluation.

Run from lits_llm/ with demos/ on PYTHONPATH:
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_answer_voting
    PYTHONBREAKPOINT=0 python -m unit_test.eval.test_answer_voting --voting-model openai/local-instruct

Set OPENAI_API_BASE and OPENAI_API_KEY for an OpenAI-compatible endpoint.
Default checks make no inference calls. Live checks enable thinking on the voter
and save input text, responses, usage, and outcomes in a fresh temporary directory.
CLI/in-memory/reloaded comparisons remain in test_search_evaluation.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import tempfile

from lits.agents.tree.node import MCTSNode
from lits.benchmarks.registry import get_evaluator
from lits.eval.answer_voting import vote_answers, _count_answer_votes, _raw_terminal_answers
from lits.eval.search import evaluate_search_result, retrieve_answer_from_tool_use_node
from lits.lm import get_lm, setup_inference_logging
from lits.structures.base import TrajectoryState
from lits.structures.qa import ThoughtStep

import lits_benchmark.math_qa_eval  # noqa: F401 -- registers real numeric QA checkers


MODEL_NAME = "openai/local-instruct"


def terminal_answer(text):
    """Build a real terminal node containing an unmodified ThoughtStep answer."""
    step = ThoughtStep(action=text, terminate=True)
    node = MCTSNode(state=TrajectoryState([step]), action=step.get_action(),
                    fast_reward=0.5, is_terminal=True)
    node.step = step
    return node


def check_voting_contract():
    """Check search.py::evaluate_search_result model validation without inference."""
    model = get_lm(MODEL_NAME)
    cases = (
        ("missing_voter", {"task_type": "language_grounded"}, "voting_model is required"),
        ("legacy_retriever", {"task_type": "language_grounded", "voting_model": model,
                              "retrieve_answer": retrieve_answer_from_tool_use_node},
         "not retrieve_answer"),
        ("tool_voter", {"task_type": "tool_use", "voting_model": model},
         "only supported for language_grounded"),
    )
    checks = []
    for name, options, message in cases:
        try:
            evaluate_search_result([], "Test query", "42", **options)
        except ValueError as error:
            if message not in str(error):
                raise RuntimeError(f"{name}: unexpected error {error}") from error
            checks.append({"case": name, "rejected": str(error)})
        else:
            raise RuntimeError(f"{name}: invalid voting configuration accepted")
    breakpoint()  # inspect: checks; invalid configurations must be rejected before inference
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
    nodes = [terminal_answer(f"The answer is {answer}") for answer in ("1/2", "1/3", "1e-3")]
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

    Enable thinking when constructing the voter, not as an ignored per-call option.
    Check fraction equivalence and complete membership of majority/minority answers.
    Hand-written complex last steps cover corrections, intermediate numbers, LaTeX,
    percentages, incorrect conclusions, and text without a final answer.
    Store input text, native tool arguments, inference usage, and evaluation outcomes.
    A failed semantic or format check is reported, not replaced by a mock or fallback.
    """
    case_dir = output_dir / "live_answer_voting"
    case_dir.mkdir()
    print(f"Live voting artifacts: {case_dir}", flush=True)
    model = get_lm(model_name, enable_thinking=True)
    setup_inference_logging(model, root_dir=str(case_dir))
    if hasattr(getattr(model, "client", None), "with_options"):
        model.client = model.client.with_options(timeout=60, max_retries=0)
    cases = (
        ("half", "What is one divided by two?",
         {"n1": "The answer is 1/2.", "n2": "The answer is 0.5.",
          "n3": "The answer is 1/3.", "n4": ""}, {"1/2": 2, "1/3": 1}),
        ("unseen_fraction", "What is three divided by four?",
         {"n1": "The answer is 3/4.", "n2": "The answer is 0.75.",
          "n3": "The answer is 2/3.", "n4": ""}, {"3/4": 2, "2/3": 1}),
        ("majority_membership", "Test query",
         {"n0": "The answer is 42.", "n1": "The answer is 43.",
          "n2": "The answer is 42."}, {"42": 2, "43": 1}),
        ("complex_last_steps",
         "A bag contains three red tokens and five blue tokens. One token is drawn "
         "uniformly. What is the probability that it is red?",
         {
             "n0": (
                 "There are 3 favorable outcomes and 3 + 5 = 8 possible outcomes.\n"
                 "I initially wrote 3/5, but that denominator counts only blue tokens, "
                 "not the full sample space. Correcting it gives 3/8.\n"
                 "The decimal check is 0.375; 0.625 instead describes drawing blue. "
                 "I therefore report 3/8 for the requested red probability."
             ),
             "n1": (
                 "Consider the complementary event first. Five of the eight tokens "
                 "are blue, so its probability is 5/8.\n"
                 r"Subtracting from one gives $1-\frac{5}{8}=\frac{3}{8}$. "
                 r"For red, the final result is $\boxed{\frac{3}{8}}$."
             ),
             "n2": (
                 "Dividing the three favorable outcomes by the eight total outcomes "
                 "gives 0.375. Multiplying by 100 expresses that probability as a "
                 "percentage, rather than changing the event. My conclusion is "
                 "37.5 percent. The other 62.5 percent corresponds to blue."
             ),
             "n3": (
                 "I treated the sample space as twelve equally likely positions, "
                 "of which three are favorable. That gives 3/12, which reduces to "
                 "1/4. I commit to 1/4 as the red probability."
             ),
             "n4": (
                 "I am unsure whether the denominator should count all eight tokens "
                 "or only the five blue ones. I have not settled on a probability "
                 "and cannot give a final result yet."
             ),
         }, {"3/8": 3, "1/4": 1}),
    )
    checks = []
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
        for name, query, raw, expected_votes in cases:
            (case_dir / f"{name}_inputs.json").write_text(json.dumps(raw, indent=2) + "\n")
            votes = vote_answers(query, raw, model=model)
            (case_dir / f"{name}_votes.json").write_text(json.dumps(votes, indent=2) + "\n")
            if votes != expected_votes:
                raise RuntimeError(f"{name}: {votes} != {expected_votes}; inspect {case_dir}")
            prediction = max(expected_votes, key=expected_votes.get)
            nodes = [terminal_answer(text) for text in raw.values()]
            outcome = evaluate_search_result(
                nodes, query, prediction, task_type="language_grounded", voting_model=model,
                custom_evaluator=get_evaluator("math500"),
            )
            (case_dir / f"{name}_outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
            if outcome["prediction"] != prediction or not outcome["correct"]:
                raise RuntimeError(f"{name}: unexpected semantic-voting evaluation: {outcome}")
            checks.extend([
                {"name": f"live_{name}_voting", "votes": votes},
                {"name": f"live_{name}_evaluation", "outcome": outcome},
            ])
            print(f"PASS {name}: {votes}; prediction={outcome['prediction']}", flush=True)
        usage = model.inference_logger.get_metrics_by_role("evaluator_voting")
        if not 2 * len(cases) <= usage["num_calls"] <= 4 * len(cases):
            raise RuntimeError(f"Expected at most one retry per voting evaluation: {usage}")
    finally:
        voter_logger.removeHandler(handler)
        voter_logger.setLevel(previous_level)
        handler.close()
    breakpoint()  # inspect: votes, outcome, case_dir
    return checks


def check_saved_answer_voting(terminal_file, output_dir, model_name):
    """Evaluate saved terminals with answer_voting.py::vote_answers, without search.

    Retain responses and votes separately; never rewrite the source terminal file.
    This checks real grouping, but only exercises retry when the model is invalid.
    """
    from lits.cli.eval_search import load_terminal_nodes_from_file

    original = terminal_file.read_bytes()
    data = load_terminal_nodes_from_file(terminal_file)
    raw = _raw_terminal_answers(data["terminal_nodes"])
    model = get_lm(model_name, enable_thinking=False)
    model.client = model.client.with_options(timeout=120, max_retries=0)
    setup_inference_logging(model, root_dir=str(output_dir))
    voter_logger = logging.getLogger("lits.eval.answer_voting")
    handler = logging.FileHandler(output_dir / "voting.log")
    previous_level = voter_logger.level
    voter_logger.setLevel(logging.DEBUG)
    voter_logger.addHandler(handler)
    try:
        votes = vote_answers(data["query"], raw, model=model)
    finally:
        voter_logger.removeHandler(handler)
        voter_logger.setLevel(previous_level)
        handler.close()
    if terminal_file.read_bytes() != original:
        raise RuntimeError("Source terminal file changed")
    result = {"source": str(terminal_file), "raw_answers": raw, "votes": votes,
              "prediction": max(votes, key=votes.get) if votes else "",
              "usage": model.inference_logger.get_metrics_by_role("evaluator_voting")}
    print(json.dumps(result, indent=2), flush=True)
    breakpoint()  # inspect: raw, votes, result; compare with the saved cohort reference
    return result


def main():
    """Run deterministic voting checks and optional real-model checks; retain results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voting-model", help="Real model for thinking-enabled semantic voting checks")
    parser.add_argument("--terminal-file", type=Path,
                        help="Evaluate saved terminals only, using the pilot's non-thinking voter")
    args = parser.parse_args()
    os.environ.setdefault("OPENAI_API_KEY", "no-key")
    os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8000/v1")
    output_dir = Path(tempfile.mkdtemp(prefix="lits-answer-voting-"))
    print(f"Voting artifacts: {output_dir}", flush=True)
    checks = check_voting_contract()
    checks.extend(check_answer_groups())
    if args.terminal_file:
        checks.append(check_saved_answer_voting(
            args.terminal_file, output_dir, args.voting_model or MODEL_NAME,
        ))
    elif args.voting_model:
        checks.extend(check_live_answer_voting(output_dir, args.voting_model))
    (output_dir / "results.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"Passed {len(checks)} checks. Results: {output_dir / 'results.json'}")
    breakpoint()  # inspect: checks, output_dir


if __name__ == "__main__":
    main()
