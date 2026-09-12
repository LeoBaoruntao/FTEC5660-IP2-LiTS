"""Terminal selection, answer retrieval, and correctness checking for tree-search results.

Input paths::

    Search execution -> nodes ----------+
                                        +-> lits.eval.search
    Saved files -> CLI loads nodes -----+

File loading remains the caller's responsibility.
Callers also supply configured evaluators; this module does not create models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from lits.eval.answer_voting import _raw_terminal_answers, vote_answers
from lits.structures.env_grounded import EnvState

if TYPE_CHECKING:
    from logging import Logger

    from lits.agents.tree.node import SearchNode
    from lits.components.registry import TaskType
    from lits.eval.general_eval import GeneralEvaluator
    from lits.lm.base import LanguageModel


def evaluate_search_result(
    terminal_nodes: Sequence[SearchNode],
    query: str,
    ground_truth: Any,
    *,
    task_type: TaskType,
    retrieve_answer: Callable[[Any, str], Any] | None = None,
    voting_model: LanguageModel | None = None,
    terminal_selection: str = "q-mean",
    custom_evaluator: Callable[[Any, Any], bool | float] | None = None,
    llm_evaluator: GeneralEvaluator | None = None,
    llm_eval_mode: str = "binary",
    goal_check: Callable[[str, Any], tuple[bool, float]] | None = None,
    eval_logger: Logger | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Select a prediction from one tree result and evaluate it.

    Args:
        terminal_nodes: Collected terminal nodes, in their original order.
        query: Task question or environment goal description.
        ground_truth: Reference passed to correctness checking, never selection.
            The caller applies dataset-specific formatting. Ignored for env_grounded.
        task_type: "language_grounded", "tool_use", or "env_grounded".
        retrieve_answer: Required for language_grounded without voting_model:
            (state, query) -> answer,
            e.g. the callback returned by get_fn_retrieve_answer(model). For tool_use,
            an optional (node, query) -> answer callback overrides step retrieval.
            Unused for env_grounded.
        voting_model: Optional caller-configured model for semantic language-answer
            voting. Reads raw last-step answers and calls vote_answers once, bypassing
            retrieve_answer and its numeric extraction. Omit to retain legacy voting.
            Ground truth is never passed to the voter; correctness checking is unchanged.
        terminal_selection: Tool-use ranking: "first", "q-max", or "q-mean" (default).
            Language tasks vote over answers; environment tasks use maximum returns.
            Return-based ranking falls back per node to fast_reward when its
            cum_rewards are missing, None, or an empty list.
        custom_evaluator: Optional (prediction, ground_truth) -> bool or float checker.
            When absent, exact-string comparison runs before any LLM judging.
        llm_evaluator: Optional configured GeneralEvaluator; see check_answer for
            fallback precedence. Omit it to disable LLM judging.
        llm_eval_mode: "binary" or "f1", used only by the supplied LLM evaluator.
        goal_check: Required for env_grounded: (query, final_env_state) ->
            (goal_reached, progress). Overrides other correctness checkers.
        eval_logger: Optional logger passed to check_answer.
        label: Optional per-example label for checking logs.

    Returns:
        A dictionary with prediction, correct (bool), score (float or None),
        and environment_progress (float or None). score is a continuous checker
        score, not a search reward or float(correct). Environment prediction is
        "correct"/"incorrect", with score=None. Other tasks have environment_progress=None.

        Empty terminals produce "incorrect" with progress=0.0 for environment tasks,
        or an empty prediction for other tasks. That empty prediction still goes
        through configured correctness checking. Required callbacks/models must be supplied
        even for empty terminals, but retrieval/goal checking is not invoked on them.
        Without a custom checker, non-environment tasks use exact string matching.
        A configured LLM judge runs only when the deterministic check fails.

    Raises:
        ValueError: Unsupported task_type, a missing required callback, or invalid
            terminal inputs/selection options rejected by the selection helpers.
        Other retrieval, goal-checking, and LLM errors propagate to the caller.

    Example:
        outcome = evaluate_search_result(
            result.terminal_nodes_collected, query, reference,
            task_type="language_grounded", retrieve_answer=answer_retriever,
        )
    """
    if voting_model is not None and task_type != "language_grounded":
        raise ValueError("voting_model is only supported for language_grounded evaluation")
    environment_progress = None
    if task_type == "env_grounded":
        if goal_check is None:
            raise ValueError("goal_check is required for env_grounded evaluation")
        best_node = select_env_terminal(terminal_nodes)
        prediction, environment_progress = retrieve_answer_from_env_node(
            best_node, query, goal_check,
        )
    elif task_type == "tool_use":
        best_node = select_tool_use_terminal(terminal_nodes, terminal_selection)
        if retrieve_answer is None:
            retrieve_answer = retrieve_answer_from_tool_use_node
        prediction = retrieve_answer(best_node, query) if best_node is not None else ""
    elif task_type == "language_grounded":
        if voting_model is not None:
            votes = vote_answers(query, _raw_terminal_answers(terminal_nodes), model=voting_model)
        elif retrieve_answer is None:
            raise ValueError("retrieve_answer is required for language_grounded evaluation")
        else:
            from lits.agents.tree.common import extract_answers_from_terminal_nodes

            votes, *_ = extract_answers_from_terminal_nodes(
                terminal_nodes_collected=terminal_nodes,
                retrieve_answer=retrieve_answer,
                query=query,
            )
        prediction = max(votes, key=votes.get) if votes else ""
    else:
        raise ValueError(f"Unknown task_type={task_type!r}")

    score = None
    if task_type == "env_grounded":
        correct = prediction == "correct"
    else:
        correct, score = check_answer(
            prediction, ground_truth,
            custom_evaluator=custom_evaluator,
            llm_evaluator=llm_evaluator,
            llm_eval_mode=llm_eval_mode,
            eval_logger=eval_logger,
            label=label,
        )

    return {
        "prediction": prediction,
        "correct": correct,
        "score": score,
        "environment_progress": environment_progress,
    }


def select_tool_use_terminal(
    terminal_nodes: Sequence[SearchNode], strategy: str,
) -> SearchNode | None:
    """Pick which terminal node's answer to evaluate for tool_use tasks.

    Strategies:
      - "first":   terminal_nodes[0] (legacy save-order behaviour).
      - "q-max":   argmax over max(cum_rewards) per node - best single
                   rollout that reached this terminal.
      - "q-mean":  argmax over mean(cum_rewards) per node - matches
                   MCTSNode.DEFAULT_Q_FUNC (np.mean), i.e. the same
                   Q-value MCTS uses for UCT during search.

    Why "q-*" and not "prm-*": the values in `cum_rewards` are per-rollout
    returns (per-step PRM scores aggregated along the path by
    `backprop_reward_func`), not raw PRM scores. Across rollouts they're
    aggregated into the node's Q-value by `cross_rollout_q_func`. So the
    correct MCTS terminology for "rank terminals by their accumulated
    score" is "rank by Q-value".

    Missing, None, or empty-list `cum_rewards` falls back to that node's
    `fast_reward`. Available returns, including zero, take precedence for that
    node. A fallback score can outrank another node's backed-up score. Ties retain
    input order; neither score source nor the node's statistics are modified.

    Returns:
        The selected node, or None if `terminal_nodes` is empty.
    """
    if not terminal_nodes:
        return None
    if strategy == "first":
        return terminal_nodes[0]

    import numpy as _np
    if strategy == "q-max":
        def _score(n):
            """Return the best backed-up return, falling back to fast_reward."""
            cr = getattr(n, "cum_rewards", None)
            if cr is None or (isinstance(cr, list) and not cr):
                return float(n.fast_reward)
            return float(max(cr) if isinstance(cr, list) else cr)
    elif strategy == "q-mean":
        def _score(n):
            """Return the mean backed-up return, falling back to fast_reward."""
            cr = getattr(n, "cum_rewards", None)
            if cr is None or (isinstance(cr, list) and not cr):
                return float(n.fast_reward)
            if isinstance(cr, list):
                return float(_np.mean(cr))
            return float(cr)
    else:
        raise ValueError(
            f"Unknown terminal_selection={strategy!r}; "
            "expected one of 'first', 'q-max', 'q-mean'"
        )
    return max(terminal_nodes, key=_score)


def select_env_terminal(terminal_nodes: Sequence[SearchNode]) -> SearchNode | None:
    """Select the environment terminal with the maximum backed-up return.

    Missing, None, or empty-list returns fall back per node to fast_reward.
    Available returns, including zero, retain maximum-return ranking. Equal
    scores preserve input order. Return None when there are no terminals.

    Args:
        terminal_nodes: Nodes marked is_terminal=True, not necessarily successful.

    Raises:
        ValueError: If any supplied node is not terminal.
    """
    if not terminal_nodes:
        return None
    for index, node in enumerate(terminal_nodes):
        if not node.is_terminal:
            raise ValueError(f"terminal_nodes[{index}] is not terminal")

    def get_best_reward(node):
        """Return maximum backed-up return or fast_reward without changing its type."""
        cr = getattr(node, "cum_rewards", None)
        if cr is None or (isinstance(cr, list) and not cr):
            return node.fast_reward
        return max(cr) if isinstance(cr, list) else cr

    return sorted(terminal_nodes, key=get_best_reward, reverse=True)[0]


def retrieve_answer_from_tool_use_node(node: SearchNode, query: str) -> Any:
    """Read the terminal answer from node.step, then a final dictionary step.

    query is accepted for the common answer-retrieval callback signature.
    Return the stored value unchanged, or an empty string if no answer is found.
    """
    # Check if node has a step with an answer
    if hasattr(node, 'step') and node.step:
        step = node.step
        if hasattr(step, 'answer') and step.answer:
            return step.answer
        # Try to get answer from step dict if not deserialized
        if isinstance(step, dict) and 'answer' in step:
            return step['answer']
    # Fallback: check state
    if hasattr(node, 'state') and node.state:
        if isinstance(node.state, list) and len(node.state) > 0:
            last_step = node.state[-1]
            if isinstance(last_step, dict) and 'answer' in last_step:
                return last_step['answer']
    return ""


def retrieve_answer_from_env_node(
    node: SearchNode | None,
    query: str,
    goal_check: Callable[[str, Any], tuple[bool, float]],
) -> tuple[str, float]:
    """Check the final environment state with a caller-supplied goal checker.

    Return ("correct" or "incorrect", environment progress). Missing, empty,
    or non-EnvState state returns ("incorrect", 0.0) without calling goal_check.
    """
    if hasattr(node, 'state') and node.state and isinstance(node.state, EnvState):
        # Get the final env_state (after all steps)
        is_reached, env_score = goal_check(query, node.state.env_state)
        return ("correct" if is_reached else "incorrect"), env_score
    return "incorrect", 0.0


def check_answer(pred: str, truth, custom_evaluator=None, llm_evaluator=None,
                 llm_eval_mode: str = "binary",
                 eval_logger=None, label: str = "") -> tuple:
    """Evaluate a single predicted answer against ground truth.

    Use the dataset-specific custom_evaluator first, with an optional LLM
    fallback when it reports an incorrect answer or raises an exception.
    Without a custom evaluator, use exact string matching. A deterministic match
    skips the LLM judge.

    Args:
        pred: Predicted answer string.
        truth: Ground truth (any type - passed directly to custom_evaluator;
            converted to str for default exact-string comparison).
        custom_evaluator: Task-specific checker function, or None for exact matching.
        llm_evaluator: Configured GeneralEvaluator instance, or None.
        llm_eval_mode: "binary" (yes/no) or "f1" (float score).
        eval_logger: Logger for debug messages (optional).
        label: Human-readable label for log lines (e.g. "[3_a2]").

    Returns:
        (correct, score), where correct is bool and score is Optional[float].
        Float-returning custom evaluators and F1 mode supply a score; only
        1.0 counts as correct. F1 fallback retains the greater custom/LLM
        score; binary fallback leaves any prior custom score unchanged.

    Exceptions from the LLM evaluator propagate to the caller.
    """
    score = None

    if custom_evaluator:
        try:
            result = custom_evaluator(pred, truth)
            if isinstance(result, float):
                score = result
                correct = (result == 1.0)
            else:
                correct = bool(result)
        except Exception as e:
            correct = False
            if eval_logger:
                eval_logger.debug(f"  {label} custom evaluator failed: {e}")
    else:
        correct = pred == str(truth)

    if not correct and llm_evaluator:
        correct, score = _llm_judge(llm_evaluator, llm_eval_mode, pred, truth, score)
        if eval_logger:
            eval_logger.debug(f"  {label} LLM fallback: correct={correct}, score={score}")
    return correct, score


def _llm_judge(llm_evaluator, llm_eval_mode, pred, truth, prior_score=None):
    """Return (correct, score) from a caller-supplied LLM evaluator.

    F1 mode retains the higher of prior_score and the LLM score. Binary
    judging changes correctness without replacing prior_score.
    """
    if llm_eval_mode == "f1":
        llm_score = llm_evaluator.check_score(pred, truth)
        final = max(prior_score, llm_score) if prior_score is not None else llm_score
        return (final == 1.0), final
    else:
        return llm_evaluator.check_correct(pred, truth), prior_score
