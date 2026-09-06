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

from lits.structures.env_grounded import EnvState

if TYPE_CHECKING:
    from lits.agents.tree.node import SearchNode


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

    Empty `cum_rewards` is treated as -inf, so any node with at least one
    rollout will outrank a never-visited terminal. Ties retain input order.

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
            """Return the best backed-up return, or negative infinity."""
            cr = getattr(n, "cum_rewards", None)
            if not cr:
                return -float("inf")
            return float(max(cr) if isinstance(cr, list) else cr)
    elif strategy == "q-mean":
        def _score(n):
            """Return the mean backed-up return, or negative infinity."""
            cr = getattr(n, "cum_rewards", None)
            if not cr:
                return -float("inf")
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

    Missing/empty returns rank as negative infinity. Equal scores preserve
    input order. Return None when there are no terminals.

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
        """Return a terminal node's maximum backed-up return without changing its type."""
        if hasattr(node, 'cum_rewards') and node.cum_rewards:
            return max(node.cum_rewards) if isinstance(node.cum_rewards, list) else node.cum_rewards
        return -float('inf')

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
    Without a custom evaluator, use the supplied LLM evaluator. Exact string
    matching applies only when neither evaluator is supplied.

    Args:
        pred: Predicted answer string.
        truth: Ground truth (any type - passed directly to custom_evaluator;
            converted to str only for the fallback exact-match path).
        custom_evaluator: Registered evaluator function, or None.
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
        if not correct and llm_evaluator:
            correct, score = _llm_judge(llm_evaluator, llm_eval_mode, pred, truth, score)
            if eval_logger:
                eval_logger.debug(f"  {label} LLM fallback: correct={correct}, score={score}")
        return correct, score

    if llm_evaluator:
        correct, score = _llm_judge(llm_evaluator, llm_eval_mode, pred, truth, score)
        if eval_logger:
            eval_logger.debug(f"  {label} LLM evaluator: correct={correct}, score={score}")
        return correct, score

    return (pred == str(truth)), None


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
