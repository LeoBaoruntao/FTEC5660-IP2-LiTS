"""Inspect default selection hooks and backup after a selected terminal transition.

Run from ``lits_llm/`` with:
``python -m unit_test.agents.tree.test_selection_hooks``.
Set ``PYTHONBREAKPOINT=0`` to skip manual inspection.

References:
- ``bfs.py::BFSSearch._do_select_beam``
- ``mcts.py::MCTSSearch._do_select_path``
- ``mcts.py::MCTSSearch.search``
- ``mcts.py::_select``
"""

import os
from types import SimpleNamespace

from lits import get_lm
from lits.agents.tree.bfs import BFSSearch
from lits.agents.tree.common import _world_modeling
from lits.agents.tree.mcts import MCTSConfig, MCTSSearch, _select
from lits.agents.tree.node import MCTSNode, SearchNode
from lits.components.policy.concat import ConcatPolicy
from lits.components.reward.generative import GenerativePRM
from lits.components.transition.concat import ConcatTransition
from lits.structures import ThoughtStep


def inspect_search_node_path() -> None:
    """Check reusable root-to-node path reconstruction."""
    root = SearchNode(state="root", action=None)
    child = SearchNode(state="child", action="a", parent=root)
    leaf = SearchNode(state="leaf", action="b", parent=child)

    path = leaf.get_path()
    assert path == [root, child, leaf]
    assert root.get_path() == [root]
    print(f"SearchNode path: {[node.id for node in path]}")


def inspect_search_node_walk() -> None:
    """Check preorder traversal across sibling and descendant nodes."""
    root = SearchNode(state="root", action=None)
    left = SearchNode(state="left", action="left", parent=root)
    right = SearchNode(state="right", action="right", parent=root)
    leaf = SearchNode(state="leaf", action="leaf", parent=left)
    root.children.extend([left, right])
    left.children.append(leaf)

    visited_states = [node.state for node in root.walk()]
    print(
        f"SearchNode walk: {visited_states}; "
        "expected: ['root', 'left', 'leaf', 'right']"
    )
    breakpoint()  # inspect: visited_states


def inspect_bfs_default() -> None:
    """Inspect descending reward order and beam cardinality."""
    parent = SearchNode(state="root", action=None)
    frontier = [
        SearchNode(state="a", action="a", parent=parent, fast_reward=0.2),
        SearchNode(state="b", action="b", parent=parent, fast_reward=0.9),
        SearchNode(state="c", action="c", parent=parent, fast_reward=0.5),
    ]
    search = SimpleNamespace(
        config=SimpleNamespace(beam_size=2),
        reward_model=None,
    )

    selected = BFSSearch._do_select_beam(search, "query", 0, 1, frontier)
    selected_actions = [node.action for node in selected]
    print(f"BFS selected: {selected_actions}; expected: ['b', 'c']")
    breakpoint()  # inspect: selected_actions, selected


def inspect_mcts_default() -> None:
    """Compare the hook with the original module-level UCT dispatch."""
    root = MCTSNode(state="root", action=None)
    root.children.extend(
        [
            MCTSNode(state=None, action="low", parent=root, fast_reward=0.2),
            MCTSNode(state=None, action="high", parent=root, fast_reward=0.9),
        ]
    )
    config = SimpleNamespace(
        w_exp=1.0,
        max_steps=5,
        force_terminating_on_depth_limit=False,
    )
    search = SimpleNamespace(config=config, root=root)

    direct_path = _select(
        config.w_exp,
        root,
        config.max_steps,
        config.force_terminating_on_depth_limit,
    )
    default_path = MCTSSearch._do_select_path(search, "query", 0, 0)
    direct_ids = [node.id for node in direct_path]
    default_ids = [node.id for node in default_path]
    print(f"MCTS direct: {direct_ids}; default method: {default_ids}")
    breakpoint()  # inspect: direct_ids, default_ids, direct_path, default_path


def check_selected_terminal_backup() -> None:
    """Check terminal completion and revisits in ``mcts.py::MCTSSearch.search``.

    Initialize through ``search_base.py::BaseTreeSearch._setup``, attach a
    pre-scored root -> parent -> lazy endpoint path, then execute the real loop.
    ``concat.py::ConcatTransition`` uses its reward threshold here so completion
    is discovered during world modeling, not already signaled before selection.
    ``generative.py::GenerativePRM.reward`` reuses the supplied fast reward;
    neither action generation nor LLM evaluation is needed.

    Both stop settings run one iteration to isolate this exit from later revisits.
    A third case materializes two equal-reward terminal siblings with
    ``common.py::_world_modeling`` before three search iterations. UCT should
    select A, B, A: visit the unvisited sibling, then break the equal-score tie
    in tree order. A's second traversal must also be backed up.
    A failing check raises only after reporting all cases and their statistics.
    Configure the real client with LITS_TEST_MODEL_NAME and, optionally,
    LITS_TEST_BASE_URL / LITS_TEST_API_KEY; no inference is expected.
    Set HF_HUB_OFFLINE=1 to skip optional tokenizer metadata lookups.
    """
    model_kwargs = {}
    if base_url := os.environ.get("LITS_TEST_BASE_URL"):
        model_kwargs.update(
            base_url=base_url,
            api_key=os.environ.get("LITS_TEST_API_KEY", "no-key"),
        )
    base_model = get_lm(
        os.environ.get(
            "LITS_TEST_MODEL_NAME",
            "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        ),
        **model_kwargs,
    )
    policy = ConcatPolicy(base_model=base_model, task_name="math500")
    reward_model = GenerativePRM(base_model=base_model, task_name="math500")
    world_model = ConcatTransition(
        base_model=base_model,
        terminate_constraints=["reward_threshold"],
        r_terminating=0.9,
    )
    query = "What is 6 multiplied by 7?"
    failures = []
    cases = (
        ("transition_stop", True, 1, False),
        ("transition_continue", False, 1, False),
        ("terminal_revisits", False, 3, True),
    )
    for label, stop_on_terminal, n_iters, revisit in cases:
        config = MCTSConfig(
            n_iters=n_iters,
            max_steps=5,
            force_terminating_on_depth_limit=False,
            terminate_on_terminal_node=stop_on_terminal,
        )
        search = MCTSSearch(config, world_model, policy, reward_model)
        search._setup(query, query_idx=0)
        try:
            root = search.root
            root.reward = 0.0
            step = ThoughtStep(action="Compute six groups of seven.")
            parent = MCTSNode(state=[step], action=step.action, parent=root, fast_reward=0.5)
            root.add_child(parent)
            answer = ThoughtStep(action="The answer is 42.")
            endpoint = MCTSNode(state=None, action=answer.action, parent=parent, fast_reward=1.0)
            endpoint.step = answer
            parent.add_child(endpoint)
            terminals = [endpoint]
            if revisit:
                other_answer = ThoughtStep(action="Six times seven equals 42.")
                sibling = MCTSNode(
                    state=None, action=other_answer.action, parent=parent, fast_reward=1.0,
                )
                sibling.step = other_answer
                parent.add_child(sibling)
                terminals.append(sibling)
                for node in terminals:
                    _world_modeling(query, 0, node, world_model, reward_model)
                if any(not node.is_terminal or node.cum_rewards or node.visit_count for node in terminals):
                    raise RuntimeError("Expected materialized terminals with no previous backup")
            result = search.search(query, query_idx=0)
        finally:
            search._teardown()

        if any(node.state is None or not node.is_terminal or node.reward != 1.0 for node in terminals):
            raise RuntimeError("Transition did not materialize the expected rewarded terminal")
        if result.terminal_nodes_collected != terminals or any(node.children for node in terminals):
            raise RuntimeError("Search did not collect and close the terminal endpoint")
        if len(result.trace_in_each_iter) != n_iters:
            raise RuntimeError(f"Expected {n_iters} completed search iterations")

        # Default per-node mean backup of rewards [0.0, 0.5, 1.0].
        observed = [(node.visit_count, node.cum_rewards) for node in [root, parent, *terminals]]
        endpoint_visits = 2 if revisit else 1
        expected = [
            (n_iters, [0.5] * n_iters),
            (n_iters, [0.75] * n_iters),
            (endpoint_visits, [1.0] * endpoint_visits),
        ]
        expected_ids = [endpoint.id]
        if revisit:
            expected.append((1, [1.0]))
            expected_ids = [endpoint.id, sibling.id, endpoint.id]
        selected_ids = [path[-1].id for path in result.trace_in_each_iter]
        print(f"{label}: stop_on_terminal={stop_on_terminal}, terminal reward={endpoint.reward}")
        print(f"  selected IDs: {selected_ids}; expected: {expected_ids}")
        print(f"  root, parent, terminal(s) observed: {observed}")
        print(f"  expected:                         {expected}")
        breakpoint()  # inspect: selected_ids, expected_ids, observed, expected, result.trace_in_each_iter
        if observed != expected or selected_ids != expected_ids:
            failures.append(label)
            print("  FAIL: selected-terminal backup accounting differs")
        else:
            print("  PASS: selected path backed up once per iteration")

    if failures:
        raise RuntimeError("Selected-terminal backup failed: " + ", ".join(failures))


def main() -> None:
    inspect_search_node_path()
    inspect_search_node_walk()
    inspect_bfs_default()
    inspect_mcts_default()
    check_selected_terminal_backup()


if __name__ == "__main__":
    main()
