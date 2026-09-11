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

import json
import os
import warnings
from cProfile import Profile
from itertools import product
from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace

from lits import get_lm
from lits.agents.tree.augmentor_setup import build_search_callbacks
from lits.agents.tree.bfs import BFSSearch
from lits.agents.tree.common import _world_modeling
from lits.agents.tree.continuation import _continuation
from lits.agents.tree.mcts import MCTSConfig, MCTSSearch, _expand, _select
from lits.agents.tree.node import MCTSNode, SearchNode
from lits.components.context_augmentor.reflection import ReflectionAugmentor
from lits.components.policy.concat import ConcatPolicy
from lits.components.reward.generative import GenerativePRM
from lits.components.transition.concat import ConcatTransition
from lits.config import ExperimentConfig
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

    Exercise cumulative/decay updates with per-node/terminal broadcasting.
    World-model completion records one iteration under both stop settings;
    stop cases leave unused iterations to verify the early break.
    Continuation materializes its selected endpoint at entry, without generating
    another action; its continue case then revisits that terminal. Two sibling
    cases materialize equal-reward terminals with
    ``common.py::_world_modeling`` before three search iterations. UCT should
    select A, B, A: visit the unvisited sibling, then break the equal-score tie
    in tree order. One case seeds earlier root/parent estimates to check decay
    values, not just return-list length. Each selected traversal gets one backup.

    Profiling counts the real completion callback and reflection analysis calls.
    Successful rewards skip reflection generation. Fresh checkpoint files retain
    each iteration's post-backup statistics in the printed temporary directory.
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
    check_stop_configuration(base_model, policy, reward_model)
    query = "What is 6 multiplied by 7?"
    output_dir = Path(mkdtemp(prefix="lits-mcts-backup-modes-"))
    print(f"Backup-mode checkpoint checks: {output_dir}")
    failures = []
    cases = (
        ("transition_stop", True, 1, False, False),
        ("transition_continue", False, 1, False, False),
        ("continuation_stop", True, 1, False, True),
        ("continuation_continue", False, 2, False, True),
        ("terminal_revisits", False, 3, True, False),
        ("seeded_revisits", False, 3, True, False),
    )
    modes = product(cases, ("cumulative", "decay"), ("per_node", "terminal"))
    for (label, stop_on_terminal, n_iters, revisit, continuation), mode, broadcast in modes:
        case_name = f"{label}-{mode}-{broadcast}"
        case_dir = output_dir / case_name
        seeded = label == "seeded_revisits"
        config = MCTSConfig(
            n_iters=3 if stop_on_terminal else n_iters,
            max_steps=5,
            force_terminating_on_depth_limit=False,
            terminate_on_first_solution=stop_on_terminal,
            add_continuation=continuation,
            backprop_mode=mode,
            backprop_broadcast_mode=broadcast,
            decay_gamma=0.5,
        )
        search = MCTSSearch(
            config, world_model, policy, reward_model,
            checkpoint_dir=str(case_dir),
            augmentors=[ReflectionAugmentor(base_model, persist=False)],
        )
        search._setup(query, query_idx=0)
        try:
            root = search.root
            root.reward = 0.0
            step = ThoughtStep(action="Compute six groups of seven.")
            parent = MCTSNode(state=[step], action=step.action, parent=root, fast_reward=0.5)
            parent.step = step
            root.add_child(parent)
            if seeded:
                for node in (root, parent):
                    node.cum_rewards = [0.0]
                    node.visit_count = 1
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
            profiler = Profile()
            result = profiler.runcall(search.search, query, query_idx=0)
        finally:
            search._teardown()

        if any(node.state is None or not node.is_terminal or node.reward != 1.0 for node in terminals):
            raise RuntimeError("Transition did not materialize the expected rewarded terminal")
        if result.terminal_nodes_collected != terminals or any(node.children for node in terminals):
            raise RuntimeError("Search did not collect and close the terminal endpoint")
        if len(result.trace_in_each_iter) != n_iters:
            raise RuntimeError(f"Expected {n_iters} completed search iterations")

        # Explicit return oracles for rewards [0, .5, 1] and gamma=.5.
        if mode == "cumulative":
            root_updates = [0.5 if broadcast == "per_node" else 1.0] * n_iters
            parent_updates = [0.75 if broadcast == "per_node" else 1.0] * n_iters
        elif seeded:
            root_updates = [0.25, 0.5, 0.6875] if broadcast == "per_node" else [0.5, 0.75, 0.875]
            parent_updates = [0.5, 0.75, 0.875]
        else:
            root_updates = parent_updates = [1.0] * n_iters

        expected_ids = [endpoint.id, sibling.id, endpoint.id] if revisit else [endpoint.id] * n_iters
        selected_ids = [path[-1].id for path in result.trace_in_each_iter]
        errors = []
        if selected_ids != expected_ids:
            errors.append(f"selected IDs {selected_ids} != {expected_ids}")
        for iteration, path in enumerate(result.trace_in_each_iter):
            count = iteration + 1
            root_returns = ([0.0] if seeded else []) + root_updates[:count]
            parent_returns = ([0.0] if seeded else []) + parent_updates[:count]
            leaf_count = expected_ids[:count].count(path[-1].id)
            leaf_returns = [1.0] * leaf_count
            if mode == "decay":
                root_returns, parent_returns, leaf_returns = (
                    root_returns[-1:], parent_returns[-1:], leaf_returns[-1:],
                )
            expected = [
                (count + int(seeded), root_returns),
                (count + int(seeded), parent_returns),
                (leaf_count, leaf_returns),
            ]
            observed = [(node.visit_count, node.cum_rewards) for node in path]
            saved = json.loads((case_dir / "checkpoints" / f"0_{iteration}.json").read_text())
            saved_stats = [(node["visit_count"], node["cum_rewards"]) for node in saved]
            if observed != expected or saved_stats != expected:
                errors.append(f"iteration {iteration}: trace={observed}, saved={saved_stats}, expected={expected}")
            if saved[-1]["id"] != expected_ids[iteration]:
                errors.append(f"iteration {iteration}: saved the wrong endpoint")

        # Compare live statistics with each node's last recorded traversal.
        for node in (root, parent, *terminals):
            recorded = next(
                item for path in reversed(result.trace_in_each_iter)
                for item in path if item.id == node.id
            )
            if (node.visit_count, node.cum_rewards) != (recorded.visit_count, recorded.cum_rewards):
                errors.append(f"node {node.id}: live statistics differ from its last trace")

        callback_count = sum(
            entry.callcount for entry in profiler.getstats()
            if getattr(entry.code, "co_filename", None) == build_search_callbacks.__code__.co_filename
            and getattr(entry.code, "co_name", None) == "on_trajectory_complete"
        )
        call_counts = {entry.code: entry.callcount for entry in profiler.getstats()}
        expected_callbacks = 0 if revisit else 1
        if callback_count != expected_callbacks:
            errors.append(f"completion callbacks {callback_count} != {expected_callbacks}")
        if call_counts.get(ReflectionAugmentor._analyze.__code__, 0) != expected_callbacks:
            errors.append("completion callback did not reach reflection analysis once")
        if call_counts.get(_continuation.__code__, 0) != int(continuation):
            errors.append("unexpected number of continuation invocations")
        print(f"  {'FAIL' if errors else 'PASS'}: {case_name}; iterations={n_iters}, callbacks={callback_count}")
        for error in errors:
            print(f"    {error}")
        breakpoint()  # inspect: errors, result.trace_in_each_iter, call_counts, case_dir
        if errors:
            failures.append(case_name)

    if failures:
        raise RuntimeError("Selected-terminal backup failed: " + ", ".join(failures))
    check_early_exit_recording(world_model, policy, reward_model)
    check_terminal_stopping(world_model, policy, reward_model)


def check_early_exit_recording(world_model, policy, reward_model) -> None:
    """Check ``mcts.py::MCTSSearch._handle_early_exit`` ordering without inference.

    Reuse real search components; the supplied callback checks backup statistics
    and writes an inspection marker. Check trace/checkpoint capture of that marker
    for all four stages, both stop settings, and checkpointing enabled/disabled,
    with the callback supplied or omitted independently of stage. Search-loop
    checks above verify the actual selection caller omits the callback.
    Fresh checkpoint files are retained in the printed temporary directory.
    """
    output_dir = Path(mkdtemp(prefix="lits-mcts-early-exit-"))
    print(f"Early-exit checkpoint checks: {output_dir}")
    stages = (
        ("select", "select"), ("continuation", "continuation"),
        ("world_model", "expand"), ("pre_simulate", "expand"),
    )
    for (stage, phase), stop, save, with_callback in product(
        stages, (False, True), (False, True), (False, True),
    ):
        case_dir = output_dir / f"{stage}-stop-{stop}-save-{save}-callback-{with_callback}"
        checkpoint = case_dir / "checkpoints" / "0_0.json"
        config = MCTSConfig(n_iters=1, terminate_on_first_solution=stop)
        search = MCTSSearch(
            config, world_model, policy, reward_model,
            checkpoint_dir=str(case_dir) if save else None,
        )
        query = "What is 6 multiplied by 7?"
        search._setup(query, query_idx=0)
        traces, calls = [], []

        def record_completion(path, reward, query_idx, *, from_phase):
            """Check backup precedes this callback and mark the recorded endpoint."""
            if [(node.visit_count, node.cum_rewards) for node in path] != expected:
                raise RuntimeError("Completion callback ran before backup")
            if traces or checkpoint.exists():
                raise RuntimeError("Trace or checkpoint was captured before the callback")
            calls.append((reward, query_idx, from_phase))
            path[-1].step.error = "completion recorded"

        callback_args = {"on_trajectory_complete": record_completion} if with_callback else {}
        try:
            root = search.root
            root.reward = 0.0
            step = ThoughtStep(action="The answer is 42.")
            endpoint = MCTSNode(state=None, action=step.action, parent=root, fast_reward=1.0)
            endpoint.step = step
            root.add_child(endpoint)
            path = endpoint.get_path()
            expected = [(1, [0.5]), (1, [1.0])]

            if search._handle_early_exit(
                path, 0, 0, traces, stage=stage, **callback_args,
            ):
                raise RuntimeError("A nonterminal path unexpectedly ended its iteration")
            if traces or calls or checkpoint.exists() or any(node.visit_count or node.cum_rewards for node in path):
                raise RuntimeError("A non-stopping check changed statistics or records")

            _world_modeling(query, 0, endpoint, world_model, reward_model)
            ended = search._handle_early_exit(
                path, 0, 0, traces, stage=stage, **callback_args,
            )
            if not ended or [(node.visit_count, node.cum_rewards) for node in path] != expected:
                raise RuntimeError("Early exit did not back up its terminal path once")
            expected_calls = [(1.0, 0, phase)] if with_callback else []
            if calls != expected_calls:
                raise RuntimeError(f"Incorrect completion callback: {calls} != {expected_calls}")
            marker = "completion recorded" if with_callback else None
            if len(traces) != 1 or [(node.visit_count, node.cum_rewards) for node in traces[0]] != expected:
                raise RuntimeError("Trace did not capture post-backup statistics")
            endpoint.step.error = "changed after exit"
            if traces[0][-1].step.error != marker:
                raise RuntimeError("Trace did not preserve an independent post-callback copy")
            if save:
                saved = json.loads(checkpoint.read_text())
                if [(node["visit_count"], node["cum_rewards"]) for node in saved] != expected:
                    raise RuntimeError("Checkpoint did not capture post-backup statistics")
                if saved[-1]["step"].get("error") != marker:
                    raise RuntimeError("Checkpoint did not capture the callback update")
            elif case_dir.exists():
                raise RuntimeError("Disabled checkpointing created an output directory")
            print(f"  PASS: {stage}, stop={stop}, checkpoint={save}, callback={with_callback}")
            breakpoint()  # inspect: calls, traces, checkpoint, expected
        finally:
            search._teardown()


def check_terminal_stopping(world_model, policy, reward_model) -> None:
    """Check terminal quality does not prevent iteration completion.

    ``concat.py::ConcatTransition`` with step_signal marks a low-scoring answer
    terminal independently of its score. Call ``mcts.py::MCTSSearch._handle_early_exit``
    directly so a rejected exit cannot proceed to LLM expansion during this check.
    Separately run ``mcts.py::MCTSSearch.search`` with a qualifying answer completed
    during world modeling, before simulation; global early stopping should still apply.
    Profiling checks that these full-search cases never invoke simulation.
    Controls disable the search-stop flag or its threshold. All actions/rewards are supplied;
    no inference is needed. Report every case before raising on failed expectations.
    """
    output_dir = Path(mkdtemp(prefix="lits-mcts-terminal-stopping-"))
    print(f"Terminal-stopping checks: {output_dir}")
    signal_transition = ConcatTransition(
        base_model=policy.base_model, terminate_constraints=["step_signal"],
    )
    # Name, full search or exit helper, search threshold, early stop, expected iterations.
    cases = (
        ("low_terminal_control", False, None, True, 1),
        ("low_terminal_with_threshold", False, 0.9, True, 1),
        ("qualifying_terminal_control", True, 0.9, False, 3),
        ("qualifying_terminal_early_stop", True, 0.9, True, 1),
    )
    failures = []
    for label, full_search, threshold, early_stop, expected_iterations in cases:
        case_dir = output_dir / label
        score = 1.0 if full_search else 0.2
        transition = world_model if full_search else signal_transition
        config = MCTSConfig(
            n_iters=3, max_steps=5, force_terminating_on_depth_limit=False,
            terminate_on_first_solution=early_stop, early_stop_reward=threshold,
        )
        search = MCTSSearch(
            config, transition, policy, reward_model, checkpoint_dir=str(case_dir),
        )
        query = "What is 6 multiplied by 7?"
        search._setup(query, query_idx=0)
        try:
            root = search.root
            root.reward = 0.0
            step = ThoughtStep(
                action="The answer is 42." if full_search else "The answer is 41.",
                terminate=not full_search,
            )
            endpoint = MCTSNode(state=None, action=step.action, parent=root, fast_reward=score)
            endpoint.step = step
            root.add_child(endpoint)
            if full_search:
                profiler = Profile()
                result = profiler.runcall(search.search, query, query_idx=0)
                traces = result.trace_in_each_iter
            else:
                _world_modeling(query, 0, endpoint, transition, reward_model)
                traces = []
                ended = search._handle_early_exit(
                    endpoint.get_path(), 0, 0, traces, stage="world_model",
                )
                print(f"  {label}: exit helper returned {ended}; expected True")

            if endpoint.state is None or not endpoint.is_terminal or endpoint.reward != score:
                raise RuntimeError(f"{label}: fixture did not produce the expected scored terminal")
            checkpoints = sorted((case_dir / "checkpoints").glob("0_[0-9]*.json"))
            saved = json.loads(checkpoints[-1].read_text()) if checkpoints else []
            expected_stats = [
                (expected_iterations, [score / 2] * expected_iterations),
                (expected_iterations, [score] * expected_iterations),
            ]
            observed = {
                "iteration_traces": len(traces),
                "node_stats": [(node.visit_count, node.cum_rewards) for node in (root, endpoint)],
                "checkpoint_count": len(checkpoints),
                "saved_stats": [(node["visit_count"], node["cum_rewards"]) for node in saved],
                "terminal_children": len(endpoint.children),
            }
            expected = {
                "iteration_traces": expected_iterations, "node_stats": expected_stats,
                "checkpoint_count": expected_iterations, "saved_stats": expected_stats,
                "terminal_children": 0,
            }
            if full_search:
                observed["simulation_calls"] = sum(
                    entry.callcount for entry in profiler.getstats()
                    if entry.code == MCTSSearch._do_simulate.__code__
                )
                expected["simulation_calls"] = 0
            passed = observed == expected and (full_search or ended is True)
            print(f"  {'PASS' if passed else 'FAIL'}: {label}, terminal={endpoint.is_terminal}, reward={score}")
            print(f"    observed: {observed}")
            print(f"    expected: {expected}")
            breakpoint()  # inspect: observed, expected, traces, checkpoints, config
            if not passed:
                failures.append(label)
        finally:
            search._teardown()
    failures.extend(check_rollout_terminal_stopping(signal_transition, policy, reward_model))
    failures.extend(check_rollout_iteration_recording(signal_transition, policy, reward_model))
    failures.extend(check_rollout_path_cutoffs(signal_transition, policy, reward_model))
    failures.extend(check_nonterminal_cutoffs(world_model, policy, reward_model))
    if failures:
        raise RuntimeError("Terminal-stopping regressions failed: " + ", ".join(failures))


def check_rollout_terminal_stopping(world_model, policy, reward_model) -> list[str]:
    """Check terminal entry and completion in ``mcts.py::MCTSSearch._do_simulate``.

    Reuse pre-scored children through the real expansion implementation, then let
    the real step-signal transition detect completion. No inference is needed.
    A two-step allowance exposes attempts to continue after a low-score terminal.
    Neither case sets the separate repetition flag. Return failures so the caller
    can report all terminal-stopping regressions together.
    """
    failures = []
    for at_entry in (True, False):
        label = "terminal_at_rollout_entry" if at_entry else "terminal_during_rollout"
        config = MCTSConfig(
            early_stop_reward=0.9, roll_out_steps=2, max_steps=5,
            force_terminating_on_depth_limit=False, terminate_on_first_solution=False,
        )
        search = MCTSSearch(config, world_model, policy, reward_model)
        query = "What is 6 multiplied by 7?"
        search._setup(query, query_idx=0)
        try:
            root = search.root
            step = ThoughtStep(action="The answer is 41.", terminate=True)
            endpoint = MCTSNode(state=None, action=step.action, parent=root, fast_reward=0.2)
            endpoint.step = step
            endpoint.from_expand = True
            root.add_child(endpoint)
            path = [root]
            if at_entry:
                _world_modeling(query, 0, endpoint, world_model, reward_model)
                path.append(endpoint)
            try:
                repeated, siblings = search._do_simulate(
                    query, 0, path, config, world_model, policy, reward_model,
                    roll_out_steps=config.roll_out_steps,
                )
            except ValueError as error:
                print(f"  FAIL: {label}: {type(error).__name__}: {error}")
                failures.append(label)
                continue
            observed = (path == [root, endpoint], endpoint.is_terminal,
                        endpoint.is_terminal_for_repeat, len(endpoint.children),
                        repeated, len(siblings), root.visit_count, endpoint.visit_count)
            expected = (True, True, False, 0, False, 0, 0, 0)
            print(f"  {'PASS' if observed == expected else 'FAIL'}: {label}: {observed}")
            breakpoint()  # inspect: path, endpoint, observed, expected
            if observed != expected:
                failures.append(label)
        finally:
            search._teardown()
    return failures


def check_rollout_iteration_recording(world_model, policy, reward_model) -> list[str]:
    """Check full-loop rollout backup and recording without generating actions.

    A selection-hook subclass chooses the prepared parent on each iteration;
    it tests ``mcts.py::MCTSSearch.search`` phase execution, not UCT traversal.
    Real expansion reuses pre-scored children. Only the best child is simulated; siblings must keep
    empty returns and zero visits. A one-step rollout also tests nonterminal
    exhaustion. Check terminal and repetition exits, the stop flag, its threshold, and saved
    post-backup statistics with checkpointing on/off. No inference is needed.
    """
    class PreparedParentSearch(MCTSSearch):
        """Fix the selected path while retaining the real expansion and rollout."""

        def _do_select_path(self, query, query_idx, iteration):
            """Return the prepared parent's path for this iteration."""
            return self.root.children[0].get_path()

    output_dir = Path(mkdtemp(prefix="lits-mcts-rollout-recording-"))
    print(f"Rollout iteration checks: {output_dir}")
    # Name, terminal signal, reward, stop flag, threshold, iterations, repeat flag.
    cases = (
        ("low_terminal", True, 0.2, True, 0.9, 3, False),
        ("first_solution", True, 1.0, True, 0.9, 1, False),
        ("stopping_disabled", True, 1.0, False, 0.9, 3, False),
        ("threshold_equal", True, 0.9, True, 0.9, 1, False),
        ("no_threshold", True, 0.2, True, None, 1, False),
        ("nonterminal_exhaustion", False, 0.2, True, None, 3, False),
        ("low_repeat", False, 0.2, True, 0.9, 3, True),
        ("repeat_threshold_equal", False, 0.9, True, 0.9, 3, True),
        ("high_score_repeat", False, 1.0, True, 0.9, 3, True),
        ("repeat_stopping_disabled", False, 1.0, False, None, 3, True),
        ("terminal_with_repeat_flag", True, 1.0, True, 0.9, 1, True),
    )
    failures = []
    for (label, terminal, score, stop, threshold, expected_iters, repeated), save in product(
        cases, (False, True),
    ):
        case_dir = output_dir / f"{label}-save-{save}"
        config = MCTSConfig(
            n_iters=3, max_steps=5, roll_out_steps=1,
            terminate_on_first_solution=stop, early_stop_reward=threshold,
        )
        search = PreparedParentSearch(
            config, world_model, policy, reward_model,
            checkpoint_dir=str(case_dir) if save else None,
        )
        query = "What is 6 multiplied by 7?"
        search._setup(query, query_idx=0)
        try:
            root = search.root
            root.reward = 0.0
            parent_step = ThoughtStep(action="Compute six groups of seven.")
            parent = MCTSNode(state=[parent_step], action=parent_step.action, parent=root, fast_reward=0.5)
            parent.step = parent_step
            root.add_child(parent)
            for index in range(policy.n_actions):
                step = ThoughtStep(action=f"Candidate {index}: six times seven is 42.", terminate=terminal)
                child = MCTSNode(state=None, action=step.action, parent=parent, fast_reward=score - index * 0.05)
                child.step = step
                child.from_expand = True
                parent.add_child(child)
            endpoint = parent.children[0]
            endpoint.is_terminal_for_repeat = repeated
            profiler = Profile()
            result = profiler.runcall(search.search, query, query_idx=0)
            expected_stats = [
                (expected_iters, [(0.5 + score) / 3] * expected_iters),
                (expected_iters, [(0.5 + score) / 2] * expected_iters),
                (expected_iters, [score] * expected_iters),
            ]
            checkpoints = sorted((case_dir / "checkpoints").glob("0_[0-9]*.json"))
            saved = [json.loads(path.read_text()) for path in checkpoints]
            observed = {
                "iterations": len(result.trace_in_each_iter),
                "node_stats": [(node.visit_count, node.cum_rewards) for node in (root, parent, endpoint)],
                "terminal": endpoint.is_terminal,
                "simulation_calls": sum(entry.callcount for entry in profiler.getstats()
                                        if entry.code == MCTSSearch._do_simulate.__code__),
                "checkpoints": len(saved),
                "siblings_untouched": all(node.visit_count == 0 and not node.cum_rewards
                                          for node in parent.children[1:]),
            }
            expected = {
                "iterations": expected_iters, "node_stats": expected_stats,
                "terminal": terminal, "simulation_calls": expected_iters,
                "checkpoints": expected_iters if save else 0, "siblings_untouched": True,
            }
            copies_match = all(
                [(node.visit_count, node.cum_rewards) for node in trace]
                == [(i, returns[:i]) for _, returns in expected_stats]
                for i, trace in enumerate(result.trace_in_each_iter, start=1)
            )
            files_match = all(
                [(node["visit_count"], node["cum_rewards"]) for node in nodes]
                == [(i, returns[:i]) for _, returns in expected_stats]
                for i, nodes in enumerate(saved, start=1)
            )
            passed = (observed == expected and copies_match and files_match
                      and (save or not case_dir.exists()))
            print(f"  {'PASS' if passed else 'FAIL'}: {label}, save={save}: {observed}")
            breakpoint()  # inspect: observed, expected, result, saved, copies_match, files_match
            if not passed:
                failures.append(f"{label}-save-{save}")
        finally:
            search._teardown()
    return failures


def check_nonterminal_cutoffs(world_model, policy, reward_model) -> list[str]:
    """Check nonterminal cutoffs never qualify for completion-based search stopping."""
    failures = []
    # Name, depth limit, repetition, threshold, reward, path exit, global stop.
    cases = (
        ("depth_stop", True, False, None, 0.2, True, False),
        ("depth_below_threshold", True, False, 0.9, 0.2, True, False),
        ("depth_above_threshold", True, False, 0.9, 1.0, True, False),
        ("repeat_no_threshold", False, True, None, 0.2, True, False),
        ("repeat_below_threshold", False, True, 0.9, 0.2, True, False),
        ("repeat_at_threshold", False, True, 0.9, 0.9, True, False),
        ("repeat_above_threshold", False, True, 0.9, 1.0, True, False),
        ("nonterminal_no_cutoff", False, False, None, 1.0, False, False),
    )
    for label, depth_limit, repeated, threshold, score, expected_exit, expected_stop in cases:
        config = MCTSConfig(max_steps=1, force_terminating_on_depth_limit=depth_limit,
                            early_stop_reward=threshold, terminate_on_first_solution=True)
        search = MCTSSearch(config, world_model, policy, reward_model)
        search._setup("What is 6 multiplied by 7?", query_idx=0)
        try:
            step = ThoughtStep(action="Compute six groups of seven.")
            endpoint = MCTSNode(state=[step], action=step.action, parent=search.root, fast_reward=score)
            endpoint.step = step
            endpoint.is_terminal_for_repeat = repeated
            search.root.add_child(endpoint)
            path, traces = endpoint.get_path(), []
            ended = search._handle_early_exit(path, 0, 0, traces, stage="select")
            observed = (ended, len(traces), endpoint.is_terminal,
                        [(node.visit_count, node.cum_rewards) for node in path],
                        search._should_stop_search(endpoint))
            expected = (expected_exit, int(expected_exit), False, [(0, []), (0, [])], expected_stop)
            print(f"  {'PASS' if observed == expected else 'FAIL'}: {label}: {observed}")
            breakpoint()  # inspect: observed, expected, path, traces
            if observed != expected:
                failures.append(label)
        finally:
            search._teardown()
    return failures


def check_rollout_path_cutoffs(world_model, policy, reward_model) -> list[str]:
    """Check depth/repetition stops in ``mcts.py::MCTSSearch._do_simulate``.

    Pre-scored children avoid inference. The first transition is nonterminal at
    depth one; the second is terminal. A depth bound or repetition flag must stop
    rollout at the first transition regardless of score. At entry, neither should
    allow expansion. With neither cutoff, rollout reaches the second transition.
    """
    failures = []
    for cutoff, score, at_entry in product(("depth", "repeat", "neither"), (0.2, 0.9, 1.0), (False, True)):
        enabled = cutoff == "depth"
        config = MCTSConfig(max_steps=1, force_terminating_on_depth_limit=enabled,
                            early_stop_reward=0.9, roll_out_steps=2)
        search = MCTSSearch(config, world_model, policy, reward_model)
        query = "What is 6 multiplied by 7?"
        search._setup(query, query_idx=0)
        try:
            parent = search.root
            for text, terminal, reward in (
                ("Compute six groups of seven.", False, score),
                ("The answer is 42.", True, 1.0),
            ):
                step = ThoughtStep(action=text, terminate=terminal)
                child = MCTSNode(state=None, action=step.action, parent=parent, fast_reward=reward)
                child.step = step
                child.from_expand = True
                parent.add_child(child)
                parent = child
            endpoint = search.root.children[0]
            endpoint.is_terminal_for_repeat = cutoff == "repeat"
            path = [search.root]
            if at_entry:
                _world_modeling(query, 0, endpoint, world_model, reward_model)
                path.append(endpoint)
            profiler = Profile()
            profiler.runcall(search._do_simulate, query, 0, path, config,
                             world_model, policy, reward_model, roll_out_steps=config.roll_out_steps)
            expansion_calls = sum(entry.callcount for entry in profiler.getstats()
                                  if entry.code == _expand.__code__)
            depth = 2 if cutoff == "neither" else 1
            observed = (path[-1].depth, path[-1].is_terminal, expansion_calls)
            expected = (depth, cutoff == "neither", depth - int(at_entry))
            print(f"  {'PASS' if observed == expected else 'FAIL'}: rollout cutoff={cutoff}, "
                  f"score={score}, at_entry={at_entry}: {observed}")
            breakpoint()  # inspect: path, observed, expected, config
            if observed != expected:
                failures.append(f"rollout-{cutoff}-{score}-entry-{at_entry}")
        finally:
            search._teardown()
    return failures


def check_stop_configuration(base_model, policy, reward_model) -> None:
    """Check direct/CLI config migration and independent transition/search thresholds.

    Exercise ``config.py::ExperimentConfig.create_search_config`` and
    ``concat.py::ConcatTransition.from_config`` using the configured real client
    without inference. Inspect serialized defaults and explicit migration errors.
    """
    failures = []

    def record(label, passed):
        """Report a check and retain failures for the final exception."""
        print(f"  {'PASS' if passed else 'FAIL'}: {label}")
        breakpoint()  # inspect: label, passed, failures
        if not passed:
            failures.append(label)

    def experiment(search_args):
        """Build the same configuration entry point used by the search CLI."""
        return ExperimentConfig(dataset="math500", policy_model_name="openai/local-instruct",
                                eval_model_name="openai/local-instruct", search_args=search_args)

    for overrides, expected_stop in (({}, True), ({"terminate_on_first_solution": False}, False)):
        for source, config in (("direct", MCTSConfig(**overrides)),
                               ("CLI", experiment(overrides).create_search_config())):
            saved = json.loads(json.dumps(config.to_dict()))
            record(f"{source} canonical stop={expected_stop}",
                   saved["terminate_on_first_solution"] is expected_stop
                   and saved["early_stop_reward"] is None
                   and "terminate_on_terminal_node" not in saved)

    for legacy in (True, False):
        for source in ("direct", "CLI"):
            try:
                overrides = {"terminate_on_terminal_node": legacy}
                if source == "direct":
                    MCTSConfig(**overrides)
                else:
                    experiment(overrides).get_search_args()
            except (TypeError, ValueError) as error:
                record(f"{source} rejects retired flag={legacy}",
                       "terminate_on_terminal_node" in str(error))
            else:
                record(f"{source} rejects retired flag={legacy}", False)

    exp = experiment({"terminate_constraints": ["reward_threshold"],
                      "r_terminating": 0.7, "early_stop_reward": 0.9})
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        config = exp.create_search_config()
    saved = json.loads(json.dumps(exp.to_dict()))["search_args"]
    record("transition threshold retained with explicit migration warning",
           saved["r_terminating"] == 0.7 and saved["early_stop_reward"] == 0.9
           and any("r_terminating only configures transition" in str(n.message) for n in notices))
    transition = ConcatTransition.from_config(base_model, exp.get_search_args(), exp.get_component_args())
    search = MCTSSearch(config, transition, policy, reward_model)
    state = [ThoughtStep(action="The answer is 42.", terminate=True)]
    for score in (0.6, 0.7, 0.8, 0.9):
        terminal = transition.is_terminal(state, "What is 6 multiplied by 7?", fast_reward=score)
        node = MCTSNode(state=state, action=state[-1].action,
                        fast_reward=score, is_terminal=terminal)
        record(f"independent thresholds at reward={score}",
               terminal == (score >= 0.7) and search._should_stop_search(node) == (score >= 0.9))
    if failures:
        raise RuntimeError("Stop-configuration checks failed: " + ", ".join(failures))


def main() -> None:
    inspect_search_node_path()
    inspect_search_node_walk()
    inspect_bfs_default()
    inspect_mcts_default()
    check_selected_terminal_backup()


if __name__ == "__main__":
    main()
