"""Per-example tree search, saved outputs, and registered answer resolution."""

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from lits.agents.tree.search_base import SearchResult
    from lits.components.base import Transition, Policy, RewardModel
    from lits.components.bn_evaluator import BNEvaluatorBase
    from lits.eval import TreeToJsonl

logger = logging.getLogger(__name__)


def run_example(
    example: dict, query_idx: int, repetition_index: int, *,
    task_type: str, experiment_config, world_model: Transition, policy: Policy,
    evaluator: RewardModel, result_saver: TreeToJsonl, result_dir: str,
    bn_evaluator: BNEvaluatorBase = None,
    result_saver_unselected=None, augmentors: list = None,
    prepare_tool_state: Callable | None = None, resolve_answer: Callable | None = None,
    memory_manager=None, run_logger=None, search_kwargs: dict | None = None,
    search_context: Callable | None = None,
) -> SearchResult | None:
    """Run preparation, search, answer resolution, and memory saving.

    The CLI uses one repetition. Resume checking stays inside run_tree_search:
    resource preparation and memory saving still run for completed examples,
    while answer resolution runs only for newly saved terminal files.

    Args:
        example: Original dataset entry. Extract its question/goal and pass the
            entry unchanged to resource preparation and world-model initialization.
        query_idx: Original dataset index, also used in saved filenames.
        repetition_index: Dataset-loop compatibility argument. It does not change
            filenames or seed models; bind a separate result_dir and savers per
            repetition to prevent one repetition resuming another's output.
        task_type: Use query_or_goals (falling back to question) for env_grounded;
            otherwise use question.
        world_model, policy, evaluator, bn_evaluator: Caller-built components
            forwarded to the registered search.
        result_saver, result_saver_unselected: Existing path writers; the second
            optionally saves unselected MCTS simulation paths.
        result_dir: Root for checkpoints, terminal files, and memory snapshots.
        augmentors: Optional search augmentors.
        prepare_tool_state: Resource callback accepting the original example.
        resolve_answer: Resource callback accepting raw answer and ToolUseState;
            resolved answers replace saved answers only for newly saved terminals.
        memory_manager: Optional manager whose backend.save writes a snapshot.
        run_logger: Optional logger; defaults to this module's logger.
        search_kwargs: Extra registered-search constructor arguments, such as
            beam_selector, frontier_selector, or selection_events. Forwarded
            without copying their values; cannot replace explicit search inputs.
        search_context: Optional zero-argument context-manager factory wrapping
            only search.run(); see run_tree_search. Not entered on resume.
        experiment_config: Full ExperimentConfig supplying the registered algorithm
            and create_search_config(). Also saved and checked before search/resume.
            Matching config.json files are unchanged; conflicts raise ValueError.

    Returns:
        The newly executed SearchResult, after saving; None when search is skipped.
        Answer resolution updates saved terminal answers only, not this object.
        Terminal-file existence is the current resume rule, not a completeness
        guarantee. No correctness judging or experiment metrics are added here.

    Bind keyword dependencies with functools.partial to obtain the three-argument
    callback accepted by runner.py::run_benchmark. Model/resource construction
    remains outside this function. Memory-save errors are logged; other uncaught
    errors propagate to the caller.

    Saving and search-only timing:
        This function does not measure time. The timing column describes a timer
        around BaseTreeSearch.run(), not around this entire workflow. References
        use symbols/calls so they remain useful when source lines move.

        Operation                 Save?             Timed?  Code reference
        ------------------------  ----------------  ------  -----------------------------
        Internal checkpoints      When configured   Yes     search_base.py::save_checkpoint
        Internal full-tree save   When invoked      Yes     search_base.py::save_full_tree
        Final paths               Keep              No      run_tree_search: result_saver.append_result
        Terminal nodes            Keep              No      save_terminal_nodes
        Unselected rollout paths  Optional writer   No      run_tree_search: result_saver_unselected
        Memory snapshot           Optional manager  No      run_example: memory_manager.backend.save
        Answer resolution         Saved if changed  No      run_example: resolve_answer(raw, tool_state)
        Correctness evaluation    External caller   No      Not called by this module

        Internal writes remain part of run() wall time; final paths and terminal
        files are written after it returns. Optional writers/managers can be None.
    """
    run_logger = run_logger or logger
    experiment_config.save_config(result_dir, overwrite=False)
    if task_type == "env_grounded":
        query_or_goals = example.get("query_or_goals", example.get("question", ""))
    else:
        query_or_goals = example["question"]

    if prepare_tool_state is not None:
        prepare_tool_state(example)

    tn_file = Path(result_dir) / "terminal_nodes" / f"terminal_nodes_{query_idx}.json"
    tn_existed_before = tn_file.exists()

    algo_output = run_tree_search(
        query_or_goals=query_or_goals,
        query_idx=query_idx,
        search_config=experiment_config.create_search_config(),
        world_model=world_model,
        policy=policy,
        evaluator=evaluator,
        bn_evaluator=bn_evaluator,
        result_saver=result_saver,
        result_dir=result_dir,
        result_saver_unselected=result_saver_unselected,
        init_state_kwargs=example,
        augmentors=augmentors,
        search_algorithm=experiment_config.search_algorithm,
        run_logger=run_logger,
        search_kwargs=search_kwargs,
        search_context=search_context,
    )

    # Skip resolve on resume (search was a no-op). See docs/cli/search.md.
    if resolve_answer is not None and not tn_existed_before:
        tn_file = Path(result_dir) / "terminal_nodes" / f"terminal_nodes_{query_idx}.json"
        if tn_file.exists():
            with open(tn_file) as f:
                tn_data = json.load(f)
            modified = False
            for node in tn_data.get("terminal_nodes", []):
                state_data = node.get("state", {})
                # state can be a dict {"__type__": ..., "steps": [...]} or a list
                if isinstance(state_data, dict):
                    state_steps = state_data.get("steps", [])
                else:
                    state_steps = state_data
                if state_steps:
                    last_step = state_steps[-1]
                    raw = last_step.get("answer")
                    if raw:
                        # Deserialize to ToolUseState so resolve_answer
                        # always receives the same type as chain.py
                        from lits.structures.tool_use import ToolUseState
                        tool_state = ToolUseState.from_dict({"steps": state_steps})
                        resolved = resolve_answer(raw, tool_state)
                        if resolved != raw:
                            last_step["answer"] = resolved
                            # Also update node.step.answer (eval reads this first)
                            node_step = node.get("step")
                            if isinstance(node_step, dict) and "answer" in node_step:
                                node_step["answer"] = resolved
                            modified = True
            if modified:
                with open(tn_file, "w") as f:
                    json.dump(tn_data, f, indent=2)
                run_logger.info(f"[{query_idx}] Resolved answers in terminal nodes")

    # Save memory backend per-example (facts persist for inspection/debugging)
    if memory_manager is not None and hasattr(memory_manager, 'backend'):
        memory_dir = os.path.join(result_dir, "memory", str(query_idx))
        try:
            memory_manager.backend.save(memory_dir)
        except Exception as e:
            run_logger.warning(f"[{query_idx}] Failed to save memory: {e}")
    return algo_output


def save_terminal_nodes(algo_output, query_or_goals, query_idx, result_dir, run_logger):
    """Save terminal nodes separately for post-search evaluation.

    Terminal nodes are saved to result_dir/terminal_nodes/terminal_nodes_{query_idx}.json
    for use by lits-eval to perform answer extraction and voting without needing
    to reconstruct the full search tree.

    Args:
        algo_output: MCTSResult or BFSResult containing terminal_nodes_collected
        query_or_goals: Original query/question
        query_idx: Query index
        result_dir: Directory to save results
        run_logger: Logger instance
    """
    terminal_nodes_data = {
        'terminal_nodes': [node.to_dict() for node in algo_output.terminal_nodes_collected],
        'query': query_or_goals,
        'query_idx': query_idx
    }

    terminal_nodes_dir = Path(result_dir) / "terminal_nodes"
    terminal_nodes_dir.mkdir(parents=True, exist_ok=True)
    terminal_nodes_file = terminal_nodes_dir / f"terminal_nodes_{query_idx}.json"

    with open(terminal_nodes_file, 'w') as f:
        json.dump(terminal_nodes_data, f, indent=2)

    run_logger.debug(f"Saved {len(algo_output.terminal_nodes_collected)} terminal nodes to {terminal_nodes_file}")


def run_tree_search(
    query_or_goals: str, query_idx: int, search_config, world_model: Transition, policy: Policy,
    evaluator: RewardModel, bn_evaluator: BNEvaluatorBase, result_saver: TreeToJsonl, result_dir: str,
    result_saver_unselected=None, init_state_kwargs: dict = None,
    augmentors: list = None,
    search_algorithm: str = "mcts",
    run_logger=None,
    search_kwargs: dict | None = None,
    search_context: Callable | None = None,
) -> SearchResult | None:
    """Run tree search and save results to checkpoint files.

    This function only performs tree search and saves results. Answer extraction
    and evaluation should be done separately as post-processing using the saved
    checkpoint files.
    The caller saves the full experiment configuration before invoking this function.

    Args:
        query_or_goals: Question to solve
        query_idx: Example index for logging/tracking
        search_config: Search configuration (MCTSConfig or BFSConfig)
        world_model: World model
        policy: Policy
        evaluator: Evaluator/reward model
        bn_evaluator: BN evaluator (optional)
        result_saver: Result saver (TreeToJsonl for both MCTS and BFS)
        result_dir: Directory to save results
        result_saver_unselected: Unselected paths saver (MCTS only)
        init_state_kwargs: Optional kwargs passed to world_model.init_state().
                           For env_grounded tasks, should include 'init_state_str'.
        augmentors: List of ContextAugmentor instances to wire into the search loop
                   via setup_augmentors().
        search_algorithm: Search algorithm to use ("mcts", "bfs", or custom registered algorithm)
        run_logger: Logger instance (falls back to module logger if None)
        search_kwargs: Additional constructor arguments. The mapping is not mutated
            and its values retain identity. Conflicts with explicit inputs raise
            ValueError before construction; unsupported keys follow the registered
            constructor's validation. Skipped searches do not construct anything.
        search_context: Optional zero-argument factory returning a context manager.
            Entered after construction, around search.run() only, and exited before
            final serialization. Allows caller-owned accounting without importing
            its metrics here. Not invoked on resume; exceptions propagate normally.

    Returns:
        The newly executed SearchResult after writing outputs, or None when an
        existing terminal file causes search to be skipped. Saving errors still
        propagate; returning a result does not certify successful execution.
    """
    # Load agents only when executing, not when importing benchmark registration.
    from lits.agents import AgentRegistry

    _logger = run_logger or logger

    # Skip if terminal nodes already exist for this query (resume support)
    tn_file = Path(result_dir) / "terminal_nodes" / f"terminal_nodes_{query_idx}.json"
    if tn_file.exists():
        _logger.info(f"[{query_idx}] terminal_nodes already exists, skipping")
        return

    # Construct separately so caller accounting excludes constructor work.
    search_class = AgentRegistry.get_search_class(search_algorithm)

    constructor_kwargs = {
        "config": search_config,
        "world_model": world_model,
        "policy": policy,
        "reward_model": evaluator,
        "bn_evaluator": bn_evaluator,
        "init_state_kwargs": init_state_kwargs,
        "checkpoint_dir": result_dir,
        "augmentors": augmentors or [],
    }
    extra = dict(search_kwargs or {})
    conflicts = extra.keys() & (constructor_kwargs.keys() | {"query", "query_idx"})
    if conflicts:
        raise ValueError(f"search_kwargs duplicates explicit inputs: {sorted(conflicts)}")
    constructor_kwargs.update(extra)

    search = search_class(**constructor_kwargs)
    with search_context() if search_context is not None else nullcontext():
        algo_output = search.run(query_or_goals, query_idx)

    # Save unselected paths if available (MCTS-specific attribute)
    if result_saver_unselected and hasattr(algo_output, 'unselected_terminal_paths_during_simulate'):
        result_saver_unselected.append_result(algo_output.unselected_terminal_paths_during_simulate)

    # Convert to paths for saving — all results support to_paths()
    paths = algo_output.to_paths()
    result_saver.append_result(paths)

    # Save terminal nodes separately for post-search evaluation
    save_terminal_nodes(algo_output, query_or_goals, query_idx, result_dir, _logger)
    return algo_output
