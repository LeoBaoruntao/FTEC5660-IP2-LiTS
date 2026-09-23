"""Per-example tree search, saved outputs, and registered answer resolution."""

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Callable

from filelock import FileLock, SoftFileLock
from lits import observability as tracing

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
    collect_search_facts: Callable[[SearchResult], dict] | None = None,
    query_or_goals: str | None = None, init_state_kwargs: dict | None = None,
) -> SearchResult | None:
    """Run preparation, search, answer resolution, and memory saving.

    The CLI uses one repetition. Resume checking stays inside this function:
    resource preparation and memory saving still run for completed examples,
    while answer resolution runs only for newly saved terminal files.

    Args:
        example: Original dataset entry, passed unchanged to resource preparation.
            Supplies the default query and world-model initialization arguments.
        query_idx: Original dataset index, also used in saved filenames.
        repetition_index: Dataset-loop compatibility argument. It does not change
            filenames or seed models; bind a separate result_dir and savers per
            repetition to prevent one repetition resuming another's output.
        task_type: Use query_or_goals (falling back to question) for env_grounded;
            otherwise use question, unless an explicit query_or_goals is supplied.
        query_or_goals: Optional caller-extracted query, supporting other dataset
            schemas without modifying example or including its reference in search.
        init_state_kwargs: Explicit world-model initialization arguments, passed
            without copying. None uses example for CLI compatibility; {} supplies
            no arguments. Callers may exclude reference fields independently of
            the original example received by prepare_tool_state. For env_grounded
            tasks, include init_state_str.
        world_model, policy, evaluator, bn_evaluator: Caller-built components
            forwarded to the registered search.
        result_saver, result_saver_unselected: Existing path writers; the second
            optionally saves unselected MCTS simulation paths.
        result_dir: Root for checkpoints, terminal files, and memory snapshots.
        augmentors: Optional ContextAugmentor instances wired via setup_augmentors().
        prepare_tool_state: Resource callback accepting the original example.
        resolve_answer: Resource callback accepting raw answer and ToolUseState;
            resolved answers replace saved answers only for newly saved terminals.
        memory_manager: Optional manager whose backend.save writes a snapshot.
        run_logger: Optional logger; defaults to this module's logger.
        search_kwargs: Extra registered-search constructor arguments, such as
            beam_selector, frontier_selector, or selection_events. Forwarded
            without copying their values or mutating the mapping; cannot replace
            explicit search inputs. Unsupported keys follow constructor validation.
            Skipped searches do not construct anything.
        search_context: Optional zero-argument context-manager factory wrapping
            only search.run(). Entered after construction and exited before final
            serialization; exceptions propagate normally. Not entered on resume.
            A nested benchmark.search tracing span records algorithm, query index,
            and repetition index. It uses the caller's tracing configuration and
            closes before measurement ends; it is a no-op when tracing is disabled.
        collect_search_facts: Optional callback receiving the newly saved result and
            returning JSON-serializable evidence (e.g. measured costs and selection
            events). Runs after saving, outside search_context, before completion.
            Execution status and query/repetition indices belong to this function.
            Not called on resume; correctness evaluation belongs to the caller.
        experiment_config: Full ExperimentConfig supplying the registered algorithm
            and create_search_config(). Also saved and checked before search/resume.
            Matching config.json files are unchanged; conflicts raise ValueError.

    Returns:
        The newly executed SearchResult, after saving; None when search is skipped.
        Answer resolution updates saved terminal answers only, not this object.
        Terminal-file existence remains the legacy resume rule, not an artifact
        validation guarantee. No correctness judging is performed here.
        Search and saving errors propagate; retained checkpoints alone do not
        certify successful execution.

    Bind keyword dependencies with functools.partial to obtain the three-argument
    callback accepted by runner.py::run_benchmark. Model/resource construction
    remains outside this function. Memory-save errors are logged; other uncaught
    errors propagate to the caller.

    Execution evidence and local ownership:
        search_facts_<query_idx>.json is atomically replaced after flushing/fsync.
        The separate OS-backed .lock file is never deleted or replaced. Its mere
        existence says nothing about ownership; acquisition is nonblocking.

        Lock acquired?  Saved status  Meaning / action
        --------------  ------------  -------------------------------------------
        No              Any           Active owner; filelock.Timeout propagates
        Yes             running       Interrupted; refuse to overwrite evidence
        Yes             failed        Recorded failure; refuse to overwrite
        Yes             complete      Require matching query/repetition and terminal file

        The lock covers preparation, search, saving, and collect_search_facts.
        New execution writes running, then complete after all required calls return.
        Exceptions write failed best-effort and propagate unchanged; interrupts may
        leave running, but release ownership. A dead process need not write an
        interrupted status. Malformed or incomplete existing evidence is preserved;
        legacy terminal files without a completion status are refused for manual review.
        Completion follows successful execution and saving, not a full read-back
        comparison. Serialization coverage is checked in tests; resume is separate.
        Optional memory-save warnings retain their existing nonfatal behavior.

        Ownership inference assumes original local directories and cooperating
        writers on an OS-lock-capable filesystem, not copied or remote archives.
        It does not prove progress or coordinate shared path writers across examples.
        Telemetry closure and later correctness evaluation do not alter this status.

    Saving and search-only timing:
        This function does not measure time. The timing column describes a timer
        around BaseTreeSearch.run(), not around this entire workflow. References
        use symbols/calls so they remain useful when source lines move.

        Operation                 Save?             Timed?  Code reference
        ------------------------  ----------------  ------  -----------------------------
        Internal checkpoints      When configured   Yes     search_base.py::save_checkpoint
        Internal full-tree save   When invoked      Yes     search_base.py::save_full_tree
        Final paths               Keep              No      run_example: result_saver.append_result
        Terminal nodes            Keep              No      save_terminal_nodes
        Unselected rollout paths  Optional writer   No      run_example: result_saver_unselected
        Memory snapshot           Optional manager  No      run_example: memory_manager.backend.save
        Answer resolution         Saved if changed  No      run_example: resolve_answer(raw, tool_state)
        Correctness evaluation    External caller   No      Not called by this module

        Internal writes remain part of run() wall time; final paths and terminal
        files are written after it returns. Optional writers/managers can be None.
    """
    run_logger = run_logger or logger
    experiment_config.save_config(result_dir, overwrite=False)
    if query_or_goals is None:
        if task_type == "env_grounded":
            query_or_goals = example.get("query_or_goals", example.get("question", ""))
        else:
            query_or_goals = example["question"]

    facts_path = Path(result_dir) / f"search_facts_{query_idx}.json"
    tn_file = Path(result_dir) / "terminal_nodes" / f"terminal_nodes_{query_idx}.json"
    lock = FileLock(facts_path.with_suffix(".lock"), timeout=0)
    if isinstance(lock, SoftFileLock):
        raise NotImplementedError("Search ownership requires an OS-backed file lock")
    with lock:
        # Read only after acquiring ownership; never overwrite an unfinished attempt.
        tn_existed_before = tn_file.exists()
        saved = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.exists() else {}
        if tn_existed_before and saved.get("status") is None:
            raise FileExistsError(
                f"Cannot verify completion for example {query_idx}, repetition {repetition_index}: "
                f"terminal output exists at {tn_file}, but {facts_path} is missing or has no status. "
                "Existing output was not overwritten and search was not resumed. "
                "Inspect the saved run manually; use a new result directory to rerun."
            )
        if facts_path.exists():
            status = saved.get("status")
            if status == "running":
                raise FileExistsError(f"Interrupted search (no active owner): {facts_path}")
            if status == "failed":
                raise FileExistsError(f"Failed search; retained evidence: {facts_path}")
            if status not in (None, "complete") or not tn_existed_before:
                raise FileExistsError(f"Unverified existing search evidence: {facts_path}")
            saved_identity = (saved.get("query_idx"), saved.get("repetition_index"))
            if saved_identity != (query_idx, repetition_index):
                raise FileExistsError(
                    f"Search identity mismatch in {facts_path}: saved query/repetition "
                    f"{saved_identity}, requested {(query_idx, repetition_index)}. "
                    "Existing output was not overwritten. Use a separate result directory "
                    "for a different example or repetition."
                )

        facts = {"query_idx": query_idx, "repetition_index": repetition_index, "status": "running"}
        algo_output = None
        try:
            if not tn_existed_before:
                _write_search_facts(facts_path, facts)
            if prepare_tool_state is not None:
                prepare_tool_state(example)

            if tn_existed_before:
                run_logger.info(f"[{query_idx}] terminal_nodes already exists, skipping")
            else:
                # Load agents only when executing, not when importing benchmark registration.
                from lits.agents import AgentRegistry

                # Construct separately so caller accounting excludes constructor work.
                search_class = AgentRegistry.get_search_class(experiment_config.search_algorithm)
                constructor_kwargs = {
                    "config": experiment_config.create_search_config(),
                    "world_model": world_model,
                    "policy": policy,
                    "reward_model": evaluator,
                    "bn_evaluator": bn_evaluator,
                    "init_state_kwargs": example if init_state_kwargs is None else init_state_kwargs,
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
                    with tracing.span(
                        "benchmark.search", algorithm=experiment_config.search_algorithm,
                        query_idx=query_idx, repetition_index=repetition_index,
                    ):
                        algo_output = search.run(query_or_goals, query_idx)

                # Save unselected paths if available (MCTS-specific attribute)
                if result_saver_unselected and hasattr(algo_output, 'unselected_terminal_paths_during_simulate'):
                    result_saver_unselected.append_result(algo_output.unselected_terminal_paths_during_simulate)

                # Convert to paths for saving — all results support to_paths()
                paths = algo_output.to_paths()
                result_saver.append_result(paths)

                # Save terminal nodes separately for post-search evaluation
                save_terminal_nodes(algo_output, query_or_goals, query_idx, result_dir, run_logger)

                if resolve_answer is not None:
                    _resolve_terminal_answers(tn_file, resolve_answer, query_idx, run_logger)

            # Save memory backend per-example (facts persist for inspection/debugging)
            if memory_manager is not None and hasattr(memory_manager, 'backend'):
                memory_dir = os.path.join(result_dir, "memory", str(query_idx))
                try:
                    memory_manager.backend.save(memory_dir)
                except Exception as e:
                    run_logger.warning(f"[{query_idx}] Failed to save memory: {e}")
            if not tn_existed_before:
                if collect_search_facts is not None:
                    supplied = collect_search_facts(algo_output)
                    if not isinstance(supplied, dict):
                        raise ValueError("collect_search_facts must return a dictionary")
                    for key in facts.keys() & supplied.keys():
                        if supplied[key] != facts[key]:
                            raise ValueError(f"Search facts conflict with execution: {key}")
                    # Reject bad caller data before it can also break failed-status saving.
                    json.dumps(supplied, allow_nan=False)
                    facts.update(supplied)
                facts["status"] = "complete"
                _write_search_facts(facts_path, facts)
        except Exception as error:
            if not tn_existed_before:
                facts.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
                try:
                    _write_search_facts(facts_path, facts)
                except Exception:
                    run_logger.warning(f"[{query_idx}] Could not save failure status", exc_info=True)
            raise
        return algo_output


def _resolve_terminal_answers(
    tn_file: Path, resolve_answer: Callable, query_idx: int, run_logger,
) -> None:
    """Resolve saved answer references using each terminal's complete trajectory.

    Used for KGQA-style tasks: an answer such as '#3' references a trajectory
    variable rather than concrete answer values. The benchmark resolver rebuilds
    that variable from ToolUseState and queries the knowledge graph for entity
    results. This resolves references; it does not judge correctness.

    Args:
        tn_file: Terminal-node JSON file; a missing file is a no-op.
        resolve_answer: Callback accepting (raw_answer, ToolUseState) and returning
            the resolved answer. Only truthy last-step answers are passed to it.
        query_idx: Dataset index used in the log message.
        run_logger: Logger for reporting a changed file.

    流程：
        读取 terminal_nodes 文件
            ↓
        遍历每个终点，取出完整轨迹的 steps
            ↓
        读取最后一步的 answer
            ↓
        resolve_answer(原始答案, 完整轨迹对象)
            ↓
        答案发生变化？
            ├─ 否：不修改
            └─ 是：更新轨迹最后一步及节点 step 中已有的 answer
                        ↓
        至少一个答案变化 → 写回整个 JSON 文件

    Only this file changes, not in-memory search results, paths, or checkpoints.
    Unchanged files are not rewritten; read, resolver, and write errors propagate.
    The caller skips this operation on resumed searches.
    """
    if not tn_file.exists():
        return
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


def _write_search_facts(path: Path, facts: dict) -> None:
    """Flush JSON to a temporary file, then atomically replace the facts file.

    The caller holds the separate, stable OS lock. Failed serialization or writes
    leave the previous JSON intact and remove the temporary file when possible.
    """
    payload = json.dumps(facts, sort_keys=True) + "\n"
    temporary = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    try:
        with temporary as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary.name).replace(path)
    finally:
        Path(temporary.name).unlink(missing_ok=True)


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
