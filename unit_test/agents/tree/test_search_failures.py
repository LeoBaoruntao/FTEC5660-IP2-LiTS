"""Check search_base.py::BaseTreeSearch.run error propagation and cleanup.

Run with PYTHONBREAKPOINT=0 python -m unit_test.agents.tree.test_search_failures.
Real BFS/MCTS selection runs before an injected exception, without inference.
The delegating hook saves a real checkpoint; benchmarks/search.py::run_example
must preserve it without writing final outputs. CUDA OOM is injected, not allocated.
"""

from cProfile import Profile
from contextlib import contextmanager
import json
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock, Timeout

from lits import get_lm
from lits.agents.registry import AgentRegistry
from lits.agents.tree import search_base
from lits.benchmarks import run_example
from lits.components.factory import create_components
from lits.config import ExperimentConfig
from lits.eval import TreeToJsonl


def check_search_failures(model):
    """Verify original exceptions, one teardown, retained checkpoints, and no final files."""
    error_types = [ValueError, RuntimeError, KeyboardInterrupt]
    error_types.extend(search_base._optional_cuda_errors())
    failures = []
    with TemporaryDirectory(prefix="lits-search-failures-") as directory:
        for algorithm, hook in (("bfs", "_do_select_beam"), ("mcts", "_do_select_path")):
            config = ExperimentConfig(
                dataset="math500", search_algorithm=algorithm, search_framework="rest",
                policy_model_name=model.model_name, eval_model_name=model.model_name,
                search_args={"max_steps": 1, "add_continuation": False,
                             "terminate_constraints": ["step_signal"]},
            )
            world_model, policy, reward_model = create_components(
                config=config, task_type="language_grounded", task_name="math500",
                base_model=model, eval_base_model=model, terminal_model=None, tool_use_spec=None,
            )
            search_class = AgentRegistry.get_search_class(algorithm)
            original_select = getattr(search_class, hook)
            for error_type in error_types:
                case = f"{algorithm}-{error_type.__name__}"
                run_dir = Path(directory) / case
                run_dir.mkdir()
                saver = TreeToJsonl(run_id="", root_dir=str(run_dir), override=False)
                expected_error = error_type("Injected after real selection")
                snapshots = []
                profiler, caught = Profile(), None

                def fail_after_selection(search, query, query_idx, *args):
                    """Delegate real selection, save root evidence, then inject the failure."""
                    original_select(search, query, query_idx, *args)
                    search.save_checkpoint(query_idx, 0, search.root.to_dict())
                    snapshots.append((run_dir / "checkpoints/0_0.json").read_bytes())
                    raise expected_error

                setattr(search_class, hook, fail_after_selection)
                try:
                    profiler.runcall(
                        run_example, {"question": "What is 1 + 1?"}, 0, 0,
                        task_type="language_grounded", experiment_config=config,
                        world_model=world_model, policy=policy, evaluator=reward_model,
                        result_dir=str(run_dir), result_saver=saver,
                    )
                except BaseException as error:
                    caught = error
                finally:
                    setattr(search_class, hook, original_select)

                calls = {entry.code: entry.callcount for entry in profiler.getstats()}
                facts_path = run_dir / "search_facts_0.json"
                facts = json.loads(facts_path.read_text())
                checks = {
                    "same_exception": caught is expected_error,
                    "one_cleanup": calls.get(search_base.BaseTreeSearch._teardown.__code__, 0) == 1,
                    "checkpoint_retained": snapshots == [(run_dir / "checkpoints/0_0.json").read_bytes()],
                    "no_terminal_file": not (run_dir / "terminal_nodes/terminal_nodes_0.json").exists(),
                    "no_final_paths": Path(saver.filepath).stat().st_size == 0,
                    "execution_status": facts["status"] == (
                        "running" if error_type is KeyboardInterrupt else "failed"
                    ),
                }
                with FileLock(facts_path.with_suffix(".lock"), timeout=0):
                    checks["ownership_released"] = True
                if error_type is not KeyboardInterrupt:
                    checks["original_error_recorded"] = facts["error"] == {
                        "type": error_type.__name__, "message": str(expected_error),
                    }
                if error_type in search_base._optional_cuda_errors():
                    checks["cuda_cleanup"] = calls.get(search_base.torch.cuda.empty_cache.__code__, 0) == 1
                print(f"{'PASS' if all(checks.values()) else 'FAIL'} {case}: {checks}")
                breakpoint()  # inspect: case, caught, checks, run_dir
                if not all(checks.values()):
                    failures.append(case)
    if failures:
        raise RuntimeError("Search failure checks failed: " + ", ".join(failures))


def _zero_iteration_execution(model, directory):
    """Build actual factory components and executor arguments without requesting inference."""
    config = ExperimentConfig(
        dataset="math500", search_algorithm="mcts", search_framework="rest",
        policy_model_name=model.model_name, eval_model_name=model.model_name,
        search_args={"n_iters": 0, "add_continuation": False},
    )
    world_model, policy, reward_model = create_components(
        config=config, task_type="language_grounded", task_name="math500",
        base_model=model, eval_base_model=model, terminal_model=None, tool_use_spec=None,
    )
    return dict(
        task_type="language_grounded", experiment_config=config,
        world_model=world_model, policy=policy, evaluator=reward_model,
        result_dir=str(directory),
        result_saver=TreeToJsonl(run_id="", root_dir=str(directory), override=False),
    )


def check_facts_callback_failures(model):
    """Check search.py::run_example preserves callback errors, even if status writing fails."""
    for block_status_write in (False, True):
        with TemporaryDirectory(prefix="lits-facts-failure-") as directory:
            root = Path(directory)
            kwargs = _zero_iteration_execution(model, root)
            facts_path = root / "search_facts_0.json"
            expected = RuntimeError("Injected while collecting saved-result facts")

            def fail_after_saving(result):
                """Observe real saved output, then optionally block the JSON replacement."""
                if result is None or not (root / "terminal_nodes/terminal_nodes_0.json").exists():
                    raise RuntimeError("Facts callback preceded final saving")
                if json.loads(facts_path.read_text())["status"] != "running":
                    raise RuntimeError("Completion was recorded before fact collection")
                if block_status_write:
                    facts_path.rename(root / "retained_running.json")
                    facts_path.mkdir()  # force a real filesystem error, without replacing the writer
                raise expected

            try:
                run_example({"question": "What is 1 + 1?"}, 0, 0,
                            **kwargs, collect_search_facts=fail_after_saving)
            except RuntimeError as error:
                if error is not expected:
                    raise
            else:
                raise RuntimeError("Facts collection failure did not propagate")
            if block_status_write:
                facts_path = root / "retained_running.json"
            facts = json.loads(facts_path.read_text())
            if facts["status"] != ("running" if block_status_write else "failed"):
                raise RuntimeError(f"Unexpected failed-save evidence: {facts}")
            if not block_status_write:
                before = facts_path.read_bytes()
                try:
                    run_example({"question": "What is 1 + 1?"}, 0, 0, **kwargs)
                except FileExistsError as error:
                    if "Failed search" not in str(error):
                        raise
                else:
                    raise RuntimeError("Terminal-file presence hid the failed fact collection")
                if facts_path.read_bytes() != before:
                    raise RuntimeError("Retry overwrote failed evidence")
            if list(root.glob("tmp*")):
                raise RuntimeError("Failed atomic replacement left a temporary file")
            with FileLock(root / "search_facts_0.lock", timeout=0):
                pass
            print(f"PASS facts callback failure: original error retained; blocked writer={block_status_write}")
            breakpoint()  # inspect: facts, root, expected


def _hold_execution(directory, phase, connection):
    """Run real zero-iteration search in a child, pausing at the selected lifecycle boundary."""
    model = get_lm("openai/local-instruct", base_url="http://localhost:8080/v1", api_key="no-key")
    kwargs = _zero_iteration_execution(model, directory)

    def pause():
        """Notify the parent only after running status exists and ownership is held."""
        connection.send("ready")
        connection.recv()

    @contextmanager
    def search_context():
        """Pause inside the actual search accounting context, before the run call."""
        if phase == "search":
            pause()
        yield

    def collect_facts(result):
        """Pause after real search and terminal saving, before committing completion."""
        if phase == "saving":
            pause()
        return {"terminal_count": len(result.terminal_nodes_collected)}

    try:
        run_example({"question": "What is 1 + 1?"}, 0, 0, **kwargs,
                    search_context=search_context, collect_search_facts=collect_facts)
    finally:
        connection.close()


def check_process_ownership(model):
    """Check search.py::run_example active ownership, completion, and hard-kill interruption.

    Spawn actual executor processes; no model calls or simulated lock implementations.
    Kill both before search execution and after terminal saving, so file presence
    alone cannot hide an interrupted save. The stable lock inode must survive.
    """
    context = get_context("spawn")
    for phase, kill in (("search", True), ("saving", True), ("saving", False)):
        with TemporaryDirectory(prefix="lits-search-owner-") as directory:
            root = Path(directory)
            kwargs = _zero_iteration_execution(model, root)
            parent, child = context.Pipe()
            worker = context.Process(target=_hold_execution, args=(directory, phase, child))
            worker.start()
            child.close()
            try:
                if not parent.poll(60) or parent.recv() != "ready":
                    raise RuntimeError("Executor child did not reach the requested boundary")
                facts_path = root / "search_facts_0.json"
                before = facts_path.read_bytes()
                if json.loads(before)["status"] != "running":
                    raise RuntimeError("Active execution did not persist running")
                lock_path = facts_path.with_suffix(".lock")
                inode = lock_path.stat().st_ino
                try:
                    run_example({"question": "What is 1 + 1?"}, 0, 0, **kwargs)
                except Timeout:
                    pass
                else:
                    raise RuntimeError("Active executor ownership was ignored")
                if facts_path.read_bytes() != before:
                    raise RuntimeError("Contending caller rewrote active evidence")
                if kill:
                    worker.kill()
                else:
                    parent.send("continue")
                worker.join(30)
                if worker.is_alive() or (not kill and worker.exitcode != 0):
                    raise RuntimeError(f"Executor child did not exit as expected: {worker.exitcode}")
                if kill:
                    try:
                        run_example({"question": "What is 1 + 1?"}, 0, 0, **kwargs)
                    except FileExistsError as error:
                        if "Interrupted search (no active owner)" not in str(error):
                            raise
                    else:
                        raise RuntimeError("Interrupted execution was treated as complete")
                    if facts_path.read_bytes() != before:
                        raise RuntimeError("Interrupted evidence was overwritten")
                else:
                    before = facts_path.read_bytes()
                    completed = json.loads(before)
                    if completed["status"] != "complete" or completed["terminal_count"] != 0:
                        raise RuntimeError("Completion omitted callback facts")
                    if run_example({"question": "What is 1 + 1?"}, 0, 0, **kwargs) is not None:
                        raise RuntimeError("Completed search was rerun")
                    if facts_path.read_bytes() != before:
                        raise RuntimeError("Resume rewrote completed evidence")
                if lock_path.stat().st_ino != inode:
                    raise RuntimeError("Executor replaced or deleted its coordination lock")
                print(f"PASS subprocess ownership: phase={phase}, killed={kill}, lock preserved")
                breakpoint()  # inspect: root, before, worker.exitcode
            finally:
                if worker.is_alive():
                    worker.kill()
                worker.join()
                parent.close()


if __name__ == "__main__":
    model = get_lm(
        "openai/local-instruct", base_url="http://localhost:8080/v1", api_key="no-key",
    )
    check_search_failures(model)
    check_facts_callback_failures(model)
    check_process_ownership(model)
