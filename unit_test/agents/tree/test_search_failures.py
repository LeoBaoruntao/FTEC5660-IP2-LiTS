"""Check search_base.py::BaseTreeSearch.run error propagation and cleanup.

Run with PYTHONBREAKPOINT=0 python -m unit_test.agents.tree.test_search_failures.
Real BFS/MCTS selection runs before an injected exception, without inference.
The delegating hook saves a real checkpoint; benchmarks/search.py::run_example
must preserve it without writing final outputs. CUDA OOM is injected, not allocated.
"""

from cProfile import Profile
from pathlib import Path
from tempfile import TemporaryDirectory

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
                checks = {
                    "same_exception": caught is expected_error,
                    "one_cleanup": calls.get(search_base.BaseTreeSearch._teardown.__code__, 0) == 1,
                    "checkpoint_retained": snapshots == [(run_dir / "checkpoints/0_0.json").read_bytes()],
                    "no_terminal_file": not (run_dir / "terminal_nodes/terminal_nodes_0.json").exists(),
                    "no_final_paths": Path(saver.filepath).stat().st_size == 0,
                }
                if error_type in search_base._optional_cuda_errors():
                    checks["cuda_cleanup"] = calls.get(search_base.torch.cuda.empty_cache.__code__, 0) == 1
                print(f"{'PASS' if all(checks.values()) else 'FAIL'} {case}: {checks}")
                breakpoint()  # inspect: case, caught, checks, run_dir
                if not all(checks.values()):
                    failures.append(case)
    if failures:
        raise RuntimeError("Search failure checks failed: " + ", ".join(failures))


if __name__ == "__main__":
    check_search_failures(get_lm(
        "openai/local-instruct", base_url="http://localhost:8080/v1", api_key="no-key",
    ))
