"""Check ``runner.py::run_benchmark`` using a real registry and deterministic callbacks.

Run: ``PYTHONBREAKPOINT=0 python -m unit_test.benchmarks.test_runner``.
No models or search implementations are mocked. These checks exercise orchestration,
not inference; ``test_search_loop.py::check_dataset_loop`` covers actual CLI execution.
"""

from contextlib import redirect_stderr
from io import StringIO

from lits.benchmarks import register_dataset, run_benchmark


EXAMPLES = [{"value": index} for index in range(4)]
LOADS = []


@register_dataset("benchmark_loop_check")
def load_cohort(split="test"):
    """Record loader arguments and return original objects without task-specific fields."""
    LOADS.append(split)
    return EXAMPLES


def check_order():
    """Check original indices, loader kwargs, slicing, and example-major repetitions."""
    cases = (
        ({"offset": 1, "limit": 2}, [(1, 0), (2, 0)]),
        ({"eval_idx": [3, 1]}, [(3, 0), (1, 0)]),
        ({"eval_idx": [3, 1, 2], "offset": 1, "limit": 1}, [(1, 0)]),
        ({"limit": 0}, []),
        ({"offset": 8}, []),
        ({"eval_idx": []}, [(0, 0), (1, 0), (2, 0), (3, 0)]),
        ({"eval_idx": [3, 1], "repetitions": 2}, [(3, 0), (3, 1), (1, 0), (1, 1)]),
    )
    for settings, expected in cases:
        observed = []

        def execute(example, idx, rep):
            """Check identity and record each dispatched pair."""
            if example is not EXAMPLES[idx]:
                raise RuntimeError("Example identity no longer matches the original index")
            observed.append((idx, rep))

        loads_before = len(LOADS)
        result = run_benchmark(
            "benchmark_loop_check", execute, dataset_kwargs={"split": "local"}, **settings,
        )
        if observed != expected or result is not None or LOADS[loads_before:] != ["local"]:
            raise RuntimeError(f"Incorrect execution: {settings}, {observed}, {LOADS}")
    print(f"PASS ordering/loading: {len(cases)} cases")


def check_resume_and_errors():
    """Keep successes and errors caller-owned; retry only the unsuccessful pair on resume."""
    completed, attempts, errors = {}, [], []
    failure = RuntimeError("Execution failed")

    def execute(example, idx, rep):
        """Record an attempt; fail once before the caller writes its completed result."""
        pair = (idx, rep)
        attempts.append(pair)
        if pair == (1, 0) and attempts.count(pair) == 1:
            raise failure
        completed[pair] = {"correct": False}

    def is_complete(example, idx, rep):
        """An incorrect answer still counts as completed; no correctness-based skipping."""
        return (idx, rep) in completed

    def record_error(example, idx, rep, error):
        """Retain the original exception without adding a completed result."""
        if example is not EXAMPLES[idx] or error is not failure:
            raise RuntimeError("Error callback lost the failing example or exception")
        errors.append((idx, rep))

    settings = dict(eval_idx=[1, 2], repetitions=2, is_complete=is_complete, on_error=record_error)
    run_benchmark("benchmark_loop_check", execute, **settings)
    if attempts != [(1, 0), (1, 1), (2, 0), (2, 1)] or errors != [(1, 0)] or (1, 0) in completed:
        raise RuntimeError("Handled failure altered completion or continuation behavior")
    run_benchmark("benchmark_loop_check", execute, **settings)
    run_benchmark("benchmark_loop_check", execute, **settings)
    if attempts != [(1, 0), (1, 1), (2, 0), (2, 1), (1, 0)] or len(completed) != 4:
        raise RuntimeError("Resume repeated a completed pair or missed the failed pair")
    print("PASS handled errors and resume: failed pair retried, incorrect results skipped")
    breakpoint()  # inspect: attempts, completed, errors


def check_loaded_dataset():
    """Reuse caller-loaded data without registry calls, including an empty cohort."""
    loads_before = len(LOADS)
    for dataset, expected in ((EXAMPLES, [(1, 0)]), ([], [])):
        observed = []
        run_benchmark(
            "benchmark_loop_check", lambda example, idx, rep: observed.append((idx, rep)),
            dataset=dataset, offset=1, limit=1, show_progress=True,
        )
        if observed != expected or len(LOADS) != loads_before:
            raise RuntimeError("Loaded dataset was reloaded or sliced incorrectly")
    try:
        run_benchmark(
            "benchmark_loop_check", lambda *args: None,
            dataset=[], dataset_kwargs={"split": "test"},
        )
    except ValueError:
        pass
    else:
        raise RuntimeError("Loader arguments should not be ignored with preloaded data")
    print("PASS preloaded data and progress: 3 cases")


def check_slicing_and_progress():
    """Check ``runner.py::_slice_dataset`` aliases and default/disabled progress output."""
    from lits.benchmarks.runner import _slice_dataset
    from lits.eval import _slice_dataset as eval_slice
    from lits.eval.results import _slice_dataset as results_slice

    if eval_slice is not _slice_dataset or results_slice is not _slice_dataset:
        raise RuntimeError("Evaluation imports must reuse the benchmark slicing function")
    cases = ((0, None, [3, 1, 2]), (1, None, [1, 2]), (1, 1, [1]),
             (0, 0, []), (5, None, []), (-1, None, [2]))
    for offset, limit, expected in cases:
        if _slice_dataset([3, 1, 2], offset, limit) != expected:
            raise RuntimeError(f"Changed slicing behavior: {offset}, {limit}")
    for settings, visible in (({}, True), ({"show_progress": False}, False)):
        output, observed = StringIO(), []
        with redirect_stderr(output):
            run_benchmark(
                "benchmark_loop_check", lambda example, idx, rep: observed.append((idx, rep)),
                dataset=EXAMPLES, limit=1, **settings,
            )
        if observed != [(0, 0)] or bool(output.getvalue()) != visible:
            raise RuntimeError(f"Incorrect progress behavior: {settings}, {output.getvalue()!r}")
    print("PASS shared slicing aliases, 6 slicing cases, and 2 progress modes")


def check_propagation():
    """Propagate execution, completion-check, handler, loader errors, and interrupts."""
    failure = RuntimeError("Deliberate callback failure")
    interrupt = KeyboardInterrupt()
    events = []

    def fail(*args):
        """Record a single invocation and raise the original failure."""
        events.append("fail")
        raise failure

    def succeed(*args):
        """Record execution without side effects outside this check."""
        events.append("success")

    def handle(*args):
        """Record a handled error so unexpected interception is visible."""
        events.append("handled")

    def stop(*args):
        """Exercise a user interruption, which must not be handled as a run failure."""
        raise interrupt

    cases = (
        (fail, {}, failure, ["fail"]),
        (succeed, {"is_complete": fail, "on_error": handle}, failure, ["fail"]),
        (fail, {"on_error": fail}, failure, ["fail", "fail"]),
        (stop, {"on_error": handle}, interrupt, []),
    )
    for execute, settings, expected_error, expected_events in cases:
        events.clear()
        try:
            run_benchmark("benchmark_loop_check", execute, **settings)
        except BaseException as error:
            if error is not expected_error or events != expected_events:
                raise RuntimeError(f"Incorrect exception propagation: {events}") from error
        else:
            raise RuntimeError("An exception was swallowed")
    events.clear()
    try:
        run_benchmark("benchmark_loop_check", succeed, dataset_kwargs={"unknown_argument": True}, on_error=handle)
    except TypeError:
        if events:
            raise RuntimeError("Dataset-loading errors must not invoke per-example callbacks")
    else:
        raise RuntimeError("Dataset loader should reject unsupported arguments")
    print("PASS exception propagation: 5 cases")


def check_invalid_settings():
    """Reject ambiguous or invalid run identities before any execution callback."""
    cases = (
        {"offset": -1}, {"offset": True}, {"limit": -1}, {"limit": 1.5},
        {"repetitions": 0}, {"repetitions": -1}, {"repetitions": True},
        {"eval_idx": [0, -1]}, {"eval_idx": [0, 4]}, {"eval_idx": [0, 0]},
        {"eval_idx": [False]}, {"eval_idx": [1.5]},
    )
    for settings in cases:
        executed = []
        try:
            run_benchmark("benchmark_loop_check", lambda *args: executed.append(args), **settings)
        except ValueError:
            if executed:
                raise RuntimeError(f"Invalid settings reached execution: {settings}")
        else:
            raise RuntimeError(f"Accepted invalid settings: {settings}")
    print(f"PASS input validation: {len(cases)} cases")


def check_search_example():
    """Exercise search.py::run_example with real zero-iteration MCTS.

    Bind components exactly as a benchmark caller would. Verify query extraction,
    returned results, saved terminal metadata, preparation on every invocation, and unchanged files
    on resume. No inference is requested; nonempty answer resolution and memory
    persistence are not exercised by these zero-iteration searches.
    Explicit query/init inputs also support arbitrary example schemas and keep
    references outside world-model initialization while preparation sees the original.
    """
    from functools import partial
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from lits import get_lm
    from lits.agents import AgentRegistry, MCTSSearch
    from lits.config import ExperimentConfig
    from lits.benchmarks import run_example
    from lits.components.policy.concat import ConcatPolicy
    from lits.components.reward.generative import GenerativePRM
    from lits.components.transition.concat import ConcatTransition
    from lits.eval import TreeToJsonl

    model = get_lm("openai/local-instruct", base_url="http://localhost:8080/v1", api_key="no-key")
    examples = [
        ("language_grounded", {"question": "What is 1 + 1?"}, "What is 1 + 1?", {}),
        ("env_grounded", {"query_or_goals": "Goal text", "question": "Fallback"}, "Goal text", {}),
        ("env_grounded", {"question": "Fallback"}, "Fallback", {}),
        ("env_grounded", {"prompt": "Custom goal", "reference": "private"}, "Custom goal",
         {"query_or_goals": "Custom goal", "init_state_kwargs": {"init_state_str": "Initial state"}}),
        ("language_grounded", {"prompt": "Custom question", "reference": "private"}, "Custom question",
         {"query_or_goals": "Custom question", "init_state_kwargs": {}}),
    ]
    prepared = []
    initialized = []
    world_model = ConcatTransition(base_model=model, terminate_constraints=["step_signal"])
    original_init = world_model.init_state

    def record_init(**kwargs):
        """Observe actual initialization inputs, then run the real transition initializer."""
        initialized.append(kwargs)
        return original_init(**kwargs)

    world_model.init_state = record_init

    def prepare(value):
        """Record original examples received before new or resumed searches."""
        prepared.append(value)

    config = ExperimentConfig(
        dataset="math500", policy_model_name="openai/local-instruct",
        eval_model_name="openai/local-instruct", search_algorithm="mcts",
        search_args={"n_iters": 0, "add_continuation": False},
        import_modules=["lits.benchmarks"], dataset_kwargs={"subset": "test"},
    )
    derived = config.create_search_config()
    if derived.import_modules != config.import_modules or derived.dataset_kwargs != config.dataset_kwargs:
        raise RuntimeError("Derived search configuration lost reconstruction metadata")
    kwargs = dict(
        experiment_config=config,
        world_model=world_model,
        policy=ConcatPolicy(base_model=model, task_name="math500"),
        evaluator=GenerativePRM(base_model=model, task_name="math500"),
        prepare_tool_state=prepare,
    )
    with TemporaryDirectory(prefix="lits-run-example-") as directory:
        for index, (task_type, example, expected_query, inputs) in enumerate(examples):
            result_dir = Path(directory) / str(index)
            result_dir.mkdir()
            execute = partial(
                run_example, task_type=task_type, result_dir=str(result_dir),
                result_saver=TreeToJsonl(run_id="", root_dir=str(result_dir), override=False),
                **kwargs, **inputs,
            )
            result = execute(example, index, 0)
            if result is None or result.root.action != expected_query or result.terminal_nodes_collected:
                raise RuntimeError("New search did not return its zero-iteration result")
            if initialized[-1] != inputs.get("init_state_kwargs", example):
                raise RuntimeError("World-model initialization did not use the selected inputs")
            terminal_file = result_dir / "terminal_nodes" / f"terminal_nodes_{index}.json"
            saved = json.loads(terminal_file.read_text())
            if saved != {"query": expected_query, "query_idx": index, "terminal_nodes": []}:
                raise RuntimeError(f"Unexpected saved output: {saved}")
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in result_dir.rglob("*") if p.is_file()}
            if execute(example, index, 0) is not None:
                raise RuntimeError("Skipped search must return None, not a new result")
            after = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                     for p in result_dir.rglob("*") if p.is_file()}
            if before != after:
                raise RuntimeError("Resumed example rewrote saved output")
            if len(prepared) != 2 * (index + 1) or any(p is not example for p in prepared[-2:]):
                raise RuntimeError("Preparation must receive the original example once per call")
            if len(initialized) != index + 1:
                raise RuntimeError("Resume unexpectedly initialized the world model")
        breakpoint()  # inspect: saved, prepared, result_dir
    if AgentRegistry.get_search_class("mcts") is not MCTSSearch or config.search_args["n_iters"] != 0:
        raise RuntimeError("Registry lookup or caller configuration changed")
    for lookup in (AgentRegistry.get_search, AgentRegistry.get_search_class):
        try:
            lookup("unregistered_search_check")
        except KeyError:
            pass
        else:
            raise RuntimeError("Unknown search name was accepted")
    print("PASS run_example: 5 query/init cases, saved outputs, preparation, unchanged resume")


def check_config_saving():
    """Check config.py::ExperimentConfig.save_config overwrite and resume behavior."""
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from lits.config import ExperimentConfig

    config = ExperimentConfig(
        dataset="math500", policy_model_name="openai/local-instruct",
        eval_model_name="openai/local-instruct", dataset_kwargs={"subset": (1, 2)},
    )
    with TemporaryDirectory() as directory:
        target = Path(directory) / "nested"
        config.save_config(target, overwrite=False)
        path = target / "config.json"
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        config.save_config(target, overwrite=False)
        if (path.read_bytes(), path.stat().st_mtime_ns) != before:
            raise RuntimeError("Matching JSON-normalized config was rewritten")
        config.dataset = "gsm8k"
        try:
            config.save_config(target, overwrite=False)
        except ValueError:
            pass
        else:
            raise RuntimeError("Conflicting config was accepted")
        if (path.read_bytes(), path.stat().st_mtime_ns) != before:
            raise RuntimeError("Conflict handling changed the saved config")
        config.save_config(target)
        if json.loads(path.read_text()) != json.loads(json.dumps(config.to_dict())):
            raise RuntimeError("Default overwrite did not save the new config")
        breakpoint()  # inspect: config, path, before
    print("PASS config saving: directory creation, normalization, resume, conflict, overwrite")


if __name__ == "__main__":
    check_config_saving()
    check_order()
    check_loaded_dataset()
    check_slicing_and_progress()
    check_resume_and_errors()
    check_propagation()
    check_invalid_settings()
    check_search_example()
