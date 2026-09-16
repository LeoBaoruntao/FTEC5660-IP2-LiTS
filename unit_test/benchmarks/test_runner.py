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


if __name__ == "__main__":
    check_order()
    check_loaded_dataset()
    check_slicing_and_progress()
    check_resume_and_errors()
    check_propagation()
    check_invalid_settings()
