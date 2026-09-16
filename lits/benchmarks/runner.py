"""Sequential execution over registered datasets, with caller-owned search and persistence."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from lits.benchmarks.registry import load_dataset


def _slice_dataset(dataset: Sequence[Any], offset: int, limit: int | None) -> Sequence[Any]:
    """Slice examples or indices by offset/limit; None keeps every remaining entry."""
    if limit is None:
        return dataset[offset:]
    return dataset[offset : offset + limit]


def run_benchmark(
    dataset_name: str,
    run_example: Callable[[Any, int, int], Any],
    *,
    dataset_kwargs: Mapping[str, Any] | None = None,
    dataset: Sequence[Any] | None = None,
    eval_idx: Sequence[int] | None = None,
    offset: int = 0,
    limit: int | None = None,
    repetitions: int = 1,
    is_complete: Callable[[Any, int, int], bool] | None = None,
    on_error: Callable[[Any, int, int, Exception], None] | None = None,
    show_progress: bool = True,
) -> None:
    """Execute selected example/repetition pairs from a registered or preloaded dataset.

    Args:
        dataset_name: Registered dataset name; import its registration module first.
        run_example: Called as ``run_example(example, query_idx, repetition_index)``.
            The example is the original loaded object, not a copy. The caller owns
            resource preparation, model calls, seeds, evaluation, and result saving.
        dataset_kwargs: Keyword arguments forwarded to the registered loader.
        dataset: Optional already-loaded full dataset, preserving original indices.
            Reuses caller-side loading without invoking the registry again. Do not
            combine with nonempty dataset_kwargs, which would otherwise be ignored.
        eval_idx: Unique, nonnegative original dataset indices, in execution order.
            None or an empty sequence selects all indices, matching the search CLI.
        offset: Number of entries to skip in the selected index sequence.
        limit: Maximum number of examples after offset; None means no limit.
        repetitions: Positive number of runs per example. Zero-based repetitions
            run consecutively for each example before moving to the next example.
        is_complete: Optional ``is_complete(example, query_idx, repetition_index)``.
            A true result skips run_example, including its resource preparation.
        on_error: Optional ``on_error(example, query_idx, repetition_index, error)``.
            Handles an execution exception; returning normally continues to the next
            pair. Without this callback, execution errors propagate immediately.
        show_progress: Display a tqdm progress bar by default; False disables it.

    Returns:
        None. Callback results are not accumulated or written by this loop.
        An incorrect final answer is an ordinary result, not an execution error.

    Example:
        pairs = []
        run_benchmark(
            "math500", lambda example, idx, rep: pairs.append((idx, rep)),
            eval_idx=[3, 1, 2], offset=1, limit=1, repetitions=2,
        )
        # pairs == [(1, 0), (1, 1)]

    Invalid indices/settings fail before any per-example callback. Dataset-loading,
    completion-check, and error-handler exceptions propagate; only run_example is
    covered by on_error. Interrupts also propagate. No completion state is inferred
    from callback returns or errors: resume requires caller-provided saved state.
    """
    for name, value in (("offset", offset), ("limit", limit), ("repetitions", repetitions)):
        if name == "limit" and value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if repetitions == 0:
        raise ValueError("repetitions must be positive")

    if dataset is None:
        dataset = load_dataset(dataset_name, **(dataset_kwargs or {}))
    elif dataset_kwargs:
        raise ValueError("dataset_kwargs cannot be combined with an already-loaded dataset")
    indices = list(eval_idx) if eval_idx is not None else []
    if not indices:
        indices = list(range(len(dataset)))
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("eval_idx must contain nonnegative integers")
        if index >= len(dataset):
            raise ValueError(f"Dataset index {index} is out of range for {len(dataset)} examples")
    if len(set(indices)) != len(indices):
        raise ValueError("eval_idx must not contain duplicate indices")
    indices = _slice_dataset(indices, offset, limit)

    if show_progress:
        from tqdm import tqdm
        indices = tqdm(indices)

    for query_idx in indices:
        example = dataset[query_idx]
        for repetition_index in range(repetitions):
            if is_complete is not None and is_complete(example, query_idx, repetition_index):
                continue
            try:
                run_example(example, query_idx, repetition_index)
            except Exception as error:
                if on_error is None:
                    raise
                on_error(example, query_idx, repetition_index, error)
