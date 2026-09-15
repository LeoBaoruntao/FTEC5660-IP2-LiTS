"""Shared summaries of already-evaluated predictions, without task or model dependencies."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def aggregate_results(
    records: Iterable[Mapping[str, Any]],
    *,
    pass_at_n: bool = False,
    mean_fields: Sequence[str] = (),
) -> dict[str, Any]:
    """Summarize prediction correctness and optional numeric measurements.

    Args:
        records: Evaluated rows with ``idx`` and boolean ``correct``. Optional
            ``score`` is binary correctness, defaulting to float(correct), not an
            F1 or reward value. Additional numeric fields can hold scores or costs.
        pass_at_n: Group attempts by idx instead of computing flat accuracy.
            Optional ``attempt`` orders each group; None is ordered as zero.
            pass_at_1 uses the first available attempt, even if attempt zero is
            absent. pass_at_n measures any successful observed attempt, not an
            unbiased estimator or correctness of a selected final answer.
        mean_fields: Numeric fields averaged over their non-None observations.
            Zero is retained; an entirely missing field returns None.

    Returns:
        ``total``, sanitized ``examples``, and ``means``. Flat mode adds
        ``correct_count`` and ``accuracy``; attempt mode adds pass_at_1/pass_at_n
        rates and their corresponding *_count fields. Empty inputs return zero
        counts/rates. Means always describe input rows, not per-example totals.

    This function neither mutates inputs nor evaluates predictions. Budget curves
    call flat mode separately for each method/budget using comparable example sets::

        summary = aggregate_results(rows, mean_fields=("output_tokens", "seconds"))
        success_rate = summary["accuracy"]
    """
    points = list(records)
    means = {}
    for field in mean_fields:
        values = [point[field] for point in points if point.get(field) is not None]
        means[field] = sum(values) / len(values) if values else None

    if not pass_at_n:
        correct_count = sum(1 for point in points if point["correct"])
        return {
            "total": len(points),
            "correct_count": correct_count,
            "accuracy": correct_count / len(points) if points else 0.0,
            "examples": [
                {"idx": point["idx"], "correct": point["correct"],
                 "score": point.get("score", float(point["correct"]))}
                for point in points
            ],
            "means": means,
        }

    groups = defaultdict(list)
    for point in points:
        groups[point["idx"]].append(point)
    examples = []
    for idx in sorted(groups):
        ordered = sorted(
            groups[idx],
            key=lambda point: point["attempt"] if point.get("attempt") is not None else 0,
        )
        attempts = [point.get("score", float(point["correct"])) for point in ordered]
        examples.append({
            "idx": idx, "attempts": attempts,
            "pass_at_1": attempts[0] == 1.0,
            "pass_at_n": any(score == 1.0 for score in attempts),
        })
    summary = {"total": len(examples), "examples": examples, "means": means}
    for metric in ("pass_at_1", "pass_at_n"):
        count = sum(1 for example in examples if example[metric])
        summary[metric + "_count"] = count
        summary[metric] = count / len(examples) if examples else 0
    return summary
