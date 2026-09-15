"""Check shared budget analysis and generate a professional illustrative figure.

Run ``PYTHONBREAKPOINT=0 python -m unit_test.eval.test_budget_curves``.
Controlled evaluated records exercise arithmetic and rendering, not model behavior;
no fake models or inference calls are used. Figures are explicitly labeled as
illustrative test data. Fresh records, points, figures, and checks are retained.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from lits.eval.budget import prepare_budget_curves, prepare_cost_threshold_curves, plot_budget_curves


COUNTS = {
    "beam_pointwise": (5, 8, 10, 13),
    "beam_cluster": (6, 10, 13, 16),
    "beam_llm": (6, 12, 15, 19),
    "uct": (4, 10, 16, 15),
    "frontier_llm": (5, 11, 17, 20),
}
BUDGETS = (1, 2, 4, 8)
PLOT_CONFIG = {
    "panels": {
        "Study I | Joint beam allocation": ["beam_pointwise", "beam_cluster", "beam_llm"],
        "Study II | Sequential allocation": ["uct", "frontier_llm"],
    },
    "method_labels": {
        "beam_pointwise": "Pointwise beam", "beam_cluster": "Clustered beam",
        "beam_llm": "LLM joint", "uct": "UCT", "frontier_llm": "LLM frontier",
    },
    "title": "Success, search budget, and realized cost",
    "subtitle": "ILLUSTRATIVE TEST DATA ONLY | 12 examples, 2 repeats | Not benchmark results",
}


def make_records():
    """Construct evaluated-row fixtures for ``budget.py::prepare_budget_curves``.

    Correctness counts and costs are chosen explicitly for validation, not sampled
    from searches. UCT's final count decreases to test preservation of nonmonotonicity.
    """
    rows = []
    for method_index, (method, counts) in enumerate(COUNTS.items()):
        for budget, count in zip(BUDGETS, counts):
            for idx in range(12):
                for repeat in range(2):
                    tokens = (70 + method_index * 15) * budget + idx * 3 + repeat
                    rows.append({
                        "dataset": "illustrative_fixture", "method": method,
                        "budget_unit": "beam_steps" if method.startswith("beam_") else "search_iterations",
                        "budget": budget, "idx": idx, "repeat": repeat,
                        "correct": 2 * idx + repeat < count,
                        "output_tokens": tokens,
                        "num_calls": (2 + method_index) * budget + repeat,
                        "wall_time_seconds": tokens / 75 + method_index * budget,
                    })
    return rows


def check_records(rows):
    """Check cell summaries and invalid coverage without duplicating CLI evaluation tests."""
    original = deepcopy(rows)
    points = prepare_budget_curves(iter(rows))
    for point in points:
        count = COUNTS[point["method"]][BUDGETS.index(point["budget"])]
        cell = [r for r in rows if (r["method"], r["budget"]) == (point["method"], point["budget"])]
        if (point["num_examples"], point["num_runs"], point["success_rate"]) != (12, 24, count / 24):
            raise RuntimeError(f"Incorrect final-output success summary: {point}")
        if point["means"] != {
            field: sum(r[field] for r in cell) / 24
            for field in ("output_tokens", "num_calls", "wall_time_seconds")
        }:
            raise RuntimeError(f"Incorrect measured-cost summary: {point}")
    if rows != original or len(points) != 20:
        raise RuntimeError("Input mutation or missing curve points")
    checks = [{"case": "cell_summaries", "passed": True, "points": len(points)}]

    invalid = [
        ("empty", [], "nonempty"),
        ("duplicate", rows + [rows[0]], "Duplicate run"),
        ("incomplete_cohort", rows[:-1], "coverage differs"),
        ("unequal_repeats", [r for r in rows if (r["idx"], r["repeat"]) != (0, 1)], "same repeat"),
        ("missing_budget", [r for r in rows if not (r["method"] == "beam_pointwise" and r["budget"] == 8)], "Budget grid differs"),
    ]
    for name, field, value, message in (
        ("mixed_datasets", "dataset", "another_dataset", "one dataset"),
        ("changed_unit", "budget_unit", "another_unit", "unit changes"),
        ("nonboolean_correctness", "correct", 1, "correct must be boolean"),
        ("partial_cost", "output_tokens", None, "Partial output_tokens"),
        ("nonfinite_cost", "wall_time_seconds", float("nan"), "finite nonnegative"),
        ("negative_cost", "num_calls", -1, "finite nonnegative"),
        ("zero_budget", "budget", 0, "positive"),
    ):
        changed = [dict(row) for row in rows]
        changed[0][field] = value
        invalid.append((name, changed, message))
    for name, changed, message in invalid:
        try:
            prepare_budget_curves(changed)
        except ValueError as error:
            if message not in str(error):
                raise RuntimeError(f"Unexpected {name} error: {error}") from error
        else:
            raise RuntimeError(f"Invalid {name} input was accepted")
        checks.append({"case": name, "passed": True})

    missing = prepare_budget_curves([{**r, "output_tokens": None} for r in rows])
    if any(p["means"]["output_tokens"] is not None for p in missing):
        raise RuntimeError("Unavailable costs were replaced with numbers")
    try:
        plot_budget_curves(missing, **PLOT_CONFIG)
    except ValueError as error:
        if "unavailable output_tokens" not in str(error):
            raise
    else:
        raise RuntimeError("Plot silently omitted missing costs")
    checks.append({"case": "unavailable_cost", "passed": True})
    breakpoint()  # inspect: points, checks; rates are fixed illustrative expectations
    return points, checks


def check_cost_thresholds():
    """Check ``budget.py::prepare_cost_threshold_curves`` on completed-run records.

    The four answers cost 80/180/120/350 tokens; only the 120-token answer is
    incorrect. Two repeats per example must retain eight runs in each denominator.
    """
    rows = [
        {"dataset": "toy", "method": method, "idx": idx, "repeat": repeat,
         "correct": correct, "output_tokens": tokens, "num_calls": calls,
         "wall_time_seconds": tokens / 100}
        for method in ("uct", "frontier_llm")
        for idx, (correct, tokens, calls) in enumerate([
            (True, 80, 1), (True, 180, 2), (False, 120, 1), (True, 350, 4),
        ])
        for repeat in (0, 1)
    ]
    grids = {"output_tokens": [400, 100, 200], "num_calls": [1, 2, 4],
             "wall_time_seconds": [1, 2, 4]}
    original = deepcopy(rows)
    points = prepare_cost_threshold_curves(iter(rows), thresholds=grids)
    for method in ("uct", "frontier_llm"):
        for field, grid in grids.items():
            actual = [(p["threshold"], p["success_count"], p["num_runs"], p["num_examples"], p["success_rate"])
                      for p in points if (p["method"], p["cost_field"]) == (method, field)]
            expected = [(value, count, 8, 4, count / 8) for value, count in zip(sorted(grid), (2, 4, 6))]
            if actual != expected:
                raise RuntimeError(f"Incorrect threshold counts or denominator: {actual}")
    if rows != original or grids["output_tokens"] != [400, 100, 200] or len(points) != 18:
        raise RuntimeError("Threshold preparation mutated inputs or lost points")
    checks = [{"case": "cost_threshold_counts", "passed": True}]

    single = [{key: value for key, value in row.items() if key != "repeat"}
              for row in rows if row["method"] == "uct" and row["repeat"] == 0]
    single[0]["output_tokens"] = 0
    boundary = prepare_cost_threshold_curves(single, thresholds={"output_tokens": [0, 179, 180, 350]})
    if [p["success_rate"] for p in boundary] != [0.25, 0.25, 0.5, 0.75]:
        raise RuntimeError("Zero cost or inclusive threshold comparison is incorrect")
    failed = prepare_cost_threshold_curves([{**r, "correct": False} for r in rows], thresholds=grids)
    if any(p["success_rate"] != 0 or p["num_runs"] != 8 for p in failed):
        raise RuntimeError("All-failed groups were dropped or normalized incorrectly")
    checks.extend([{"case": name, "passed": True}
                   for name in ("inclusive_zero_cost", "all_failed_thresholds")])

    token_only = plot_budget_curves(points, x_fields=("output_tokens",))
    if len([axis for axis in token_only.axes if axis.lines]) != 1:
        raise RuntimeError("Selecting one cost axis did not produce one panel")
    if "Cumulative success" not in token_only.texts[0].get_text():
        raise RuntimeError("The default title does not identify threshold analysis")
    checks.append({"case": "single_cost_axis", "passed": True})
    for name, changed, options in (
        ("mixed_point_types", [*points, {"curve_type": "horizon"}], {}),
        ("missing_threshold_point", points[:-1], {}),
        ("threshold_budget_axis", points, {"x_fields": ("budget",)}),
    ):
        try:
            plot_budget_curves(changed, **options)
        except ValueError:
            checks.append({"case": name, "passed": True})
        else:
            raise RuntimeError(f"Invalid plot {name} was accepted")

    invalid = [
        ("threshold_empty_records", [], grids, "nonempty"),
        ("threshold_duplicate_run", rows + [rows[0]], grids, "Duplicate run"),
        ("threshold_incomplete_cohort", rows[:-1], grids, "coverage differs"),
        ("threshold_unbalanced_repeats", [r for r in rows if (r["idx"], r["repeat"]) != (0, 1)], grids, "same repeat"),
        ("threshold_missing_cost", [{**r, "output_tokens": None} for r in rows], grids, "Missing output_tokens"),
        ("threshold_partial_cost", [{**rows[0], "num_calls": None}, *rows[1:]], grids, "Missing num_calls"),
    ]
    for name, values, message in (
        ("empty_mapping", {}, "nonempty thresholds"),
        ("unsupported_field", {"unknown": [1]}, "Unsupported cost field"),
        ("empty_grid", {"output_tokens": []}, "Empty thresholds"),
        ("duplicate_threshold", {"num_calls": [1, 1]}, "Duplicate thresholds"),
        ("negative_threshold", {"num_calls": [-1]}, "finite nonnegative"),
        ("boolean_threshold", {"num_calls": [True]}, "finite nonnegative"),
        ("nonfinite_threshold", {"wall_time_seconds": [float("inf")]}, "finite nonnegative"),
    ):
        invalid.append((name, rows, values, message))
    for name, field, value in (
        ("threshold_negative_cost", "num_calls", -1),
        ("threshold_nonfinite_cost", "wall_time_seconds", float("nan")),
        ("threshold_boolean_cost", "output_tokens", True),
    ):
        invalid.append((name, [{**rows[0], field: value}, *rows[1:]], grids, "finite nonnegative"))
    for name, records, thresholds, message in invalid:
        try:
            prepare_cost_threshold_curves(records, thresholds=thresholds)
        except ValueError as error:
            if message not in str(error):
                raise RuntimeError(f"Unexpected {name} error: {error}") from error
        else:
            raise RuntimeError(f"Invalid {name} input was accepted")
        checks.append({"case": name, "passed": True})
    breakpoint()  # inspect: points, boundary, checks; thresholds retain failed runs
    return checks


def check_figure(points, output_dir, *, plot_config=PLOT_CONFIG):
    """Inspect curves, canvas regions, and text bounds from ``budget.py::plot_budget_curves``."""
    threshold_mode = points[0].get("curve_type") == "cost_threshold"
    name = "cost_threshold" if threshold_mode else "success_budget"
    figure = plot_budget_curves(points, **plot_config)
    for extension in ("png", "svg"):
        figure.savefig(output_dir / f"{name}_illustrative.{extension}", dpi=200, bbox_inches="tight")
    # Restore the display-DPI canvas after exports before checking pixel coordinates.
    figure.canvas.draw()
    axes = [axis for axis in figure.axes if axis.lines]
    columns = 3 if threshold_mode else 4
    if len(axes) != 2 * columns or [len(axis.lines) for axis in axes] != [3] * columns + [2] * columns:
        raise RuntimeError("Missing study panels or method curves")
    if not threshold_mode and list(axes[4].lines[0].get_ydata()) != [count / 24 for count in COUNTS["uct"]]:
        raise RuntimeError("The plot changed the nonmonotonic success measurements")
    if threshold_mode:
        for axis in axes:
            if "threshold" not in axis.get_xlabel().lower() or "mean" in axis.get_xlabel().lower():
                raise RuntimeError("Threshold axis is mislabeled as mean cost")
            if axis.get_ylabel() and "Cumulative" not in axis.get_ylabel():
                raise RuntimeError("Threshold success is not labeled cumulative")
            for line in axis.lines:
                method = next(m for m, label in PLOT_CONFIG["method_labels"].items() if label == line.get_label())
                field = ("output_tokens", "num_calls", "wall_time_seconds")[axes.index(axis) % columns]
                expected = [p for p in points if (p["method"], p["cost_field"]) == (method, field)]
                if (list(line.get_xdata()) != [p["threshold"] for p in expected]
                        or list(line.get_ydata()) != [p["success_rate"] for p in expected]
                        or line.get_drawstyle() != "steps-post"):
                    raise RuntimeError("Threshold plot changed computed coordinates")
    renderer = figure.canvas.get_renderer()
    pixels = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
    for axis in axes:
        box = axis.get_window_extent(renderer)
        region = pixels[pixels.shape[0] - int(box.y1):pixels.shape[0] - int(box.y0), int(box.x0):int(box.x1)]
        if not region.size or np.mean(np.any(region < 230, axis=2)) < 0.01:
            raise RuntimeError("A curve panel is blank")
    # Axis tight bounds exclude locator ticks outside the visible data range.
    for artist in [*figure.axes, *figure.texts]:
        box = artist.get_tightbbox(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > figure.bbox.x1 + 1 or box.y1 > figure.bbox.y1 + 1:
            raise RuntimeError("Rendered text exceeds figure bounds")
    for header in (axis for axis in figure.axes if not axis.lines):
        if header.texts[0].get_window_extent(renderer).overlaps(header.get_legend().get_window_extent(renderer)):
            raise RuntimeError("Panel heading overlaps its legend")
    breakpoint()  # inspect: figure, axes, output_dir; inspect the generated PNG visually
    return {"case": f"{name}_figure_layout_and_pixels", "passed": True, "panels": len(axes)}


def check_module(rows, points, output_dir, config, *, prefix="module"):
    """Compare ``budget.py::main`` with the imported preparer and protect existing outputs."""
    records_path = output_dir / f"{prefix}_records.jsonl"
    with records_path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    config_path = output_dir / f"{prefix}_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    command = [sys.executable, "-m", "lits.eval.budget", "--input", str(records_path),
               "--output", str(output_dir / f"{prefix}_figure.png"), "--config", str(config_path)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    print(completed.stdout)
    if json.loads((output_dir / f"{prefix}_figure.points.json").read_text()) != points:
        raise RuntimeError("Module entry point disagrees with the imported function")
    repeated = subprocess.run(command, capture_output=True, text=True)
    if repeated.returncode != 2 or "Output already exists" not in repeated.stderr:
        raise RuntimeError("Module entry point did not protect existing output")
    return [{"case": f"{prefix}_{name}", "passed": True}
            for name in ("entry_point", "existing_output_protection")]


def main():
    """Run sequential checks, exercise the module entry point, and retain all artifacts."""
    root = Path(__file__).resolve().parents[2] / "benchmark_contract_results"
    root.mkdir(exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="budget-curves-", dir=root))
    print(f"Artifacts: {output_dir}", flush=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib_cache"))
    rows = make_records()
    points, checks = check_records(rows)
    checks.append(check_figure(points, output_dir))
    checks.extend(check_module(rows, points, output_dir, PLOT_CONFIG))
    checks.extend(check_cost_thresholds())
    completed_rows = [{k: v for k, v in row.items() if k not in ("budget", "budget_unit")}
                      for row in rows if row["budget"] == BUDGETS[-1]]
    thresholds = {field: sorted({0, *(row[field] for row in completed_rows)})
                  for field in ("output_tokens", "num_calls", "wall_time_seconds")}
    threshold_points = prepare_cost_threshold_curves(completed_rows, thresholds=thresholds)
    threshold_plot = {k: v for k, v in PLOT_CONFIG.items() if k != "title"}
    checks.append(check_figure(threshold_points, output_dir, plot_config=threshold_plot))
    checks.extend(check_module(completed_rows, threshold_points, output_dir,
                               {**threshold_plot, "thresholds": thresholds}, prefix="threshold"))
    (output_dir / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"Passed {len(checks)} checks; {len(points)} horizon points and {len(threshold_points)} threshold points.")
    print(f"Artifacts: {output_dir}")
    breakpoint()  # inspect: checks, points, output_dir


if __name__ == "__main__":
    main()
