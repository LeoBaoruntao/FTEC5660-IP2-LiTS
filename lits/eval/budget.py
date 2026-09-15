"""Analyze evaluated search runs without executing search or model inference.

Completed-run cost thresholds
-----------------------------
``prepare_cost_threshold_curves(records, thresholds=...)`` reuses one completed
run per method/example/repeat. For each threshold c, it computes::

    success(c) = count(correct final output AND recorded cost <= c) / all runs

Failed and over-threshold runs stay in the denominator. No budget reruns,
checkpoint replay, or first-correct-terminal timestamps are needed. The result
describes completed runs, not accuracy when search is interrupted at c.

Configured search horizons
-------------------------
``prepare_budget_curves(records)`` groups separately evaluated horizons such as
two versus ten iterations. Its cost axes show per-group mean costs, not thresholds.

Input and cost scope
-------------------
Both modes accept dataset, method, idx, boolean correct, optional repeat (default
0), and recorded output_tokens, num_calls, or wall_time_seconds. Horizon rows also
require budget and budget_unit. Threshold rows need every requested cost value.

Rows describe selected final outputs, not any correct terminal inside the tree.
每条记录对应一次运行最终选出的答案及其正确性；不是只要搜索树中
存在一个正确的终止节点，就将这次运行算作成功。

Keep cost scope consistent across methods and state it in the plot: search-only
or including final-answer selection, always excluding reference-based judging.
各方法必须使用相同的成本统计范围，并在图中注明：仅统计搜索成本，
或同时计入最终答案选择（如投票）的成本。两种范围都不计入使用
 标准答案判定正确性的评测成本。

Plotting and module execution
----------------------------
``plot_budget_curves(points)`` recognizes either point type and labels it accordingly.
It requires the optional ``lits-llm[analysis]`` dependency. Import the functions or run::

    python -m lits.eval.budget --input horizons.jsonl --output horizons.png
    python -m lits.eval.budget --input runs.jsonl --output costs.svg --config plot.json

Cost-threshold configuration (plot.json)::

    {
      "thresholds": {
        "output_tokens": [100, 200, 400]
      },
      "title": "Success by Generated-Token Cost",
      "subtitle": "Search-only cost; evaluation excluded"
    }

Thresholds select the analysis mode; title and subtitle label the figure.
The input records must already use the cost scope stated in the subtitle.

Horizon configuration (plot.json, without thresholds)::

    {
      "x_fields": ["budget", "output_tokens"],
      "title": "Success across Search Horizons"
    }

Each x_fields entry creates a separate subplot, not a mixed x-axis. Horizon
records require budget and budget_unit even when only output_tokens is plotted.
Grouping remains (method, budget); x_fields only chooses the displayed axes::

             Records grouped by (method, budget)
                              |
                       Aggregated results
                              |
                 +------------+------------+
                 |                         |
              Left plot                 Right plot
       x: configured budget       x: mean output_tokens
       y: success rate            y: same success rate

For two runs of the same method at budget 2 costing 100 and 200 tokens, the
respective x-coordinates are 2 and 150, not token thresholds.

Other plot arguments include panels and method_labels. JSONL is strict;
existing outputs are never overwritten.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import json
import math
from numbers import Real
from typing import Any

from .aggregation import aggregate_results


COST_FIELDS = ("output_tokens", "num_calls", "wall_time_seconds")


def _nonnegative_number(value: Any, field: str) -> None:
    """Reject booleans, nonnumeric, nonfinite, and negative budget/cost values."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")


def _group_runs(records, *, group_fields, cost_fields):
    """Group rows and require one dataset and identical, balanced example/repeat cohorts.

    Return (groups, num_examples). Optional costs may be missing but never invalid;
    each public preparer applies its own completeness rule. Input rows are copied.
    """
    groups = defaultdict(dict)
    datasets = set()
    for row in records:
        for field in ("dataset", "method"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"{field} must be a nonempty string")
        if isinstance(row["idx"], bool) or not isinstance(row["idx"], (int, str)):
            raise ValueError("idx must be an integer or string example ID")
        repeat = row.get("repeat", 0)
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
            raise ValueError("repeat must be a nonnegative integer")
        if not isinstance(row["correct"], bool):
            raise ValueError("correct must be boolean final-output correctness")
        key = tuple(row[field] for field in group_fields)
        identity = (row["idx"], repeat)
        if identity in groups[key]:
            raise ValueError(f"Duplicate run for group={key}, identity={identity}")
        for field in cost_fields:
            if row.get(field) is not None:
                _nonnegative_number(row[field], field)
        groups[key][identity] = dict(row)
        datasets.add(row["dataset"])
    if not groups or len(datasets) != 1:
        raise ValueError("Provide nonempty records from exactly one dataset")
    cohort = set(next(iter(groups.values())))
    repeats = defaultdict(set)
    for idx, repeat in cohort:
        repeats[idx].add(repeat)
    if len({frozenset(values) for values in repeats.values()}) != 1:
        raise ValueError("Every example must have the same repeat IDs")
    for key, group in groups.items():
        if set(group) != cohort:
            raise ValueError(f"Example/repeat coverage differs for group={key}")
    return groups, len(repeats)


def prepare_budget_curves(
    records: Iterable[Mapping[str, Any]], *, mean_fields: Sequence[str] = COST_FIELDS,
) -> list[dict[str, Any]]:
    """Validate one dataset and summarize each method/budget with aggregate_results.

    Args:
        records: Rows described in the module docstring. idx identifies an example;
            repeat identifies an independent run of that example, not pass-at-N.
            Budget is positive; units describe the host's limit, not realized cost.
        mean_fields: Nonnegative numeric fields to average within each method/budget.
            Example: ``mean_fields=("output_tokens",)`` with two runs costing 100
            and 200 tokens at the same method/budget yields
            ``means={"output_tokens": 150.0}``. An entirely missing field stays
            None; partially missing measurements are rejected.

    Returns:
        Points containing dataset, method, budget_unit, budget, num_examples,
        num_runs, success_rate, and means. Order follows method's first appearance,
        then increasing budget. Missing measurements remain None.

    Every cell must contain the same (idx, repeat) set, with equal repeats per
    example. Methods sharing a budget unit must share the same budget grid.
    Duplicates, incomplete cells, mixed datasets, and per-method unit changes fail
    explicitly. Call separately for different datasets or experimental settings.
    Rates are flat final-output accuracy, never cumulative maxima or pass-at-N.
    This function prepares horizon-grouped points, not completed-run cost-threshold
    points; see the module docstring for the distinction and current support.

    Example:
        Two examples at one budget produce one curve point. Omitted repeat IDs
        default to 0; additional budgets must contain these same example IDs.

        >>> records = [
        ...     {"dataset": "toy", "method": "uct", "budget_unit": "iterations",
        ...      "budget": 2, "idx": 0, "correct": True,
        ...      "output_tokens": 100, "num_calls": 1, "wall_time_seconds": 2.0},
        ...     {"dataset": "toy", "method": "uct", "budget_unit": "iterations",
        ...      "budget": 2, "idx": 1, "correct": False,
        ...      "output_tokens": 200, "num_calls": 3, "wall_time_seconds": 4.0},
        ... ]
        >>> from pprint import pprint
        >>> pprint(prepare_budget_curves(records), sort_dicts=False)
        [{'dataset': 'toy',
          'method': 'uct',
          'budget_unit': 'iterations',
          'budget': 2,
          'num_examples': 2,
          'num_runs': 2,
          'success_rate': 0.5,
          'means': {'output_tokens': 150.0,
                    'num_calls': 2.0,
                    'wall_time_seconds': 3.0}}]
    """
    rows = list(records)
    units = {}
    grids = defaultdict(set)
    for row in rows:
        method, unit, budget = row["method"], row["budget_unit"], row["budget"]
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError("budget_unit must be a nonempty string")
        _nonnegative_number(budget, "budget")
        if budget == 0:
            raise ValueError("budget must be positive")
        if units.setdefault(method, unit) != unit:
            raise ValueError(f"Budget unit changes within method {method}")
        grids[method].add(budget)
    cells, num_examples = _group_runs(
        rows, group_fields=("method", "budget"), cost_fields=mean_fields,
    )
    unit_grids = {}
    for method, grid in grids.items():
        if unit_grids.setdefault(units[method], grid) != grid:
            raise ValueError(f"Budget grid differs for method {method}")

    points = []
    for method in units:
        for budget in sorted(grids[method]):
            cell = cells[(method, budget)]
            for field in mean_fields:
                observed = sum(row.get(field) is not None for row in cell.values())
                if observed not in (0, len(cell)):
                    raise ValueError(f"Partial {field} measurements for {method}, budget={budget}")
            summary = aggregate_results(cell.values(), mean_fields=mean_fields)
            points.append({
                "dataset": rows[0]["dataset"], "method": method,
                "budget_unit": units[method], "budget": budget,
                "num_examples": num_examples, "num_runs": summary["total"],
                "success_rate": summary["accuracy"], "means": summary["means"],
            })
    return points


def prepare_cost_threshold_curves(
    records: Iterable[Mapping[str, Any]], *,
    thresholds: Mapping[str, Sequence[float]],
) -> list[dict[str, Any]]:
    """Count correct completed runs whose recorded cost is at most each threshold.

    Args:
        records: One full-horizon row per method/example/repeat, with dataset,
            method, idx, correct, optional repeat, and all requested cost fields.
            Use one fixed search configuration per method. Duplicate runs or
            mismatched example/repeat cohorts fail; input records are not changed.
        thresholds: Cost field to nonempty sequence of distinct finite nonnegative
            thresholds. Allowed fields: output_tokens, num_calls, wall_time_seconds.
            Every method uses the same thresholds, sorted ascending per field.

    Returns:
        Points with curve_type="cost_threshold", dataset, method, cost_field,
        threshold, num_examples, num_runs, success_count, and success_rate.
        Order follows first-seen method, field order, then increasing threshold.
        num_runs includes failures and over-threshold runs. Repeats count as
        separate runs, not pass-at-N attempts. Missing requested costs raise errors.

    A threshold is a post-hoc filter, never a search-time stopping rule. Costs are
    per completed run, not per terminal or first success. Equality counts as within
    threshold. Each rate is computed directly from records, not a monotonic envelope.

    Example:
        >>> rows = [
        ...     {"dataset": "toy", "method": "uct", "idx": i,
        ...      "correct": correct, "output_tokens": tokens}
        ...     for i, (correct, tokens) in enumerate([
        ...         (True, 80), (True, 180), (False, 120), (True, 350),
        ...     ])
        ... ]
        >>> points = prepare_cost_threshold_curves(
        ...     rows, thresholds={"output_tokens": [100, 200, 400]},
        ... )
        >>> [(p["threshold"], p["success_count"], p["num_runs"], p["success_rate"])
        ...  for p in points]
        [(100, 1, 4, 0.25), (200, 2, 4, 0.5), (400, 3, 4, 0.75)]
    """
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise ValueError("Provide a nonempty thresholds mapping")
    grids = {}
    for field, values in thresholds.items():
        if field not in COST_FIELDS:
            raise ValueError(f"Unsupported cost field: {field}")
        values = list(values)
        if not values:
            raise ValueError(f"Empty thresholds for {field}")
        for value in values:
            _nonnegative_number(value, f"{field} threshold")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate thresholds for {field}")
        grids[field] = sorted(values)
    groups, num_examples = _group_runs(
        records, group_fields=("method",), cost_fields=grids,
    )
    points = []
    for (method,), group in groups.items():
        for field, values in grids.items():
            if any(row.get(field) is None for row in group.values()):
                raise ValueError(f"Missing {field} measurements for {method}")
            for threshold in values:
                summary = aggregate_results(
                    {"idx": row["idx"], "correct": row["correct"] and row[field] <= threshold}
                    for row in group.values()
                )
                points.append({
                    "curve_type": "cost_threshold",
                    "dataset": next(iter(group.values()))["dataset"], "method": method,
                    "cost_field": field, "threshold": threshold,
                    "num_examples": num_examples, "num_runs": summary["total"],
                    "success_count": summary["correct_count"], "success_rate": summary["accuracy"],
                })
    return points


def plot_budget_curves(
    points: Sequence[Mapping[str, Any]], *,
    panels: Mapping[str, Sequence[str]] | None = None,
    method_labels: Mapping[str, str] | None = None,
    x_fields: Sequence[str] | None = None,
    axis_labels: Mapping[str, str] | None = None,
    title: str | None = None,
    subtitle: str = "",
):
    """Return a Matplotlib Figure from prepared points; do not save files or show it.

    Args:
        points: Output of either preparer for one dataset; do not mix point types.
        panels: Display titles mapped to method IDs. Include every method exactly
            once. Horizon panels require one budget unit; threshold panels share
            cost fields. Default: group horizons by unit, or show all methods.
        method_labels: Optional display names, without changing method identities.
        x_fields: Budget and/or names from each point's means. Missing values fail
            rather than silently omitting points. Cost axes show per-run means
            in horizon mode. Threshold mode uses cost-field names and plots the
            thresholds themselves. Defaults: budget/all means, or all cost fields.
        axis_labels: Optional x-axis labels keyed by field name.
        title: Figure heading; default names the selected analysis mode.
        subtitle: Context such as dataset, sample size, or illustrative-test status.

    Colors, markers, and line styles distinguish methods across all panels. Lines
    connect increasing budgets even when measured costs or success decrease; no
    smoothing, monotonic envelope, or confidence intervals are invented. Threshold
    curves use post-step segments between evaluated thresholds; markers are thinned
    on dense grids. Labels and footer distinguish thresholds from mean costs.
    Matplotlib is imported only when plotting. Save via figure.savefig(path, dpi=200).
    """
    if not points:
        raise ValueError("Nonempty points are required")
    kinds = {point.get("curve_type", "horizon") for point in points}
    if len(kinds) != 1 or not kinds <= {"horizon", "cost_threshold"}:
        raise ValueError("Do not mix horizon and cost-threshold point types")
    threshold_mode = kinds == {"cost_threshold"}
    coordinate_key = "threshold" if threshold_mode else "budget"
    if x_fields is None:
        x_fields = (
            tuple(dict.fromkeys(p["cost_field"] for p in points))
            if threshold_mode else ("budget", *COST_FIELDS)
        )
    if not x_fields:
        raise ValueError("Nonempty x_fields are required")
    if threshold_mode and not set(x_fields) <= set(COST_FIELDS):
        raise ValueError("Threshold axes must name recorded cost fields")
    if len({point["dataset"] for point in points}) != 1:
        raise ValueError("Plot one dataset at a time")
    by_method = defaultdict(list)
    for point in points:
        by_method[point["method"]].append(point)
    if panels is None:
        panels = defaultdict(list)
        for method, rows in by_method.items():
            panels["Completed runs" if threshold_mode else rows[0]["budget_unit"]].append(method)
    methods = [method for group in panels.values() for method in group]
    if len(methods) != len(set(methods)) or set(methods) != set(by_method):
        raise ValueError("Panels must include every method exactly once")
    for group in panels.values():
        if not group:
            raise ValueError("Panels must contain at least one method")
        if not threshold_mode and len({p["budget_unit"] for m in group for p in by_method[m]}) != 1:
            raise ValueError("Each panel must contain methods with one common budget unit")
    if threshold_mode:
        for field in x_fields:
            grids = [tuple(sorted(p["threshold"] for p in rows if p["cost_field"] == field))
                     for rows in by_method.values()]
            if any(not grid or len(set(grid)) != len(grid) for grid in grids) or len(set(grids)) != 1:
                raise ValueError(f"Missing, duplicate, or inconsistent thresholds for {field}")
    for point in points:
        for field in ((point["cost_field"],) if threshold_mode else x_fields):
            value = (
                point[coordinate_key] if threshold_mode or field == "budget"
                else point["means"].get(field)
            )
            if value is None:
                raise ValueError(f"Cannot plot unavailable {field} for {point['method']}")
            _nonnegative_number(value, field)
        if not 0 <= point["success_rate"] <= 1:
            raise ValueError("success_rate must be between zero and one")

    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    labels = ({
        "output_tokens": "Generated-token threshold", "num_calls": "Model-call threshold",
        "wall_time_seconds": "Time threshold (s)",
    } if threshold_mode else {
        "output_tokens": "Mean generated tokens", "num_calls": "Mean model calls",
        "wall_time_seconds": "Mean search time (s)",
    }) | (axis_labels or {})
    title = title or (
        "Cumulative success by recorded cost" if threshold_mode else "Success across search budgets"
    )
    colors = ("#0072B2", "#D55E00", "#009E73", "#555555", "#CC79A7", "#56B4E9")
    markers = ("o", "s", "^", "D", "v", "P")
    styles = ("-", "--", "-.")
    with rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                     "pdf.fonttype": 42, "axes.labelcolor": "#303840", "text.color": "#202830"}):
        figure = Figure(figsize=(max(8, 3.6 * len(x_fields)), 3.7 * len(panels) + 1.4))
        FigureCanvasAgg(figure)
        grid = figure.add_gridspec(
            2 * len(panels), len(x_fields), height_ratios=[0.20, 1] * len(panels),
            left=0.065, right=0.985, bottom=0.10, top=0.84, hspace=0.35, wspace=0.22,
        )
        figure.text(0.065, 0.962, title, fontsize=19, weight="bold", va="top")
        context = subtitle or (
            f"{points[0]['dataset']} | {points[0]['num_examples']} examples | "
            f"{points[0]['num_runs']} runs per method" + ("" if threshold_mode else "/budget")
        )
        figure.text(0.065, 0.913, context, fontsize=11, color="#59636D", va="top")
        for row_index, (panel_title, group) in enumerate(panels.items()):
            header = figure.add_subplot(grid[2 * row_index, :])
            header.set_axis_off()
            header.text(0, 0.5, panel_title, fontsize=13, weight="bold", va="center")
            handles = []
            for column, field in enumerate(x_fields):
                axis = figure.add_subplot(grid[2 * row_index + 1, column])
                for method in group:
                    rows = sorted(
                        (p for p in by_method[method] if not threshold_mode or p["cost_field"] == field),
                        key=lambda point: point[coordinate_key],
                    )
                    xs = [
                        p[coordinate_key] if threshold_mode or field == "budget" else p["means"][field]
                        for p in rows
                    ]
                    index = methods.index(method)
                    line, = axis.plot(
                        xs, [p["success_rate"] for p in rows],
                        color=colors[index % len(colors)], marker=markers[index % len(markers)],
                        linestyle=styles[index % len(styles)], linewidth=1.9, markersize=6,
                        markerfacecolor="white", markeredgewidth=1.5,
                        drawstyle="steps-post" if threshold_mode else "default",
                        markevery=max(1, len(rows) // 12) if threshold_mode else 1,
                        label=(method_labels or {}).get(method, method),
                    )
                    if column == 0:
                        handles.append(line)
                unit = "" if threshold_mode else rows[0]["budget_unit"].replace("_", " ")
                axis.set_xlabel(labels.get(field, f"Budget ({unit})" if field == "budget" else field))
                axis.set_ylim(-0.035, 1.035)
                axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
                axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                if column == 0:
                    axis.set_ylabel(
                        "Cumulative final-output success" if threshold_mode else "Final-output success"
                    )
                else:
                    axis.tick_params(labelleft=False)
                if field == "budget":
                    axis.set_xticks(sorted({p["budget"] for m in group for p in by_method[m]}))
                    axis.margins(x=0.07)
                else:
                    axis.set_xlim(left=0)
                    axis.xaxis.set_major_locator(MaxNLocator(4))
                axis.grid(axis="y", color="#E1E5E9", linewidth=0.8)
                axis.set_axisbelow(True)
                for side in ("top", "right"):
                    axis.spines[side].set_visible(False)
                for side in ("bottom", "left"):
                    axis.spines[side].set_color("#B8C0C8")
                axis.tick_params(colors="#59636D", length=3)
            header.legend(handles=handles, loc="center right", ncol=min(3, len(group)),
                          frameon=False, fontsize=10, handlelength=2.4, columnspacing=1.4)
        figure.text(
            0.065, 0.025,
            ("Markers evaluate cost thresholds. Failed and over-threshold runs remain in the denominator."
             if threshold_mode else
             "Points summarize evaluated final outputs. Lines follow increasing budgets; costs are per-run means."),
            fontsize=9, color="#59636D",
        )
        figure.canvas.draw()
    return figure


def main() -> None:
    """Read strict JSONL and optional plot configuration; save points and one figure.

    Existing output files are refused. Malformed records propagate as errors instead
    of disappearing from the denominator. No separate console command is registered.

    See the module docstring for configuration examples and subplot layout.
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON object of plot arguments and optional cost thresholds")
    args = parser.parse_args()
    points_path = args.output.with_suffix(".points.json")
    if args.output.exists() or points_path.exists():
        parser.error("Output already exists; choose a new output path")
    rows = []
    for path in args.input:
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    config = json.loads(args.config.read_text()) if args.config else {}
    if "thresholds" in config:
        points = prepare_cost_threshold_curves(rows, thresholds=config.pop("thresholds"))
    else:
        # Without thresholds, group runs by configured search horizon. Average only
        # the requested cost axes (all COST_FIELDS by default); "budget" is the
        # grouping value, not a measured cost to average. For example,
        # x_fields=["budget", "output_tokens"] gives mean_fields=("output_tokens",).
        x_fields = config.get("x_fields") or ("budget", *COST_FIELDS)
        mean_fields = tuple(field for field in x_fields if field != "budget")
        points = prepare_budget_curves(rows, mean_fields=mean_fields)
    figure = plot_budget_curves(points, **config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    with points_path.open("x", encoding="utf-8") as stream:
        json.dump(points, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Figure: {args.output}\nCurve points: {points_path}")


if __name__ == "__main__":
    main()
