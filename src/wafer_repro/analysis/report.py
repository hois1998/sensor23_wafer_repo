from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pandas as pd

from wafer_repro.analysis.collector import METRIC_COLUMNS


def _fmt(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame[columns].iterrows():
        rows.append("| " + " | ".join(_fmt(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def _leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ["run", "suite", "model", "seed", "fold", "split_strategy", *METRIC_COLUMNS, "best_epoch"] if column in frame]
    leaderboard = frame.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    leaderboard.insert(0, "rank", leaderboard.index + 1)
    return leaderboard[["rank", *columns]].head(20)


def _axis_summary(frame: pd.DataFrame, axis_column: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(axis_column, dropna=False)[["macro_f1", "accuracy"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = [
        axis_column,
        "macro_f1_mean",
        "macro_f1_std",
        "macro_f1_count",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_count",
    ]
    return grouped.sort_values("macro_f1_mean", ascending=False)


def _paired_axis_analysis(frame: pd.DataFrame, axis_column: str) -> pd.DataFrame:
    methods = sorted(str(value) for value in frame[axis_column].dropna().unique())
    if len(methods) < 2:
        return pd.DataFrame(columns=["axis", "method_a", "method_b", "mean_delta_macro_f1", "std_delta_macro_f1", "n_pairs"])

    key_candidates = ["suite", "model", "seed", "fold", "split_strategy"]
    key_columns = [column for column in key_candidates if column in frame and column != axis_column]
    rows = []
    for method_a, method_b in itertools.combinations(methods, 2):
        left = frame[frame[axis_column].astype(str) == method_a]
        right = frame[frame[axis_column].astype(str) == method_b]
        paired = left.merge(right, on=key_columns, suffixes=("_a", "_b"))
        if paired.empty:
            continue
        deltas = paired["macro_f1_b"] - paired["macro_f1_a"]
        rows.append(
            {
                "axis": axis_column.removeprefix("axis_"),
                "method_a": method_a,
                "method_b": method_b,
                "mean_delta_macro_f1": float(deltas.mean()),
                "std_delta_macro_f1": float(deltas.std()) if len(deltas) > 1 else None,
                "n_pairs": int(len(deltas)),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_report(
    frame: pd.DataFrame,
    grouped: pd.DataFrame,
    summary: dict[str, Any],
    out: str | Path,
) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    axis_columns = sorted(column for column in frame.columns if column.startswith("axis_"))
    lines = [
        f"# Suite Report: {frame['suite'].dropna().iloc[0] if 'suite' in frame and not frame['suite'].dropna().empty else out.stem}",
        "",
        "## Summary",
        f"- Runs: {summary.get('run_count')}",
        f"- Completed: {summary.get('completed_count')}",
        f"- Seeds: {', '.join(summary.get('seeds') or []) or 'none'}",
        f"- Folds: {', '.join(summary.get('folds') or []) or 'none'}",
        "",
        "## Metric Summary",
    ]
    metric_rows = []
    for metric, values in summary.get("metrics", {}).items():
        metric_rows.append({"metric": metric, **values})
    lines.append(_markdown_table(pd.DataFrame(metric_rows), ["metric", "mean", "std", "min", "max", "count"]))

    lines.extend(["", "## Leaderboard", _markdown_table(_leaderboard(frame), [column for column in _leaderboard(frame).columns])])

    lines.extend(["", "## Grouped Results"])
    grouped_frame = grouped.reset_index()
    flattened_columns = []
    for column in grouped_frame.columns:
        if isinstance(column, tuple):
            flattened_columns.append("_".join(str(part) for part in column if part))
        else:
            flattened_columns.append(str(column))
    grouped_frame.columns = flattened_columns
    display_columns = [column for column in grouped_frame.columns if column in summary.get("group_keys", []) or column.endswith("_mean") or column.endswith("_count")]
    lines.append(_markdown_table(grouped_frame.head(20), display_columns))

    lines.extend(["", "## Per-axis Analysis"])
    for axis_column in axis_columns:
        lines.extend([f"### {axis_column.removeprefix('axis_')}", _markdown_table(_axis_summary(frame, axis_column), [axis_column, "macro_f1_mean", "macro_f1_std", "macro_f1_count", "accuracy_mean"])])

    lines.extend(["", "## Paired Axis Analysis"])
    paired_frames = [_paired_axis_analysis(frame, axis_column) for axis_column in axis_columns]
    nonempty_paired = [item for item in paired_frames if not item.empty]
    paired = pd.concat(nonempty_paired, ignore_index=True) if nonempty_paired else pd.DataFrame()
    lines.append(_markdown_table(paired, ["axis", "method_a", "method_b", "mean_delta_macro_f1", "std_delta_macro_f1", "n_pairs"]))

    lines.extend(["", "## Warnings"])
    warnings = summary.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No warnings.")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
