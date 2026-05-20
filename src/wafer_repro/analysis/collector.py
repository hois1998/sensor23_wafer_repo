from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_COLUMNS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_sweep_axes(runs_dir: Path) -> dict[str, dict[str, str]]:
    manifest = _read_json(runs_dir / "sweep_manifest.json")
    axes_by_run: dict[str, dict[str, str]] = {}
    for trial in manifest.get("trials", []):
        axes_by_run[trial["name"]] = {f"axis_{key}": value for key, value in trial.get("axes", {}).items()}
    return axes_by_run


def collect_run_rows(runs_dir: str | Path) -> list[dict[str, Any]]:
    runs_dir = Path(runs_dir)
    axes_by_run = _load_sweep_axes(runs_dir)
    rows = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir() and not path.name.startswith("_")):
        summary = _read_json(run_dir / "test_summary.json")
        manifest = _read_json(run_dir / "run_manifest.json")
        config = _read_json(run_dir / "config.json")
        data_summary = _read_json(run_dir / "data_summary.json")
        data_identity = _read_json(run_dir / "data_identity.json")
        artifact_manifest = _read_json(run_dir / "artifact_manifest.json")
        if not summary and manifest.get("test_summary"):
            summary = manifest["test_summary"]
        if not summary:
            continue
        predictions_path = Path(artifact_manifest.get("test_predictions", run_dir / "predictions" / "test_predictions.csv"))
        row = {
                "run": run_dir.name,
                "status": manifest.get("status"),
                "suite": manifest.get("suite"),
                "model": config.get("model"),
                "fold": config.get("fold"),
                "seed": config.get("seed"),
                "split_strategy": config.get("split_strategy") or manifest.get("split_strategy"),
                "accuracy": summary.get("accuracy"),
                "macro_precision": summary.get("macro_precision"),
                "macro_recall": summary.get("macro_recall"),
                "macro_f1": summary.get("macro_f1"),
                "weighted_f1": summary.get("weighted_f1"),
                "best_epoch": summary.get("best_epoch") or manifest.get("best_epoch"),
                "config_hash": manifest.get("config_hash") or config.get("config_hash"),
                "train_records_after_augmentation": data_summary.get("train_records_after_augmentation"),
                "test_records": data_summary.get("test_records"),
                "data_path": data_identity.get("path"),
                "data_size_bytes": data_identity.get("size_bytes"),
                "has_data_identity": bool(data_identity),
                "has_split_hashes": (run_dir / "splits" / "split_hashes.json").exists(),
                "has_test_predictions": predictions_path.exists(),
            }
        row.update(axes_by_run.get(run_dir.name, {}))
        rows.append(row)
    return rows


def _json_number(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def build_suite_summary(frame: pd.DataFrame, group_keys: list[str]) -> dict[str, Any]:
    metric_summary: dict[str, dict[str, Any]] = {}
    for metric in METRIC_COLUMNS:
        if metric not in frame:
            continue
        series = frame[metric].dropna()
        metric_summary[metric] = {
            "mean": _json_number(series.mean()) if not series.empty else None,
            "std": _json_number(series.std()) if len(series) > 1 else None,
            "min": _json_number(series.min()) if not series.empty else None,
            "max": _json_number(series.max()) if not series.empty else None,
            "count": int(series.count()),
        }

    warnings = []
    seeds = sorted(str(value) for value in frame["seed"].dropna().unique()) if "seed" in frame else []
    folds = sorted(str(value) for value in frame["fold"].dropna().unique()) if "fold" in frame else []
    if len(seeds) <= 1:
        warnings.append("Suite has one or zero explicit train seeds.")
    if "split_strategy" in frame and any(frame["split_strategy"].astype(str).str.contains("kfold")) and len(folds) <= 1:
        warnings.append("K-fold suite has one or zero explicit fold indices.")

    return {
        "run_count": int(len(frame)),
        "completed_count": int((frame.get("status") == "completed").sum()) if "status" in frame else None,
        "seeds": seeds,
        "folds": folds,
        "group_keys": group_keys,
        "metrics": metric_summary,
        "warnings": warnings,
    }


def write_comparison(runs_dir: str | Path, out: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = collect_run_rows(runs_dir)
    if not rows:
        raise FileNotFoundError(f"No completed run summaries found under {runs_dir}")

    frame = pd.DataFrame(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    axis_columns = sorted(column for column in frame.columns if column.startswith("axis_"))
    group_keys = [column for column in ["suite", *axis_columns, "model", "split_strategy"] if column in frame.columns]
    grouped = (
        frame.groupby(group_keys, dropna=False)[METRIC_COLUMNS]
        .agg(["mean", "std", "min", "max", "count"])
        .sort_values(("macro_f1", "mean"), ascending=False)
    )
    grouped.to_csv(out.with_name(out.stem + "_grouped.csv"))
    summary = build_suite_summary(frame, group_keys)
    (out.with_name(out.stem + "_summary.json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return frame, grouped
