from __future__ import annotations

import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wafer_repro.core.config import deep_merge, read_yaml, set_path, write_yaml
from wafer_repro.utils import ensure_dir, write_json


@dataclass(frozen=True)
class TrialSpec:
    name: str
    config: dict[str, Any]
    axes: dict[str, str]
    seed: int | None = None
    fold: int | None = None


def _load_base_config(sweep_path: Path, sweep_config: dict[str, Any]) -> dict[str, Any]:
    sweep = sweep_config["sweep"]
    base_config = sweep.get("base_config")
    if not base_config:
        raise ValueError("sweep.base_config is required.")
    base_path = Path(base_config)
    if not base_path.is_absolute():
        base_path = sweep_path.parent / base_path
        if not base_path.exists():
            base_path = sweep_path.parent.parent / base_config
        if not base_path.exists():
            base_path = Path(base_config)
    return read_yaml(base_path)


def _apply_dotted_values(config: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    updated = deep_merge({}, config)
    for dotted_path, value in values.items():
        set_path(updated, dotted_path, value)
    return updated


def _sanitize_name(name: str) -> str:
    keep = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "trial"


def _axis_trials(sweep: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, str]]]:
    expansion = sweep.get("expansion", {})
    mode = expansion.get("mode", "grid")

    if mode == "manual":
        trials = []
        for item in sweep.get("trials", []):
            name = item["name"]
            trials.append((name, item.get("set", {}), item.get("axes", {"manual": name})))
        return trials

    if mode != "grid":
        raise ValueError(f"Unsupported sweep expansion mode: {mode}")

    axes = sweep.get("axes", {})
    include = expansion.get("include") or list(axes)
    selected_axes = [(axis_name, axes[axis_name]) for axis_name in include]
    trials = []
    for combination in itertools.product(*(methods for _, methods in selected_axes)):
        values: dict[str, Any] = {}
        axis_names: dict[str, str] = {}
        parts = []
        for (axis_name, _), method in zip(selected_axes, combination):
            method_name = method["name"]
            values.update(method.get("set", {}))
            axis_names[axis_name] = method_name
            parts.append(f"{axis_name}={method_name}")
        trials.append(("__".join(parts), values, axis_names))
    return trials


def expand_sweep(sweep_path: str | Path) -> tuple[dict[str, Any], list[TrialSpec]]:
    sweep_path = Path(sweep_path)
    sweep_config = read_yaml(sweep_path)
    sweep = sweep_config["sweep"]
    base = _load_base_config(sweep_path, sweep_config)
    fixed = sweep.get("fixed", {})
    base = _apply_dotted_values(base, fixed)
    if fixed:
        base.setdefault("fixed", {}).setdefault("controls", {}).update(fixed)

    repeat_config = sweep.get("repeats", {})
    seeds = repeat_config.get("seeds") or [None]
    folds = repeat_config.get("folds") or [None]

    trial_specs: list[TrialSpec] = []
    for axis_name, values, axes in _axis_trials(sweep):
        for seed, fold in itertools.product(seeds, folds):
            config = _apply_dotted_values(base, values)
            name_parts = [axis_name]
            if seed is not None:
                set_path(config, "train.seed", seed)
                set_path(config, "data.split.seed", seed)
                name_parts.append(f"seed={seed}")
            if fold is not None:
                set_path(config, "data.split.strategy", "stratified_kfold")
                set_path(config, "data.split.fold_index", fold)
                name_parts.append(f"fold={fold}")
            trial_name = _sanitize_name("__".join(name_parts))
            set_path(config, "experiment.suite", sweep["name"])
            set_path(config, "runtime.output_dir", str(Path("outputs") / "experiments" / sweep["name"]))
            set_path(config, "runtime.run_name", trial_name)
            trial_specs.append(TrialSpec(trial_name, config, axes=axes, seed=seed, fold=fold))
    return sweep_config, trial_specs


def write_trial_configs(sweep_path: str | Path, out_dir: str | Path | None = None) -> tuple[Path, list[TrialSpec]]:
    sweep_path = Path(sweep_path)
    sweep_config, trials = expand_sweep(sweep_path)
    sweep_name = sweep_config["sweep"]["name"]
    suite_dir = ensure_dir(out_dir or Path("outputs") / "experiments" / sweep_name)
    config_dir = ensure_dir(suite_dir / "_trial_configs")
    for trial in trials:
        write_yaml(config_dir / f"{trial.name}.yaml", trial.config)
    write_json(
        suite_dir / "sweep_manifest.json",
        {
            "sweep_name": sweep_name,
            "sweep_config": str(sweep_path),
            "trial_count": len(trials),
            "trials": [
                {
                    "name": trial.name,
                    "config": str(config_dir / f"{trial.name}.yaml"),
                    "axes": trial.axes,
                    "seed": trial.seed,
                    "fold": trial.fold,
                }
                for trial in trials
            ],
        },
    )
    return suite_dir, trials


def run_sweep(sweep_path: str | Path, dry_run: bool = False) -> Path:
    sweep_path = Path(sweep_path)
    sweep_config = read_yaml(sweep_path)
    sweep = sweep_config["sweep"]
    suite_dir, trials = write_trial_configs(sweep_path)
    config_dir = suite_dir / "_trial_configs"
    continue_on_error = bool(sweep.get("execution", {}).get("continue_on_error", False))

    statuses = []
    for trial in trials:
        config_path = config_dir / f"{trial.name}.yaml"
        command = [sys.executable, "-m", "wafer_repro.train", "--config", str(config_path)]
        print("Running:", " ".join(command), flush=True)
        if dry_run:
            statuses.append({"trial": trial.name, "status": "dry_run", "config": str(config_path)})
            continue
        result = subprocess.run(command)
        status = "completed" if result.returncode == 0 else "failed"
        statuses.append({"trial": trial.name, "status": status, "returncode": result.returncode, "config": str(config_path)})
        write_json(suite_dir / "sweep_status.json", {"trials": statuses})
        if result.returncode != 0 and not continue_on_error:
            raise subprocess.CalledProcessError(result.returncode, command)
    write_json(suite_dir / "sweep_status.json", {"trials": statuses})
    return suite_dir

