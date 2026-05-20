from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wafer_repro.core.config import deep_merge, read_yaml, set_path, write_yaml
from wafer_repro.core.validation import validate_sweep_config
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

    axes = sweep.get("axes", {})
    include = expansion.get("include") or list(axes)
    selected_axes = [(axis_name, axes[axis_name]) for axis_name in include]

    if mode == "random":
        rng = random.Random(int(expansion.get("seed", 42)))
        num_trials = int(expansion.get("num_trials", expansion.get("n_trials", 1)))
        trials = []
        for trial_index in range(1, num_trials + 1):
            values: dict[str, Any] = {}
            axis_names: dict[str, str] = {}
            parts = [f"random={trial_index:03d}"]
            for axis_name, methods in selected_axes:
                method = rng.choice(methods)
                method_name = method["name"]
                values.update(method.get("set", {}))
                axis_names[axis_name] = method_name
                parts.append(f"{axis_name}={method_name}")
            trials.append(("__".join(parts), values, axis_names))
        return trials

    if mode != "grid":
        raise ValueError(f"Unsupported sweep expansion mode: {mode}")

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
    validate_sweep_config(sweep_config)
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
    trial_manifest_dir = ensure_dir(suite_dir / "_trial_manifests")
    for trial in trials:
        write_yaml(config_dir / f"{trial.name}.yaml", trial.config)
        write_json(
            trial_manifest_dir / f"{trial.name}.json",
            {
                "name": trial.name,
                "status": "pending",
                "config": str(config_dir / f"{trial.name}.yaml"),
                "run_dir": str(_trial_run_dir(trial)),
                "axes": trial.axes,
                "seed": trial.seed,
                "fold": trial.fold,
            },
        )
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


def _trial_run_dir(trial: TrialSpec) -> Path:
    runtime = trial.config.get("runtime", {})
    output_dir = runtime.get("output_dir")
    run_name = runtime.get("run_name") or trial.name
    if not output_dir:
        raise ValueError(f"Trial {trial.name!r} does not define runtime.output_dir.")
    return Path(output_dir) / run_name


def _read_run_manifest(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _status_summary(statuses: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {"total": len(statuses)}
    for status in statuses:
        key = str(status["status"])
        summary[key] = summary.get(key, 0) + 1
    return summary


def _write_sweep_status(suite_dir: Path, statuses: list[dict[str, Any]]) -> None:
    write_json(
        suite_dir / "sweep_status.json",
        {
            "summary": _status_summary(statuses),
            "trials": statuses,
        },
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_trial_status(suite_dir: Path, status: dict[str, Any]) -> None:
    trial_manifest_dir = ensure_dir(suite_dir / "_trial_manifests")
    write_json(trial_manifest_dir / f"{status['trial']}.json", status)


def _run_trial_process(trial: TrialSpec, config_path: Path, run_dir: Path, retry_failed: int) -> dict[str, Any]:
    command = [sys.executable, "-m", "wafer_repro.train", "--config", str(config_path)]
    attempts = []
    final_returncode = 1
    started_at = _now_iso()
    for attempt in range(1, retry_failed + 2):
        print("Running:", " ".join(command), f"(attempt {attempt})", flush=True)
        start = time.monotonic()
        result = subprocess.run(command)
        elapsed = time.monotonic() - start
        final_returncode = result.returncode
        attempts.append({"attempt": attempt, "returncode": result.returncode, "duration_seconds": elapsed})
        if result.returncode == 0:
            break
    status = "completed" if final_returncode == 0 else "failed"
    return {
        "trial": trial.name,
        "status": status,
        "returncode": final_returncode,
        "attempts": attempts,
        "started_at": started_at,
        "finished_at": _now_iso(),
        "config": str(config_path),
        "run_dir": str(run_dir),
        "axes": trial.axes,
        "seed": trial.seed,
        "fold": trial.fold,
    }


def run_sweep(sweep_path: str | Path, dry_run: bool = False, skip_completed: bool | None = None) -> Path:
    sweep_path = Path(sweep_path)
    sweep_config = read_yaml(sweep_path)
    sweep = sweep_config["sweep"]
    suite_dir, trials = write_trial_configs(sweep_path)
    config_dir = suite_dir / "_trial_configs"
    execution_config = sweep.get("execution", {})
    continue_on_error = bool(execution_config.get("continue_on_error", False))
    retry_failed = int(execution_config.get("retry_failed", 0))
    max_workers = max(1, int(execution_config.get("max_workers", 1)))
    if skip_completed is None:
        skip_completed = bool(execution_config.get("skip_completed", False))

    statuses = []
    runnable_trials: list[tuple[TrialSpec, Path, Path]] = []
    for trial in trials:
        config_path = config_dir / f"{trial.name}.yaml"
        run_dir = _trial_run_dir(trial)
        run_manifest = _read_run_manifest(run_dir)
        if skip_completed and run_manifest and run_manifest.get("status") == "completed":
            status = {
                "trial": trial.name,
                "status": "skipped_completed",
                "config": str(config_path),
                "run_dir": str(run_dir),
                "manifest": str(run_dir / "run_manifest.json"),
                "config_hash": run_manifest.get("config_hash"),
            }
            print(f"Skipping completed trial: {trial.name} ({run_dir})", flush=True)
            statuses.append(status)
            _write_trial_status(suite_dir, status)
            _write_sweep_status(suite_dir, statuses)
            continue

        if dry_run:
            status = {
                "trial": trial.name,
                "status": "dry_run",
                "config": str(config_path),
                "run_dir": str(run_dir),
                "axes": trial.axes,
                "seed": trial.seed,
                "fold": trial.fold,
            }
            statuses.append(status)
            _write_trial_status(suite_dir, status)
            continue
        runnable_trials.append((trial, config_path, run_dir))

    if dry_run:
        _write_sweep_status(suite_dir, statuses)
        return suite_dir

    if max_workers == 1:
        for trial, config_path, run_dir in runnable_trials:
            status = _run_trial_process(trial, config_path, run_dir, retry_failed)
            statuses.append(status)
            _write_trial_status(suite_dir, status)
            _write_sweep_status(suite_dir, statuses)
            if status["returncode"] != 0 and not continue_on_error:
                raise subprocess.CalledProcessError(status["returncode"], [sys.executable, "-m", "wafer_repro.train", "--config", str(config_path)])
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_trial = {
                executor.submit(_run_trial_process, trial, config_path, run_dir, retry_failed): (trial, config_path)
                for trial, config_path, run_dir in runnable_trials
            }
            failures = []
            for future in as_completed(future_to_trial):
                trial, config_path = future_to_trial[future]
                status = future.result()
                statuses.append(status)
                _write_trial_status(suite_dir, status)
                _write_sweep_status(suite_dir, statuses)
                if status["returncode"] != 0:
                    failures.append((status, config_path))
            if failures and not continue_on_error:
                status, config_path = failures[0]
                raise subprocess.CalledProcessError(status["returncode"], [sys.executable, "-m", "wafer_repro.train", "--config", str(config_path)])

    _write_sweep_status(suite_dir, statuses)
    return suite_dir
