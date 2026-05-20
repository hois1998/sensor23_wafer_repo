from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wafer_repro.core.config import MISSING, get_path, validate_fixed_controls
from wafer_repro.training.callbacks import monitor_to_history_key


HISTORY_KEYS = {
    "train_loss",
    "train_accuracy",
    "train_macro_f1",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
}
OPTIMIZER_NAMES = {"adam", "adamw", "sgd"}
SCHEDULER_NAMES = {"none", "step_lr", "cosine_annealing", "reduce_on_plateau"}
MODES = {"max", "min"}
RUNTIME_DEVICES = {"auto", "cuda", "mps", "directml", "dml", "cpu"}
WM811K_SPLITS = {"stratified_holdout", "single", "single_6_2_2", "stratified_kfold", "paper_kfold", "predefined_files", "external_test_with_train_val_split"}
IMAGE_FOLDER_SPLITS = {"stratified_holdout", "single", "single_6_2_2"}


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class ConfigValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        super().__init__(format_validation_issues(issues))


def format_validation_issues(issues: list[ValidationIssue]) -> str:
    details = "\n".join(f"- {issue.path}: {issue.message}" for issue in issues)
    return f"Config validation failed:\n{details}"


def _add_missing(issues: list[ValidationIssue], path: str) -> None:
    issues.append(ValidationIssue(path, "required value is missing"))


def _get(config: dict[str, Any], path: str, issues: list[ValidationIssue]) -> Any:
    value = get_path(config, path, default=MISSING)
    if value is MISSING:
        _add_missing(issues, path)
    return value


def _validate_choice(
    issues: list[ValidationIssue],
    config: dict[str, Any],
    path: str,
    choices: set[str] | tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    value = get_path(config, path, default=MISSING)
    if value is MISSING:
        if required:
            _add_missing(issues, path)
        return None
    normalized = str(value).lower()
    if normalized not in set(choices):
        issues.append(ValidationIssue(path, f"unknown value {value!r}; available: {', '.join(sorted(choices))}"))
    return normalized


def _validate_positive_int(issues: list[ValidationIssue], config: dict[str, Any], path: str, *, required: bool = False) -> None:
    value = get_path(config, path, default=MISSING)
    if value is MISSING:
        if required:
            _add_missing(issues, path)
        return
    try:
        if int(value) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        issues.append(ValidationIssue(path, f"expected a positive integer, got {value!r}"))


def _validate_fraction(issues: list[ValidationIssue], config: dict[str, Any], path: str, *, required: bool = False) -> None:
    value = get_path(config, path, default=MISSING)
    if value is MISSING:
        if required:
            _add_missing(issues, path)
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        issues.append(ValidationIssue(path, f"expected a float in (0, 1), got {value!r}"))
        return
    if numeric <= 0.0 or numeric >= 1.0:
        issues.append(ValidationIssue(path, f"expected a float in (0, 1), got {value!r}"))


def _validate_path_exists(issues: list[ValidationIssue], config: dict[str, Any], path: str, *, kind: str) -> None:
    value = get_path(config, path, default=MISSING)
    if value is MISSING:
        _add_missing(issues, path)
        return
    resolved = Path(value)
    if kind == "file" and not resolved.is_file():
        issues.append(ValidationIssue(path, f"file does not exist: {value}"))
    elif kind == "dir" and not resolved.is_dir():
        issues.append(ValidationIssue(path, f"directory does not exist: {value}"))
    elif kind == "path" and not resolved.exists():
        issues.append(ValidationIssue(path, f"path does not exist: {value}"))


def _validate_monitor(issues: list[ValidationIssue], config: dict[str, Any], path: str, *, required: bool = False) -> None:
    value = get_path(config, path, default=MISSING)
    if value is MISSING:
        if required:
            _add_missing(issues, path)
        return
    key = monitor_to_history_key(str(value))
    if key not in HISTORY_KEYS:
        allowed = ", ".join(sorted(key.replace("_", "/") for key in HISTORY_KEYS))
        issues.append(ValidationIssue(path, f"monitor {value!r} is not produced by the current trainer; available: {allowed}"))


def _registered_task_names() -> tuple[str, ...]:
    import wafer_repro.tasks.classification  # noqa: F401
    from wafer_repro.tasks.registry import TASK_REGISTRY

    return TASK_REGISTRY.keys()


def _registered_trainer_names() -> tuple[str, ...]:
    import wafer_repro.training.supervised  # noqa: F401
    from wafer_repro.training.registry import TRAINER_REGISTRY

    return TRAINER_REGISTRY.keys()


def _registered_model_names() -> tuple[str, ...]:
    from wafer_repro.models import MODEL_REGISTRY

    return MODEL_REGISTRY.keys()


def _registered_data_module_names() -> tuple[str, ...]:
    from wafer_repro.datasets.registry import registered_data_module_names

    return registered_data_module_names()


def validate_experiment_config(
    config: dict[str, Any],
    *,
    check_paths: bool = False,
    raise_on_error: bool = True,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    schema_version = _get(config, "schema_version", issues)
    if schema_version is not MISSING and schema_version != 1:
        issues.append(ValidationIssue("schema_version", f"unsupported schema version {schema_version!r}; expected 1"))

    for section in ("experiment", "data", "task", "model", "train", "runtime"):
        if not isinstance(config.get(section), dict):
            issues.append(ValidationIssue(section, "required mapping is missing"))

    try:
        validate_fixed_controls(config)
    except ValueError as exc:
        issues.append(ValidationIssue("fixed.controls", str(exc)))

    data_module = _validate_choice(issues, config, "data.module", _registered_data_module_names())
    model_name = _validate_choice(issues, config, "model.name", _registered_model_names())
    task_type = _validate_choice(issues, config, "task.type", _registered_task_names())
    trainer_name = _validate_choice(issues, config, "train.trainer", _registered_trainer_names())
    _ = (model_name, task_type, trainer_name)

    _validate_choice(issues, config, "train.optimizer.name", OPTIMIZER_NAMES)
    scheduler_name = _validate_choice(issues, config, "train.scheduler.name", SCHEDULER_NAMES, required=False) or "none"
    _validate_choice(issues, config, "runtime.device", RUNTIME_DEVICES, required=False)
    _validate_choice(issues, config, "task.loss.class_weights", {"none", "balanced"}, required=False)

    _validate_positive_int(issues, config, "train.max_epochs", required=True)
    _validate_positive_int(issues, config, "data.dataloader.batch_size", required=True)
    _validate_positive_int(issues, config, "data.split.n_splits", required=False)
    _validate_fraction(issues, config, "data.split.test_size", required=False)
    _validate_fraction(issues, config, "data.split.val_fraction_of_trainval", required=False)

    split_strategy = str(get_path(config, "data.split.strategy", default="stratified_holdout"))
    if data_module == "wm811k" and split_strategy not in WM811K_SPLITS:
        issues.append(ValidationIssue("data.split.strategy", f"WM-811K does not support split strategy {split_strategy!r}"))
    if data_module == "image_folder" and split_strategy not in IMAGE_FOLDER_SPLITS:
        issues.append(ValidationIssue("data.split.strategy", f"image_folder currently supports: {', '.join(sorted(IMAGE_FOLDER_SPLITS))}"))

    if split_strategy == "predefined_files":
        for name in ("train", "val", "test"):
            split_path = f"data.split.files.{name}"
            if check_paths:
                _validate_path_exists(issues, config, split_path, kind="file")
            else:
                _get(config, split_path, issues)

    if split_strategy == "external_test_with_train_val_split":
        external_path = "data.split.external_test.path"
        if check_paths:
            _validate_path_exists(issues, config, external_path, kind="file")
        else:
            _get(config, external_path, issues)

    if check_paths:
        if data_module == "wm811k":
            _validate_path_exists(issues, config, "data.source.path", kind="file")
        elif data_module == "image_folder":
            root = get_path(config, "data.source.path", default=get_path(config, "data.source.root", default=MISSING))
            if root is MISSING:
                issues.append(ValidationIssue("data.source.path", "image_folder requires data.source.path or data.source.root"))
            elif not Path(root).is_dir():
                issues.append(ValidationIssue("data.source.path", f"directory does not exist: {root}"))

    _validate_choice(issues, config, "train.checkpoint.mode", MODES, required=False)
    _validate_monitor(issues, config, "train.checkpoint.monitor", required=False)

    early_enabled = bool(get_path(config, "train.early_stopping.enabled", default=False))
    if early_enabled:
        _validate_choice(issues, config, "train.early_stopping.mode", MODES, required=False)
        _validate_positive_int(issues, config, "train.early_stopping.patience", required=False)
        _validate_monitor(issues, config, "train.early_stopping.monitor", required=True)

    if scheduler_name == "reduce_on_plateau":
        _validate_choice(issues, config, "train.scheduler.mode", MODES, required=False)
        _validate_monitor(issues, config, "train.scheduler.monitor", required=True)

    if issues and raise_on_error:
        raise ConfigValidationError(issues)
    return issues


def _iter_sweep_method_sets(sweep: dict[str, Any]):
    expansion = sweep.get("expansion", {})
    mode = expansion.get("mode", "grid")
    if mode == "manual":
        for trial in sweep.get("trials", []):
            yield f"trial:{trial.get('name', '<unnamed>')}", trial.get("set", {})
        return
    for axis_name, methods in sweep.get("axes", {}).items():
        for method in methods:
            yield f"axis:{axis_name}/{method.get('name', '<unnamed>')}", method.get("set", {})


def _flatten_dotted_mapping(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        dotted_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_dotted_mapping(value, dotted_key))
        else:
            flattened[dotted_key] = value
    return flattened


def validate_sweep_config(
    config: dict[str, Any],
    *,
    raise_on_error: bool = True,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    sweep = config.get("sweep")
    if not isinstance(sweep, dict):
        issues.append(ValidationIssue("sweep", "required mapping is missing"))
        if raise_on_error:
            raise ConfigValidationError(issues)
        return issues

    if not sweep.get("name"):
        _add_missing(issues, "sweep.name")
    if not sweep.get("base_config"):
        _add_missing(issues, "sweep.base_config")

    expansion_mode = str(sweep.get("expansion", {}).get("mode", "grid"))
    if expansion_mode not in {"grid", "manual"}:
        issues.append(ValidationIssue("sweep.expansion.mode", f"unsupported expansion mode {expansion_mode!r}; available: grid, manual"))

    fixed = sweep.get("fixed", {})
    if fixed is not None and not isinstance(fixed, dict):
        issues.append(ValidationIssue("sweep.fixed", "expected a mapping"))
        fixed = {}
    fixed_flat = _flatten_dotted_mapping(fixed)

    for owner, values in _iter_sweep_method_sets(sweep):
        if not isinstance(values, dict):
            issues.append(ValidationIssue(owner, "set must be a mapping"))
            continue
        for key, value in values.items():
            if key in fixed_flat and fixed_flat[key] != value:
                issues.append(
                    ValidationIssue(
                        f"{owner}.set.{key}",
                        f"overrides fixed value {fixed_flat[key]!r} with {value!r}",
                    )
                )

    if issues and raise_on_error:
        raise ConfigValidationError(issues)
    return issues
