from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs
    raise RuntimeError("YAML config support requires PyYAML. Install project dependencies first.") from exc


MISSING = object()
_NO_DEFAULT = object()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_path(payload: dict[str, Any], dotted_path: str, default: Any = _NO_DEFAULT) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            if default is _NO_DEFAULT:
                raise KeyError(dotted_path)
            return default
        current = current[part]
    return current


def set_path(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = payload
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def parse_override(override: str) -> tuple[str, Any]:
    if "=" not in override:
        raise ValueError(f"Config override must be KEY=VALUE, got: {override}")
    key, raw_value = override.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Config override key cannot be empty: {override}")
    value = yaml.safe_load(raw_value)
    return key, value


def apply_overrides(config: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for override in overrides or []:
        key, value = parse_override(override)
        set_path(updated, key, value)
    return updated


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level YAML payload must be a mapping: {path}")
    return payload


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    config = read_yaml(path)
    base_config = config.pop("base_config", None)
    if base_config:
        base_path = Path(base_config)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        config = deep_merge(load_config(base_path), config)
    config = apply_overrides(config, overrides)
    validate_fixed_controls(config)
    return config


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _drop_paths(payload: dict[str, Any], paths: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    for dotted_path in paths:
        current: Any = result
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            current = current.get(part) if isinstance(current, dict) else None
            if current is None:
                break
        if isinstance(current, dict):
            current.pop(parts[-1], None)
    return result


def config_hash(payload: dict[str, Any], exclude_paths: Iterable[str] | None = None) -> str:
    stable_payload = _drop_paths(payload, exclude_paths or [])
    encoded = json.dumps(stable_payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def validate_fixed_controls(config: dict[str, Any]) -> None:
    """Validate optional fixed controls.

    The initial platform implementation supports a conservative form:

    fixed:
      controls:
        data.split.seed: 42

    Each key under fixed.controls must exist in the resolved config and match
    exactly. This gives us a first-class place to lock comparison conditions
    without making the rest of the config schema heavy yet.
    """

    fixed = config.get("fixed")
    if not isinstance(fixed, dict):
        return
    controls = fixed.get("controls")
    if controls is None:
        return
    if not isinstance(controls, dict):
        raise ValueError("fixed.controls must be a mapping of dotted config paths to expected values.")

    mismatches = []
    for dotted_path, expected in controls.items():
        actual = get_path(config, dotted_path, default=MISSING)
        if actual is MISSING:
            mismatches.append(f"{dotted_path}: missing, expected {expected!r}")
        elif actual != expected:
            mismatches.append(f"{dotted_path}: expected {expected!r}, got {actual!r}")
    if mismatches:
        details = "\n".join(f"- {item}" for item in mismatches)
        raise ValueError(f"Fixed control validation failed:\n{details}")
