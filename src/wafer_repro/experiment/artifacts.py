from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wafer_repro.core.config import get_path


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def build_data_identity(config: dict[str, Any]) -> dict[str, Any]:
    data_module = get_path(config, "data.module", default="unknown")
    source = get_path(config, "data.source", default={})
    raw_path = source.get("path") or source.get("root")
    include_sha256 = bool(get_path(config, "runtime.hash_data_files", default=False))

    identity: dict[str, Any] = {
        "module": data_module,
        "source": source,
        "path": raw_path,
        "exists": False,
        "kind": None,
    }
    if not raw_path:
        return identity

    path = Path(raw_path)
    identity["resolved_path"] = str(path.resolve()) if path.exists() else str(path)
    identity["exists"] = path.exists()
    if not path.exists():
        return identity

    if path.is_file():
        identity.update(
            {
                "kind": "file",
                "size_bytes": path.stat().st_size,
                "modified_time_utc": _mtime_iso(path),
            }
        )
        if include_sha256:
            identity["sha256"] = _sha256_file(path)
        return identity

    if path.is_dir():
        files = [item for item in path.rglob("*") if item.is_file()]
        identity.update(
            {
                "kind": "directory",
                "file_count": len(files),
                "size_bytes": sum(item.stat().st_size for item in files),
                "modified_time_utc": _mtime_iso(path),
            }
        )
    return identity


def build_preprocessing_manifest(config: dict[str, Any]) -> dict[str, Any]:
    data = get_path(config, "data", default={})
    return {
        "module": data.get("module"),
        "preprocessing": data.get("preprocessing", {}),
        "augmentation": data.get("augmentation", {}),
        "dataloader": data.get("dataloader", {}),
        "split": data.get("split", {}),
    }


def file_sha256(path: str | Path) -> str:
    return _sha256_file(Path(path))


def build_split_hashes(split_dir: str | Path) -> dict[str, dict[str, Any]]:
    split_dir = Path(split_dir)
    hashes: dict[str, dict[str, Any]] = {}
    for path in sorted(split_dir.glob("*.csv")):
        hashes[path.stem] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return hashes
