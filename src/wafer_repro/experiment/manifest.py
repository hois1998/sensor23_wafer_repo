from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from wafer_repro.utils import write_json


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> None:
    write_json(Path(run_dir) / "run_manifest.json", manifest)

