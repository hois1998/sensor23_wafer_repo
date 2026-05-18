from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "Pillow",
    "scikit-learn",
    "matplotlib",
    "tqdm",
    "torch",
    "torchvision",
    "PyYAML",
)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_info(cwd: str | Path) -> dict[str, Any]:
    cwd = Path(cwd)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(status), "status_short": status}
    except Exception:
        return {"commit": None, "dirty": None, "status_short": None}


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"available": False}

    info: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return info


def capture_environment(cwd: str | Path) -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": _package_versions(),
        "torch": _torch_info(),
        "git": _git_info(cwd),
    }

