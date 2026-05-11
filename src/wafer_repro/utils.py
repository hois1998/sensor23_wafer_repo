from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DeviceChoice:
    name: str
    backend: str
    device: Any


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def choose_device(requested: str = "auto") -> DeviceChoice:
    import torch

    requested = requested.lower()
    if requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        device = torch.device("cuda")
        return DeviceChoice(torch.cuda.get_device_name(device), "cuda", device)

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available.")
        return DeviceChoice("Apple MPS", "mps", torch.device("mps"))

    if requested in {"directml", "dml"}:
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError("DirectML was requested. Install torch-directml first.") from exc
        device = torch_directml.device()
        return DeviceChoice(str(device), "directml", device)

    if requested == "cpu":
        return DeviceChoice(os.environ.get("PROCESSOR_IDENTIFIER", "CPU"), "cpu", torch.device("cpu"))

    if requested != "auto":
        raise ValueError(f"Unknown device request: {requested}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        return DeviceChoice(torch.cuda.get_device_name(device), "cuda", device)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceChoice("Apple MPS", "mps", torch.device("mps"))
    return DeviceChoice(os.environ.get("PROCESSOR_IDENTIFIER", "CPU"), "cpu", torch.device("cpu"))


def amp_is_enabled(device: DeviceChoice, requested: bool) -> bool:
    return bool(requested and device.backend == "cuda")

