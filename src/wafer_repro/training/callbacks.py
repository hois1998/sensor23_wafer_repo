from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def monitor_to_history_key(monitor: str) -> str:
    return monitor.replace("/", "_").replace(".", "_")


def resolve_monitor_value(row: dict[str, Any], monitor: str) -> float:
    key = monitor_to_history_key(monitor)
    if key not in row:
        choices = ", ".join(sorted(row))
        raise KeyError(f"Monitor {monitor!r} resolved to {key!r}, but history row has: {choices}")
    return float(row[key])


def is_improvement(value: float, best: float | None, mode: str, min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    if mode == "max":
        return value > best + min_delta
    if mode == "min":
        return value < best - min_delta
    raise ValueError(f"mode must be 'max' or 'min', got: {mode}")


@dataclass
class EarlyStopping:
    monitor: str
    mode: str = "max"
    patience: int = 10
    min_delta: float = 0.0
    best: float | None = None
    bad_epochs: int = 0
    stopped_epoch: int | None = None

    def update(self, value: float, epoch: int) -> bool:
        if is_improvement(value, self.best, self.mode, self.min_delta):
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            self.stopped_epoch = epoch
            return True
        return False


def build_early_stopper(config: dict[str, Any]) -> EarlyStopping | None:
    if not bool(config.get("enabled", False)):
        return None
    return EarlyStopping(
        monitor=str(config.get("monitor", "val/macro_f1")),
        mode=str(config.get("mode", "max")).lower(),
        patience=int(config.get("patience", 10)),
        min_delta=float(config.get("min_delta", 0.0)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    max_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    name = str(config.get("name", "none")).lower()
    if name in {"none", "null", ""}:
        return None
    if name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("step_size", 10)),
            gamma=float(config.get("gamma", 0.1)),
        )
    if name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("t_max") or max_epochs),
            eta_min=float(config.get("eta_min", 0.0)),
        )
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(config.get("mode", "max")).lower(),
            factor=float(config.get("factor", 0.1)),
            patience=int(config.get("patience", 10)),
            threshold=float(config.get("threshold", 1e-4)),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def step_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    config: dict[str, Any],
    row: dict[str, Any],
) -> None:
    if scheduler is None:
        return
    name = str(config.get("name", "none")).lower()
    if name == "reduce_on_plateau":
        monitor = str(config.get("monitor", "val/macro_f1"))
        scheduler.step(resolve_monitor_value(row, monitor))
        return
    scheduler.step()
