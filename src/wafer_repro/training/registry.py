from __future__ import annotations

from typing import Any

from wafer_repro.core.registry import Registry

TRAINER_REGISTRY: Registry[type] = Registry("trainer")


def create_trainer(trainer_type: str, **kwargs: Any):
    # Import built-in trainer modules lazily so their registration side effects run.
    import wafer_repro.training.supervised  # noqa: F401

    trainer_cls = TRAINER_REGISTRY.get(trainer_type)
    return trainer_cls(**kwargs)

