from __future__ import annotations

from typing import Any

from wafer_repro.core.registry import Registry

TASK_REGISTRY: Registry[type] = Registry("task")


def create_task(task_type: str, **kwargs: Any):
    # Import built-in task modules lazily so their registration side effects run.
    import wafer_repro.tasks.classification  # noqa: F401

    task_cls = TASK_REGISTRY.get(task_type)
    return task_cls(**kwargs)

