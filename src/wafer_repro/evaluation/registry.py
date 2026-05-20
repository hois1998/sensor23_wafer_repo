from __future__ import annotations

from typing import Any

from wafer_repro.core.registry import Registry


EVALUATOR_REGISTRY: Registry[type] = Registry("evaluator")


def create_evaluator(evaluator_type: str, **kwargs: Any):
    import wafer_repro.evaluation.classification  # noqa: F401

    evaluator_cls = EVALUATOR_REGISTRY.get(evaluator_type)
    return evaluator_cls(**kwargs)
