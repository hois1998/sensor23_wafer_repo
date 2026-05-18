from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from wafer_repro.labels import PAPER_CLASSES
from wafer_repro.tasks.registry import TASK_REGISTRY


@TASK_REGISTRY.register("classification")
@dataclass(frozen=True)
class ClassificationTask:
    labels: tuple[str, ...] = PAPER_CLASSES
    class_weights: str = "none"

    def build_criterion(self, train_records: pd.DataFrame, device) -> nn.Module:
        if self.class_weights == "none":
            return nn.CrossEntropyLoss()
        if self.class_weights != "balanced":
            raise ValueError(f"Unknown classification class_weights mode: {self.class_weights}")

        counts = train_records["label"].value_counts().reindex(self.labels).fillna(0).to_numpy(dtype=np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    def summarize_epoch(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        total_loss: float,
        total_correct: int,
        total_count: int,
    ) -> dict[str, float]:
        return {
            "loss": total_loss / max(total_count, 1),
            "accuracy": total_correct / max(total_count, 1),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
