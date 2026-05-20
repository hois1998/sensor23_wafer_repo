from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tqdm import tqdm

from wafer_repro.evaluation.registry import EVALUATOR_REGISTRY


@torch.no_grad()
def predict_probabilities(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for images, labels in tqdm(loader, desc="predict", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        y_true.append(labels.numpy())
        probabilities.append(probs.detach().cpu().numpy())
    return np.concatenate(y_true), np.concatenate(probabilities)


def _safe_probability_column(label: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in label).strip("_").lower()
    return f"prob_{safe or 'class'}"


def save_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    out_dir: str | Path,
    prefix: str = "",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""

    y_pred = probabilities.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true)),
            "true_index": y_true.astype(int),
            "true_label": [labels[int(index)] for index in y_true],
            "pred_index": y_pred.astype(int),
            "pred_label": [labels[int(index)] for index in y_pred],
            "pred_confidence": probabilities.max(axis=1),
        }
    )
    for index, label in enumerate(labels):
        frame[_safe_probability_column(label)] = probabilities[:, index]
    path = out_dir / f"{stem}predictions.csv"
    frame.to_csv(path, index=False)
    return path


def save_evaluation(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    out_dir: str | Path,
    prefix: str = "",
    predictions_dir: str | Path | None = None,
) -> dict[str, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""

    y_pred = probabilities.argmax(axis=1)
    save_predictions(y_true, probabilities, labels, predictions_dir or out_dir, prefix=prefix)

    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / f"{stem}classification_report.csv")
    (out_dir / f"{stem}classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_dir / f"{stem}confusion_matrix.csv")

    normalized = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(normalized, index=labels, columns=labels).to_csv(out_dir / f"{stem}confusion_matrix_normalized.csv")

    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
        values_format="d",
    )
    ax.set_title("Classification Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(normalized, display_labels=labels).plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
        values_format=".2f",
    )
    ax.set_title("Classification Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}confusion_matrix_normalized.png", dpi=180)
    plt.close(fig)

    summary = {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }
    (out_dir / f"{stem}summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@EVALUATOR_REGISTRY.register("classification")
class ClassificationEvaluator:
    def __init__(self, labels: tuple[str, ...]) -> None:
        self.labels = labels

    def predict_probabilities(self, model, loader, device) -> tuple[np.ndarray, np.ndarray]:
        return predict_probabilities(model, loader, device)

    def save(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        out_dir: str | Path,
        prefix: str = "",
        predictions_dir: str | Path | None = None,
    ) -> dict[str, float]:
        return save_evaluation(y_true, probabilities, self.labels, out_dir, prefix=prefix, predictions_dir=predictions_dir)
