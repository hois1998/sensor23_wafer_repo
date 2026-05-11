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


def save_evaluation(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    out_dir: str | Path,
    prefix: str = "",
) -> dict[str, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""

    y_pred = probabilities.argmax(axis=1)
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
    ax.set_title("WM-811K Confusion Matrix")
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
    ax.set_title("WM-811K Normalized Confusion Matrix")
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

