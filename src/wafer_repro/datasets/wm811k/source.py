from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from wafer_repro.labels import LABEL_ALIASES, PAPER_CLASSES

WAFER_COLUMN_CANDIDATES = ("waferMap", "WaferMap", "wafer_map")
FAILURE_COLUMN_CANDIDATES = ("failureType", "FailureType", "failure_type")
SPLIT_COLUMN_CANDIDATES = ("trianTestLabel", "trainTestLabel", "TrainTestLabel")


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    columns = list(columns)
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise KeyError(f"Could not find one of {tuple(candidates)} in columns: {columns}")


def scalarize(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return scalarize(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return scalarize(value[0])
    text = str(value).strip().strip("'\"")
    if text == "" or text.lower() in {"nan", "none[]", "[]"}:
        return None
    return text


def normalize_failure_label(value) -> str | None:
    text = scalarize(value)
    if text is None:
        return None
    return LABEL_ALIASES.get(text.lower(), text)


def load_lswmd(path: str | Path) -> tuple[pd.DataFrame, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {path}")

    df = pd.read_pickle(path)
    wafer_col = find_column(df.columns, WAFER_COLUMN_CANDIDATES)
    failure_col = find_column(df.columns, FAILURE_COLUMN_CANDIDATES)

    df = df.copy()
    df["original_index"] = np.arange(len(df), dtype=np.int64)
    df["failure_label"] = df[failure_col].map(normalize_failure_label)
    labeled = df[df["failure_label"].isin(PAPER_CLASSES)].reset_index(drop=True)
    labeled["labeled_index"] = np.arange(len(labeled), dtype=np.int64)
    if labeled.empty:
        raise ValueError("No labeled WM-811K rows were found after filtering failureType.")
    return labeled, wafer_col

