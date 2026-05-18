from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wafer_repro.labels import DEFECT_CLASSES, PAPER_CLASSES


def sample_per_class(df: pd.DataFrame, max_per_class: int | None, seed: int) -> pd.DataFrame:
    if max_per_class is None or max_per_class <= 0:
        return df.reset_index(drop=True)
    sampled = (
        df.groupby("failure_label", group_keys=False, observed=True)
        .apply(lambda group: group.sample(min(len(group), max_per_class), random_state=seed))
        .reset_index(drop=True)
    )
    sampled["labeled_index"] = np.arange(len(sampled), dtype=np.int64)
    return sampled


def base_records(df: pd.DataFrame) -> pd.DataFrame:
    records = pd.DataFrame(
        {
            "row_id": np.arange(len(df), dtype=np.int64),
            "label": df["failure_label"].to_numpy(),
            "augmented": np.zeros(len(df), dtype=bool),
            "aug_seq": np.zeros(len(df), dtype=np.int64),
        }
    )
    for column in ("original_index", "labeled_index"):
        if column in df.columns:
            records[column] = df[column].to_numpy()
    return records


def augment_training_records(
    records: pd.DataFrame,
    target_defect_count: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    if target_defect_count <= 0:
        return records.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    for label in PAPER_CLASSES:
        group = records[records["label"] == label].copy()
        if group.empty:
            continue
        pieces.append(group)
        if label not in DEFECT_CLASSES:
            continue
        extra_count = max(0, target_defect_count - len(group))
        if extra_count == 0:
            continue
        chosen = group.iloc[rng.integers(0, len(group), size=extra_count)].copy()
        chosen["augmented"] = True
        chosen["aug_seq"] = np.arange(1, extra_count + 1, dtype=np.int64)
        pieces.append(chosen)

    out = pd.concat(pieces, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def record_counts(records: pd.DataFrame) -> dict[str, int]:
    return records["label"].value_counts().reindex(PAPER_CLASSES).fillna(0).astype(int).to_dict()


def save_records(records: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(path, index=False)


def load_records(path: str | Path) -> pd.DataFrame:
    records = pd.read_csv(path, keep_default_na=False)
    if "augmented" in records.columns:
        records["augmented"] = records["augmented"].astype(bool)
    else:
        records["augmented"] = False
    if "row_id" in records.columns:
        records["row_id"] = records["row_id"].astype(np.int64)
    if "aug_seq" in records.columns:
        records["aug_seq"] = records["aug_seq"].astype(np.int64)
    else:
        records["aug_seq"] = 0
    for column in ("original_index", "labeled_index"):
        if column in records.columns:
            records[column] = records[column].astype(np.int64)
    return records

