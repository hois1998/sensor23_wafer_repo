from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from wafer_repro.datasets.wm811k.records import base_records, load_records


def _stratify_or_none(labels: pd.Series):
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def make_single_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_fraction_of_trainval: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = base_records(df)
    trainval, test = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=_stratify_or_none(records["label"]),
    )
    train, val = train_test_split(
        trainval,
        test_size=val_fraction_of_trainval,
        random_state=seed,
        shuffle=True,
        stratify=_stratify_or_none(trainval["label"]),
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_kfold_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    n_splits: int = 4,
    seed: int = 42,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    records = base_records(df)
    trainval, test = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=_stratify_or_none(records["label"]),
    )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    labels = trainval["label"].to_numpy()
    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(trainval)), labels)):
        train = trainval.iloc[train_idx].reset_index(drop=True)
        val = trainval.iloc[val_idx].reset_index(drop=True)
        splits.append((fold_id, train, val, test.reset_index(drop=True)))
    return splits


def make_predefined_file_split(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = load_records(train_path)
    val = load_records(val_path)
    test = load_records(test_path)
    _validate_disjoint_records(train, val, test, id_column="row_id")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_external_test_split(
    df: pd.DataFrame,
    external_test_path: str | Path,
    id_column: str = "original_index",
    val_fraction_of_trainval: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = base_records(df)
    if id_column not in records.columns:
        raise KeyError(f"External split id column '{id_column}' is not available in records.")

    external_ids = _load_id_values(external_test_path, id_column)
    record_ids = records[id_column].astype(str)
    is_test = record_ids.isin(external_ids)
    test = records[is_test].copy()
    trainval = records[~is_test].copy()
    if test.empty:
        raise ValueError(f"External test file did not match any records using id column '{id_column}'.")
    missing = external_ids.difference(set(record_ids.tolist()))
    if missing:
        examples = ", ".join(map(str, sorted(missing)[:5]))
        raise ValueError(f"External test file contains ids that are not in the dataset: {examples}")
    train, val = train_test_split(
        trainval,
        test_size=val_fraction_of_trainval,
        random_state=seed,
        shuffle=True,
        stratify=_stratify_or_none(trainval["label"]),
    )
    _validate_disjoint_records(train, val, test, id_column=id_column)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _load_id_values(path: str | Path, id_column: str) -> set:
    frame = pd.read_csv(path, keep_default_na=False)
    if id_column in frame.columns:
        values = frame[id_column]
    elif len(frame.columns) == 1:
        values = frame.iloc[:, 0]
    else:
        raise KeyError(f"Could not find id column '{id_column}' in {path}.")
    return set(values.astype(str).tolist())


def _validate_disjoint_records(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    id_column: str,
) -> None:
    if id_column not in train.columns or id_column not in val.columns or id_column not in test.columns:
        return
    train_ids = set(train[id_column].tolist())
    val_ids = set(val[id_column].tolist())
    test_ids = set(test[id_column].tolist())
    overlaps = {
        "train_val": train_ids & val_ids,
        "train_test": train_ids & test_ids,
        "val_test": val_ids & test_ids,
    }
    bad = {name: ids for name, ids in overlaps.items() if ids}
    if bad:
        details = ", ".join(f"{name}={len(ids)}" for name, ids in bad.items())
        raise ValueError(f"Split records must be disjoint by '{id_column}', but overlaps were found: {details}")
