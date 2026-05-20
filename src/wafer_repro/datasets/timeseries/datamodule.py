from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from wafer_repro.datasets.base import DataBundle
from wafer_repro.datasets.registry import DATA_MODULE_REGISTRY


def _stratify_or_none(labels: pd.Series):
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def _split_records(records: pd.DataFrame, test_size: float, val_fraction: float, seed: int):
    trainval, test = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=_stratify_or_none(records["label"]),
    )
    train, val = train_test_split(
        trainval,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=_stratify_or_none(trainval["label"]),
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class TimeSeriesRecordsDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        records: pd.DataFrame,
        feature_columns: list[str],
        label_map: dict[str, int],
    ) -> None:
        self.frame = frame
        self.records = records.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records.iloc[index]
        row = self.frame.iloc[int(record["row_id"])]
        values = row[self.feature_columns].to_numpy(dtype="float32")
        tensor = torch.from_numpy(values).unsqueeze(0)
        target = torch.tensor(self.label_map[str(record["label"])], dtype=torch.long)
        return tensor, target


def _record_counts(records: pd.DataFrame, labels: tuple[str, ...]) -> dict[str, int]:
    return records["label"].value_counts().reindex(labels).fillna(0).astype(int).to_dict()


@DATA_MODULE_REGISTRY.register("timeseries_window")
def build_timeseries_bundle(config: dict[str, Any]) -> DataBundle:
    data_config = config.get("data", {})
    source_config = data_config.get("source", {})
    path = source_config.get("path")
    if not path:
        raise ValueError("timeseries_window requires data.source.path.")

    frame = pd.read_csv(path)
    label_column = source_config.get("label_column", "label")
    id_column = source_config.get("id_column", "sample_id")
    feature_prefix = source_config.get("feature_prefix", "x_")
    feature_columns = [column for column in frame.columns if str(column).startswith(feature_prefix)]
    if not feature_columns:
        raise ValueError(f"No time-series feature columns found with prefix {feature_prefix!r}.")
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")

    labels = tuple(sorted(str(value) for value in frame[label_column].unique()))
    records = pd.DataFrame(
        {
            "row_id": range(len(frame)),
            "label": frame[label_column].astype(str),
            "augmented": False,
            "aug_seq": 0,
        }
    )
    if id_column in frame.columns:
        records[id_column] = frame[id_column]

    split_config = data_config.get("split", {})
    strategy = split_config.get("strategy", "stratified_holdout")
    if strategy not in {"stratified_holdout", "single", "single_6_2_2"}:
        raise ValueError(f"timeseries_window currently supports stratified_holdout only, got: {strategy}")
    train_base, val_records, test_records = _split_records(
        records,
        test_size=float(split_config.get("test_size", 0.2)),
        val_fraction=float(split_config.get("val_fraction_of_trainval", 0.25)),
        seed=int(split_config.get("seed", 42)),
    )
    train_records = train_base.copy()

    label_map = {label: index for index, label in enumerate(labels)}
    train_dataset = TimeSeriesRecordsDataset(frame, train_records, feature_columns, label_map)
    val_dataset = TimeSeriesRecordsDataset(frame, val_records, feature_columns, label_map)
    test_dataset = TimeSeriesRecordsDataset(frame, test_records, feature_columns, label_map)

    data_summary = {
        "raw_records": int(len(records)),
        "split_strategy": "single_6_2_2",
        "feature_count": len(feature_columns),
        "train_base_records": int(len(train_base)),
        "train_records_after_augmentation": int(len(train_records)),
        "val_records": int(len(val_records)),
        "test_records": int(len(test_records)),
        "class_counts_raw": _record_counts(records, labels),
        "class_counts_train_augmented": _record_counts(train_records, labels),
        "class_counts_val": _record_counts(val_records, labels),
        "class_counts_test": _record_counts(test_records, labels),
    }
    return DataBundle(
        labels=labels,
        train_base=train_base,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        data_summary=data_summary,
        split_strategy="single_6_2_2",
        metadata={"feature_columns": feature_columns, "label_column": label_column},
    )
