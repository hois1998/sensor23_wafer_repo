from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from wafer_repro.datasets.base import DataBundle
from wafer_repro.datasets.registry import DATA_MODULE_REGISTRY
from wafer_repro.datasets.wm811k.dataset import WaferMapDataset
from wafer_repro.datasets.wm811k.records import augment_training_records, sample_per_class
from wafer_repro.datasets.wm811k.source import load_lswmd
from wafer_repro.datasets.wm811k.split import (
    make_external_test_split,
    make_kfold_splits,
    make_predefined_file_split,
    make_single_split,
)
from wafer_repro.labels import PAPER_CLASSES, label_to_index


@dataclass
class SplitBundle:
    strategy: str
    train_base: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class WM811KDataModule:
    config: dict[str, Any]

    def load_raw(self) -> tuple[pd.DataFrame, str]:
        data_path = self.config["data"]["source"]["path"]
        df, wafer_col = load_lswmd(data_path)
        max_per_class = self.config.get("data", {}).get("debug", {}).get("max_samples_per_class")
        seed = int(self.config.get("data", {}).get("split", {}).get("seed", 42))
        return sample_per_class(df, max_per_class, seed), wafer_col

    def split(self, df: pd.DataFrame) -> SplitBundle:
        split_config = self.config.get("data", {}).get("split", {})
        strategy = split_config.get("strategy", "stratified_holdout")
        seed = int(split_config.get("seed", 42))
        test_size = float(split_config.get("test_size", 0.2))
        val_fraction = float(split_config.get("val_fraction_of_trainval", 0.25))

        if strategy in {"stratified_holdout", "single", "single_6_2_2"}:
            train, val, test = make_single_split(df, test_size=test_size, val_fraction_of_trainval=val_fraction, seed=seed)
            return SplitBundle("single_6_2_2", train, val, test)

        if strategy in {"stratified_kfold", "paper_kfold"}:
            n_splits = int(split_config.get("n_splits", 4))
            fold_index = int(split_config.get("fold_index", 0))
            splits = make_kfold_splits(df, test_size=test_size, n_splits=n_splits, seed=seed)
            if fold_index < 0 or fold_index >= len(splits):
                raise ValueError(f"fold_index must be between 0 and {len(splits) - 1}.")
            fold_id, train, val, test = splits[fold_index]
            return SplitBundle(f"kfold_{fold_id}_of_{n_splits}", train, val, test)

        if strategy == "predefined_files":
            files = split_config.get("files", {})
            train, val, test = make_predefined_file_split(files["train"], files["val"], files["test"])
            return SplitBundle("predefined_files", train, val, test)

        if strategy == "external_test_with_train_val_split":
            external_test = split_config.get("external_test", {})
            test_path = external_test["path"] if isinstance(external_test, dict) else external_test
            id_column = external_test.get("id_column", "original_index") if isinstance(external_test, dict) else "original_index"
            train, val, test = make_external_test_split(
                df,
                Path(test_path),
                id_column=id_column,
                val_fraction_of_trainval=val_fraction,
                seed=seed,
            )
            return SplitBundle("external_test_with_train_val_split", train, val, test)

        raise ValueError(f"Unknown WM-811K split strategy: {strategy}")

    def augment_train(self, train_base: pd.DataFrame) -> pd.DataFrame:
        augmentation_config = self.config.get("data", {}).get("augmentation", {})
        if not augmentation_config.get("enabled", True):
            return train_base
        target_count = int(augmentation_config.get("target_defect_count", 10_000))
        seed = int(self.config.get("train", {}).get("seed", self.config.get("data", {}).get("split", {}).get("seed", 42)))
        return augment_training_records(train_base, target_defect_count=target_count, seed=seed)


def _record_counts(records: pd.DataFrame, labels: tuple[str, ...]) -> dict[str, int]:
    return records["label"].value_counts().reindex(labels).fillna(0).astype(int).to_dict()


@DATA_MODULE_REGISTRY.register("wm811k")
def build_wm811k_bundle(config: dict[str, Any]) -> DataBundle:
    data_module = WM811KDataModule(config)
    df, wafer_col = data_module.load_raw()
    split_bundle = data_module.split(df)
    train_records = data_module.augment_train(split_bundle.train_base)

    data_config = config.get("data", {})
    preprocessing = data_config.get("preprocessing", {})
    augmentation = data_config.get("augmentation", {})
    transforms_config = augmentation.get("transforms", {})
    random_rotation = transforms_config.get("random_rotation", {})
    random_crop = transforms_config.get("random_crop", {})
    gaussian_blur = transforms_config.get("gaussian_blur", {})
    random_erasing = transforms_config.get("random_erasing", {})

    labels = tuple(PAPER_CLASSES)
    label_map = label_to_index(labels)
    common_dataset_kwargs = {
        "df": df,
        "wafer_col": wafer_col,
        "label_map": label_map,
        "image_size": int(preprocessing.get("image_size", 224)),
        "channel_mode": preprocessing.get("channel_mode", "colormap"),
        "rotation_degrees": float(random_rotation.get("degrees", 180.0)),
        "crop_padding": int(random_crop.get("padding", 16)),
        "blur_prob": float(gaussian_blur.get("p", 0.2)),
        "erase_prob": float(random_erasing.get("p", 0.25)),
    }
    augmentation_enabled = bool(augmentation.get("enabled", True))
    train_dataset = WaferMapDataset(
        records=train_records,
        train=True,
        augmentation=augmentation_enabled,
        **common_dataset_kwargs,
    )
    val_dataset = WaferMapDataset(
        records=split_bundle.val,
        train=False,
        augmentation=False,
        **common_dataset_kwargs,
    )
    test_dataset = WaferMapDataset(
        records=split_bundle.test,
        train=False,
        augmentation=False,
        **common_dataset_kwargs,
    )

    data_summary = {
        "raw_labeled_rows_after_optional_sampling": int(len(df)),
        "split_strategy": split_bundle.strategy,
        "train_base_records": int(len(split_bundle.train_base)),
        "train_records_after_augmentation": int(len(train_records)),
        "val_records": int(len(split_bundle.val)),
        "test_records": int(len(split_bundle.test)),
        "class_counts_raw": _record_counts(split_bundle.train_base, labels) | {"__note__": "train_base only"},
        "class_counts_train_augmented": _record_counts(train_records, labels),
        "class_counts_val": _record_counts(split_bundle.val, labels),
        "class_counts_test": _record_counts(split_bundle.test, labels),
    }
    return DataBundle(
        labels=labels,
        train_base=split_bundle.train_base,
        train_records=train_records,
        val_records=split_bundle.val,
        test_records=split_bundle.test,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        data_summary=data_summary,
        split_strategy=split_bundle.strategy,
        metadata={"wafer_column": wafer_col},
    )
