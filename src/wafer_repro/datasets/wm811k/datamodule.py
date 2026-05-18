from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from wafer_repro.datasets.wm811k.records import augment_training_records, sample_per_class
from wafer_repro.datasets.wm811k.source import load_lswmd
from wafer_repro.datasets.wm811k.split import (
    make_external_test_split,
    make_kfold_splits,
    make_predefined_file_split,
    make_single_split,
)


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

