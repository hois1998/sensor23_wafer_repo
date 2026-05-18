from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_image_folder(root: str | Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image folder root was not found: {root}")

    labels = tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))
    if not labels:
        raise ValueError(f"No class folders found under {root}")

    rows = []
    for label in labels:
        class_dir = root / label
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"image_path": str(image_path), "label": label})
    if not rows:
        raise ValueError(f"No images found under {root}")

    records = pd.DataFrame(rows)
    records["row_id"] = range(len(records))
    records["augmented"] = False
    records["aug_seq"] = 0
    return records, labels


def _stratify_or_none(labels: pd.Series):
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def split_records(
    records: pd.DataFrame,
    test_size: float,
    val_fraction_of_trainval: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def build_image_transform(image_size: int, train: bool, augmentation: bool):
    if train and augmentation:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ]
    )


class ImageFolderRecordsDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        label_map: dict[str, int],
        image_size: int,
        train: bool = False,
        augmentation: bool = True,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.label_map = label_map
        self.transform = build_image_transform(image_size, train=train, augmentation=augmentation)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch

        record = self.records.iloc[index]
        image = Image.open(record["image_path"]).convert("RGB")
        tensor = self.transform(image)
        target = torch.tensor(self.label_map[str(record["label"])], dtype=torch.long)
        return tensor, target


def record_counts(records: pd.DataFrame, labels: tuple[str, ...]) -> dict[str, int]:
    return records["label"].value_counts().reindex(labels).fillna(0).astype(int).to_dict()


@dataclass
class ImageFolderBundle:
    labels: tuple[str, ...]
    train_base: pd.DataFrame
    train_records: pd.DataFrame
    val_records: pd.DataFrame
    test_records: pd.DataFrame
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    data_summary: dict[str, Any]
    split_strategy: str


def build_image_folder_bundle(config: dict[str, Any]) -> ImageFolderBundle:
    data_config = config.get("data", {})
    source_config = data_config.get("source", {})
    root = source_config.get("path") or source_config.get("root")
    if not root:
        raise ValueError("image_folder data.source.path or data.source.root is required.")

    records, labels = scan_image_folder(root)
    split_config = data_config.get("split", {})
    strategy = split_config.get("strategy", "stratified_holdout")
    if strategy not in {"stratified_holdout", "single", "single_6_2_2"}:
        raise ValueError(f"image_folder currently supports stratified_holdout only, got: {strategy}")
    train_base, val_records, test_records = split_records(
        records,
        test_size=float(split_config.get("test_size", 0.2)),
        val_fraction_of_trainval=float(split_config.get("val_fraction_of_trainval", 0.25)),
        seed=int(split_config.get("seed", 42)),
    )
    train_records = train_base.copy()

    label_map = {label: index for index, label in enumerate(labels)}
    preprocessing = data_config.get("preprocessing", {})
    augmentation = data_config.get("augmentation", {})
    image_size = int(preprocessing.get("image_size", 64))
    augmentation_enabled = bool(augmentation.get("enabled", True))

    train_dataset = ImageFolderRecordsDataset(train_records, label_map, image_size, train=True, augmentation=augmentation_enabled)
    val_dataset = ImageFolderRecordsDataset(val_records, label_map, image_size, train=False, augmentation=False)
    test_dataset = ImageFolderRecordsDataset(test_records, label_map, image_size, train=False, augmentation=False)

    data_summary = {
        "raw_records": int(len(records)),
        "split_strategy": "single_6_2_2",
        "train_base_records": int(len(train_base)),
        "train_records_after_augmentation": int(len(train_records)),
        "val_records": int(len(val_records)),
        "test_records": int(len(test_records)),
        "class_counts_raw": record_counts(records, labels),
        "class_counts_train_augmented": record_counts(train_records, labels),
        "class_counts_val": record_counts(val_records, labels),
        "class_counts_test": record_counts(test_records, labels),
    }
    return ImageFolderBundle(
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
    )

