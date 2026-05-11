from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from wafer_repro.labels import DEFECT_CLASSES, LABEL_ALIASES, PAPER_CLASSES

WAFER_COLUMN_CANDIDATES = ("waferMap", "WaferMap", "wafer_map")
FAILURE_COLUMN_CANDIDATES = ("failureType", "FailureType", "failure_type")
SPLIT_COLUMN_CANDIDATES = ("trianTestLabel", "trainTestLabel", "TrainTestLabel")


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
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
    wafer_col = _find_column(df.columns, WAFER_COLUMN_CANDIDATES)
    failure_col = _find_column(df.columns, FAILURE_COLUMN_CANDIDATES)

    df = df.copy()
    df["original_index"] = np.arange(len(df), dtype=np.int64)
    df["failure_label"] = df[failure_col].map(normalize_failure_label)
    labeled = df[df["failure_label"].isin(PAPER_CLASSES)].reset_index(drop=True)
    labeled["labeled_index"] = np.arange(len(labeled), dtype=np.int64)
    if labeled.empty:
        raise ValueError("No labeled WM-811K rows were found after filtering failureType.")
    return labeled, wafer_col


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
    return pd.DataFrame(
        {
            "row_id": np.arange(len(df), dtype=np.int64),
            "label": df["failure_label"].to_numpy(),
            "augmented": np.zeros(len(df), dtype=bool),
            "aug_seq": np.zeros(len(df), dtype=np.int64),
        }
    )


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
    records["augmented"] = records["augmented"].astype(bool)
    records["row_id"] = records["row_id"].astype(np.int64)
    records["aug_seq"] = records["aug_seq"].astype(np.int64)
    return records


def wafer_to_rgb_array(wafer_map, channel_mode: str = "colormap") -> np.ndarray:
    array = np.asarray(wafer_map, dtype=np.float32)
    if array.ndim != 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D wafer map, got shape {array.shape}.")

    if channel_mode == "replicate":
        normalized = np.clip(array / 2.0, 0.0, 1.0)
        rgb = np.repeat(normalized[..., None], 3, axis=-1)
        return (rgb * 255.0).astype(np.uint8)

    if channel_mode != "colormap":
        raise ValueError(f"Unknown channel mode: {channel_mode}")

    rgb = np.zeros((*array.shape, 3), dtype=np.float32)
    pass_mask = np.isclose(array, 1.0)
    fail_mask = array >= 1.5
    rgb[pass_mask] = (0.0, 0.75, 0.15)
    rgb[fail_mask] = (1.0, 0.9, 0.0)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def build_transform(
    image_size: int,
    train: bool,
    augmentation: bool,
    rotation_degrees: float,
    crop_padding: int,
    blur_prob: float,
    erase_prob: float,
):
    if train and augmentation:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.RandomCrop(image_size, padding=crop_padding, padding_mode="constant"),
                transforms.RandomRotation(rotation_degrees, fill=0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=blur_prob),
                transforms.ToTensor(),
                transforms.RandomErasing(p=erase_prob, scale=(0.005, 0.04), ratio=(0.3, 3.3), value=0),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ]
    )


class WaferMapDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        wafer_col: str,
        records: pd.DataFrame,
        label_map: dict[str, int],
        image_size: int = 224,
        train: bool = False,
        augmentation: bool = True,
        channel_mode: str = "colormap",
        rotation_degrees: float = 180.0,
        crop_padding: int = 16,
        blur_prob: float = 0.2,
        erase_prob: float = 0.25,
    ) -> None:
        self.df = df
        self.wafer_col = wafer_col
        self.records = records.reset_index(drop=True)
        self.label_map = label_map
        self.channel_mode = channel_mode
        self.transform = build_transform(
            image_size=image_size,
            train=train,
            augmentation=augmentation,
            rotation_degrees=rotation_degrees,
            crop_padding=crop_padding,
            blur_prob=blur_prob,
            erase_prob=erase_prob,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch

        record = self.records.iloc[index]
        row = self.df.iloc[int(record["row_id"])]
        image = Image.fromarray(wafer_to_rgb_array(row[self.wafer_col], self.channel_mode), mode="RGB")
        tensor = self.transform(image)
        target = torch.tensor(self.label_map[str(record["label"])], dtype=torch.long)
        return tensor, target


def make_inference_tensor(wafer_map, image_size: int, channel_mode: str = "colormap"):
    import torch

    image = Image.fromarray(wafer_to_rgb_array(wafer_map, channel_mode), mode="RGB")
    transform = build_transform(
        image_size=image_size,
        train=False,
        augmentation=False,
        rotation_degrees=0.0,
        crop_padding=0,
        blur_prob=0.0,
        erase_prob=0.0,
    )
    return transform(image).unsqueeze(0).to(dtype=torch.float32)
