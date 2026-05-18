from __future__ import annotations

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from wafer_repro.datasets.wm811k.transforms import build_transform, wafer_to_rgb_array


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

