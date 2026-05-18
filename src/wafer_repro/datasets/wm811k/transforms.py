from __future__ import annotations

import numpy as np
from torchvision import transforms


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

