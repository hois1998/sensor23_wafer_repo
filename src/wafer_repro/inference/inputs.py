from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from wafer_repro.data import load_lswmd, make_inference_tensor, sample_per_class
from wafer_repro.datasets.image_folder.datamodule import build_image_transform


@dataclass
class InferenceInput:
    tensor: torch.Tensor
    metadata: dict[str, Any]
    source_image: Image.Image | None = None
    wafer_map: Any | None = None


def _load_wafer_from_args(args, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    if args.npy:
        wafer = np.load(args.npy)
        return wafer, {"source": str(args.npy), "true_label": None}

    df, wafer_col = load_lswmd(args.data)
    df = sample_per_class(df, config.get("max_samples_per_class"), int(config.get("seed", 42)))
    if args.original_index is not None:
        matches = df.index[df["original_index"] == args.original_index].to_list()
        if not matches:
            raise ValueError(f"Original index {args.original_index} is not present in the labeled dataframe.")
        row = df.iloc[matches[0]]
    else:
        if args.row_index is None:
            raise ValueError("Provide one of --row-index, --original-index, or --npy.")
        row = df.iloc[args.row_index]
    return row[wafer_col], {
        "source": "LSWMD.pkl",
        "row_index": int(row.name),
        "original_index": int(row["original_index"]),
        "true_label": str(row["failure_label"]),
    }


def _make_image_input(path: str, image_size: int) -> InferenceInput:
    image = Image.open(path).convert("RGB")
    transform = build_image_transform(image_size=image_size, train=False, augmentation=False)
    tensor = transform(image).unsqueeze(0).to(dtype=torch.float32)
    return InferenceInput(
        tensor=tensor,
        source_image=image,
        metadata={
            "source": str(path),
            "true_label": Path(path).parent.name if Path(path).parent else None,
        },
    )


def load_inference_input(args, config: dict[str, Any], data_module: str) -> InferenceInput:
    if data_module == "image_folder" or args.image:
        if not args.image:
            raise ValueError("Provide --image for image_folder checkpoints.")
        return _make_image_input(args.image, image_size=int(config.get("image_size", 224)))

    if data_module == "wm811k":
        wafer_map, metadata = _load_wafer_from_args(args, config)
        tensor = make_inference_tensor(
            wafer_map,
            image_size=int(config.get("image_size", 224)),
            channel_mode=config.get("channel_mode", "colormap"),
        )
        return InferenceInput(tensor=tensor, metadata=metadata, wafer_map=wafer_map)

    raise ValueError(f"Unsupported checkpoint data_module for infer: {data_module}")
