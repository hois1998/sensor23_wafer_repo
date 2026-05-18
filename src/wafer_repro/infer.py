from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from wafer_repro.data import load_lswmd, make_inference_tensor, sample_per_class, wafer_to_rgb_array
from wafer_repro.datasets.image_folder.datamodule import build_image_transform
from wafer_repro.labels import PAPER_CLASSES
from wafer_repro.models import create_model
from wafer_repro.utils import choose_device, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-sample inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="../LSWMD.pkl")
    parser.add_argument("--row-index", type=int, default=None, help="Index in the filtered labeled dataframe.")
    parser.add_argument("--original-index", type=int, default=None, help="Original WM-811K dataframe index.")
    parser.add_argument("--npy", default=None, help="Optional .npy wafer map path instead of a dataframe row.")
    parser.add_argument("--image", default=None, help="Optional image path for image_folder checkpoints.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "directml", "dml", "cpu"])
    return parser.parse_args()


def load_wafer_from_args(args, config):
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


def make_image_tensor(path: str, image_size: int):
    image = Image.open(path).convert("RGB")
    transform = build_image_transform(image_size=image_size, train=False, augmentation=False)
    return transform(image).unsqueeze(0).to(dtype=torch.float32), image


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    labels = tuple(checkpoint.get("labels", PAPER_CLASSES))
    device_choice = choose_device(args.device)

    data_module = config.get("data_module", "wm811k")
    source_image = None
    wafer_map = None
    if data_module == "image_folder" or args.image:
        if not args.image:
            raise ValueError("Provide --image for image_folder checkpoints.")
        tensor, source_image = make_image_tensor(args.image, image_size=int(config.get("image_size", 224)))
        tensor = tensor.to(device_choice.device)
        meta = {
            "source": str(args.image),
            "true_label": Path(args.image).parent.name if Path(args.image).parent else None,
        }
    elif data_module == "wm811k":
        wafer_map, meta = load_wafer_from_args(args, config)
        tensor = make_inference_tensor(
            wafer_map,
            image_size=int(config.get("image_size", 224)),
            channel_mode=config.get("channel_mode", "colormap"),
        ).to(device_choice.device)
    else:
        raise ValueError(f"Unsupported checkpoint data_module for infer: {data_module}")

    model = create_model(
        model_name=config["model"],
        num_classes=len(labels),
        pretrained=False,
        dropout=float(config.get("dropout", 0.35)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device_choice.device)
    model.eval()

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).detach().cpu().numpy()[0]

    order = probabilities.argsort()[::-1][: args.top_k]
    result = {
        **meta,
        "checkpoint": str(args.checkpoint),
        "model": config["model"],
        "top_k": [
            {"label": labels[idx], "probability": float(probabilities[idx])}
            for idx in order
        ],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.out_dir:
        out_dir = ensure_dir(args.out_dir)
        (out_dir / "prediction.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(4, 4))
        if source_image is not None:
            ax.imshow(source_image)
        else:
            rgb = wafer_to_rgb_array(wafer_map, config.get("channel_mode", "colormap"))
            ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(result["top_k"][0]["label"])
        fig.tight_layout()
        fig.savefig(out_dir / "prediction.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
