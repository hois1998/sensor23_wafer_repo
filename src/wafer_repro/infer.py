from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from wafer_repro.data import wafer_to_rgb_array
from wafer_repro.inference.inputs import load_inference_input
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


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    labels = tuple(checkpoint.get("labels", PAPER_CLASSES))
    device_choice = choose_device(args.device)

    data_module = config.get("data_module", "wm811k")
    inference_input = load_inference_input(args, config, data_module)
    tensor = inference_input.tensor.to(device_choice.device)

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
        **inference_input.metadata,
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
        if inference_input.source_image is not None:
            ax.imshow(inference_input.source_image)
        else:
            rgb = wafer_to_rgb_array(inference_input.wafer_map, config.get("channel_mode", "colormap"))
            ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(result["top_k"][0]["label"])
        fig.tight_layout()
        fig.savefig(out_dir / "prediction.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
