from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wafer_repro.data import WaferMapDataset, load_lswmd, load_records, sample_per_class
from wafer_repro.labels import PAPER_CLASSES, label_to_index
from wafer_repro.metrics import predict_probabilities, save_evaluation
from wafer_repro.models import create_model
from wafer_repro.utils import choose_device, ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained wafer classifier checkpoint.")
    parser.add_argument("--data", default="../LSWMD.pkl")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--split", default="test", choices=["train_base", "train_augmented", "val", "test"])
    parser.add_argument("--split-file", default=None, help="Override split CSV path.")
    parser.add_argument("--out-dir", default=None, help="Metric output directory. Defaults to the run metrics folder.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "directml", "dml", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    labels = tuple(checkpoint.get("labels", PAPER_CLASSES))

    device_choice = choose_device(args.device)
    df, wafer_col = load_lswmd(args.data)
    df = sample_per_class(df, config.get("max_samples_per_class"), int(config.get("seed", 42)))

    split_file = Path(args.split_file) if args.split_file else checkpoint_path.parent / "splits" / f"{args.split}.csv"
    records = load_records(split_file)

    model = create_model(
        model_name=config["model"],
        num_classes=len(labels),
        pretrained=False,
        dropout=float(config.get("dropout", 0.35)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device_choice.device)

    dataset = WaferMapDataset(
        df=df,
        wafer_col=wafer_col,
        records=records,
        label_map=label_to_index(labels),
        image_size=int(config.get("image_size", 224)),
        channel_mode=config.get("channel_mode", "colormap"),
        train=False,
        augmentation=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config.get("batch_size", 128)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device_choice.backend == "cuda",
    )

    out_dir = ensure_dir(args.out_dir or checkpoint_path.parent / "metrics")
    y_true, probabilities = predict_probabilities(model, loader, device_choice.device)
    summary = save_evaluation(y_true, probabilities, labels, out_dir, prefix=args.split)
    write_json(out_dir / f"{args.split}_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

