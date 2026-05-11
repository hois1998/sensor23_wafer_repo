from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from wafer_repro.data import (
    WaferMapDataset,
    augment_training_records,
    load_lswmd,
    make_kfold_splits,
    make_single_split,
    record_counts,
    sample_per_class,
    save_records,
)
from wafer_repro.labels import PAPER_CLASSES, label_to_index
from wafer_repro.metrics import predict_probabilities, save_evaluation
from wafer_repro.models import create_model
from wafer_repro.utils import amp_is_enabled, choose_device, ensure_dir, set_seed, timestamp, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Sensors 2023-style WM-811K classifier.")
    parser.add_argument("--data", default="../LSWMD.pkl", help="Path to LSWMD.pkl.")
    parser.add_argument("--out-dir", default="outputs/runs", help="Directory for run outputs.")
    parser.add_argument("--run-name", default=None, help="Optional run folder name.")
    parser.add_argument("--model", default="mobilenet_v3_small")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--channel-mode", choices=["colormap", "replicate"], default="colormap")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-fraction-of-trainval", type=float, default=0.25)
    parser.add_argument("--fold", type=int, default=None, help="Use one fold from the paper 4-fold split.")
    parser.add_argument("--num-folds", type=int, default=4)
    parser.add_argument("--target-defect-count", type=int, default=10_000)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--rotation-degrees", type=float, default=180.0)
    parser.add_argument("--crop-padding", type=int, default=16)
    parser.add_argument("--blur-prob", type=float, default=0.2)
    parser.add_argument("--erase-prob", type=float, default=0.25)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights for torchvision models.")
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="none")
    parser.add_argument("--max-samples-per-class", type=int, default=None, help="Debug option for quick smoke runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "directml", "dml", "cpu"])
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision.")
    parser.add_argument("--skip-test", action="store_true", help="Do not evaluate the best checkpoint on the test split.")
    return parser.parse_args()


def select_splits(args: argparse.Namespace, df):
    if args.fold is None:
        train, val, test = make_single_split(
            df,
            test_size=args.test_size,
            val_fraction_of_trainval=args.val_fraction_of_trainval,
            seed=args.seed,
        )
        return "single_6_2_2", train, val, test

    splits = make_kfold_splits(df, test_size=args.test_size, n_splits=args.num_folds, seed=args.seed)
    if args.fold < 0 or args.fold >= len(splits):
        raise ValueError(f"--fold must be between 0 and {len(splits) - 1}.")
    fold_id, train, val, test = splits[args.fold]
    return f"kfold_{fold_id}_of_{args.num_folds}", train, val, test


def build_criterion(args: argparse.Namespace, train_records, device):
    if args.class_weights == "none":
        return nn.CrossEntropyLoss()
    counts = train_records["label"].value_counts().reindex(PAPER_CLASSES).fillna(0).to_numpy(dtype=np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def run_epoch(model, loader, criterion, optimizer, device, use_amp: bool, train: bool) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    desc = "train" if train else "val"

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((preds == labels).sum().detach().cpu())
        total_count += batch_size
        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())

    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
    }


def cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device_choice = choose_device(args.device)
    use_amp = amp_is_enabled(device_choice, args.amp)

    run_name = args.run_name or f"{args.model}_{timestamp()}"
    run_dir = ensure_dir(Path(args.out_dir) / run_name)
    split_dir = ensure_dir(run_dir / "splits")
    metrics_dir = ensure_dir(run_dir / "metrics")

    print(f"Device: {device_choice.name} ({device_choice.backend})")
    print(f"Run directory: {run_dir}")

    df, wafer_col = load_lswmd(args.data)
    df = sample_per_class(df, args.max_samples_per_class, args.seed)
    split_strategy, train_base, val_records, test_records = select_splits(args, df)
    train_records = (
        train_base
        if args.no_augment
        else augment_training_records(train_base, target_defect_count=args.target_defect_count, seed=args.seed)
    )

    save_records(train_base, split_dir / "train_base.csv")
    save_records(train_records, split_dir / "train_augmented.csv")
    save_records(val_records, split_dir / "val.csv")
    save_records(test_records, split_dir / "test.csv")

    labels = PAPER_CLASSES
    label_map = label_to_index(labels)
    common_dataset_kwargs = {
        "df": df,
        "wafer_col": wafer_col,
        "label_map": label_map,
        "image_size": args.image_size,
        "channel_mode": args.channel_mode,
        "rotation_degrees": args.rotation_degrees,
        "crop_padding": args.crop_padding,
        "blur_prob": args.blur_prob,
        "erase_prob": args.erase_prob,
    }
    train_ds = WaferMapDataset(
        records=train_records,
        train=True,
        augmentation=not args.no_augment,
        **common_dataset_kwargs,
    )
    val_ds = WaferMapDataset(records=val_records, train=False, augmentation=False, **common_dataset_kwargs)
    test_ds = WaferMapDataset(records=test_records, train=False, augmentation=False, **common_dataset_kwargs)

    pin_memory = device_choice.backend == "cuda"
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, pin_memory)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, pin_memory)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers, pin_memory)

    data_summary = {
        "raw_labeled_rows_after_optional_sampling": int(len(df)),
        "split_strategy": split_strategy,
        "train_base_records": int(len(train_base)),
        "train_records_after_augmentation": int(len(train_records)),
        "val_records": int(len(val_records)),
        "test_records": int(len(test_records)),
        "class_counts_raw": record_counts(train_base) | {"__note__": "train_base only"},
        "class_counts_train_augmented": record_counts(train_records),
        "class_counts_val": record_counts(val_records),
        "class_counts_test": record_counts(test_records),
    }
    write_json(run_dir / "data_summary.json", data_summary)
    print(json.dumps(data_summary, indent=2, ensure_ascii=False))

    config = vars(args).copy()
    config.update(
        {
            "run_dir": str(run_dir),
            "wafer_column": wafer_col,
            "split_strategy": split_strategy,
            "device_name": device_choice.name,
            "device_backend": device_choice.backend,
            "labels": labels,
        }
    )
    write_json(run_dir / "config.json", config)

    model = create_model(args.model, num_classes=len(labels), pretrained=args.pretrained, dropout=args.dropout)
    model = model.to(device_choice.device)
    criterion = build_criterion(args, train_records, device_choice.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = run_dir / "history.csv"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    best_macro_f1 = -1.0
    fieldnames = ["epoch", "train_loss", "train_accuracy", "train_macro_f1", "val_loss", "val_accuracy", "val_macro_f1"]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device_choice.device, use_amp, train=True)
            val_metrics = run_epoch(model, val_loader, criterion, None, device_choice.device, False, train=False)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
                f"train_f1={row['train_macro_f1']:.4f} val_loss={row['val_loss']:.4f} "
                f"val_acc={row['val_accuracy']:.4f} val_f1={row['val_macro_f1']:.4f}"
            )

            checkpoint = {
                "epoch": epoch,
                "model_state": cpu_state_dict(model),
                "config": config,
                "labels": labels,
                "val_macro_f1": row["val_macro_f1"],
            }
            torch.save(checkpoint, last_path)
            if row["val_macro_f1"] > best_macro_f1:
                best_macro_f1 = row["val_macro_f1"]
                torch.save(checkpoint, best_path)

    if args.skip_test:
        return

    print("Evaluating best checkpoint on the unaugmented test split...")
    best_model, checkpoint = create_model(args.model, len(labels), False, args.dropout), torch.load(best_path, map_location="cpu")
    best_model.load_state_dict(checkpoint["model_state"])
    best_model = best_model.to(device_choice.device)
    y_true, probs = predict_probabilities(best_model, test_loader, device_choice.device)
    summary = save_evaluation(y_true, probs, labels, metrics_dir, prefix="test")
    write_json(run_dir / "test_summary.json", summary | {"best_epoch": int(checkpoint["epoch"])})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

