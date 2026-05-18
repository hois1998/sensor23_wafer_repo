from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from wafer_repro.data import (
    WaferMapDataset,
    augment_training_records,
    load_lswmd,
    make_external_test_split,
    make_kfold_splits,
    make_predefined_file_split,
    make_single_split,
    record_counts,
    sample_per_class,
    save_records,
)
from wafer_repro.core.config import (
    MISSING,
    apply_overrides,
    config_hash,
    deep_merge,
    get_path,
    load_config,
    set_path,
    validate_fixed_controls,
    write_yaml,
)
from wafer_repro.core.environment import capture_environment
from wafer_repro.labels import PAPER_CLASSES, label_to_index
from wafer_repro.metrics import predict_probabilities, save_evaluation
from wafer_repro.models import create_model
from wafer_repro.tasks.registry import create_task
from wafer_repro.training.registry import create_trainer
from wafer_repro.training.supervised import build_optimizer, cpu_state_dict, make_loader
from wafer_repro.utils import amp_is_enabled, choose_device, ensure_dir, set_seed, timestamp, write_json


CONFIG_ARG_MAP = {
    "data.source.path": "data",
    "runtime.output_dir": "out_dir",
    "runtime.run_name": "run_name",
    "model.name": "model",
    "model.pretrained": "pretrained",
    "model.dropout": "dropout",
    "task.type": "task_type",
    "data.preprocessing.image_size": "image_size",
    "data.preprocessing.channel_mode": "channel_mode",
    "train.max_epochs": "epochs",
    "data.dataloader.batch_size": "batch_size",
    "train.trainer": "trainer",
    "train.optimizer.name": "optimizer",
    "train.optimizer.lr": "lr",
    "train.optimizer.weight_decay": "weight_decay",
    "data.split.strategy": "split_strategy",
    "data.split.test_size": "test_size",
    "data.split.val_fraction_of_trainval": "val_fraction_of_trainval",
    "data.split.fold_index": "fold",
    "data.split.n_splits": "num_folds",
    "data.split.files.train": "train_split_file",
    "data.split.files.val": "val_split_file",
    "data.split.files.test": "test_split_file",
    "data.split.external_test.path": "external_test_path",
    "data.split.external_test.id_column": "external_id_column",
    "data.augmentation.target_defect_count": "target_defect_count",
    "data.augmentation.transforms.random_rotation.degrees": "rotation_degrees",
    "data.augmentation.transforms.random_crop.padding": "crop_padding",
    "data.augmentation.transforms.gaussian_blur.p": "blur_prob",
    "data.augmentation.transforms.random_erasing.p": "erase_prob",
    "task.loss.class_weights": "class_weights",
    "data.debug.max_samples_per_class": "max_samples_per_class",
    "train.seed": "seed",
    "data.dataloader.num_workers": "num_workers",
    "runtime.device": "device",
    "train.amp.enabled": "amp",
    "evaluation.skip_test": "skip_test",
}


def config_to_arg_defaults(config: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for config_path, arg_name in CONFIG_ARG_MAP.items():
        value = get_path(config, config_path, default=MISSING)
        if value is not MISSING:
            defaults[arg_name] = value

    augmentation_enabled = get_path(config, "data.augmentation.enabled", default=MISSING)
    if augmentation_enabled is not MISSING:
        defaults["no_augment"] = not bool(augmentation_enabled)
    return defaults


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Sensors 2023-style WM-811K classifier.")
    parser.add_argument("--config", default=None, help="Path to an experiment YAML config.")
    parser.add_argument("--set", action="append", default=[], dest="config_overrides", help="Override config value as KEY=VALUE.")
    parser.add_argument("--data", default="../LSWMD.pkl", help="Path to LSWMD.pkl.")
    parser.add_argument("--out-dir", default="outputs/runs", help="Directory for run outputs.")
    parser.add_argument("--run-name", default=None, help="Optional run folder name.")
    parser.add_argument("--model", default="mobilenet_v3_small")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--channel-mode", choices=["colormap", "replicate"], default="colormap")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-fraction-of-trainval", type=float, default=0.25)
    parser.add_argument(
        "--split-strategy",
        default=None,
        choices=["stratified_holdout", "stratified_kfold", "predefined_files", "external_test_with_train_val_split"],
    )
    parser.add_argument("--fold", type=int, default=None, help="Use one fold from the paper 4-fold split.")
    parser.add_argument("--num-folds", type=int, default=4)
    parser.add_argument("--train-split-file", default=None, help="CSV file for predefined train records.")
    parser.add_argument("--val-split-file", default=None, help="CSV file for predefined validation records.")
    parser.add_argument("--test-split-file", default=None, help="CSV file for predefined test records.")
    parser.add_argument("--external-test-path", default=None, help="CSV containing fixed external test ids.")
    parser.add_argument("--external-id-column", default="original_index", help="ID column used by external split files.")
    parser.add_argument("--target-defect-count", type=int, default=10_000)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--rotation-degrees", type=float, default=180.0)
    parser.add_argument("--crop-padding", type=int, default=16)
    parser.add_argument("--blur-prob", type=float, default=0.2)
    parser.add_argument("--erase-prob", type=float, default=0.25)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights for torchvision models.")
    parser.add_argument("--task-type", choices=["classification"], default="classification")
    parser.add_argument("--trainer", choices=["supervised_torch"], default="supervised_torch")
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="none")
    parser.add_argument("--max-samples-per-class", type=int, default=None, help="Debug option for quick smoke runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "directml", "dml", "cpu"])
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision.")
    parser.add_argument("--skip-test", action="store_true", help="Do not evaluate the best checkpoint on the test split.")
    return parser


def parse_args() -> tuple[argparse.Namespace, dict[str, Any]]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_parser.add_argument("--set", action="append", default=[], dest="config_overrides")
    pre_args, _ = pre_parser.parse_known_args()

    loaded_config: dict[str, Any] = {}
    if pre_args.config:
        loaded_config = load_config(pre_args.config, pre_args.config_overrides)
    elif pre_args.config_overrides:
        loaded_config = apply_overrides({}, pre_args.config_overrides)

    parser = build_parser()
    parser.set_defaults(**config_to_arg_defaults(loaded_config))
    args = parser.parse_args()
    return args, loaded_config


def build_resolved_config(args: argparse.Namespace, loaded_config: dict[str, Any]) -> dict[str, Any]:
    resolved = deep_merge({}, loaded_config)
    set_path(resolved, "schema_version", get_path(resolved, "schema_version", default=1))
    set_path(resolved, "experiment.name", get_path(resolved, "experiment.name", default=args.run_name or args.model))
    set_path(resolved, "experiment.suite", get_path(resolved, "experiment.suite", default="adhoc"))

    set_path(resolved, "data.module", get_path(resolved, "data.module", default="wm811k"))
    set_path(resolved, "data.source.path", args.data)
    set_path(resolved, "data.preprocessing.image_size", args.image_size)
    set_path(resolved, "data.preprocessing.channel_mode", args.channel_mode)
    split_strategy = args.split_strategy or ("stratified_kfold" if args.fold is not None else "stratified_holdout")
    set_path(resolved, "data.split.strategy", split_strategy)
    set_path(resolved, "data.split.seed", args.seed)
    set_path(resolved, "data.split.test_size", args.test_size)
    set_path(resolved, "data.split.val_fraction_of_trainval", args.val_fraction_of_trainval)
    set_path(resolved, "data.split.fold_index", args.fold)
    set_path(resolved, "data.split.n_splits", args.num_folds)
    if args.train_split_file:
        set_path(resolved, "data.split.files.train", args.train_split_file)
    if args.val_split_file:
        set_path(resolved, "data.split.files.val", args.val_split_file)
    if args.test_split_file:
        set_path(resolved, "data.split.files.test", args.test_split_file)
    if args.external_test_path:
        set_path(resolved, "data.split.external_test.path", args.external_test_path)
        set_path(resolved, "data.split.external_test.id_column", args.external_id_column)
    set_path(resolved, "data.augmentation.enabled", not args.no_augment)
    set_path(resolved, "data.augmentation.target_defect_count", args.target_defect_count)
    set_path(resolved, "data.augmentation.train_only", True)
    set_path(resolved, "data.augmentation.transforms.random_rotation.degrees", args.rotation_degrees)
    set_path(resolved, "data.augmentation.transforms.random_crop.padding", args.crop_padding)
    set_path(resolved, "data.augmentation.transforms.gaussian_blur.p", args.blur_prob)
    set_path(resolved, "data.augmentation.transforms.random_erasing.p", args.erase_prob)
    set_path(resolved, "data.dataloader.batch_size", args.batch_size)
    set_path(resolved, "data.dataloader.num_workers", args.num_workers)
    set_path(resolved, "data.debug.max_samples_per_class", args.max_samples_per_class)

    set_path(resolved, "task.type", args.task_type)
    set_path(resolved, "task.class_order", list(PAPER_CLASSES))
    set_path(resolved, "task.loss.name", get_path(resolved, "task.loss.name", default="cross_entropy"))
    set_path(resolved, "task.loss.class_weights", args.class_weights)
    set_path(resolved, "task.metrics", get_path(resolved, "task.metrics", default=["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]))

    set_path(resolved, "model.name", args.model)
    set_path(resolved, "model.pretrained", args.pretrained)
    set_path(resolved, "model.dropout", args.dropout)

    set_path(resolved, "train.trainer", args.trainer)
    set_path(resolved, "train.seed", args.seed)
    set_path(resolved, "train.max_epochs", args.epochs)
    set_path(resolved, "train.optimizer.name", args.optimizer)
    set_path(resolved, "train.optimizer.lr", args.lr)
    set_path(resolved, "train.optimizer.weight_decay", args.weight_decay)
    set_path(resolved, "train.amp.enabled", args.amp)
    set_path(resolved, "train.checkpoint.monitor", get_path(resolved, "train.checkpoint.monitor", default="val/macro_f1"))
    set_path(resolved, "train.checkpoint.mode", get_path(resolved, "train.checkpoint.mode", default="max"))

    set_path(resolved, "evaluation.primary_metric", get_path(resolved, "evaluation.primary_metric", default="macro_f1"))
    set_path(resolved, "evaluation.splits", get_path(resolved, "evaluation.splits", default=[] if args.skip_test else ["test"]))
    set_path(resolved, "evaluation.skip_test", args.skip_test)

    set_path(resolved, "runtime.device", args.device)
    set_path(resolved, "runtime.output_dir", args.out_dir)
    set_path(resolved, "runtime.run_name", args.run_name)
    set_path(resolved, "runtime.config_path", args.config)
    set_path(resolved, "runtime.cli_overrides", args.config_overrides)
    set_path(resolved, "runtime.deterministic", True)
    return resolved


def select_splits(args: argparse.Namespace, df):
    strategy = args.split_strategy or ("stratified_kfold" if args.fold is not None else "stratified_holdout")

    if strategy == "predefined_files":
        missing = [
            name
            for name, value in {
                "--train-split-file": args.train_split_file,
                "--val-split-file": args.val_split_file,
                "--test-split-file": args.test_split_file,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"predefined_files split requires: {', '.join(missing)}")
        train, val, test = make_predefined_file_split(args.train_split_file, args.val_split_file, args.test_split_file)
        return "predefined_files", train, val, test

    if strategy == "external_test_with_train_val_split":
        if not args.external_test_path:
            raise ValueError("external_test_with_train_val_split requires --external-test-path.")
        train, val, test = make_external_test_split(
            df,
            args.external_test_path,
            id_column=args.external_id_column,
            val_fraction_of_trainval=args.val_fraction_of_trainval,
            seed=args.seed,
        )
        return "external_test_with_train_val_split", train, val, test

    if strategy == "stratified_holdout":
        train, val, test = make_single_split(
            df,
            test_size=args.test_size,
            val_fraction_of_trainval=args.val_fraction_of_trainval,
            seed=args.seed,
        )
        return "single_6_2_2", train, val, test

    if strategy != "stratified_kfold":
        raise ValueError(f"Unknown split strategy: {strategy}")

    fold = 0 if args.fold is None else args.fold
    splits = make_kfold_splits(df, test_size=args.test_size, n_splits=args.num_folds, seed=args.seed)
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"--fold must be between 0 and {len(splits) - 1}.")
    fold_id, train, val, test = splits[fold]
    return f"kfold_{fold_id}_of_{args.num_folds}", train, val, test


def main() -> None:
    args, loaded_config = parse_args()
    set_seed(args.seed)
    device_choice = choose_device(args.device)
    use_amp = amp_is_enabled(device_choice, args.amp)

    run_name = args.run_name or f"{args.model}_{timestamp()}"
    run_dir = ensure_dir(Path(args.out_dir) / run_name)
    split_dir = ensure_dir(run_dir / "splits")
    metrics_dir = ensure_dir(run_dir / "metrics")
    resolved_config = build_resolved_config(args, loaded_config)
    set_path(resolved_config, "runtime.run_dir", str(run_dir))
    validate_fixed_controls(resolved_config)

    print(f"Device: {device_choice.name} ({device_choice.backend})")
    print(f"Run directory: {run_dir}")

    df, wafer_col = load_lswmd(args.data)
    df = sample_per_class(df, args.max_samples_per_class, args.seed)
    split_strategy, train_base, val_records, test_records = select_splits(args, df)
    set_path(resolved_config, "data.split.resolved_strategy", split_strategy)
    resolved_hash = config_hash(
        resolved_config,
        exclude_paths={"runtime.output_dir", "runtime.run_dir", "runtime.config_hash"},
    )
    set_path(resolved_config, "runtime.config_hash", resolved_hash)
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
    if loaded_config:
        write_yaml(run_dir / "source_config.yaml", loaded_config)
    write_yaml(run_dir / "resolved_config.yaml", resolved_config)
    (run_dir / "config_hash.txt").write_text(resolved_hash + "\n", encoding="utf-8")
    write_json(run_dir / "environment.json", capture_environment(Path.cwd()))

    config = vars(args).copy()
    config.update(
        {
            "run_dir": str(run_dir),
            "resolved_config_path": str(run_dir / "resolved_config.yaml"),
            "config_hash": resolved_hash,
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
    task = create_task(args.task_type, labels=labels, class_weights=args.class_weights)
    trainer = create_trainer(args.trainer, task=task, device=device_choice.device, use_amp=use_amp)
    criterion = task.build_criterion(train_records, device_choice.device)
    optimizer = build_optimizer(
        model,
        {
            "name": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    )

    history_path = run_dir / "history.csv"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    best_macro_f1 = -1.0
    fieldnames = ["epoch", "train_loss", "train_accuracy", "train_macro_f1", "val_loss", "val_accuracy", "val_macro_f1"]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = trainer.train_epoch(model, train_loader, criterion, optimizer)
            val_metrics = trainer.validate(model, val_loader, criterion)
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
