from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from wafer_repro.core.config import (
    config_hash,
    deep_merge,
    get_path,
    set_path,
    validate_fixed_controls,
    write_yaml,
)
from wafer_repro.core.environment import capture_environment
from wafer_repro.data import save_records
from wafer_repro.datasets.registry import create_data_bundle
from wafer_repro.experiment.manifest import now_iso, write_manifest
from wafer_repro.labels import PAPER_CLASSES
from wafer_repro.metrics import predict_probabilities, save_evaluation
from wafer_repro.models import create_model
from wafer_repro.tasks.registry import create_task
from wafer_repro.training.callbacks import (
    build_early_stopper,
    build_scheduler,
    is_improvement,
    resolve_monitor_value,
    step_scheduler,
)
from wafer_repro.training.registry import create_trainer
from wafer_repro.training.supervised import build_optimizer, cpu_state_dict, make_loader
from wafer_repro.utils import amp_is_enabled, choose_device, ensure_dir, set_seed, timestamp, write_json


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
    set_path(
        resolved,
        "task.metrics",
        get_path(resolved, "task.metrics", default=["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]),
    )

    set_path(resolved, "model.name", args.model)
    set_path(resolved, "model.pretrained", args.pretrained)
    set_path(resolved, "model.dropout", args.dropout)

    set_path(resolved, "train.trainer", args.trainer)
    set_path(resolved, "train.seed", args.seed)
    set_path(resolved, "train.max_epochs", args.epochs)
    set_path(resolved, "train.optimizer.name", args.optimizer)
    set_path(resolved, "train.optimizer.lr", args.lr)
    set_path(resolved, "train.optimizer.weight_decay", args.weight_decay)
    set_path(resolved, "train.scheduler.name", args.scheduler)
    set_path(resolved, "train.scheduler.step_size", args.scheduler_step_size)
    set_path(resolved, "train.scheduler.gamma", args.scheduler_gamma)
    set_path(resolved, "train.scheduler.t_max", args.scheduler_t_max)
    set_path(resolved, "train.scheduler.eta_min", args.scheduler_eta_min)
    set_path(resolved, "train.scheduler.monitor", args.scheduler_monitor)
    set_path(resolved, "train.scheduler.mode", args.scheduler_mode)
    set_path(resolved, "train.scheduler.factor", args.scheduler_factor)
    set_path(resolved, "train.scheduler.patience", args.scheduler_patience)
    set_path(resolved, "train.early_stopping.enabled", args.early_stopping)
    set_path(resolved, "train.early_stopping.monitor", args.early_stopping_monitor)
    set_path(resolved, "train.early_stopping.mode", args.early_stopping_mode)
    set_path(resolved, "train.early_stopping.patience", args.early_stopping_patience)
    set_path(resolved, "train.early_stopping.min_delta", args.early_stopping_min_delta)
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


def _save_common_records(split_dir: Path, train_base, train_records, val_records, test_records) -> None:
    save_records(train_base, split_dir / "train_base.csv")
    save_records(train_records, split_dir / "train_augmented.csv")
    save_records(val_records, split_dir / "val.csv")
    save_records(test_records, split_dir / "test.csv")


class ExperimentRunner:
    def __init__(self, args: argparse.Namespace, loaded_config: dict[str, Any]) -> None:
        self.args = args
        self.loaded_config = loaded_config

    def run(self) -> Path:
        args = self.args
        set_seed(args.seed)
        device_choice = choose_device(args.device)
        use_amp = amp_is_enabled(device_choice, args.amp)

        run_name = args.run_name or f"{args.model}_{timestamp()}"
        run_dir = ensure_dir(Path(args.out_dir) / run_name)
        split_dir = ensure_dir(run_dir / "splits")
        metrics_dir = ensure_dir(run_dir / "metrics")
        resolved_config = build_resolved_config(args, self.loaded_config)
        set_path(resolved_config, "runtime.run_dir", str(run_dir))

        manifest: dict[str, Any] = {
            "experiment_name": get_path(resolved_config, "experiment.name", default=run_name),
            "suite": get_path(resolved_config, "experiment.suite", default="adhoc"),
            "run_name": run_name,
            "run_dir": str(run_dir),
            "status": "running",
            "started_at": now_iso(),
            "finished_at": None,
            "config_hash": None,
            "primary_metric": get_path(resolved_config, "evaluation.primary_metric", default="macro_f1"),
            "best_epoch": None,
            "best_checkpoint": None,
            "error": None,
        }
        write_manifest(run_dir, manifest)

        try:
            validate_fixed_controls(resolved_config)
            print(f"Device: {device_choice.name} ({device_choice.backend})")
            print(f"Run directory: {run_dir}")

            data_module = get_path(resolved_config, "data.module", default="wm811k")
            data_bundle = create_data_bundle(data_module, resolved_config)
            labels = data_bundle.labels
            train_base = data_bundle.train_base
            train_records = data_bundle.train_records
            val_records = data_bundle.val_records
            test_records = data_bundle.test_records
            train_ds = data_bundle.train_dataset
            val_ds = data_bundle.val_dataset
            test_ds = data_bundle.test_dataset
            split_strategy = data_bundle.split_strategy
            data_summary = data_bundle.data_summary
            wafer_col = data_bundle.metadata.get("wafer_column")

            set_path(resolved_config, "data.split.resolved_strategy", split_strategy)
            set_path(resolved_config, "task.class_order", list(labels))
            resolved_hash = config_hash(
                resolved_config,
                exclude_paths={"runtime.output_dir", "runtime.run_dir", "runtime.config_hash"},
            )
            set_path(resolved_config, "runtime.config_hash", resolved_hash)
            manifest["config_hash"] = resolved_hash
            manifest["split_strategy"] = split_strategy
            write_manifest(run_dir, manifest)

            _save_common_records(split_dir, train_base, train_records, val_records, test_records)

            pin_memory = device_choice.backend == "cuda"
            train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, pin_memory)
            val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, pin_memory)
            test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers, pin_memory)

            write_json(run_dir / "data_summary.json", data_summary)
            print(json.dumps(data_summary, indent=2, ensure_ascii=False))
            if self.loaded_config:
                write_yaml(run_dir / "source_config.yaml", self.loaded_config)
            write_yaml(run_dir / "resolved_config.yaml", resolved_config)
            (run_dir / "config_hash.txt").write_text(resolved_hash + "\n", encoding="utf-8")
            write_json(run_dir / "environment.json", capture_environment(Path.cwd()))

            legacy_config = vars(args).copy()
            legacy_config.update(
                {
                    "run_dir": str(run_dir),
                    "resolved_config_path": str(run_dir / "resolved_config.yaml"),
                    "config_hash": resolved_hash,
                    "wafer_column": wafer_col,
                    "data_module": data_module,
                    "split_strategy": split_strategy,
                    "device_name": device_choice.name,
                    "device_backend": device_choice.backend,
                    "labels": labels,
                }
            )
            write_json(run_dir / "config.json", legacy_config)

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
            scheduler_config = get_path(resolved_config, "train.scheduler", default={})
            scheduler = build_scheduler(optimizer, scheduler_config, max_epochs=args.epochs)
            early_stopper = build_early_stopper(get_path(resolved_config, "train.early_stopping", default={}))
            checkpoint_monitor = get_path(resolved_config, "train.checkpoint.monitor", default="val/macro_f1")
            checkpoint_mode = str(get_path(resolved_config, "train.checkpoint.mode", default="max")).lower()

            history_path = run_dir / "history.csv"
            best_path = run_dir / "best.pt"
            last_path = run_dir / "last.pt"
            best_score = None
            best_epoch = None
            trained_epochs = 0
            stopped_early = False
            fieldnames = [
                "epoch",
                "lr",
                "train_loss",
                "train_accuracy",
                "train_macro_f1",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
            ]
            with history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for epoch in range(1, args.epochs + 1):
                    train_metrics = trainer.train_epoch(model, train_loader, criterion, optimizer)
                    val_metrics = trainer.validate(model, val_loader, criterion)
                    trained_epochs = epoch
                    row = {
                        "epoch": epoch,
                        "lr": optimizer.param_groups[0]["lr"],
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
                        f"lr={row['lr']:.6g} "
                        f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
                        f"train_f1={row['train_macro_f1']:.4f} val_loss={row['val_loss']:.4f} "
                        f"val_acc={row['val_accuracy']:.4f} val_f1={row['val_macro_f1']:.4f}"
                    )

                    checkpoint_value = resolve_monitor_value(row, checkpoint_monitor)
                    checkpoint = {
                        "epoch": epoch,
                        "model_state": cpu_state_dict(model),
                        "config": legacy_config,
                        "labels": labels,
                        "val_macro_f1": row["val_macro_f1"],
                        "monitor": checkpoint_monitor,
                        "monitor_value": checkpoint_value,
                    }
                    torch.save(checkpoint, last_path)
                    if is_improvement(checkpoint_value, best_score, checkpoint_mode):
                        best_score = checkpoint_value
                        best_epoch = epoch
                        torch.save(checkpoint, best_path)
                        manifest["best_epoch"] = best_epoch
                        manifest["best_checkpoint"] = str(best_path)
                        manifest["best_monitor"] = checkpoint_monitor
                        manifest["best_metric_value"] = best_score
                        manifest["best_val_macro_f1"] = row["val_macro_f1"]
                        write_manifest(run_dir, manifest)

                    step_scheduler(scheduler, scheduler_config, row)
                    if early_stopper is not None:
                        early_value = resolve_monitor_value(row, early_stopper.monitor)
                        if early_stopper.update(early_value, epoch):
                            stopped_early = True
                            manifest["stopped_early"] = True
                            manifest["stopped_epoch"] = epoch
                            manifest["stopped_monitor"] = early_stopper.monitor
                            manifest["stopped_metric_value"] = early_value
                            write_manifest(run_dir, manifest)
                            print(
                                f"early stopping at epoch {epoch:03d}: "
                                f"{early_stopper.monitor}={early_value:.6g}"
                            )
                            break

            manifest["trained_epochs"] = trained_epochs
            manifest["stopped_early"] = stopped_early
            if args.skip_test:
                manifest["status"] = "completed"
                manifest["finished_at"] = now_iso()
                write_manifest(run_dir, manifest)
                return run_dir

            print("Evaluating best checkpoint on the unaugmented test split...")
            best_model = create_model(args.model, len(labels), False, args.dropout)
            checkpoint = torch.load(best_path, map_location="cpu")
            best_model.load_state_dict(checkpoint["model_state"])
            best_model = best_model.to(device_choice.device)
            y_true, probs = predict_probabilities(best_model, test_loader, device_choice.device)
            summary = save_evaluation(y_true, probs, labels, metrics_dir, prefix="test")
            summary_with_epoch = summary | {"best_epoch": int(checkpoint["epoch"])}
            write_json(run_dir / "test_summary.json", summary_with_epoch)
            print(json.dumps(summary, indent=2))

            manifest["status"] = "completed"
            manifest["finished_at"] = now_iso()
            manifest["trained_epochs"] = trained_epochs
            manifest["stopped_early"] = stopped_early
            manifest["test_summary"] = summary_with_epoch
            write_manifest(run_dir, manifest)
            return run_dir
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["finished_at"] = now_iso()
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
            write_manifest(run_dir, manifest)
            raise
