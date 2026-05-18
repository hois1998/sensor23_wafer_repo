from __future__ import annotations

import argparse
from typing import Any

from wafer_repro.core.config import MISSING, apply_overrides, get_path, load_config
from wafer_repro.experiment.runner import ExperimentRunner


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
    "train.scheduler.name": "scheduler",
    "train.scheduler.step_size": "scheduler_step_size",
    "train.scheduler.gamma": "scheduler_gamma",
    "train.scheduler.t_max": "scheduler_t_max",
    "train.scheduler.eta_min": "scheduler_eta_min",
    "train.scheduler.monitor": "scheduler_monitor",
    "train.scheduler.mode": "scheduler_mode",
    "train.scheduler.factor": "scheduler_factor",
    "train.scheduler.patience": "scheduler_patience",
    "train.early_stopping.enabled": "early_stopping",
    "train.early_stopping.monitor": "early_stopping_monitor",
    "train.early_stopping.mode": "early_stopping_mode",
    "train.early_stopping.patience": "early_stopping_patience",
    "train.early_stopping.min_delta": "early_stopping_min_delta",
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
    parser.add_argument("--scheduler", choices=["none", "step_lr", "cosine_annealing", "reduce_on_plateau"], default="none")
    parser.add_argument("--scheduler-step-size", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--scheduler-t-max", type=int, default=None)
    parser.add_argument("--scheduler-eta-min", type=float, default=0.0)
    parser.add_argument("--scheduler-monitor", default="val/macro_f1")
    parser.add_argument("--scheduler-mode", choices=["max", "min"], default="max")
    parser.add_argument("--scheduler-factor", type=float, default=0.1)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--early-stopping", action="store_true", help="Stop when the monitored validation metric stops improving.")
    parser.add_argument("--early-stopping-monitor", default="val/macro_f1")
    parser.add_argument("--early-stopping-mode", choices=["max", "min"], default="max")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
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


def main() -> None:
    args, loaded_config = parse_args()
    ExperimentRunner(args, loaded_config).run()


if __name__ == "__main__":
    main()
