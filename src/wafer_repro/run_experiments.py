from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from wafer_repro.models import PAPER_MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-style model comparison sequentially.")
    parser.add_argument("--data", default="../LSWMD.pkl")
    parser.add_argument("--out-dir", default="outputs/paper_runs")
    parser.add_argument("--models", nargs="+", default=["paper"], help="Model names or 'paper'.")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--target-defect-count", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--extra-train-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def expand_models(models: list[str]) -> list[str]:
    if len(models) == 1 and models[0] == "paper":
        return list(PAPER_MODEL_NAMES)
    return models


def main() -> None:
    args = parse_args()
    models = expand_models(args.models)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for model in models:
        for fold in range(args.folds):
            run_name = f"{model}_fold{fold}"
            command = [
                sys.executable,
                "-m",
                "wafer_repro.train",
                "--data",
                args.data,
                "--out-dir",
                args.out_dir,
                "--run-name",
                run_name,
                "--model",
                model,
                "--fold",
                str(fold),
                "--num-folds",
                str(args.folds),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--image-size",
                str(args.image_size),
                "--target-defect-count",
                str(args.target_defect_count),
                "--device",
                args.device,
                "--num-workers",
                str(args.num_workers),
            ]
            command.extend(args.extra_train_args)
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

