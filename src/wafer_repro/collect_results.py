from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect run summaries into comparison CSV files.")
    parser.add_argument("--runs-dir", default="outputs/paper_runs")
    parser.add_argument("--out", default="outputs/comparison_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    rows = []
    for summary_path in sorted(runs_dir.glob("*/test_summary.json")):
        run_dir = summary_path.parent
        config_path = run_dir / "config.json"
        data_summary_path = run_dir / "data_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        data_summary = json.loads(data_summary_path.read_text(encoding="utf-8")) if data_summary_path.exists() else {}
        rows.append(
            {
                "run": run_dir.name,
                "model": config.get("model"),
                "fold": config.get("fold"),
                "split_strategy": config.get("split_strategy"),
                "accuracy": summary.get("accuracy"),
                "macro_precision": summary.get("macro_precision"),
                "macro_recall": summary.get("macro_recall"),
                "macro_f1": summary.get("macro_f1"),
                "weighted_f1": summary.get("weighted_f1"),
                "best_epoch": summary.get("best_epoch"),
                "train_records_after_augmentation": data_summary.get("train_records_after_augmentation"),
                "test_records": data_summary.get("test_records"),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No test_summary.json files found under {runs_dir}")

    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    grouped = (
        frame.groupby("model", dropna=False)[["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]]
        .agg(["mean", "std"])
        .sort_values(("macro_f1", "mean"), ascending=False)
    )
    grouped.to_csv(out.with_name(out.stem + "_by_model.csv"))
    print(frame)
    print(grouped)


if __name__ == "__main__":
    main()

