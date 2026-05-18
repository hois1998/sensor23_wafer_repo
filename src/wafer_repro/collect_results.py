from __future__ import annotations

import argparse
from pathlib import Path

from wafer_repro.analysis.collector import write_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect run summaries into comparison CSV files.")
    parser.add_argument("--runs-dir", default="outputs/paper_runs")
    parser.add_argument("--out", default="outputs/comparison_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, grouped = write_comparison(Path(args.runs_dir), Path(args.out))
    print(frame)
    print(grouped)


if __name__ == "__main__":
    main()
