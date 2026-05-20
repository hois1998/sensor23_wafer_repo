from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_dataset(per_class: int, length: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    labels = ("sine", "square", "saw")
    time = np.linspace(0, 1, length, dtype=np.float32)
    for label in labels:
        for index in range(per_class):
            phase = rng.uniform(0, 2 * np.pi)
            noise = rng.normal(0, 0.05, size=length)
            if label == "sine":
                signal = np.sin(2 * np.pi * 3 * time + phase)
            elif label == "square":
                signal = np.sign(np.sin(2 * np.pi * 3 * time + phase))
            else:
                signal = 2 * ((3 * time + phase / (2 * np.pi)) % 1) - 1
            values = signal + noise
            row = {"sample_id": f"{label}_{index:03d}", "label": label}
            row.update({f"x_{step:03d}": float(values[step]) for step in range(length)})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny time-series classification dataset.")
    parser.add_argument("--out", default="data/toy_timeseries.csv")
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = build_dataset(args.per_class, args.length, args.seed)
    frame.to_csv(out, index=False)
    print(f"Wrote {len(frame)} rows to {out}")


if __name__ == "__main__":
    main()
