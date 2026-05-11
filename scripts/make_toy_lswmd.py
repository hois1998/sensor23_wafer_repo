from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("Center", "Donut", "Edge-loc", "Edge-ring", "Loc", "Near-full", "Random", "Scratch", "none")


def base_map(size: int = 32) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2
    radius = size * 0.45
    wafer = np.zeros((size, size), dtype=np.uint8)
    wafer[(xx - center) ** 2 + (yy - center) ** 2 <= radius**2] = 1
    return wafer


def add_pattern(label: str, rng: np.random.Generator, size: int = 32) -> np.ndarray:
    wafer = base_map(size)
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2
    if label == "Center":
        mask = (xx - center) ** 2 + (yy - center) ** 2 < (size * 0.14) ** 2
    elif label == "Donut":
        dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        mask = (dist > size * 0.16) & (dist < size * 0.24)
    elif label == "Edge-loc":
        mask = (xx > size * 0.7) & (yy > size * 0.35) & (yy < size * 0.75)
    elif label == "Edge-ring":
        dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        mask = dist > size * 0.36
    elif label == "Loc":
        cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < (size * 0.13) ** 2
    elif label == "Near-full":
        mask = wafer == 1
    elif label == "Random":
        mask = rng.random((size, size)) < 0.12
    elif label == "Scratch":
        mask = np.abs(yy - (0.45 * xx + size * 0.2)) < 1.5
    else:
        mask = np.zeros((size, size), dtype=bool)
    wafer[mask & (wafer > 0)] = 2
    return wafer


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny LSWMD-like pickle for smoke tests.")
    parser.add_argument("--out", default="data/toy_LSWMD.pkl")
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for label in LABELS:
        for _ in range(args.per_class):
            rows.append(
                {
                    "waferMap": add_pattern(label, rng),
                    "failureType": np.array([[label]], dtype=object),
                    "trianTestLabel": np.array([["Training"]], dtype=object),
                }
            )
    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(out)
    print(f"Wrote {out} with shape {frame.shape}")


if __name__ == "__main__":
    main()

