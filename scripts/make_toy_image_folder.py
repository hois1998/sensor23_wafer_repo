from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = ("horizontal", "vertical", "diagonal")


def make_image(label: str, rng: np.random.Generator, size: int) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    noise = rng.integers(0, 25, size=(size, size, 3), dtype=np.uint8)
    image += noise
    yy, xx = np.mgrid[:size, :size]
    if label == "horizontal":
        mask = np.abs(yy - size // 2) < 4
        color = (230, 50, 50)
    elif label == "vertical":
        mask = np.abs(xx - size // 2) < 4
        color = (50, 210, 80)
    else:
        mask = np.abs(yy - xx) < 4
        color = (70, 120, 240)
    image[mask] = color
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny class-folder image dataset for smoke tests.")
    parser.add_argument("--out", default="data/toy_images")
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    for label in CLASSES:
        class_dir = out / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(args.per_class):
            image = make_image(label, rng, args.size)
            Image.fromarray(image, mode="RGB").save(class_dir / f"{index:03d}.png")
    print(f"Wrote {len(CLASSES) * args.per_class} images to {out}")


if __name__ == "__main__":
    main()

