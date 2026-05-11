from __future__ import annotations

PAPER_CLASSES: tuple[str, ...] = (
    "Center",
    "Donut",
    "Edge-loc",
    "Edge-ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "None",
)

DEFECT_CLASSES: tuple[str, ...] = tuple(label for label in PAPER_CLASSES if label != "None")

LABEL_ALIASES = {
    "center": "Center",
    "donut": "Donut",
    "edge-loc": "Edge-loc",
    "edge_loc": "Edge-loc",
    "edgeloc": "Edge-loc",
    "edge-ring": "Edge-ring",
    "edge_ring": "Edge-ring",
    "edgering": "Edge-ring",
    "loc": "Loc",
    "near-full": "Near-full",
    "near_full": "Near-full",
    "nearfull": "Near-full",
    "random": "Random",
    "scratch": "Scratch",
    "none": "None",
    "normal": "None",
}


def label_to_index(labels: tuple[str, ...] = PAPER_CLASSES) -> dict[str, int]:
    return {label: idx for idx, label in enumerate(labels)}


def index_to_label(labels: tuple[str, ...] = PAPER_CLASSES) -> dict[int, str]:
    return {idx: label for idx, label in enumerate(labels)}

