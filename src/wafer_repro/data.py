from __future__ import annotations

"""Compatibility exports for the original WM-811K data API.

New code should import dataset-specific implementations from
`wafer_repro.datasets.wm811k.*`. This module remains as the stable facade for
the existing train/evaluate/infer commands.
"""

from wafer_repro.datasets.wm811k.dataset import WaferMapDataset, make_inference_tensor
from wafer_repro.datasets.wm811k.records import (
    augment_training_records,
    base_records,
    load_records,
    record_counts,
    sample_per_class,
    save_records,
)
from wafer_repro.datasets.wm811k.source import (
    FAILURE_COLUMN_CANDIDATES,
    SPLIT_COLUMN_CANDIDATES,
    WAFER_COLUMN_CANDIDATES,
    find_column as _find_column,
    load_lswmd,
    normalize_failure_label,
    scalarize,
)
from wafer_repro.datasets.wm811k.split import (
    make_external_test_split,
    make_kfold_splits,
    make_predefined_file_split,
    make_single_split,
)
from wafer_repro.datasets.wm811k.transforms import build_transform, wafer_to_rgb_array

__all__ = [
    "FAILURE_COLUMN_CANDIDATES",
    "SPLIT_COLUMN_CANDIDATES",
    "WAFER_COLUMN_CANDIDATES",
    "WaferMapDataset",
    "_find_column",
    "augment_training_records",
    "base_records",
    "build_transform",
    "load_lswmd",
    "load_records",
    "make_external_test_split",
    "make_inference_tensor",
    "make_kfold_splits",
    "make_predefined_file_split",
    "make_single_split",
    "normalize_failure_label",
    "record_counts",
    "sample_per_class",
    "save_records",
    "scalarize",
    "wafer_to_rgb_array",
]
