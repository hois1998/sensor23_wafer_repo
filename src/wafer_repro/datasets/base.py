from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from torch.utils.data import Dataset


@dataclass
class DataBundle:
    """Standard payload returned by dataset-specific experiment builders."""

    labels: tuple[str, ...]
    train_base: pd.DataFrame
    train_records: pd.DataFrame
    val_records: pd.DataFrame
    test_records: pd.DataFrame
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    data_summary: dict[str, Any]
    split_strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)
