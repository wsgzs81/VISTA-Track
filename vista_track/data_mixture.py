"""Data-mixture helpers for the current VISTA-Track first-stage recipe."""

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class DatasetWeight:
    name: str
    ratio: float


DEFAULT_STAGE1_MIXTURE = [
    DatasetWeight("GOT10K_train_full", 4.0),
    DatasetWeight("LASOT", 2.0),
    DatasetWeight("TRACKINGNET", 3.0),
]


def normalize_mixture(items: Iterable[DatasetWeight]) -> Dict[str, float]:
    items = list(items)
    total = sum(max(0.0, item.ratio) for item in items)
    if total <= 0:
        raise ValueError("Dataset mixture must contain at least one positive ratio.")
    return {item.name: max(0.0, item.ratio) / total for item in items}


def names_and_ratios(items: Iterable[DatasetWeight]) -> tuple[List[str], List[float]]:
    items = list(items)
    return [item.name for item in items], [item.ratio for item in items]
