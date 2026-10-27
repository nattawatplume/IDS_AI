"""Detector contract shared by rules, statistical detectors, baselines and AI models."""

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, field_validator

MAX_EVIDENCE = 5

Layer = Literal["rule", "stat", "ai", "baseline"]
Track = Literal["T", "B"]


class Evidence(BaseModel):
    """Top contributing features as `(feature, value, contribution)`, at most 5, best first."""

    top: list[tuple[str, float, float]] = []

    @field_validator("top")
    @classmethod
    def _check_top(cls, v: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
        if len(v) > MAX_EVIDENCE:
            raise ValueError(f"evidence holds at most {MAX_EVIDENCE} items, got {len(v)}")
        contributions = [c for _, _, c in v]
        if any(a < b for a, b in zip(contributions, contributions[1:], strict=False)):
            raise ValueError("evidence must be sorted by contribution, descending")
        return v

    @classmethod
    def from_items(cls, items: Iterable[tuple[str, float, float]]) -> "Evidence":
        """Drop NaN items, sort by contribution descending (stable) and keep the top 5."""
        kept = [
            (str(f), float(v), float(c))
            for f, v, c in items
            if not (math.isnan(v) or math.isnan(c))
        ]
        kept.sort(key=lambda t: -t[2])
        return cls(top=kept[:MAX_EVIDENCE])


def feature_matrix(windows: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """`(n_rows, len(columns))` float64 matrix; null becomes NaN."""
    if not columns:
        return np.empty((windows.height, 0), dtype=np.float64)
    return windows.select(
        [pl.col(c).cast(pl.Float64).fill_null(float("nan")) for c in columns]
    ).to_numpy()


class Detector(ABC):
    """One anomaly detector.

    `technique_ids == ["ANY"]` marks a technique-agnostic detector. `score` returns float64 with
    NaN wherever a required feature is NaN (the Chief treats NaN as p = 1).
    """

    name: str
    layer: Layer
    track: Track = "T"
    technique_ids: list[str]
    requires: list[str]
    signature_id: int | None = None
    rev: int | None = None

    def fit(self, normal: pl.DataFrame) -> None:  # noqa: B027 - intentional no-op default
        """Fit on normal-period window rows (default: nothing to fit)."""

    @abstractmethod
    def score(self, windows: pl.DataFrame) -> np.ndarray:
        """Higher = more anomalous; float64, one value per row."""

    def fired(self, scores: np.ndarray) -> np.ndarray:
        """Whether each score counts as a hit (non-rule detectors always fire; rules override)."""
        return np.ones(np.shape(scores), dtype=bool)

    @abstractmethod
    def evidence(self, row: dict[str, Any]) -> Evidence:
        """Top contributing features for one window row."""

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, d: dict[str, Any]) -> None:  # noqa: B027 - default: stateless
        """Restore state produced by `state_dict` (default: stateless)."""
