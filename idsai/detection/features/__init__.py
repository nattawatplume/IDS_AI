"""Features: registry, 1 s bars, shared NumPy kernel, batch window driver."""

from idsai.detection.features.bars import BARS, trades_to_bars
from idsai.detection.features.kernel import BarArrays, Baseline, Kernel
from idsai.detection.features.registry import (
    REGISTRY,
    FeatureSpec,
    feature_columns,
    model_features,
    spec_for,
)
from idsai.detection.features.windows import compute_windows, windows_schema

__all__ = [
    "BARS",
    "REGISTRY",
    "BarArrays",
    "Baseline",
    "FeatureSpec",
    "Kernel",
    "compute_windows",
    "feature_columns",
    "model_features",
    "spec_for",
    "trades_to_bars",
    "windows_schema",
]
