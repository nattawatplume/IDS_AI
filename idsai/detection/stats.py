"""Statistical layer: robust-z maximum and sequential CUSUM, both read off existing features."""

from typing import Any

import numpy as np
import polars as pl

from idsai.contracts.detector import Detector, Evidence, feature_matrix

CUSUM_COLUMNS = ["cusum_ret", "cusum_imb"]


class RobustZDetector(Detector):
    """Score = row-wise max of the listed robust-z features.

    The max is of the signed z (not |z|): manipulation shows up as upward anomalies, so a very
    low volume must not raise an alert. Evidence ranks the listed features by |z|.
    """

    name = "stat.robust_z"
    layer = "stat"
    track = "T"

    def __init__(self, features: list[str]):
        if not features:
            raise ValueError("RobustZDetector needs at least one feature")
        self.technique_ids = ["ANY"]
        self.requires = list(features)

    def score(self, windows: pl.DataFrame) -> np.ndarray:
        # NaN in any required feature propagates through the max
        return feature_matrix(windows, self.requires).max(axis=1)

    def evidence(self, row: dict[str, Any]) -> Evidence:
        zs = [(f, float(row[f])) for f in self.requires]
        return Evidence.from_items((f, z, abs(z)) for f, z in zs)

    def state_dict(self) -> dict[str, Any]:
        return {"features": list(self.requires)}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.requires = list(d["features"])


class CusumDetector(Detector):
    """Score = max(cusum_ret, cusum_imb): sustained drift in return or buy imbalance."""

    name = "stat.cusum"
    layer = "stat"
    track = "T"

    def __init__(self) -> None:
        self.technique_ids = ["MM-T001"]
        self.requires = list(CUSUM_COLUMNS)

    def score(self, windows: pl.DataFrame) -> np.ndarray:
        return feature_matrix(windows, self.requires).max(axis=1)

    def evidence(self, row: dict[str, Any]) -> Evidence:
        return Evidence.from_items((f, float(row[f]), float(row[f])) for f in self.requires)
