"""Rule layer: signature-based detectors read from `configs/signatures.yaml`.

A signature is a conjunction of threshold conditions on window features. Each condition has a
normalized margin (positive = satisfied, zero = exactly at the threshold); the score of a row is
the minimum margin, so `score >= 0` means every condition holds. NaN in any required feature
gives a NaN score and the signature does not fire.
"""

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field

from idsai.config import REPO_ROOT
from idsai.contracts.detector import Detector, Evidence, feature_matrix

SIGNATURES_PATH = REPO_ROOT / "configs" / "signatures.yaml"
THR_FLOOR = 1e-9
KK_SID = 100103  # Kamps-Kleinberg baseline signature


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    op: Literal[">=", "<=", "abs<="]
    thr: float


class Signature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sid: int
    technique: str  # e.g. "MM-T001.001"
    layer: Literal["rule", "baseline"]
    rev: int = 1
    name: str
    conditions: list[Condition] = Field(min_length=1)


def load_signatures(path: str | Path | None = None) -> list[Signature]:
    """Read signatures (default: `configs/signatures.yaml` at the repo root)."""
    p = Path(path) if path is not None else SIGNATURES_PATH
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    sigs = [Signature.model_validate(s) for s in raw.get("signatures", [])]
    sids = [s.sid for s in sigs]
    if len(set(sids)) != len(sids):
        raise ValueError(f"duplicate signature ids in {p}")
    return sigs


def _margin(x: np.ndarray, c: Condition) -> np.ndarray:
    """Normalized margin of a condition: `>= 0` iff satisfied, NaN where `x` is NaN."""
    denom = max(abs(c.thr), THR_FLOOR)
    if c.op == ">=":
        return (x - c.thr) / denom
    if c.op == "<=":
        return (c.thr - x) / denom
    return (c.thr - np.abs(x)) / denom


class SignatureDetector(Detector):
    layer: Literal["rule", "baseline"]
    track = "T"

    def __init__(self, sig: Signature):
        self.sig = sig
        self.layer = sig.layer
        self.name = "baseline.kk" if sig.sid == KK_SID else f"{sig.layer}.{sig.sid}"
        self.technique_ids = [sig.technique.split(".")[0]]
        self.requires = [c.feature for c in sig.conditions]
        self.signature_id = sig.sid
        self.rev = sig.rev

    def score(self, windows: pl.DataFrame) -> np.ndarray:
        x = feature_matrix(windows, self.requires)
        margins = np.stack([_margin(x[:, i], c) for i, c in enumerate(self.sig.conditions)], axis=1)
        return margins.min(axis=1)  # NaN propagates

    def fired(self, scores: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.asarray(scores) >= 0.0  # NaN compares False

    def evidence(self, row: dict[str, Any]) -> Evidence:
        items = []
        for c in self.sig.conditions:
            x = float(row[c.feature])
            if math.isnan(x):
                continue
            items.append((c.feature, x, float(_margin(np.array([x]), c)[0])))
        return Evidence.from_items(items)
