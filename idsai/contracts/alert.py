"""Alert contract emitted by the detection pipeline."""

from typing import Literal

from pydantic import BaseModel, Field


class DetectorRef(BaseModel):
    name: str
    layer: str  # "rule" | "stat" | "ai" (fusion layers); baselines may use other names
    p: float = Field(ge=0.0, le=1.0)
    score: float | None  # None when the detector's score is non-finite (e.g. NaN feature)


class Alert(BaseModel):
    alert_id: str
    ts_event_us: int
    symbol: str
    technique_id: str
    # the peak row's own technique; it differs from `technique_id` when another row of the
    # episode classified it. None on alerts written before this field existed.
    peak_technique_id: str | None = None
    signature_id: int | None = None
    rev: int | None = None
    severity: int = Field(ge=1, le=3)
    status: Literal["open", "updated", "closed"]
    fused_p: float = Field(ge=0.0, le=1.0)
    detectors: list[DetectorRef]
    agreement: int = Field(ge=0, le=3)
    evidence: list[tuple[str, float, float]]
    window: tuple[int, int]
    dedup_key: str
    suppress_for_s: int
    config_hash: str
    model_versions: dict[str, str]
    provenance: Literal["live", "replay", "sim"]
    schema_v: Literal[1] = 1
