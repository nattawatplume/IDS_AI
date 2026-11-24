"""Alert contract emitted by the detection pipeline."""

from typing import Any, Literal

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

Phase = Literal["pump", "sell"]
TechniqueSource = Literal["naming_rule"]
PUMP_PHASE_TECHNIQUES = frozenset({"MM-T001.001", "MM-T001.002"})
SELL_PHASE_TECHNIQUES = frozenset({"MM-T001.003"})
SELL_PHASE_PARENTS = frozenset({"MM-T006"})


def phase_of(technique: str | None) -> Phase | None:
    """Market phase a (sub-)technique names: MM-T001.001/.002 -> "pump" (กำลังพุ่ง),
    MM-T001.003 and MM-T006.* -> "sell" (กำลังเทขาย), anything else -> None."""
    if technique is None:
        return None
    if technique in PUMP_PHASE_TECHNIQUES:
        return "pump"
    if technique in SELL_PHASE_TECHNIQUES or technique.split(".")[0] in SELL_PHASE_PARENTS:
        return "sell"
    return None


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
    # M4: "pump" / "sell" from the episode's signature (`phase_of`); set only when sell-side
    # rules run. None (the default, and every alert written before M4) is left out of dumps, so
    # alerts made without the sell side serialize exactly as before.
    phase: Phase | None = None
    # M3: "naming_rule" when `technique_id` / `signature_id` / `rev` come from a naming-only rule
    # (`chief.attr_rules`; layer "attr", กฎระบุเทคนิค), not from a detection row. Such a rule never
    # entered F, so `fused_p`, `detectors`, `agreement` and `evidence` are the detection's alone.
    # None (the default, and every alert before M3) is left out of dumps like `phase`.
    technique_source: TechniqueSource | None = None

    @model_serializer(mode="wrap")
    def _drop_unset_phase(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        out = handler(self)
        if self.phase is None:
            out.pop("phase", None)
        if self.technique_source is None:
            out.pop("technique_source", None)
        return out
