"""Deterministic Thai template explainer (spec section 10).

Every number in the text comes from the alert (`evidence`, `fused_p`, `agreement`, `ts_event_us`)
or from a registry description. The explainer never reads detector scores and never computes
severity or any decision. `validate_numbers` is reusable to vet other (e.g. LLM) rewrites.

Evidence value formats, by registry unit (z-score columns `z_*` are always "robust z = x.x"):
    log-return, relative -> percent, 2 decimals   | ratio, sigma -> 2 decimals
    trades, orders, pairs -> 0 decimals            | quote -> 1 decimal | other -> 3 significant
"""

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache

from idsai.detection.features.registry import WINDOW_BASES, spec_for
from idsai.detection.taxonomy import load_taxonomy

BANNED_TH = ["โกง", "มิจฉาชีพ", "ผู้ปั่น", "คนร้าย", "ฉ้อโกง"]

UNCLASSIFIED_TH = "ความผิดปกติที่ยังไม่จัดประเภท"
NO_EVIDENCE_TH = "ไม่มีหลักฐานรายคุณลักษณะแนบมา"
MAX_FACTS = 3

_UNIT_FORMATS: dict[str, str] = {
    "ratio": "{:.2f}",
    "sigma": "{:.2f}",
    "trades": "{:.0f}",
    "orders": "{:.0f}",
    "pairs": "{:.0f}",
    "quote": "{:.1f}",
}
_PERCENT_UNITS = {"log-return", "relative"}
_WINDOW_RE = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class Explanation:
    text_th: str
    facts: list[tuple[str, str]]  # (label, formatted value), same items as the text


# ---------------------------------------------------------------- number parsing / validation

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?P<sign>[-\u2212+]?)"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:[eE](?P<exp>[-+]?\d+))?"
    r"(?!\d|[A-Z]{2,})"
)


def numbers_in(text: str) -> list[float]:
    """Numbers appearing in `text`: ints, decimals, sci-notation, signs, thousands separators,
    percents (value only) and Thai digits. Digits inside identifiers (MM-T001, 1000PEPEUSDT,
    z_60s) are not numbers; a hyphen right after a letter or digit is not a sign."""
    out = []
    for m in _NUM_RE.finditer(text.translate(_THAI_DIGITS)):
        x = float(m["num"].replace(",", ""))
        if m["exp"] is not None:
            x *= 10.0 ** int(m["exp"])
        if m["sign"] in ("-", "\u2212"):
            x = -x
        out.append(x)
    return out


def validate_numbers(text: str, allowed: Iterable[float], rel_tol: float = 0.01) -> bool:
    """True when every number in `text` matches some finite allowed value within `rel_tol`
    (relative only: 0 matches only 0; sign matters)."""
    ok = [a for a in allowed if math.isfinite(a)]
    return all(
        any(math.isclose(x, a, rel_tol=rel_tol, abs_tol=0.0) for a in ok) for x in numbers_in(text)
    )


# ---------------------------------------------------------------- rendering


@lru_cache(maxsize=1)
def _names() -> dict[str, str]:
    names = {}
    for tid, entry in load_taxonomy().items():
        names[tid] = entry["name_th"]
        for sub in entry.get("sub_techniques", []):
            names[sub["id"]] = sub["name_th"]
    names["MM-T000"] = UNCLASSIFIED_TH
    return names


def _fmt(value: float, unit: str) -> tuple[str, list[float]]:
    """Formatted value and the numbers it can legitimately show (displayed and raw)."""
    if unit in _PERCENT_UNITS:
        s = f"{value * 100:.2f}%"
        return s, [float(f"{value * 100:.2f}"), value * 100, value]
    s = _UNIT_FORMATS.get(unit, "{:.3g}").format(value)
    return s, [float(s), value]


def _fact(name: str, value: float) -> tuple[str, str, list[float]]:
    """(label, formatted value, allowed numbers) for one evidence item."""
    if not math.isfinite(value):
        return _label_generic(name), "ไม่มีค่า", []
    try:
        spec = spec_for(name)
    except KeyError:
        s = f"{value:.3g}"
        return _label_generic(name), s, [float(s), value]
    label = spec.desc_th
    allowed = numbers_in(spec.desc_th)
    m = _WINDOW_RE.search(name)
    if m and spec.name in WINDOW_BASES:
        label += f" (ช่วง {m[1]} วินาที)"
        allowed.append(float(m[1]))
    if name.startswith("z_"):
        s = f"{value:.1f}"
        return label, f"robust z = {s}", [*allowed, float(s), value]
    s, nums = _fmt(value, spec.unit)
    return label, s, [*allowed, *nums]


def _label_generic(name: str) -> str:
    return f"ลักษณะที่ตรวจพบ {name}"


def _render(alert) -> tuple[str, list[tuple[str, str]], list[float]]:
    allowed: list[float] = []
    ranked = sorted(
        alert.evidence, key=lambda e: -(e[2] if math.isfinite(e[2]) else -math.inf)
    )  # stable: ties keep evidence order
    facts: list[tuple[str, str]] = []
    for name, value, _contribution in ranked[:MAX_FACTS]:
        label, shown, nums = _fact(name, value)
        facts.append((label, shown))
        allowed += nums
        allowed += numbers_in(label)
    facts_text = (
        ", ".join(f"{label} {shown}" for label, shown in facts) if facts else NO_EVIDENCE_TH
    )

    t = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=alert.ts_event_us)
    p_shown = f"{alert.fused_p:.1e}"
    technique_name = _names().get(alert.technique_id, UNCLASSIFIED_TH)
    text = (
        f"พบความผิดปกติทางสถิติ ({alert.technique_id} {technique_name}) "
        f"ที่ {alert.symbol} เวลา {t:%H:%M:%S} UTC: {facts_text}; "
        f"{alert.agreement} จาก 3 ชั้นตรวจจับเห็นพ้อง; "
        f"ความหายากเทียบตลาดปกติ p ≈ {p_shown}"
    )
    allowed += [
        float(t.hour), float(t.minute), float(t.second),
        float(alert.agreement), 3.0,
        float(p_shown), alert.fused_p,
        *numbers_in(technique_name),
    ]  # fmt: skip
    return text, facts, allowed


def explain(alert) -> Explanation:
    """Thai explanation of `alert`; facts are the top-3 evidence items by contribution."""
    text, facts, _ = _render(alert)
    return Explanation(text_th=text, facts=facts)


def allowed_numbers(alert) -> list[float]:
    """Exact values `explain(alert)` renders (displayed and raw), for `validate_numbers`."""
    return _render(alert)[2]
