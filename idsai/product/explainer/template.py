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
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
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

_MINUS_EXTRA = {0x2212, 0x02D7, 0x2796, 0x207B, 0x208B}
_TECH_ID_RE = re.compile(r"(?<![A-Za-z0-9])MM-T\d{3}(?:\.\d{3})?(?![0-9.])")
_NUM_RE = re.compile(
    r"(?:(?<![A-Za-z0-9])(?P<sign>[-+])\s*)?"
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE](?P<exp>[-+]?\d+))?"
)


def _normalise(text: str) -> str:
    """ASCII digits for every Unicode decimal digit (Nd), a plain minus for every dash-like
    character (Pd and look-alikes), and no format characters (Cf, e.g. zero-width)."""
    out = []
    for c in text:
        cat = unicodedata.category(c)
        if cat == "Cf":
            continue
        if cat == "Nd":
            out.append(str(unicodedata.digit(c)))
        elif cat == "Pd" or ord(c) in _MINUS_EXTRA:
            out.append("-")
        else:
            out.append(c)
    return "".join(out)


def numbers_in(text: str) -> list[float]:
    """Every number in `text`, strictly: ints, decimals (`.5` too), sci-notation, signs,
    thousands separators, percents (value only) and any Unicode decimal digits. The only
    exemption is a technique id (`MM-T001`, `MM-T001.002`, exactly 3 digits each); digits glued
    to letters (`1000PEPEUSDT`, `z9.9`) count, and malformed runs (`0.5.1`, `1_000`) yield every
    digit run. Dash-like characters are a minus (format characters are dropped); a minus,
    optionally followed by whitespace, before a number is a sign unless it directly follows a
    letter or digit (then it is a separator). A value that overflows, or underflows to zero
    from a nonzero mantissa, is nan."""
    text = _TECH_ID_RE.sub(" ", _normalise(text))
    out = []
    for m in _NUM_RE.finditer(text):
        num = m["num"].replace(",", "")
        x = float(f"{num}e{m['exp']}" if m["exp"] is not None else num)
        if math.isinf(x) or (x == 0.0 and any(c in "123456789" for c in num)):
            x = math.nan
        out.append(-x if m["sign"] == "-" else x)
    return out


def validate_numbers(text: str, allowed: Iterable[float], rel_tol: float = 0.01) -> bool:
    """True when every number in `text` is finite and matches some finite allowed value within
    `rel_tol` (relative only: 0 matches only 0; sign matters)."""
    ok = [a for a in allowed if math.isfinite(a)]
    return all(
        math.isfinite(x) and any(math.isclose(x, a, rel_tol=rel_tol, abs_tol=0.0) for a in ok)
        for x in numbers_in(text)
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


def _no_neg_zero(s: str) -> str:
    """Drop the sign of a formatted value that rounds to zero ("-0.00" -> "0.00")."""
    return s.replace("-", "") if float(s.rstrip("%")) == 0.0 else s


def _fmt(value: float, unit: str) -> tuple[str, list[float]]:
    """Formatted value and the numbers it can legitimately show (displayed and raw)."""
    if unit in _PERCENT_UNITS and math.isfinite(value * 100):
        s = _no_neg_zero(f"{value * 100:.2f}%")
        return s, [float(s.rstrip("%")), value * 100]
    s = _no_neg_zero(_UNIT_FORMATS.get(unit, "{:.3g}").format(value))
    return s, [float(s), value]


def _fact(name: str, value: float) -> tuple[str, str, list[float]]:
    """(label, formatted value, allowed numbers) for one evidence item."""
    if not math.isfinite(value):
        return _label_generic(name), "ไม่มีค่า", []
    try:
        spec = spec_for(name)
    except KeyError:
        s = _no_neg_zero(f"{value:.3g}")
        return _label_generic(name), s, [float(s), value]
    label = spec.desc_th
    allowed = numbers_in(spec.desc_th)
    m = _WINDOW_RE.search(name)
    if m and spec.name in WINDOW_BASES:
        label += f" (ช่วง {m[1]} วินาที)"
        allowed.append(float(m[1]))
    if name.startswith("z_"):
        s = _no_neg_zero(f"{value:.1f}")
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

    sod = (alert.ts_event_us // 1_000_000) % 86_400  # UTC seconds of day (floor, any int)
    hh, mm, ss = sod // 3600, sod // 60 % 60, sod % 60
    p_shown = f"{alert.fused_p:.1e}"
    technique_name = _names().get(alert.technique_id, UNCLASSIFIED_TH)
    text = (
        f"พบความผิดปกติทางสถิติ ({alert.technique_id} {technique_name}) "
        f"ที่ {alert.symbol} เวลา {hh:02d}:{mm:02d}:{ss:02d} UTC: {facts_text}; "
        f"{alert.agreement} จาก 3 ชั้นตรวจจับเห็นพ้อง; "
        f"ความหายากเทียบตลาดปกติ p ≈ {p_shown}"
    )
    allowed += [
        float(hh), float(mm), float(ss),
        float(alert.agreement), 3.0,
        float(p_shown), alert.fused_p,
        *numbers_in(technique_name),
        *numbers_in(alert.symbol),
        *numbers_in(alert.technique_id),
    ]  # fmt: skip
    return text, facts, allowed


def explain(alert) -> Explanation:
    """Thai explanation of `alert`; facts are the top-3 evidence items by contribution."""
    text, facts, _ = _render(alert)
    return Explanation(text_th=text, facts=facts)


def allowed_numbers(alert) -> list[float]:
    """Exact values `explain(alert)` renders (displayed and raw), for `validate_numbers`."""
    return _render(alert)[2]
