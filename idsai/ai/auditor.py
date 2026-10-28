"""Auditor: the "meta-IDS" that checks the health of the IDS itself.

Each check returns a `Check` (`ok` / `warn` / `fail`, a JSON-serialisable `value` and a Thai
`detail_th`); `Auditor.report` runs the checks whose inputs are supplied and returns a
`HealthReport` whose `overall` is the worst status. The checks look at the system (alert rate
against the budget, calibration of p-values, data gaps, latency, training exclusion, model
staleness, baseline warm-up), never at any trader.

`leakage_violations` is the label leakage guard: a pure `ast` scan proving that detector code
(`detection/` except `detection/simulator/`, `ai/` except `ai/learning.py`) never imports the
evaluation package or references the evaluation label schema by name.
"""

import ast
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, computed_field
from scipy import stats

from idsai.config import Settings

Status = Literal["ok", "warn", "fail"]
_RANK: dict[str, int] = {"ok": 0, "warn": 1, "fail": 2}
DAY_US = 86_400 * 1_000_000
ALERT_RATE_P = 0.01
CALIBRATION_P = 0.01
CALIBRATION_MIN_N = 200
LATENCY_P99_MS = 2000.0
EXCLUSION_MAX = 0.02
RESOLUTION_MARGIN = 10
NO_DATA_TH = "ข้อมูลไม่พอ"
CHECKS = (
    "alert_rate",
    "calibration",
    "calibration_resolution",
    "latency",
    "trade_gaps",
    "exclusion",
    "staleness",
    "warmup",
)


class Check(BaseModel):
    """One health check. `value` holds only JSON types (float/int/str/bool/None/list/dict)."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: Status
    value: Any = None
    detail_th: str


class HealthReport(BaseModel):
    """The checks of one audit at `ts_us`; `overall` = worst status (empty => "ok")."""

    model_config = ConfigDict(frozen=True)

    checks: list[Check]
    ts_us: int

    @computed_field
    @property
    def overall(self) -> Status:
        return max((c.status for c in self.checks), key=_RANK.__getitem__, default="ok")


def _num(x: Any) -> float | None:
    """A plain finite float, or None (NaN/inf would break strict JSON)."""
    f = float(x)
    return f if math.isfinite(f) else None


def _unreadable_inputs(
    inputs: dict[str, Any], *, counts: tuple[str, ...] = (), nonneg: tuple[str, ...] = ()
) -> list[str]:
    """Names of the inputs that are not a usable number: NaN, inf, None, bool, str or
    non-numeric; for `counts` also a negative or non-integral value; for `nonneg` a negative."""
    bad = []
    for k, v in inputs.items():
        try:
            f = float(v) if not isinstance(v, bool | str) else math.nan
        except (TypeError, ValueError):
            f = math.nan
        ok = math.isfinite(f)
        if ok and k in counts:
            ok = f >= 0 and f == int(f)
        elif ok and k in nonneg:
            ok = f >= 0
        if not ok:
            bad.append(k)
    return bad


def _unreadable(name: str, inputs: dict[str, Any], bad: list[str]) -> Check:
    """A monitor that cannot measure never reports health: warn, raw inputs kept (bad => None)."""
    value: dict[str, Any] = {
        k: None if k in bad else int(v) if isinstance(v, int | np.integer) else float(v)
        for k, v in inputs.items()
    }
    value["unreadable"] = bad
    detail = (
        f"อ่านค่าไม่ได้: {', '.join(bad)} ไม่ใช่ตัวเลขที่ใช้ได้ (NaN, inf, ว่าง, ติดลบ "
        f"หรือไม่ใช่จำนวนเต็มในช่องที่เป็นจำนวนนับ) จึงประเมิน {name} ไม่ได้"
    )
    return Check(name=name, status="warn", value=value, detail_th=detail)


def _percentile(x_sorted: np.ndarray, q: float) -> float:
    """Linear percentile (numpy's default) of ascending values that may end in +inf."""
    if np.isfinite(x_sorted[-1]):
        return float(np.percentile(x_sorted, q))
    h = (x_sorted.size - 1) * q / 100
    lo, hi = math.floor(h), math.ceil(h)
    t = h - lo
    if math.isinf(x_sorted[lo]) or (t > 0 and math.isinf(x_sorted[hi])):
        return math.inf
    return float(x_sorted[lo] + t * (x_sorted[hi] - x_sorted[lo]))


class Auditor:
    """Health checks parameterised by `Settings` (Chief budget, baseline sampling)."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg

    def alert_rate(
        self, n_alerts: int, symbol_days: float, budget_per_symbol_day: float | None = None
    ) -> Check:
        """More alerts than the budget allows: warn iff `P(X >= n_alerts) < 0.01` under
        `Poisson(budget x symbol_days)`. Budget defaults to `cfg.chief.budget_per_symbol_day`."""
        budget = (
            self.cfg.chief.budget_per_symbol_day
            if budget_per_symbol_day is None
            else budget_per_symbol_day
        )
        inputs = {
            "n_alerts": n_alerts,
            "symbol_days": symbol_days,
            "budget_per_symbol_day": budget,
        }
        if bad := _unreadable_inputs(
            inputs, counts=("n_alerts",), nonneg=("budget_per_symbol_day", "symbol_days")
        ):
            return _unreadable("alert_rate", inputs, bad)
        budget = float(budget)
        n = int(n_alerts)
        if symbol_days <= 0:
            value = {"n_alerts": n, "expected": None, "p": None}
            return Check(
                name="alert_rate",
                status="ok",
                value=value,
                detail_th=f"{NO_DATA_TH}: ยังไม่มีจำนวนวันของสัญลักษณ์ (symbol-day) ให้เทียบงบ",
            )
        mu = budget * float(symbol_days)
        p = float(stats.poisson.sf(n - 1, mu))
        if not math.isfinite(p):  # never fall through to "ok" on an unreadable p-value
            return _unreadable("alert_rate", {**inputs, "p": p}, ["p"])
        value = {"n_alerts": n, "expected": _num(mu), "p": _num(p)}
        if p < ALERT_RATE_P:
            detail = (
                f"จำนวนการแจ้งเตือน {n} สูงกว่าที่งบคาดไว้ ({mu:.1f}) อย่างมีนัย "
                f"(Poisson p = {p:.2g} < {ALERT_RATE_P:g}) ควรตรวจสอบการสอบเทียบหรือสภาพตลาด"
            )
            return Check(name="alert_rate", status="warn", value=value, detail_th=detail)
        detail = f"จำนวนการแจ้งเตือน {n} อยู่ในงบ (คาดไว้ {mu:.1f}, Poisson p = {p:.2g})"
        return Check(name="alert_rate", status="ok", value=value, detail_th=detail)

    def calibration(self, pvals: Any) -> Check:
        """Calibration drift: KS test of p-values vs uniform; p < 0.01 => warn.

        Callers pass the p-values of one continuous detector on recent normal, scored rows only
        (rule p-values are mostly 1.0 by design and are not uniform). Non-finite values are
        dropped (D3) and counted in `value["n_dropped"]`; fewer than 200 values => ok with a
        note saying nothing was assessed, or warn when dropping unreadable values is what left
        too few (the input was there but could not be measured).
        """
        raw = np.asarray(pvals, dtype=np.float64).ravel()
        p = raw[np.isfinite(raw)]
        n, n_dropped = int(p.size), int(raw.size - p.size)
        if n < CALIBRATION_MIN_N:
            value = {"n": n, "n_dropped": n_dropped, "ks_stat": None, "p": None}
            detail = (
                f"มี p-value ที่อ่านได้ {n} ค่า (ต้องมีอย่างน้อย {CALIBRATION_MIN_N}, "
                f"ตัดค่าที่อ่านไม่ได้ {n_dropped} ค่า) ยังไม่ประเมินการสอบเทียบ"
            )
            if n_dropped:
                return Check(
                    name="calibration", status="warn", value=value, detail_th=f"อ่านค่าไม่ได้: {detail}"
                )
            return Check(
                name="calibration", status="ok", value=value, detail_th=f"{NO_DATA_TH}: {detail}"
            )
        res = stats.kstest(p, "uniform")
        value = {
            "n": n,
            "n_dropped": n_dropped,
            "ks_stat": _num(res.statistic),
            "p": _num(res.pvalue),
        }
        if res.pvalue < CALIBRATION_P:
            detail = (
                f"การกระจายของ p-value เบี่ยงจาก uniform (KS p = {res.pvalue:.2g}, n = {n}) "
                "แนะนำ recalibrate"
            )
            return Check(name="calibration", status="warn", value=value, detail_th=detail)
        detail = f"การกระจายของ p-value ใกล้เคียง uniform (KS p = {res.pvalue:.2g}, n = {n})"
        return Check(name="calibration", status="ok", value=value, detail_th=detail)

    def calibration_resolution(
        self,
        n_cal: int,
        n_detectors: int,
        rows_per_symbol_day: float,
        budget: float | None = None,
    ) -> Check:
        """Can the calibration set resolve the budget? `rate = budget / rows_per_symbol_day`.

        warn iff `rate < 10 * n_detectors / (n_cal + 1)`; fail iff `rate < n_detectors /
        (n_cal + 1)` (the smallest reachable p-value is too coarse: the Chief would refuse).
        Budget defaults to `cfg.chief.budget_per_symbol_day`. Non-finite inputs => warn;
        `n_cal < 0` or `n_detectors < 1` => ValueError (a caller bug, not a health state).
        """
        b = self.cfg.chief.budget_per_symbol_day if budget is None else budget
        inputs = {
            "n_cal": n_cal,
            "n_detectors": n_detectors,
            "rows_per_symbol_day": rows_per_symbol_day,
            "budget": b,
        }
        if bad := _unreadable_inputs(inputs, nonneg=("budget", "rows_per_symbol_day")):
            return _unreadable("calibration_resolution", inputs, bad)
        if n_cal < 0 or n_detectors < 1:
            raise ValueError(f"need n_cal >= 0 and n_detectors >= 1, got {n_cal}, {n_detectors}")
        b = float(b)
        if rows_per_symbol_day <= 0:
            value = {"rate": None, "warn_below": None, "fail_below": None}
            detail = f"{NO_DATA_TH}: ไม่ทราบจำนวนแถวต่อ symbol-day"
            return Check(name="calibration_resolution", status="ok", value=value, detail_th=detail)
        rate = b / float(rows_per_symbol_day)
        fail_below = int(n_detectors) / (int(n_cal) + 1)
        warn_below = RESOLUTION_MARGIN * int(n_detectors) / (int(n_cal) + 1)
        value = {"rate": _num(rate), "warn_below": warn_below, "fail_below": fail_below}
        if rate < fail_below:
            status: Status = "fail"
            detail = (
                f"ข้อมูลสอบเทียบ {n_cal} แถวน้อยเกินไปสำหรับงบ {b:g} ต่อ symbol-day: "
                "p-value ที่เล็กที่สุดหยาบเกินกว่างบ ระบบจะไม่สอบเทียบ ต้องเพิ่มข้อมูลสอบเทียบ"
            )
        elif rate < warn_below:
            status = "warn"
            detail = (
                f"ข้อมูลสอบเทียบ {n_cal} แถวเพียงพอแบบเฉียดฉิวสำหรับงบ {b:g} ต่อ symbol-day: "
                "τ อาจหยาบหรืออิ่มตัว ควรเพิ่มข้อมูลสอบเทียบ"
            )
        else:
            status = "ok"
            detail = f"ข้อมูลสอบเทียบ {n_cal} แถวละเอียดพอสำหรับงบ {b:g} ต่อ symbol-day"
        return Check(name="calibration_resolution", status=status, value=value, detail_th=detail)

    def latency(self, samples_ms: Any) -> Check:
        """End-to-end latency: warn iff p99 > 2000 ms; no samples => ok with a note.

        NaN (and -inf) samples are dropped; +inf is the worst latency and counts (p99 = inf =>
        warn, shown as None in `value`, with the count in `n_inf`). Samples given but none
        readable => warn; any negative sample (clock skew between hosts) => warn.
        """
        raw = np.asarray(samples_ms, dtype=np.float64).ravel()
        x = np.sort(raw[~np.isnan(raw) & (raw != -np.inf)])
        if not x.size:
            value = {"n": 0, "n_inf": 0, "p50_ms": None, "p99_ms": None}
            if raw.size:
                detail = (
                    f"อ่านค่าไม่ได้: ตัวอย่างเวลาแฝง (latency) {raw.size} ค่าอ่านไม่ได้ทั้งหมด (NaN หรือ -inf)"
                )
                return Check(name="latency", status="warn", value=value, detail_th=detail)
            detail = f"{NO_DATA_TH}: ยังไม่มีตัวอย่างเวลาแฝง (latency)"
            return Check(name="latency", status="ok", value=value, detail_th=detail)
        p50, p99 = _percentile(x, 50), _percentile(x, 99)
        n_inf = int(np.isinf(x).sum())
        n_neg = int((x < 0).sum())
        value = {
            "n": int(x.size),
            "n_inf": n_inf,
            "n_negative": n_neg,
            "p50_ms": _num(p50),
            "p99_ms": _num(p99),
        }
        if n_neg:
            detail = f"อ่านค่าไม่ได้: มีเวลาแฝงติดลบ {n_neg} ค่า นาฬิกาของเครื่องอาจไม่ตรงกัน จึงเชื่อค่า p99 ไม่ได้"
            return Check(name="latency", status="warn", value=value, detail_th=detail)
        if p99 > LATENCY_P99_MS:
            detail = f"เวลาแฝง p99 = {p99:.0f} ms เกิน {LATENCY_P99_MS:.0f} ms (p50 = {p50:.0f} ms)"
            if n_inf:
                detail += f"; มีตัวอย่างที่ไม่สิ้นสุด (inf) {n_inf} ค่า"
            return Check(name="latency", status="warn", value=value, detail_th=detail)
        detail = f"เวลาแฝง p99 = {p99:.0f} ms ไม่เกิน {LATENCY_P99_MS:.0f} ms (p50 = {p50:.0f} ms)"
        return Check(name="latency", status="ok", value=value, detail_th=detail)

    def trade_gaps(self, trades: pl.DataFrame) -> Check:
        """Missing trades: per symbol, jumps > 1 between sorted unique non-simulated trade_ids
        (simulated trades have `trade_id < 0` and are ignored). Any gap => warn; rows with a
        null `trade_id` or `symbol` cannot be checked => warn."""
        cols = trades.select("symbol", "trade_id")
        if cols.schema["trade_id"].is_float():  # off-schema input: NaN ids are unreadable too
            cols = cols.with_columns(pl.col("trade_id").fill_nan(None))
        n_null = cols.filter(pl.col("trade_id").is_null() | pl.col("symbol").is_null()).height
        real = cols.filter(pl.col("trade_id") >= 0, pl.col("symbol").is_not_null())
        by_symbol: dict[str, dict[str, int]] = {}
        for (sym,), part in sorted(real.partition_by("symbol", as_dict=True).items()):
            ids = np.unique(part["trade_id"].to_numpy())
            d = np.diff(ids)
            jumps = d[d > 1]
            by_symbol[str(sym)] = {"n_gaps": int(jumps.size), "n_missing": int((jumps - 1).sum())}
        n_gaps = sum(v["n_gaps"] for v in by_symbol.values())
        n_missing = sum(v["n_missing"] for v in by_symbol.values())
        value = {
            "n_gaps": n_gaps,
            "n_missing": n_missing,
            "n_unreadable": n_null,
            "by_symbol": by_symbol,
        }
        problems = []
        if n_null:
            problems.append(f"อ่านค่าไม่ได้: trade_id หรือ symbol ว่าง {n_null} แถว")
        if n_gaps:
            gapped = ", ".join(s for s, v in by_symbol.items() if v["n_gaps"])
            problems.append(
                f"ข้อมูลการซื้อขายขาดหาย {n_gaps} ช่วง รวม {n_missing} trade_id ({gapped}) "
                "ฟีเจอร์ในช่วงนั้นอาจคลาดเคลื่อน"
            )
        if problems:
            detail = "; ".join(problems)
            return Check(name="trade_gaps", status="warn", value=value, detail_th=detail)
        detail = f"trade_id ต่อเนื่องครบ ({len(by_symbol)} สัญลักษณ์)"
        return Check(name="trade_gaps", status="ok", value=value, detail_th=detail)

    def exclusion(self, frac: float) -> Check:
        """Share of training rows excluded around labeled periods: warn iff > 0.02, or when
        `frac` is not a number in [0, 1]."""
        bad = _unreadable_inputs({"frac": frac}, nonneg=("frac",))
        if bad or float(frac) > 1:
            return _unreadable("exclusion", {"frac": frac}, ["frac"])
        f = float(frac)
        value = {"frac": _num(f), "max": EXCLUSION_MAX}
        if f > EXCLUSION_MAX:
            detail = (
                f"ตัดข้อมูลฝึกออก {f:.1%} (เกิน {EXCLUSION_MAX:.0%}) ข้อมูลตลาดปกติที่เหลืออาจน้อยหรือไม่เป็นตัวแทน"
            )
            return Check(name="exclusion", status="warn", value=value, detail_th=detail)
        detail = f"ตัดข้อมูลฝึกออก {f:.1%} ไม่เกิน {EXCLUSION_MAX:.0%}"
        return Check(name="exclusion", status="ok", value=value, detail_th=detail)

    def staleness(
        self,
        meta: dict[str, Any],
        now_us: int,
        max_age_days: float = 30,
        config_hash: str | None = None,
    ) -> Check:
        """Model age from `meta["created_at"]` (ISO-8601 string as written by ModelRegistry,
        naive = UTC, or int µs). warn iff age > `max_age_days`, or `config_hash` is given and
        differs from `meta["config_hash"]`. A missing, unparsable or future `created_at` (clock
        skew / corrupt meta) => warn, as does a non-finite `now_us` or `max_age_days`."""
        inputs = {"now_us": now_us, "max_age_days": max_age_days}
        if bad := _unreadable_inputs(inputs, nonneg=("max_age_days",)):
            return _unreadable("staleness", inputs, bad)
        created_us = _created_us(meta.get("created_at"))
        match = None if config_hash is None else meta.get("config_hash") == config_hash
        age = None if created_us is None else (int(now_us) - created_us) / DAY_US
        value = {
            "age_days": age,
            "max_age_days": float(max_age_days),
            "config_match": match,
            "version": None if meta.get("version") is None else str(meta["version"]),
        }
        problems = []
        if age is None:
            problems.append("อ่านเวลาสร้างโมเดล (created_at) ไม่ได้")
        elif age < 0:
            problems.append(
                f"เวลาสร้างโมเดล (created_at) อยู่ในอนาคต {-age:.1f} วัน "
                "นาฬิกาอาจคลาดเคลื่อนหรือ meta เสียหาย"
            )
        elif age > max_age_days:
            problems.append(f"โมเดลมีอายุ {age:.1f} วัน เกิน {max_age_days:g} วัน ควรฝึกใหม่")
        if match is False:
            problems.append("config_hash ของโมเดลไม่ตรงกับ config ปัจจุบัน ควรฝึกใหม่")
        if problems:
            detail = "; ".join(problems)
            return Check(name="staleness", status="warn", value=value, detail_th=detail)
        detail = f"โมเดลมีอายุ {age:.1f} วัน ไม่เกิน {max_age_days:g} วัน"
        return Check(name="staleness", status="ok", value=value, detail_th=detail)

    def warmup(self, valid_samples: int, min_samples: int | None = None) -> Check:
        """Baseline warm-up (LIVE / replay): ok iff `valid_samples >= min_samples`, else warn with
        the time remaining at `cfg.features.baseline_sample_s` per sample.
        `min_samples` defaults to `cfg.features.min_baseline_samples`."""
        feats = self.cfg.features
        need = feats.min_baseline_samples if min_samples is None else min_samples
        inputs = {"valid_samples": valid_samples, "min_samples": need}
        if bad := _unreadable_inputs(inputs, counts=("valid_samples", "min_samples")):
            return _unreadable("warmup", inputs, bad)
        need, have = int(need), int(valid_samples)
        remaining_s = max(need - have, 0) * feats.baseline_sample_s
        value = {"valid": have, "min": need, "hours_remaining": remaining_s / 3600}
        if have >= need:
            detail = f"baseline พร้อมใช้ ({have}/{need} ตัวอย่าง)"
            return Check(name="warmup", status="ok", value=value, detail_th=detail)
        minutes = math.ceil(remaining_s / 60)
        detail = (
            f"baseline ยังไม่พร้อม ({have}/{need} ตัวอย่าง) "
            f"เหลืออีกประมาณ {minutes // 60} ชั่วโมง {minutes % 60} นาที"
        )
        return Check(name="warmup", status="warn", value=value, detail_th=detail)

    def report(self, ts_us: int | None = None, **inputs: dict[str, Any]) -> HealthReport:
        """Run the checks whose inputs are supplied: `report(latency={"samples_ms": [...]})`.

        Keys are check method names (see `CHECKS`), values their keyword arguments; checks run in
        `CHECKS` order. Unknown keys raise ValueError. `ts_us` defaults to now.
        """
        unknown = sorted(set(inputs) - set(CHECKS))
        if unknown:
            raise ValueError(f"unknown health checks {unknown}; known: {list(CHECKS)}")
        checks = [getattr(self, name)(**inputs[name]) for name in CHECKS if name in inputs]
        ts = time.time_ns() // 1000 if ts_us is None else int(ts_us)
        return HealthReport(checks=checks, ts_us=ts)


def _created_us(created_at: Any) -> int | None:
    if isinstance(created_at, bool):
        return None
    if isinstance(created_at, int | np.integer):
        return int(created_at)
    if isinstance(created_at, str):
        try:
            dt = datetime.fromisoformat(created_at)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1_000_000)
    return None


# ---- label leakage guard

_FORBIDDEN_MODULE = "idsai.eval"
_LABEL_NAME = "LABELS"
_IMPORT_FUNCS = {"import_module", "__import__"}


# unscanned modules that read labels (learning: training mask over eval metrics; data: the
# label store; pipeline: batch loading with LABELS) - a detector importing them could leak
_LABEL_READERS = ("idsai.ai.learning", "idsai.data", "idsai.pipeline")


def _forbidden(module: str) -> bool:
    return any(
        module == m or module.startswith(m + ".") for m in (_FORBIDDEN_MODULE, *_LABEL_READERS)
    )


def _scanned(package_root: Path) -> list[Path]:
    files = [
        p
        for p in (package_root / "detection").rglob("*.py")
        if p.relative_to(package_root / "detection").parts[0] != "simulator"
    ]
    files += [
        p for p in (package_root / "ai").rglob("*.py") if p != package_root / "ai" / "learning.py"
    ]
    return sorted(files)


def _resolve(level: int, module: str | None, package: list[str]) -> str:
    """Absolute module of a relative import `level` dots + `module` from `package`."""
    keep = len(package) - (level - 1)
    base = ".".join(package[: max(keep, 0)])
    return ".".join(x for x in (base, module) if x)


def _str_const(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_arg(node: ast.Call, pos: int, keyword: str) -> ast.expr | None:
    if len(node.args) > pos:
        return node.args[pos]
    return next((k.value for k in node.keywords if k.arg == keyword), None)


def _dynamic_import_target(node: ast.Call, fname: str, package: list[str]) -> str | None:
    """Module named by `import_module(name, package)` / `__import__(name, ...)`, if constant.

    A relative `import_module(".x", pkg)` resolves against the constant `pkg`, or against the
    scanned module's own package when `pkg` is not a constant (e.g. `__package__`).
    """
    name = _str_const(_call_arg(node, 0, "name"))
    if name is None:
        return None
    if fname == "import_module" and name.startswith("."):
        level = len(name) - len(name.lstrip("."))
        pkg = _str_const(_call_arg(node, 1, "package"))
        base = pkg.split(".") if pkg else package
        return _resolve(level, name[level:] or None, base)
    if fname == "__import__":  # `__import__("idsai", fromlist=["eval"])` loads idsai.eval
        fromlist = _call_arg(node, 3, "fromlist")
        if isinstance(fromlist, ast.List | ast.Tuple):
            subs = [s for e in fromlist.elts if (s := _str_const(e)) is not None]
            hit = next((f"{name}.{s}" for s in subs if _forbidden(f"{name}.{s}")), None)
            if hit is not None:
                return hit
    return name


def _import_func_aliases(tree: ast.Module) -> dict[str, str]:
    """Local names of the dynamic-import functions -> canonical name, including aliases such as
    `from importlib import import_module as im`."""
    names = {f: f for f in _IMPORT_FUNCS}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("importlib", "builtins"):
            for a in node.names:
                if a.name in _IMPORT_FUNCS and a.asname:
                    names[a.asname] = a.name
    return names


def _package_aliases(tree: ast.Module, top: str) -> set[str]:
    """Local names bound to the top-level package (`import idsai`, `import idsai.x`,
    `import idsai as pkg`), so `idsai.eval.<...>` attribute chains can be traced."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname is None and a.name.split(".")[0] == top:
                    names.add(top)
                elif a.asname is not None and a.name == top:
                    names.add(a.asname)
    return names


def _module_violations(tree: ast.Module, package: list[str]) -> list[tuple[int, str]]:
    """`(lineno, reason)` for every forbidden import or label-schema reference in `tree`.

    `package` is the dotted package of the module (for resolving relative imports).
    """
    found: list[tuple[int, str]] = []
    top, sub = _FORBIDDEN_MODULE.split(".")
    aliases = _package_aliases(tree, top)
    import_funcs = _import_func_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if _forbidden(a.name):
                    found.append((node.lineno, f"imports {a.name}"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                module = _resolve(node.level, node.module, package)
            else:
                module = node.module or ""
            targets = [module] + [f"{module}.{a.name}" for a in node.names]
            hit = next((t for t in targets if _forbidden(t)), None)
            if hit is not None:
                found.append((node.lineno, f"imports {hit}"))
            for a in node.names:
                if _LABEL_NAME in (a.name, a.asname):
                    found.append((node.lineno, f"imports {_LABEL_NAME} from {module}"))
        elif isinstance(node, ast.Name) and node.id == _LABEL_NAME:
            found.append((node.lineno, f"references {_LABEL_NAME}"))
        elif isinstance(node, ast.Attribute):
            if node.attr == _LABEL_NAME:
                found.append((node.lineno, f"references {_LABEL_NAME}"))
            elif node.attr == sub and isinstance(node.value, ast.Name) and node.value.id in aliases:
                found.append((node.lineno, f"uses {_FORBIDDEN_MODULE} via attribute access"))
        elif isinstance(node, ast.Call):
            f = node.func
            fname = (
                f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            )
            if fname in import_funcs:
                func = import_funcs[fname]
                target = _dynamic_import_target(node, func, package)
                if target is not None and _forbidden(target):
                    found.append((node.lineno, f"imports {target} via {func}"))
            elif fname == "getattr" and _str_const(_call_arg(node, 1, "")) == _LABEL_NAME:
                found.append((node.lineno, f"references {_LABEL_NAME} via getattr"))
    return found


def leakage_violations(package_root: Path) -> list[str]:
    """Label leakage guard over the package directory `package_root` (e.g. `.../idsai`).

    Scans `detection/**/*.py` except `detection/simulator/**` and `ai/**/*.py` except
    `ai/learning.py` with `ast` only (nothing scanned is imported). A violation is:
    an import of `idsai.eval` (absolute, `from idsai import eval`, or a relative import resolving
    into it); an `importlib.import_module` / `__import__` call whose constant name (positional or
    `name=`) is `idsai.eval…`, including a relative `import_module(".eval", "idsai")` resolved
    against its package argument, `__import__("idsai", fromlist=["eval"])`, and either function
    imported under an alias (`from importlib import import_module as im`); an `idsai.eval`
    attribute chain on a name bound by `import idsai…`; any name, attribute, imported alias or
    `getattr(x, "LABELS")` of `LABELS`.
    Returns sorted `"relpath:lineno: reason"` strings, `relpath` relative to the parent of
    `package_root`; a file that cannot be decoded as UTF-8 or parsed is a violation too.

    Imports of the unscanned modules that read labels (`idsai.ai.learning`, `idsai.data`,
    `idsai.pipeline`, see `_LABEL_READERS`) are violations in the same forms.

    Limits (a static scan, not a proof): it cannot see a transitive import through another
    unscanned helper (a new package that imports `idsai.eval` and is imported by a detector),
    module names or attribute names built at run time (string concatenation, f-strings,
    variables), `exec`/`eval` of source text, `sys.modules` lookups, or the package re-bound
    through an assignment (`pkg = idsai; pkg.eval`).
    """
    root = Path(package_root)
    out: list[tuple[str, int, str]] = []
    for path in _scanned(root):
        rel = path.relative_to(root.parent).as_posix()
        # the dotted package relative imports resolve against (for `__init__.py`: itself)
        package = [root.name, *path.relative_to(root).parent.parts]
        try:
            source = path.read_bytes().decode("utf-8-sig")
        except UnicodeDecodeError as e:
            out.append((rel, 0, f"cannot decode as UTF-8 (byte {e.start})"))
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, ValueError) as e:  # ValueError: e.g. null bytes in the source
            lineno = getattr(e, "lineno", None) or 0
            out.append((rel, lineno, f"cannot parse: {getattr(e, 'msg', str(e))}"))
            continue
        out += [(rel, ln, reason) for ln, reason in set(_module_violations(tree, package))]
    return [f"{rel}:{ln}: {reason}" for rel, ln, reason in sorted(out)]
