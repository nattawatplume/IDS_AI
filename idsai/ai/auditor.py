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


def _finite(values: Any) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


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
            else float(budget_per_symbol_day)
        )
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
        dropped; fewer than 200 values => ok with a note.
        """
        p = _finite(pvals)
        n = int(p.size)
        if n < CALIBRATION_MIN_N:
            value = {"n": n, "ks_stat": None, "p": None}
            detail = f"{NO_DATA_TH}: มี p-value {n} ค่า (ต้องมีอย่างน้อย {CALIBRATION_MIN_N}) ยังไม่ประเมิน"
            return Check(name="calibration", status="ok", value=value, detail_th=detail)
        res = stats.kstest(p, "uniform")
        value = {"n": n, "ks_stat": _num(res.statistic), "p": _num(res.pvalue)}
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
        Budget defaults to `cfg.chief.budget_per_symbol_day`.
        """
        b = self.cfg.chief.budget_per_symbol_day if budget is None else float(budget)
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
        """End-to-end latency: warn iff p99 > 2000 ms; no samples => ok with a note."""
        x = _finite(samples_ms)
        if not x.size:
            value = {"n": 0, "p50_ms": None, "p99_ms": None}
            detail = f"{NO_DATA_TH}: ยังไม่มีตัวอย่างเวลาแฝง (latency)"
            return Check(name="latency", status="ok", value=value, detail_th=detail)
        p50, p99 = (float(v) for v in np.percentile(x, [50, 99]))
        value = {"n": int(x.size), "p50_ms": p50, "p99_ms": p99}
        if p99 > LATENCY_P99_MS:
            detail = f"เวลาแฝง p99 = {p99:.0f} ms เกิน {LATENCY_P99_MS:.0f} ms (p50 = {p50:.0f} ms)"
            return Check(name="latency", status="warn", value=value, detail_th=detail)
        detail = f"เวลาแฝง p99 = {p99:.0f} ms ไม่เกิน {LATENCY_P99_MS:.0f} ms (p50 = {p50:.0f} ms)"
        return Check(name="latency", status="ok", value=value, detail_th=detail)

    def trade_gaps(self, trades: pl.DataFrame) -> Check:
        """Missing trades: per symbol, jumps > 1 between sorted unique non-simulated trade_ids
        (simulated trades have `trade_id < 0` and are ignored). Any gap => warn."""
        real = trades.select("symbol", "trade_id").filter(pl.col("trade_id") >= 0)
        by_symbol: dict[str, dict[str, int]] = {}
        for (sym,), part in sorted(real.partition_by("symbol", as_dict=True).items()):
            ids = np.unique(part["trade_id"].to_numpy())
            d = np.diff(ids)
            jumps = d[d > 1]
            by_symbol[str(sym)] = {"n_gaps": int(jumps.size), "n_missing": int((jumps - 1).sum())}
        n_gaps = sum(v["n_gaps"] for v in by_symbol.values())
        n_missing = sum(v["n_missing"] for v in by_symbol.values())
        value = {"n_gaps": n_gaps, "n_missing": n_missing, "by_symbol": by_symbol}
        if n_gaps:
            gapped = ", ".join(s for s, v in by_symbol.items() if v["n_gaps"])
            detail = (
                f"ข้อมูลการซื้อขายขาดหาย {n_gaps} ช่วง รวม {n_missing} trade_id ({gapped}) "
                "ฟีเจอร์ในช่วงนั้นอาจคลาดเคลื่อน"
            )
            return Check(name="trade_gaps", status="warn", value=value, detail_th=detail)
        detail = f"trade_id ต่อเนื่องครบ ({len(by_symbol)} สัญลักษณ์)"
        return Check(name="trade_gaps", status="ok", value=value, detail_th=detail)

    def exclusion(self, frac: float) -> Check:
        """Share of training rows excluded around labeled periods: warn iff > 0.02."""
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
        differs from `meta["config_hash"]`; a missing or unparsable `created_at` => warn."""
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
        need = feats.min_baseline_samples if min_samples is None else int(min_samples)
        have = int(valid_samples)
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
    if isinstance(created_at, int):
        return created_at
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


def _forbidden(module: str) -> bool:
    return module == _FORBIDDEN_MODULE or module.startswith(_FORBIDDEN_MODULE + ".")


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


def _module_violations(tree: ast.Module, package: list[str]) -> list[tuple[int, str]]:
    """`(lineno, reason)` for every forbidden import or label-schema reference in `tree`.

    `package` is the dotted package of the module (for resolving relative imports).
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if _forbidden(a.name):
                    found.append((node.lineno, f"imports {a.name}"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package) - (node.level - 1)
                base = ".".join(package[: max(keep, 0)])
                module = ".".join(x for x in (base, node.module) if x)
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
        elif isinstance(node, ast.Attribute) and node.attr == _LABEL_NAME:
            found.append((node.lineno, f"references {_LABEL_NAME}"))
        elif isinstance(node, ast.Call):
            f = node.func
            fname = (
                f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            )
            if fname in _IMPORT_FUNCS:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if _forbidden(arg.value):
                            found.append((node.lineno, f"imports {arg.value} via {fname}"))
    return found


def leakage_violations(package_root: Path) -> list[str]:
    """Label leakage guard over the package directory `package_root` (e.g. `.../idsai`).

    Scans `detection/**/*.py` except `detection/simulator/**` and `ai/**/*.py` except
    `ai/learning.py` with `ast` only (nothing scanned is imported). A violation is an import of
    `idsai.eval` (absolute, `from idsai import eval`, or a relative import resolving into it),
    an `importlib.import_module` / `__import__` call with an `idsai.eval…` string, or any name,
    attribute or imported alias `LABELS`. Returns sorted `"relpath:lineno: reason"` strings,
    `relpath` relative to the parent of `package_root`; an unparsable file is a violation too.
    """
    root = Path(package_root)
    out: list[tuple[str, int, str]] = []
    for path in _scanned(root):
        rel = path.relative_to(root.parent).as_posix()
        # the dotted package relative imports resolve against (for `__init__.py`: itself)
        package = [root.name, *path.relative_to(root).parent.parts]
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as e:
            out.append((rel, e.lineno or 0, f"cannot parse: {e.msg}"))
            continue
        out += [(rel, ln, reason) for ln, reason in set(_module_violations(tree, package))]
    return [f"{rel}:{ln}: {reason}" for rel, ln, reason in sorted(out)]
