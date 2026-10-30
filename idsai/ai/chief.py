"""Chief: fuse detector layers into one anomaly p-value and group exceedances into alerts.

Calibration (split-conformal): `calibrate` stores the sorted finite scores of every detector on
normal-period rows. The p-value of a new score `s` is `(1 + #{s_i >= s}) / (n + 1)`; a NaN score
or a detector that did not fire (`Detector.fired`, which only signature rules override) gets
`p = 1`. This is an "anomaly p-value": how rare the row is relative to normal market data, never
a probability that manipulation happened.

Fusion: `F = max_d (-log10 p_d)` over the fused detectors, i.e. layers `{rule, stat, ai}`. A Chief
that holds only baseline-layer detectors fuses over those instead (the baselines experiment); in
a mixed Chief baselines get a `p__` column but never enter `F`, `agreement` or `technique`.
`agreement` counts the fusion layers with any `p < agreement_p` (0-3).

Batch/stream parity: a p-value is `k / (n + 1)` for an integer `k`, and `-log10 p` is read from a
per-detector lookup table built at calibration time, so `F` of a row is bit-identical whether it
is fused alone (`ChiefStream.step`) or in a batch (`fuse`). Both go through `_fuse_matrix`, and
both build alerts through `_alert`; episode ids are `f"{provenance}-{symbol}-{first_us}"`, so a
replay or sim run on the same data never overwrites a live alert.

Alert budget: `fit_threshold` picks the smallest `tau` (among observed `F` values) whose alert
episodes per symbol-day on calibration data stay within the budget. Exceedances are rows with
`F >= tau` and `F > 0`, grouped per symbol; consecutive exceedances at most `suppress_for_s`
apart belong to the same episode.

Episode technique: `technique_id`, `signature_id` and `rev` come from the episode's highest-`F`
exceedance row whose technique is not `MM-T000` (earliest on ties), so an episode whose peak row
is unclassified still names the technique a less extreme row found; `MM-T000` only when no row
classifies. `fused_p`, `window`, `detectors`, `agreement`, `severity` and `evidence` stay from the
peak row. Batch and stream both fold rows through `_Attribution.merge`.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from idsai.config import ChiefCfg
from idsai.contracts.alert import Alert, DetectorRef
from idsai.contracts.detector import Detector, feature_matrix

FUSION_LAYERS = ("rule", "stat", "ai")
UNCLASSIFIED = "MM-T000"
CUSUM_NAME = "stat.cusum"
CUSUM_TECHNIQUE = "MM-T001"
DAY_US = 86_400_000_000
DEDUP_BUCKET_US = 300_000_000
STATE_VERSION = 1

Provenance = Literal["live", "replay", "sim"]


class CalibrationError(ValueError):
    """The Chief is used before calibration/threshold, or calibration cannot honour a budget."""


def score_column(name: str) -> str:
    return f"score__{name}"


def p_column(name: str) -> str:
    return f"p__{name}"


def _symbol_days(frame: pl.DataFrame) -> int:
    """Number of unique `(symbol, UTC day of t_end_us)` pairs."""
    return (
        frame.select(pl.col("symbol"), (pl.col("t_end_us") // DAY_US).alias("_day")).unique().height
    )


def _finite_or_none(x: float) -> float | None:
    return x if math.isfinite(x) else None


def alert_id_for(provenance: str, symbol: str, first_us: int) -> str:
    return f"{provenance}-{symbol}-{first_us}"


def _parent_technique(technique_id: str) -> str:
    return technique_id.split(".")[0]


@dataclass
class _Fusion:
    """Per-row fusion results; detector axes follow `Chief.detectors` order."""

    scores: np.ndarray  # (rows, detectors) float64
    p: np.ndarray  # (rows, detectors) anomaly p-values
    F: np.ndarray  # (rows,) max -log10 p over fused detectors
    best: np.ndarray  # (rows,) index into Chief.detectors
    agreement: np.ndarray  # (rows,) int64
    technique: np.ndarray  # (rows,) str
    rule: np.ndarray  # (rows,) index into Chief.detectors of the chosen fired rule, -1 if none


@dataclass
class _Episode:
    symbol: str
    first_us: int
    last_us: int


@dataclass(frozen=True)
class _Attribution:
    """The technique an episode reports so far, and the `F` of the row it came from."""

    F: float
    technique: str
    signature_id: int | None
    rev: int | None

    @property
    def label(self) -> tuple[str, int | None, int | None]:
        """What an alert shows of this attribution."""
        return self.technique, self.signature_id, self.rev

    def merge(self, later: "_Attribution") -> "_Attribution":
        """Attribution after a later row: highest-`F` classified row, earliest on ties."""
        if later.technique == UNCLASSIFIED:
            return self
        if self.technique == UNCLASSIFIED or later.F > self.F:
            return later
        return self


class Chief:
    """Conformal fusion of detector scores into one anomaly p-value, plus an alert budget."""

    def __init__(
        self,
        detectors: list[Detector],
        cfg: ChiefCfg,
        config_hash: str,
        model_versions: dict[str, str],
    ):
        if not detectors:
            raise ValueError("Chief needs at least one detector")
        names = [d.name for d in detectors]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate detector names: {names}")
        self.detectors = list(detectors)
        self.cfg = cfg
        self.config_hash = config_hash
        self.model_versions = dict(model_versions)
        self.tau: float | None = None
        self.tau_saturated = False
        self.n_cal: int | None = None
        self.cal_symbol_days: int | None = None
        self._cal: list[np.ndarray] | None = None
        self._nlp_tables: list[np.ndarray] = []

        fusion = [j for j, d in enumerate(detectors) if d.layer in FUSION_LAYERS]
        # baselines run alone: a baseline-only Chief fuses over its baselines
        self._fused = fusion or list(range(len(detectors)))
        self._by_layer = {
            layer: [j for j in fusion if detectors[j].layer == layer] for layer in FUSION_LAYERS
        }
        self._rules = self._by_layer["rule"]
        self._rule_tech = np.array(
            [_parent_technique(detectors[j].technique_ids[0]) for j in self._rules], dtype=object
        )
        cusum = [j for j in fusion if detectors[j].name == CUSUM_NAME]
        self._cusum = cusum[0] if cusum else None
        self._index = {n: j for j, n in enumerate(names)}

    # ------------------------------------------------------------ calibration

    @property
    def fused_detectors(self) -> list[Detector]:
        return [self.detectors[j] for j in self._fused]

    def calibrate(self, cal: pl.DataFrame) -> None:
        """Store sorted finite `score__<name>` values per detector (rules: fired or not).

        Resets `tau` and `tau_saturated`, since a threshold fitted under the old calibration no
        longer applies.
        """
        arrays = []
        for d in self.detectors:
            col = score_column(d.name)
            if col not in cal.columns:
                raise KeyError(f"calibration frame lacks column {col!r}")
            s = feature_matrix(cal, [col])[:, 0]
            arrays.append(np.sort(s[np.isfinite(s)]))
        self._set_calibration(arrays, _symbol_days(cal))
        self.tau = None
        self.tau_saturated = False

    def _set_calibration(self, arrays: list[np.ndarray], cal_symbol_days: int) -> None:
        self._cal = [np.asarray(a, dtype=np.float64) for a in arrays]
        self.n_cal = min(self._cal[j].size for j in self._fused)
        self.cal_symbol_days = int(cal_symbol_days)
        # table[k] = -log10(k / (n + 1)) for k = 1..n+1 (table[0] unused); table[n+1] is +0.0
        self._nlp_tables = []
        for a in self._cal:
            n = a.size
            table = np.full(n + 2, np.nan)
            table[1:] = -np.log10(np.arange(1, n + 2, dtype=np.float64) / (n + 1))
            table[n + 1] = 0.0
            self._nlp_tables.append(table)

    def _require_calibration(self) -> list[np.ndarray]:
        if self._cal is None:
            raise CalibrationError("Chief is not calibrated: call calibrate() first")
        return self._cal

    def _ranks(self, j: int, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`(k, fired)`: p = k / (n + 1); NaN or unfired scores get k = n + 1 (p = 1)."""
        cal = self._require_calibration()[j]
        n = cal.size
        s = np.asarray(scores, dtype=np.float64)
        fired = np.asarray(self.detectors[j].fired(s), dtype=bool) & ~np.isnan(s)
        k = 1 + n - np.searchsorted(cal, s, side="left")
        return np.where(fired, k, n + 1), fired

    def pvalues(self, name: str, scores: np.ndarray) -> np.ndarray:
        """Anomaly p-values `(1 + #{s_i >= s}) / (n + 1)`; NaN or unfired scores give 1.0."""
        j = self._index[name]
        k, _ = self._ranks(j, scores)
        return k / (self._require_calibration()[j].size + 1)

    # ------------------------------------------------------------ fusion

    def _fuse_matrix(self, scores: np.ndarray) -> _Fusion:
        """Shared fusion math for batch (`fuse`) and stream (`ChiefStream.step`)."""
        cal = self._require_calibration()
        n_rows, n_det = scores.shape
        p = np.empty((n_rows, n_det))
        nlp = np.empty((n_rows, n_det))
        fired = np.empty((n_rows, n_det), dtype=bool)
        for j in range(n_det):
            k, fired[:, j] = self._ranks(j, scores[:, j])
            p[:, j] = k / (cal[j].size + 1)
            nlp[:, j] = self._nlp_tables[j][k]

        fused = np.asarray(self._fused)
        best_local = np.argmax(nlp[:, fused], axis=1)  # ties -> first fused detector
        rows = np.arange(n_rows)
        F = nlp[rows, fused[best_local]]

        agreement = np.zeros(n_rows, dtype=np.int64)
        for cols in self._by_layer.values():
            if cols:
                agreement += np.any(p[:, cols] < self.cfg.agreement_p, axis=1)

        technique = np.full(n_rows, UNCLASSIFIED, dtype=object)
        if self._cusum is not None:
            technique[nlp[:, self._cusum] >= self.cfg.attribution_min] = CUSUM_TECHNIQUE
        rule = np.full(n_rows, -1, dtype=np.int64)
        if self._rules:
            rules = np.asarray(self._rules)
            cand = np.where(fired[:, rules], nlp[:, rules], -np.inf)
            best_rule = np.argmax(cand, axis=1)  # ties -> first rule
            has_rule = fired[:, rules].any(axis=1)
            technique = np.where(has_rule, self._rule_tech[best_rule], technique)
            rule = np.where(has_rule, rules[best_rule], -1)

        return _Fusion(
            scores=scores,
            p=p,
            F=F,
            best=fused[best_local],
            agreement=agreement,
            technique=technique,
            rule=rule,
        )

    def _score_matrix(self, frame: pl.DataFrame) -> np.ndarray:
        cols = [score_column(d.name) for d in self.detectors]
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise KeyError(f"frame lacks score columns {missing}")
        return feature_matrix(frame, cols)

    def fuse(self, scored: pl.DataFrame) -> pl.DataFrame:
        """Add `p__<name>`, `F` (-log10 anomaly p-value), `best_detector`, `agreement`, `technique`.

        All input columns are kept, in input row order.
        """
        fu = self._fuse_matrix(self._score_matrix(scored))
        names = np.array([d.name for d in self.detectors], dtype=object)
        return scored.with_columns(
            *[
                pl.Series(p_column(d.name), fu.p[:, j], dtype=pl.Float64)
                for j, d in enumerate(self.detectors)
            ],
            pl.Series("F", fu.F, dtype=pl.Float64),
            pl.Series("best_detector", names[fu.best].tolist(), dtype=pl.String),
            pl.Series("agreement", fu.agreement, dtype=pl.Int64),
            pl.Series("technique", fu.technique.tolist(), dtype=pl.String),
        )

    # ------------------------------------------------------------ episodes

    def _resolve_tau(self, tau: float | None) -> float:
        if tau is None:
            tau = self.tau
        if tau is None:
            raise CalibrationError("no alert threshold: call fit_threshold() or pass tau")
        if math.isnan(tau):
            raise ValueError("tau must not be NaN")
        return float(tau)

    @staticmethod
    def _sorted_view(fused: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """`(order, symbol_code, t_end_us, F)` sorted by `(symbol, t_end_us)` (stable)."""
        view = (
            fused.select(
                pl.col("symbol").rank("dense").alias("_sym"),
                pl.col("t_end_us"),
                pl.col("F").cast(pl.Float64),
            )
            .with_row_index("_i")
            .sort(["_sym", "t_end_us"], maintain_order=True)
        )
        return (
            view["_i"].to_numpy().astype(np.int64),
            view["_sym"].to_numpy(),
            view["t_end_us"].to_numpy().astype(np.int64),
            view["F"].to_numpy(),
        )

    def _group(
        self, sym: np.ndarray, t: np.ndarray, F: np.ndarray, tau: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(pos, starts, ends)`: exceedance positions and per-episode `[start, end)` into `pos`."""
        pos = np.flatnonzero((F >= tau) & (F > 0))
        if pos.size == 0:
            empty = np.empty(0, dtype=np.int64)
            return pos, empty, empty
        s, tt = sym[pos], t[pos]
        new = np.ones(pos.size, dtype=bool)
        new[1:] = (s[1:] != s[:-1]) | ((tt[1:] - tt[:-1]) > self.cfg.suppress_for_s * 1_000_000)
        starts = np.flatnonzero(new)
        return pos, starts, np.append(starts[1:], pos.size)

    def _attribution(self, fu: _Fusion, i: int) -> _Attribution:
        """The technique row `i` of `fu` names on its own."""
        rule_j = int(fu.rule[i])
        sig = self.detectors[rule_j] if rule_j >= 0 else None
        return _Attribution(
            F=float(fu.F[i]),
            technique=str(fu.technique[i]),
            signature_id=sig.signature_id if sig is not None else None,
            rev=sig.rev if sig is not None else None,
        )

    def _alert(
        self,
        ep: _Episode,
        fu: _Fusion,
        i: int,
        row: dict[str, Any],
        attribution: _Attribution,
        status: Literal["open", "updated", "closed"],
        provenance: Provenance,
    ) -> Alert:
        """Build one alert from the peak row `i` of `fu` and the episode's `attribution` (shared
        by batch and stream)."""
        best = self.detectors[int(fu.best[i])]
        if all(c in row for c in best.requires):
            clean = {c: (math.nan if row[c] is None else row[c]) for c in best.requires}
            # evidence is explanatory only; JSON cannot carry inf/NaN, so drop such items
            evidence = [
                (f, v, c)
                for f, v, c in best.evidence(clean).top
                if math.isfinite(v) and math.isfinite(c)
            ]
        else:
            evidence = []
        agreement = int(fu.agreement[i])
        return Alert(
            alert_id=alert_id_for(provenance, ep.symbol, ep.first_us),
            ts_event_us=ep.first_us,
            symbol=ep.symbol,
            technique_id=attribution.technique,
            signature_id=attribution.signature_id,
            rev=attribution.rev,
            severity=1 + int(agreement >= 2) + int(agreement >= 3),
            status=status,
            fused_p=10.0 ** -float(fu.F[i]),
            detectors=[
                DetectorRef(
                    name=self.detectors[j].name,
                    layer=self.detectors[j].layer,
                    p=float(fu.p[i, j]),
                    score=_finite_or_none(float(fu.scores[i, j])),  # non-finite -> None
                )
                for j in self._fused
            ],
            agreement=agreement,
            evidence=evidence,
            window=(ep.first_us, ep.last_us),
            dedup_key=f"{ep.symbol}|{ep.first_us // DEDUP_BUCKET_US}",
            suppress_for_s=self.cfg.suppress_for_s,
            config_hash=self.config_hash,
            model_versions=dict(self.model_versions),
            provenance=provenance,
        )

    def episodes(
        self, fused: pl.DataFrame, tau: float | None = None, provenance: Provenance = "replay"
    ) -> list[Alert]:
        """Closed alert episodes of `fuse` output: rows with `F >= tau`, grouped per symbol.

        `tau` defaults to `self.tau`. `fused_p`, window, detectors, agreement and evidence come
        from the episode's peak row (max `F`, earliest on ties); the technique and signature from
        its highest-`F` classified row (see the module docstring). Alerts are ordered by
        `(symbol, first_us)`.
        """
        tau = self._resolve_tau(tau)
        order, sym, t, F = self._sorted_view(fused)
        pos, starts, ends = self._group(sym, t, F, tau)
        if starts.size == 0:
            return []
        fu = self._fuse_matrix(self._score_matrix(fused[order[pos].tolist()]))
        out = []
        for a, b in zip(starts, ends, strict=True):
            seg = pos[a:b]
            peak = int(a) + int(np.argmax(F[seg]))  # first max = earliest
            attribution = self._attribution(fu, int(a))
            for i in range(int(a) + 1, int(b)):
                attribution = attribution.merge(self._attribution(fu, i))
            row = fused[int(order[pos[peak]])].to_dicts()[0]
            ep = _Episode(row["symbol"], int(t[seg[0]]), int(t[seg[-1]]))
            out.append(self._alert(ep, fu, peak, row, attribution, "closed", provenance))
        return out

    # ------------------------------------------------------------ threshold

    def fit_threshold(self, cal_fused: pl.DataFrame, budget_per_symbol_day: float) -> float:
        """Lowest observed `F` reached from the top before the episode rate exceeds the budget.

        Searches the descending candidates for a within-budget / over-budget boundary (see the
        comment below); this is a boundary near the top, not a guaranteed first crossing, since
        galloping can step over a narrow over-budget region. Stores and returns `self.tau`.

        When no observed value fits (or every `F` is 0), `tau` is the next float above `max(F)`,
        which gives no calibration episodes. When no candidate exceeds the budget at all (the
        budget is at or above the data's peak episode rate), `self.tau_saturated` is set and a
        `RuntimeWarning` is issued; eval reports the flag. `tau` is then the smallest positive `F`,
        or, with no positive `F` at all (say a rule that never fires in calibration), that next
        float above `max(F)`: the budget was never tested against any episode.
        """
        self._require_calibration()
        budget = float(budget_per_symbol_day)
        if not budget > 0:
            raise ValueError(f"budget_per_symbol_day must be > 0, got {budget}")
        if cal_fused.height == 0:
            raise CalibrationError("cannot fit a threshold on an empty frame")
        symbol_days = _symbol_days(cal_fused)
        rows_per_symbol_day = cal_fused.height / symbol_days
        n_fused = len(self._fused)
        assert self.n_cal is not None
        floor = n_fused / (self.n_cal + 1)
        if budget / rows_per_symbol_day < floor:
            raise CalibrationError(
                "ชุดข้อมูลสอบเทียบเล็กเกินไปสำหรับงบการแจ้งเตือนนี้: "
                f"งบ {budget:g} ครั้ง/สัญลักษณ์-วัน จาก {rows_per_symbol_day:.0f} แถว/สัญลักษณ์-วัน "
                f"ต่ำกว่าความละเอียดของ anomaly p-value ({n_fused}/(n_cal+1) = {floor:.3g}, "
                f"n_cal = {self.n_cal}) — ต้องใช้ข้อมูลสอบเทียบมากขึ้นหรือเพิ่มงบ. "
                f"Calibration set too small for this alert budget: budget {budget:g} per "
                f"symbol-day over {rows_per_symbol_day:.0f} rows per symbol-day is below the "
                f"anomaly p-value resolution {n_fused}/(n_cal+1) = {floor:.3g} "
                f"(n_cal = {self.n_cal}); calibrate on more rows or raise the budget."
            )

        _, sym, t, F = self._sorted_view(cal_fused)
        cand = np.unique(F[F > 0])[::-1]  # descending; index grows as tau falls

        def rate(tau: float) -> float:
            return self._group(sym, t, F, tau)[1].size / symbol_days

        # The episode rate is not monotone in tau: it rises as tau falls, then drops again once
        # exceedances are dense enough to merge into a few long episodes (in the limit, one
        # episode per symbol-day). A plain binary search over all candidates can land in that
        # merged tail, so search from the top: gallop down (1, 2, 4, ...) to the first probed
        # candidate over budget, then binary-search inside that bracket, which always ends on
        # an adjacent within/over pair. `best` only ever holds indices whose rate was checked
        # to be within budget, so no separate verify / step-up pass is needed.
        best, bad = -1, cand.size
        step = 1
        while best < cand.size - 1:
            probe = min(best + step, cand.size - 1)  # clamp so the last candidate is probed
            if rate(float(cand[probe])) <= budget:
                best, step = probe, step * 2
            else:
                bad = probe
                break
        lo, hi = best + 1, bad - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if rate(float(cand[mid])) <= budget:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        saturated = bad == cand.size
        if saturated:
            why = (
                "tau saturated at the smallest positive F (near-continuous exceedances merged "
                "into few episodes)"
                if cand.size > 0
                else "no row has a positive F, so tau stays above every F"
            )
            warnings.warn(
                f"alert budget {budget:g} per symbol-day is at or above the peak episode rate "
                f"of the calibration data: {why}; tau_saturated=True",
                RuntimeWarning,
                stacklevel=2,
            )
        self.tau_saturated = bool(saturated)
        if best >= 0:
            tau = float(cand[best])
        else:
            tau = float(np.nextafter(max(float(F.max()), 0.0), np.inf))
        self.tau = tau
        return tau

    # ------------------------------------------------------------ state

    def state_dict(self) -> dict[str, Any]:
        """Pickle-able calibration state (detector objects are not included)."""
        cal = self._require_calibration()
        return {
            "version": STATE_VERSION,
            "detector_names": [d.name for d in self.detectors],
            "cal": {d.name: cal[j].copy() for j, d in enumerate(self.detectors)},
            "n_cal": self.n_cal,
            "cal_symbol_days": self.cal_symbol_days,
            "tau": self.tau,
            "tau_saturated": self.tau_saturated,
            "config_hash": self.config_hash,
            "model_versions": dict(self.model_versions),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Restore `state_dict` output.

        The detector names (in order), `config_hash` and `model_versions` must equal this
        Chief's: a calibration is valid only for the exact config and models it was made with.
        """
        if d.get("version") != STATE_VERSION:
            raise ValueError(f"unsupported Chief state version {d.get('version')!r}")
        names = [det.name for det in self.detectors]
        if list(d["detector_names"]) != names:
            raise ValueError(f"detector names {d['detector_names']} do not match Chief's {names}")
        if d["config_hash"] != self.config_hash:
            raise ValueError(
                f"config_hash mismatch: state {d['config_hash']!r}, Chief {self.config_hash!r}"
            )
        if dict(d["model_versions"]) != self.model_versions:
            raise ValueError(
                f"model_versions mismatch: state {d['model_versions']}, Chief {self.model_versions}"
            )
        self._set_calibration([d["cal"][n] for n in names], d["cal_symbol_days"])
        if self.n_cal != d["n_cal"]:
            raise ValueError(f"n_cal mismatch: state says {d['n_cal']}, arrays give {self.n_cal}")
        self.tau = None if d["tau"] is None else float(d["tau"])
        self.tau_saturated = bool(d["tau_saturated"])


@dataclass
class _Open:
    episode: _Episode
    peak_F: float
    peak: _Fusion
    peak_row: dict[str, Any]
    attribution: _Attribution


class ChiefStream:
    """Row-at-a-time alerting that reproduces `Chief.episodes` on the same time-ordered rows.

    `step(row)` takes one scored row (`symbol`, `t_end_us`, `score__<name>`, plus any feature
    columns used for evidence) and returns 0-2 alerts: a `closed` alert for this symbol's open
    episode when the row lies more than `suppress_for_s` after its last exceedance, then `open`
    (first exceedance) or `updated` (the episode's peak `F` rose, or the technique or signature
    it reports changed; see the `Chief` module docstring). `flush()` closes everything.

    Closing is driven only by later rows of the same symbol or by `flush()`: an episode of a
    symbol whose rows stop stays open. LIVE must call `flush()` on shutdown or when a symbol
    leaves the universe, or feed heartbeat rows, so final `closed` alerts are emitted.
    """

    def __init__(self, chief: Chief, provenance: Provenance):
        chief._resolve_tau(None)  # fail early without a threshold
        self.chief = chief
        self.provenance = provenance
        self._open: dict[str, _Open] = {}
        self._last_t: dict[str, int] = {}

    def _emit(self, o: _Open, status: Literal["open", "updated", "closed"]) -> Alert:
        return self.chief._alert(
            o.episode, o.peak, 0, o.peak_row, o.attribution, status, self.provenance
        )

    def step(self, row: dict[str, Any]) -> list[Alert]:
        ch = self.chief
        tau = ch._resolve_tau(None)
        symbol, t = str(row["symbol"]), int(row["t_end_us"])
        if t < self._last_t.get(symbol, t):
            raise ValueError(f"rows for {symbol} must arrive in t_end_us order")
        self._last_t[symbol] = t

        out: list[Alert] = []
        o = self._open.get(symbol)
        if o is not None and t - o.episode.last_us > ch.cfg.suppress_for_s * 1_000_000:
            out.append(self._emit(o, "closed"))
            del self._open[symbol]
            o = None

        scores = np.array(
            [
                [
                    math.nan if row[score_column(d.name)] is None else row[score_column(d.name)]
                    for d in ch.detectors
                ]
            ],
            dtype=np.float64,
        )
        fu = ch._fuse_matrix(scores)
        F = float(fu.F[0])
        if not (F >= tau and F > 0):
            return out
        attribution = ch._attribution(fu, 0)
        if o is None:
            o = _Open(_Episode(symbol, t, t), F, fu, dict(row), attribution)
            self._open[symbol] = o
            out.append(self._emit(o, "open"))
            return out
        o.episode.last_us = t
        before = o.attribution.label
        o.attribution = o.attribution.merge(attribution)
        rose = F > o.peak_F
        if rose:
            o.peak_F, o.peak, o.peak_row = F, fu, dict(row)
        if rose or o.attribution.label != before:
            out.append(self._emit(o, "updated"))
        return out

    def discard_open(self) -> list[str]:
        """Forget every open episode without an alert (a seek); returns their alert ids."""
        ids = [
            alert_id_for(self.provenance, o.episode.symbol, o.episode.first_us)
            for o in self._open.values()
        ]
        self._open.clear()
        return ids

    def flush(self) -> list[Alert]:
        """Close every open episode (end of replay or shutdown)."""
        out = [self._emit(o, "closed") for o in self._open.values()]
        self._open.clear()
        return out
