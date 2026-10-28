"""Experiment runner: detectors + Chief on time splits -> one run record per variant x budget.

Protocol shared by every experiment (one `seed` per call; the CLI loops seeds):

- Models (`ai.autoencoder_trade`, IsolationForest, PCA, the supervised RF) are fit on
  `train_normal` windows after `training_mask` (labels +- 1 h). Fit frames are thinned to the
  `ae.train_stride_s` grid: the autoencoder only trains on that grid anyway, IsolationForest
  subsamples 256 rows per tree, and it bounds memory on the real archive.
- The Chief is calibrated on `cal_normal` scores and `tau` is fit there for every budget in
  `eval.budgets_per_symbol_day`. Each set is scored once per (seed, detector set) and reused
  across budgets. A `CalibrationError` becomes a record with `metrics = None` and `error`.
- Metrics follow `idsai.eval.metrics`. `fa_per_symbol_day` counts unmatched episodes on the
  set's normal days only (REAL: clean test/accept normal days; SEMI-SYN: every injected day);
  `fa_per_symbol_day_hard` is the same rate on `test_hard_negative`.

Event days and normal days are loaded with `load_day_windows`; windows of a set are
concatenated and de-duplicated on `(symbol, t_end_us)` (first kept), and extension rows past
`D+1` are dropped when the `D+1` partition is absent. Skipped days are counted per reason.

Red-Team injection only runs here in replay mode. Eval never writes the ModelRegistry: model
versions are `"eval-" + sha256(pickle(state_dict))[:12]`.
"""

import hashlib
import math
import pickle
import subprocess
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier

from idsai.ai.chief import CalibrationError, Chief
from idsai.ai.learning import training_mask
from idsai.config import REPO_ROOT, Settings, config_hash, file_sha
from idsai.contracts.detector import Detector, Evidence, feature_matrix
from idsai.contracts.frames import LABELS, conform, empty
from idsai.data.store import DAY_US, Store
from idsai.detection.features.registry import model_features
from idsai.detection.registry import build_detectors
from idsai.detection.rules import SIGNATURES_PATH
from idsai.detection.simulator import (
    Scenario,
    assert_injection_allowed,
    load_scenario_ranges,
    sample_scenario,
)
from idsai.detection.simulator.inject import price_tick
from idsai.eval.metrics import MatchResult, match_events, summarize, window_positive, window_pr_auc
from idsai.eval.splits import SymDay, day_start_us, drift_ranges, make_split, split_def
from idsai.pipeline.batch import load_day_windows, score_windows

S_US = 1_000_000
AE = "ai.autoencoder_trade"
RF_NAME = "ref.supervised_rf"
FAMILIES = ("pump_burst", "pump_stealth", "wash_pingpong")
BASELINES = (
    "baseline.random",
    "baseline.kk",
    "baseline.iforest",
    "baseline.pca",
    "stat.robust_z",
    AE,
)
ABLATION: dict[str, list[str] | None] = {
    "rules": ["rules"],
    "stats": ["stat.robust_z", "stat.cusum"],
    "ai": [AE],
    "rules+stats": ["rules", "stat.robust_z", "stat.cusum"],
    "all": None,  # cfg.detectors
}
# detectors whose scores depend on the run seed (everything else is deterministic)
SEEDED = {AE, "baseline.iforest", "baseline.random", RF_NAME}
MASK_MARGIN_S = 3600
RF_POSITIVE_S = 300
MIN_TICKS = 2.0  # skip pumps whose impact moves the price by fewer ticks
SCENARIO_LEAD_S = 3600
SCENARIO_TAIL_S = 600
SUPERVISED_NOTE = "supervised reference (uses labels)"
REAL, SEMI = "REAL", "SEMI-SYN"


class _VariantError(Exception):
    """A variant cannot be evaluated (e.g. no rows to fit a model on); recorded, not raised."""


# ------------------------------------------------------------------ supervised reference
def _finite_x(x: np.ndarray) -> np.ndarray:
    # sklearn trees reject inf (inf is a value under NaN-only masking): clip into float32 range
    return np.clip(x, -1e30, 1e30)


class SupervisedRFDetector(Detector):
    """RandomForest probability as a detector score: the supervised reference (uses labels).

    Rows with a NaN feature score NaN; fit rows with a NaN feature are dropped.
    """

    name = RF_NAME
    layer = "baseline"
    track = "T"

    def __init__(self, features: list[str], seed: int = 0):
        if not features:
            raise ValueError("SupervisedRFDetector needs at least one feature")
        self.technique_ids = ["ANY"]
        self.requires = list(features)
        self.seed = seed
        self._model: RandomForestClassifier | None = None

    def fit_supervised(self, rows: pl.DataFrame, y: np.ndarray) -> None:
        x = feature_matrix(rows, self.requires)
        y = np.asarray(y, dtype=bool)
        keep = ~np.isnan(x).any(axis=1)
        x, y = _finite_x(x[keep]), y[keep]
        if not (y.any() and (~y).any()):
            raise ValueError(
                f"{self.name}: need positive and negative rows with finite features, got "
                f"{int(y.sum())} positive / {int((~y).sum())} negative"
            )
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=self.seed,
            class_weight="balanced",
            n_jobs=1,
        )
        model.fit(x, y)
        self._model = model

    def _need_fit(self) -> None:
        if self._model is None:
            raise RuntimeError(f"{self.name} is not fitted; call fit_supervised() first")

    def score(self, windows: pl.DataFrame) -> np.ndarray:
        self._need_fit()
        x = feature_matrix(windows, self.requires)
        out = np.full(x.shape[0], np.nan)
        ok = ~np.isnan(x).any(axis=1)
        if ok.any():
            out[ok] = self._model.predict_proba(_finite_x(x[ok]))[:, 1]
        return out

    def evidence(self, row: dict[str, Any]) -> Evidence:
        """Features ranked by the forest's impurity importance."""
        self._need_fit()
        imp = self._model.feature_importances_
        return Evidence.from_items(
            (f, float(row[f]), float(imp[i])) for i, f in enumerate(self.requires)
        )

    def state_dict(self) -> dict[str, Any]:
        return {"features": list(self.requires), "seed": self.seed, "model": self._model}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.requires = list(d["features"])
        self.seed = d["seed"]
        self._model = d["model"]


# ------------------------------------------------------------------ helpers
def git_sha() -> str:
    """`git rev-parse HEAD` (+ `-dirty` with uncommitted changes), or "unknown"."""

    def run(*args: str) -> str:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=30
        )
        return out.stdout.strip()

    try:
        sha = run("rev-parse", "HEAD")
        return f"{sha}-dirty" if run("status", "--porcelain") else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def stable_int(symbol: str, day: Any) -> int:
    """64-bit blake2b of `symbol|day` (seeding must never use Python's salted `hash()`)."""
    digest = hashlib.blake2b(f"{symbol}|{day}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def model_version(det: Detector) -> str:
    blob = pickle.dumps(det.state_dict(), protocol=pickle.HIGHEST_PROTOCOL)
    return "eval-" + hashlib.sha256(blob).hexdigest()[:12]


def _json_float(x: Any) -> Any:
    if isinstance(x, float | np.floating):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    return {k: _json_float(v) for k, v in d.items()}


def _needs_fit(det: Detector) -> bool:
    return type(det).fit is not Detector.fit


def _score_cols(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("score__")]


# ------------------------------------------------------------------ loading
@dataclass
class _Part:
    """Scored rows of one evaluation chunk (fused and grouped into episodes on its own)."""

    frame: pl.DataFrame  # symbol, t_end_us, score__* (plus features when kept)
    labels: pl.DataFrame
    days: list[SymDay]  # loaded (not skipped)
    # days whose unmatched episodes count as false alerts (None: every loaded day)
    fa_days: list[SymDay] | None = None
    skipped: Counter = field(default_factory=Counter)
    skipped_tick: int = 0


def _dedupe(frames: list[pl.DataFrame], schema: dict[str, Any]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame(schema=schema)
    return pl.concat(frames, how="vertical").unique(
        subset=["symbol", "t_end_us"], keep="first", maintain_order=True
    )


class _Ctx:
    """Per-call state: split, labels, run metadata, and lazily loaded train frames."""

    def __init__(self, cfg: Settings, store: Store, seed: int):
        self.cfg = cfg
        self.store = store
        self.seed = int(seed)
        labels = conform(store.read_labels(), LABELS)
        self.real = labels.filter(pl.col("provenance") == REAL)
        self.split = make_split(self.real, store, cfg)
        self.config_hash = config_hash(cfg)
        self.meta = {
            "git_sha": git_sha(),
            "config_hash": self.config_hash,
            "signatures_sha": file_sha(SIGNATURES_PATH),
            "split_def": split_def(self.split, cfg),
        }
        self._train: dict[tuple[SymDay, ...], tuple[pl.DataFrame, float, Counter]] = {}

    # ---- day loading

    def load(
        self,
        items: Iterable[tuple[SymDay, Scenario | None]],
        detectors: list[Detector],
        keep_features: bool = False,
        row_filter: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    ) -> _Part:
        """Load, (inject,) score and concatenate symbol-days in `(symbol, day)` order."""
        frames, labels, days = [], [], []
        skipped: Counter = Counter()
        schema: dict[str, Any] | None = None
        for sd, sc in sorted(items, key=lambda it: it[0]):
            r = load_day_windows(
                self.store,
                sd.symbol,
                sd.day,
                self.cfg,
                scenario=sc,
                include_next_minutes=sd.include_next_minutes,
            )
            if r.skipped is not None:
                skipped[r.skipped] += 1
                continue
            w = r.windows
            if not r.next_day_present:  # extension rows were built from empty bars
                w = w.filter(pl.col("t_end_us") < day_start_us(sd.day) + DAY_US)
            if row_filter is not None:
                w = row_filter(w)
            w = score_windows(w, detectors)
            if not keep_features:
                w = w.select("symbol", "t_end_us", *_score_cols(w))
            schema = schema or dict(w.schema)
            frames.append(w)
            lab = r.labels if sc is not None else r.labels.filter(pl.col("provenance") == REAL)
            labels.append(lab)
            days.append(sd)
        if schema is None:
            schema = {"symbol": pl.String, "t_end_us": pl.Int64}
            schema.update({f"score__{d.name}": pl.Float64 for d in detectors})
        lab = pl.concat(labels) if labels else empty(LABELS)
        lab = lab.unique(subset=["label_id"], keep="first", maintain_order=True)
        return _Part(_dedupe(frames, schema), lab, days, skipped=skipped)

    def clean(self, days: Iterable[SymDay], detectors: list[Detector], **kw: Any) -> _Part:
        return self.load(((sd, None) for sd in days), detectors, **kw)

    def train_frame(self, days: list[SymDay]) -> tuple[pl.DataFrame, float, Counter]:
        """Masked, grid-thinned feature rows of `days` (cached per day list)."""
        key = tuple(sorted(days))
        if key not in self._train:
            step = self.cfg.ae.train_stride_s * S_US
            part = self.clean(
                key,
                [],
                keep_features=True,
                row_filter=lambda w: w.filter(pl.col("t_end_us") % step == 0),
            )
            keep, frac = training_mask(part.frame, self.real, margin_s=MASK_MARGIN_S)
            self._train[key] = (part.frame.filter(keep), frac, part.skipped)
        return self._train[key]

    # ---- detectors

    def build(self, names: Iterable[str]) -> dict[str, list[Detector]]:
        return {n: build_detectors([n], self.cfg, self.seed) for n in dict.fromkeys(names)}

    def fit(self, dets: Iterable[Detector], days: list[SymDay]) -> tuple[dict[str, str], dict]:
        """Fit every trainable detector on the train frame of `days`; returns
        `(model_versions, train_info)`."""
        dets = [d for d in dets if _needs_fit(d)]
        info: dict[str, Any] = {}
        if not dets:
            return {}, info
        frame, frac, skipped = self.train_frame(days)
        info = {
            "train_rows": frame.height,
            "train_excluded_frac": frac,
            "_skips": {"train": skipped},
        }
        if frame.height == 0:
            raise _VariantError(f"no train_normal rows to fit {[d.name for d in dets]} on")
        for d in dets:
            try:
                d.fit(frame)
            except ValueError as e:
                raise _VariantError(f"fit of {d.name} failed: {e}") from e
        return {d.name: model_version(d) for d in dets}, info

    # ---- semi-synthetic scenarios

    def scenario(self, sd: SymDay, fam_idx: int, ranges: dict) -> Scenario | None:
        """One test-region scenario for `sd` (None: a pump too small for the price tick)."""
        family = FAMILIES[fam_idx]
        rng = np.random.default_rng([self.seed, fam_idx, stable_int(sd.symbol, sd.day)])
        d0 = day_start_us(sd.day)
        sc = sample_scenario(family, "test", d0 + SCENARIO_LEAD_S * S_US, rng, ranges)
        hi = 86_400 - math.ceil(sc.duration_s) - SCENARIO_TAIL_S
        if hi < SCENARIO_LEAD_S:
            raise ValueError(f"{family} duration {sc.duration_s} s does not fit in one day")
        sc = replace(sc, t_start_us=d0 + int(rng.integers(SCENARIO_LEAD_S, hi + 1)) * S_US)
        if family.startswith("pump"):
            trades = self.store.read_trades(sd.symbol, d0 - DAY_US, d0 + DAY_US)
            ts = trades["ts_us"].to_numpy()
            before = ts < sc.t_start_us
            if before.any():
                price = trades["price"].to_numpy()
                tick = price_tick(ts, price, sc.t_start_us)
                last = float(price[np.flatnonzero(before)[-1]])
                if last / tick * sc.impact < MIN_TICKS:
                    return None
        return sc

    def semisyn_parts(self, detectors: list[Detector]) -> list[_Part]:
        """One injected part per family over `test_normal` (FA counted on every day)."""
        assert_injection_allowed(self.cfg.mode)  # Red-Team injection: replay only
        ranges = load_scenario_ranges()
        parts = []
        for fam_idx, _family in enumerate(FAMILIES):
            items, n_tick = [], 0
            for sd in self.split.test_normal:
                sc = self.scenario(sd, fam_idx, ranges)
                if sc is None:
                    n_tick += 1
                else:
                    items.append((sd, sc))
            part = self.load(items, detectors)
            part.skipped_tick = n_tick
            parts.append(part)
        return parts

    def real_parts(
        self, events: list[SymDay], normal: list[SymDay], detectors: list[Detector]
    ) -> _Part:
        part = self.clean([*events, *normal], detectors)
        loaded = set(part.days)
        part.fa_days = [sd for sd in normal if sd in loaded]
        return part

    # ---- records

    def record(
        self,
        experiment: str,
        variant: str,
        budget: float,
        provenance: str,
        **fields_: Any,
    ) -> dict[str, Any]:
        return {
            "run_id": f"{experiment}|{variant}|b{budget:g}|s{self.seed}|{self.config_hash[:12]}",
            "experiment": experiment,
            "variant": variant,
            "budget": float(budget),
            "seed": self.seed,
            "provenance": provenance,
            **self.meta,
            "model_versions": {},
            "metrics": None,
            "per_event": {},
            "tau": None,
            "tau_saturated": None,
            "n_cal": None,
            "n_events": None,
            "n_symbol_days": None,
            "n_skipped": {},
            "error": None,
            **fields_,
        }


# ------------------------------------------------------------------ evaluation
def _normal_rows(fused: pl.DataFrame, days: list[SymDay]) -> pl.DataFrame:
    keys = pl.DataFrame(
        {
            "symbol": [sd.symbol for sd in days],
            "_day": [day_start_us(sd.day) // DAY_US for sd in days],
        },
        schema={"symbol": pl.String, "_day": pl.Int64},
    )
    return (
        fused.with_columns((pl.col("t_end_us") // DAY_US).alias("_day"))
        .join(keys, on=["symbol", "_day"], how="semi")
        .drop("_day")
    )


def _episodes(chief: Chief, fused: pl.DataFrame, tau: float) -> list:
    return chief.episodes(fused, tau, provenance="replay") if fused.height else []


def _evaluate(
    chief: Chief,
    tau: float,
    fused_parts: list[pl.DataFrame],
    parts: list[_Part],
    hard: tuple[pl.DataFrame, _Part] | None,
    before_s: int,
    after_s: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    det_any: dict[str, bool] = {}
    det_ok: dict[str, bool] = {}
    ttd: dict[str, float] = {}
    n_alerts = n_matched = n_events = 0
    fa_unmatched, fa_days = 0, 0
    Fs, ys = [], []
    info: dict[str, dict[str, Any]] = {}
    for fused, part in zip(fused_parts, parts, strict=True):
        alerts = _episodes(chief, fused, tau)
        mr = match_events(alerts, part.labels, before_s, after_s)
        det_any.update(mr.detected_any)
        det_ok.update(mr.detected_correct)
        ttd.update(mr.ttd_s)
        n_alerts += mr.n_alerts
        n_matched += mr.n_alerts_matched
        n_events += mr.n_events
        if part.fa_days is None:
            fa_unmatched += mr.n_alerts - mr.n_alerts_matched
            fa_days += len(part.days)
        else:
            normal = _episodes(chief, _normal_rows(fused, part.fa_days), tau)
            mn = match_events(normal, part.labels, before_s, after_s)
            fa_unmatched += mn.n_alerts - mn.n_alerts_matched
            fa_days += len(part.fa_days)
        y = window_positive(
            fused["symbol"].to_numpy(), fused["t_end_us"].to_numpy(), part.labels, before_s, after_s
        )
        Fs.append(fused["F"].to_numpy())
        ys.append(y)
        for lab in part.labels.select("label_id", "symbol", "family").iter_rows(named=True):
            info[lab["label_id"]] = {"symbol": lab["symbol"], "family": lab["family"]}
    mr = MatchResult(n_events, det_any, det_ok, n_alerts, n_matched, ttd)
    y = np.concatenate(ys) if ys else np.zeros(0, dtype=bool)
    F = np.concatenate(Fs) if Fs else np.zeros(0)
    metrics = summarize(mr, fa_days, int(y.sum()), int(y.size))
    # summarize counts every unmatched episode; the FA rate is defined on normal days only
    metrics["fa_per_symbol_day"] = fa_unmatched / fa_days if fa_days else math.nan
    metrics["pr_auc"] = window_pr_auc(F, y)
    metrics["n_alerts"] = n_alerts
    metrics["n_alerts_matched"] = n_matched
    if hard is not None:
        h_fused, h_part = hard
        h_alerts = _episodes(chief, h_fused, tau)
        mh = match_events(h_alerts, h_part.labels, before_s, after_s)
        n_h = len(h_part.days)
        metrics["fa_per_symbol_day_hard"] = (
            (mh.n_alerts - mh.n_alerts_matched) / n_h if n_h else math.nan
        )
    per_event = {
        lid: {
            **info.get(lid, {}),
            "detected_any": det_any[lid],
            "detected_correct": det_ok[lid],
            "ttd_s": _json_float(ttd.get(lid, math.nan)),
        }
        for lid in sorted(det_any)
    }
    return _clean(metrics), per_event


def _skips(**parts: _Part | list[_Part] | Counter | None) -> dict[str, dict[str, int]]:
    out = {}
    for name, p in parts.items():
        if p is None:
            continue
        if isinstance(p, Counter):
            c = p
        else:
            c = sum((q.skipped for q in (p if isinstance(p, list) else [p])), Counter())
        out[name] = dict(sorted(c.items()))
    return out


def _variant(
    ctx: _Ctx,
    experiment: str,
    variant: str,
    provenance: str,
    detectors: list[Detector],
    model_versions: dict[str, str],
    cal: _Part,
    parts: list[_Part],
    hard: _Part | None,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    """Calibrate one Chief on `cal`, then one record per budget."""
    cfg = ctx.cfg
    names = [d.name for d in detectors]
    deterministic = provenance != SEMI and not (SEEDED & set(names))
    skips = _skips(cal=cal, eval=parts, hard=hard, **extra.pop("_skips", {}))
    base = {
        "detectors": names,
        "deterministic": deterministic,
        "model_versions": dict(model_versions),
        "n_events": int(sum(p.labels.height for p in parts)),
        "n_symbol_days": int(sum(len(p.days) for p in parts)),
        "n_skipped": skips,
        **extra,
    }
    chief = Chief(detectors, cfg.chief, ctx.config_hash, model_versions)
    cols = ["symbol", "t_end_us", *[f"score__{n}" for n in names]]
    chief.calibrate(cal.frame.select(cols))
    cal_fused = chief.fuse(cal.frame.select(cols))
    fused_parts = [chief.fuse(p.frame.select(cols)) for p in parts]
    hard_pair = (chief.fuse(hard.frame.select(cols)), hard) if hard is not None else None
    out = []
    for budget in cfg.eval.budgets_per_symbol_day:
        rec = ctx.record(experiment, variant, budget, provenance, **base, n_cal=chief.n_cal)
        try:
            with warnings.catch_warnings():
                # saturation is reported through tau_saturated, not as a warning
                warnings.simplefilter("ignore", RuntimeWarning)
                tau = chief.fit_threshold(cal_fused, budget)
        except CalibrationError as e:
            rec["error"] = str(e)
            out.append(rec)
            continue
        metrics, per_event = _evaluate(
            chief, tau, fused_parts, parts, hard_pair, cfg.eval.before_s, cfg.eval.after_s
        )
        rec.update(
            metrics=metrics, per_event=per_event, tau=tau, tau_saturated=bool(chief.tau_saturated)
        )
        out.append(rec)
    return out


def _error_records(
    ctx: _Ctx, experiment: str, variant: str, provenance: str, err: Exception, **extra: Any
) -> list[dict[str, Any]]:
    return [
        ctx.record(experiment, variant, b, provenance, error=str(err), **extra)
        for b in ctx.cfg.eval.budgets_per_symbol_day
    ]


def _flatten(groups: dict[str, list[Detector]], names: Iterable[str]) -> list[Detector]:
    return [d for n in dict.fromkeys(names) for d in groups[n]]


def _fit_groups(
    ctx: _Ctx, groups: dict[str, list[Detector]], days: list[SymDay]
) -> tuple[dict[str, str], dict, Exception | None]:
    try:
        mv, info = ctx.fit([d for ds in groups.values() for d in ds], days)
        return mv, info, None
    except _VariantError as e:
        return {}, {}, e


def _versions_for(mv: dict[str, str], dets: list[Detector]) -> dict[str, str]:
    names = {d.name for d in dets}
    return {k: v for k, v in mv.items() if k in names}


# ------------------------------------------------------------------ experiments
def _real_sets(ctx: _Ctx, dets: list[Detector]) -> tuple[_Part, _Part, _Part]:
    sp = ctx.split
    cal = ctx.clean(sp.cal_normal, dets)
    real = ctx.real_parts(sp.test_events, sp.test_normal, dets)
    hard = ctx.clean(sp.test_hard_negative, dets)
    return cal, real, hard


def _exp_real_t001(ctx: _Ctx) -> list[dict]:
    groups = ctx.build(ctx.cfg.detectors)
    mv, info, err = _fit_groups(ctx, groups, ctx.split.train_normal)
    if err is not None:
        return _error_records(ctx, "real_t001", "full", REAL, err)
    dets = _flatten(groups, groups)
    cal, real, hard = _real_sets(ctx, dets)
    return _variant(ctx, "real_t001", "full", REAL, dets, mv, cal, [real], hard, info)


def _exp_semisyn(ctx: _Ctx) -> list[dict]:
    groups = ctx.build(ctx.cfg.detectors)
    mv, info, err = _fit_groups(ctx, groups, ctx.split.train_normal)
    if err is not None:
        return [r for f in FAMILIES for r in _error_records(ctx, "semisyn", f, SEMI, err)]
    dets = _flatten(groups, groups)
    cal = ctx.clean(ctx.split.cal_normal, dets)
    out = []
    for family, part in zip(FAMILIES, ctx.semisyn_parts(dets), strict=True):
        extra = {**info, "n_skipped_tick": part.skipped_tick}
        out += _variant(ctx, "semisyn", family, SEMI, dets, mv, cal, [part], None, extra)
    return out


def _exp_ablation(ctx: _Ctx) -> list[dict]:
    combos = {k: (v if v is not None else list(ctx.cfg.detectors)) for k, v in ABLATION.items()}
    groups = ctx.build(n for names in combos.values() for n in names)
    mv, info, err = _fit_groups(ctx, groups, ctx.split.train_normal)
    if err is not None:
        return [
            r
            for s, prov in (("real_t001", REAL), ("semisyn", SEMI))
            for c in combos
            for r in _error_records(ctx, "ablation", f"{s}:{c}", prov, err)
        ]
    every = _flatten(groups, groups)
    cal, real, hard = _real_sets(ctx, every)
    semi = ctx.semisyn_parts(every)
    out = []
    for combo, names in combos.items():
        dets = _flatten(groups, names)
        out += _variant(
            ctx,
            "ablation",
            f"real_t001:{combo}",
            REAL,
            dets,
            _versions_for(mv, dets),
            cal,
            [real],
            hard,
            dict(info),
        )
        extra = {**info, "n_skipped_tick": sum(p.skipped_tick for p in semi)}
        out += _variant(
            ctx,
            "ablation",
            f"semisyn:{combo}",
            SEMI,
            dets,
            _versions_for(mv, dets),
            cal,
            semi,
            None,
            extra,
        )
    return out


def _exp_baselines(ctx: _Ctx) -> list[dict]:
    groups = ctx.build(BASELINES)
    mv, info, err = _fit_groups(ctx, groups, ctx.split.train_normal)
    if err is not None:
        return [r for n in BASELINES for r in _error_records(ctx, "baselines", n, REAL, err)]
    every = _flatten(groups, groups)
    cal, real, hard = _real_sets(ctx, every)
    out = []
    for name in BASELINES:
        dets = groups[name]
        out += _variant(
            ctx,
            "baselines",
            name,
            REAL,
            dets,
            _versions_for(mv, dets),
            cal,
            [real],
            hard,
            dict(info),
        )
    return out


def _rf_positive(frame: pl.DataFrame, labels: pl.DataFrame) -> np.ndarray:
    y = np.zeros(frame.height, dtype=bool)
    sym = frame["symbol"].to_numpy()
    t = frame["t_end_us"].to_numpy()
    for s, t0 in labels.select("symbol", "t_start_us").iter_rows():
        y |= (sym == s) & (t >= t0) & (t <= t0 + RF_POSITIVE_S * S_US)
    return y


def _exp_ref_supervised(ctx: _Ctx) -> list[dict]:
    sp = ctx.split
    extra = {"note": SUPERVISED_NOTE}
    rf = SupervisedRFDetector(model_features(ctx.cfg.features), ctx.seed)
    try:
        neg, frac, neg_skipped = ctx.train_frame(sp.train_normal)
        ev = ctx.clean(sp.train_events, [], keep_features=True)
        pos = ev.frame.filter(pl.Series(_rf_positive(ev.frame, ev.labels), dtype=pl.Boolean))
        if pos.height == 0 or neg.height == 0:
            raise ValueError(
                f"{RF_NAME}: no rows to train on ({pos.height} positive from train_events, "
                f"{neg.height} negative from train_normal)"
            )
        rows = pl.concat([pos.select(rf.requires), neg.select(rf.requires)])
        y = np.r_[np.ones(pos.height, dtype=bool), np.zeros(neg.height, dtype=bool)]
        rf.fit_supervised(rows, y)
    except ValueError as e:
        return _error_records(ctx, "ref_supervised", RF_NAME, REAL, e, **extra)
    extra |= {
        "train_rows": rows.height,
        "train_positive_rows": pos.height,
        "train_excluded_frac": frac,
        "_skips": {"train": neg_skipped, "train_events": ev.skipped},
    }
    cal, real, hard = _real_sets(ctx, [rf])
    mv = {RF_NAME: model_version(rf)}
    return _variant(ctx, "ref_supervised", RF_NAME, REAL, [rf], mv, cal, [real], hard, extra)


def _exp_retrain_drift(ctx: _Ctx) -> list[dict]:
    sp = ctx.split
    names = list(dict.fromkeys([*ctx.cfg.detectors, AE]))
    out = []
    for variant, (lo, hi) in zip(("v1", "v2"), drift_ranges(ctx.cfg.splits), strict=True):
        days = [sd for sd in sp.train_normal if lo <= sd.day <= hi]
        extra = {"train_range": [lo.isoformat(), hi.isoformat()], "train_days": len(days)}
        groups = ctx.build(names)  # fresh instances: v1 and v2 never share a fitted AE
        mv, info, err = _fit_groups(ctx, groups, days)
        if err is not None:
            out += _error_records(ctx, "retrain_drift", variant, REAL, err, **extra)
            continue
        dets = _flatten(groups, groups)
        cal = ctx.clean(sp.cal_normal, dets)
        acc = ctx.real_parts(sp.accept_events, sp.accept_normal, dets)
        out += _variant(
            ctx, "retrain_drift", variant, REAL, dets, mv, cal, [acc], None, {**info, **extra}
        )
    return out


_RUNNERS: dict[str, Callable[[_Ctx], list[dict]]] = {
    "real_t001": _exp_real_t001,
    "semisyn": _exp_semisyn,
    "ablation": _exp_ablation,
    "baselines": _exp_baselines,
    "ref_supervised": _exp_ref_supervised,
    "retrain_drift": _exp_retrain_drift,
}
EXPERIMENTS = tuple(_RUNNERS)
INJECTING = ("semisyn", "ablation")  # experiments that run Red-Team injection


def run_experiment(
    name: str, cfg: Settings, store: Store, seed: int, *, research: Store | None = None
) -> list[dict]:
    """Run one experiment for one seed; every record is written with `insert_run`.

    Trades and labels come from `store` (REAL experiments use labels with provenance "REAL");
    records go to `research` (default: `store`). Re-running replaces records by `run_id`.
    """
    if name not in _RUNNERS:
        raise KeyError(f"unknown experiment {name!r}; valid: {list(EXPERIMENTS)}")
    if name in INJECTING:
        assert_injection_allowed(cfg.mode)  # fail before any work
    ctx = _Ctx(cfg, store, seed)
    records = _RUNNERS[name](ctx)
    sink = research if research is not None else store
    for r in records:
        sink.insert_run(r)
    return records
