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
`D+1` are dropped when the `D+1` partition is absent. Skipped days are counted per reason. A
set's events are only the labels whose `t_start` falls on one of its event days: a label read
through the near-midnight extension or an overlap belongs to another period, or was dropped from
the split (`missing_day`), and is not counted here.

`m1_fusion` (M1: false alarms the AE adds on volatile days) fits and scores the detectors once
per seed, then re-fuses the same scores under every `m1_variants` AI fusion mode (`AiFusion`),
each with its own calibration and `tau`. Records are `accept:<v>` (the accept split; hard
negatives are the `m1_hard_k` most volatile accept-normal days, picked like the test hard
negatives and computed here, so `Split` and its sha are unchanged), `test:<v>` (REAL) and
`semisyn:<v>` (the injected test-normal days, scored once and shared like the ablation). Rows
keep the regime feature (`m1_regime_feature`) for the Mondrian variant.

`m2_ai` (M2: does an AI layer earn its place) fits four AI models once per seed: today's
autoencoder (`ae`), two grid points of it (`ae_wide`: hidden 128 / bottleneck 16, `ae_narrow`:
hidden 64 / bottleneck 8; every other `ae` setting as configured) and IsolationForest on the AI
layer (`ai.iforest`). Every set is scored once with the rules, stats and all four models; each
variant then fuses those scores through its own Chief: the AI set `ae`, `ae_wide`, `ae_narrow`,
`iforest` or `ae+iforest` (both enter F) with fusion `fuse`, `shift1` or `shift2` (`AiFusion`;
the shift applies to each AI detector), plus the reference `no_ai` (today's AE scored but
excluded: exactly m1_fusion's `no_ai`). Records are `<split>:<model>:<fusion>` and
`<split>:no_ai` for split accept (accept-hard as in m1_fusion), test (REAL) and semisyn (the
injected test-normal days, scored in the same single pass), plus the diagnostics
`test:ae:alone` and `test:iforest:alone` (that AI model alone through its own Chief, on the same
scores) for the report's AE-vs-iforest line.

`fa_by_day` holds per-day unmatched-episode counts; when several parts load the same symbol-day
(the SEMI-SYN families of one variant) it also holds `normal_mult`, how many parts loaded each
day (missing keys and older records: 1), so the report's FA CI is the ratio `sum n / sum m`.

`m4_sell` (M4: the sell side) runs with `features.sell_side` on for every load (the existing
columns are bit-identical to the flag-off ones, so the AE fit and every existing score are the
`real_t001` ones), fits and scores once per seed, then fuses two detector sets: `base` (the
configured detectors, without the sell-side rules) and `sell` (with them). Records are
`test:<v>` (REAL, with the per-phase recall of `idsai.eval.sell_labels`: the sell windows are
measured once per call and shared by both variants) and `bear_raid:<v>` (SEMI-SYN: the `bear_raid`
injection over the test-normal days, FA counted on every injected day). Every record keeps the
invoking config's `config_hash` (one report block) and adds `sell_side` and `signatures_sell_sha`.

`m3_attr` (M3: name the technique of detected events) runs the live stack inside the experiment
whatever the invoking config says (`M3_LIVE`: rules, stat.robust_z, stat.cusum, ai.iforest with
`AiFusion("shift", 1.0)`), so the eval config and its `config_hash` stay as they are. It fits
`ai.iforest` once per seed and scores every set once with the live detectors, the candidate wash
rule 100501 rev 2 (`configs/signatures_wash2.yaml`, scored as `rule.100501.rev2`) and the naming
rules (`configs/signatures_attr.yaml`, whatever `chief.attr_rules` says). Variants re-fuse those
scores: `base`, `attr` (naming rules on), `wash2` (rev 2 in place of rev 1) and `attr+wash2`.
`attr` keeps the base Chief's calibration, tau and exceedances by construction; `wash2` changes an
F-rule, so its FA can move. Records are `<split>:<variant>` for accept (accept-hard as in
m1_fusion), test (REAL) and semisyn (the injected test-normal days); they add `attr_rules`,
`wash_rev`, `ai_fusion`, the two signature shas, and the FA-naming counts in `metrics`
(`n_fa_episodes`, `n_fa_named`: unmatched normal-day episodes with a technique other than
MM-T000, `n_fa_named_by_rule`: those named by a naming rule).

`retrain_drift` records the full stack with AE v1 / v2 (`v1`, `v2`) and the AE alone through
its own Chief on the same scores (`v1:ai`, `v2:ai`). Records carry `classifies`: whether the
detector set can name a technique at all (a rule or `stat.cusum`), so recall_correct of
technique-agnostic sets is not read as a real 0.

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

from idsai.ai.chief import (
    CUSUM_NAME,
    NAMING_SOURCE,
    UNCLASSIFIED,
    AiFusion,
    CalibrationError,
    Chief,
)
from idsai.ai.learning import training_mask
from idsai.config import REPO_ROOT, Settings, config_hash, file_sha
from idsai.contracts.detector import Detector, Evidence, feature_matrix
from idsai.contracts.frames import LABELS, conform, empty
from idsai.data.store import DAY_US, Store
from idsai.detection.features.registry import model_features
from idsai.detection.registry import build_detectors, sell_rule_detectors
from idsai.detection.rules import (
    NAMING_SIGNATURES_PATH,
    SELL_SIGNATURES_PATH,
    SIGNATURES_PATH,
    WASH2_SIGNATURES_PATH,
    SignatureDetector,
    load_signatures,
)
from idsai.detection.simulator import (
    Scenario,
    assert_injection_allowed,
    load_scenario_ranges,
    sample_scenario,
)
from idsai.detection.simulator.inject import price_tick
from idsai.eval.metrics import (
    MatchResult,
    match_events,
    summarize,
    unmatched_onsets,
    window_positive,
    window_pr_auc,
)
from idsai.eval.sell_labels import SellPhase, derive_sell_labels
from idsai.eval.splits import (
    SymDay,
    day_of_us,
    day_start_us,
    drift_ranges,
    make_split,
    realized_vol_1m,
    split_def,
)
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
SEEDED = {
    AE,
    "baseline.iforest",
    "baseline.random",
    RF_NAME,
    # the M2 AI models (names spelled out: the M2 constants are defined further down)
    "ai.iforest",
    "ai.autoencoder_wide",
    "ai.autoencoder_narrow",
}
MASK_MARGIN_S = 3600
RF_POSITIVE_S = 300
MIN_TICKS = 2.0  # skip pumps whose impact moves the price by fewer ticks
SCENARIO_LEAD_S = 3600
SCENARIO_TAIL_S = 600
SUPERVISED_NOTE = "supervised reference (uses labels)"
SATURATION_WARNING = r"alert budget .* is at or above the peak episode rate"
REAL, SEMI = "REAL", "SEMI-SYN"
M1 = "m1_fusion"
M1_SPLITS = ("accept", "test", "semisyn")
M2 = "m2_ai"
M2_SPLITS = ("accept", "test", "semisyn")
# the AE grid (kept small on purpose: laptop CPU); `ae` is today's config
M2_AE_GRID: dict[str, dict[str, int]] = {
    "ae": {},
    "ae_wide": {"hidden": 128, "bottleneck": 16},
    "ae_narrow": {"hidden": 64, "bottleneck": 8},
}
IFOREST_AI = "ai.iforest"
# detector name of each fitted AI model (the grid points need their own score columns)
M2_AI_NAMES = {
    "ae": AE,
    "ae_wide": "ai.autoencoder_wide",
    "ae_narrow": "ai.autoencoder_narrow",
    "iforest": IFOREST_AI,
}
# the AI set of each variant model: which fitted models enter the fusion
M2_MODELS: dict[str, tuple[str, ...]] = {
    "ae": ("ae",),
    "ae_wide": ("ae_wide",),
    "ae_narrow": ("ae_narrow",),
    "iforest": ("iforest",),
    "ae+iforest": ("ae", "iforest"),
}
M2_FUSIONS = {
    "fuse": AiFusion(),
    "shift1": AiFusion("shift", 1.0),
    "shift2": AiFusion("shift", 2.0),
}
M2_REF = "no_ai"
M2_ALONE = ("ae", "iforest")  # diagnostics on test: the AI model alone, its own Chief
M4 = "m4_sell"
M4_VARIANTS = ("base", "sell")
BEAR_RAID = "bear_raid"
BEAR_RAID_STREAM = len(FAMILIES)  # rng stream index of bear_raid scenarios (after FAMILIES)
M3 = "m3_attr"
M3_SPLITS = ("accept", "test", "semisyn")
# the live stack of feat/live-ai, built inside m3_attr (the eval config keeps its hash)
M3_LIVE = ("rules", "stat.robust_z", "stat.cusum", "ai.iforest")
M3_FUSION = AiFusion("shift", 1.0)
# variant -> (naming rules on, wash rule rev); the selection's tie order is this order
M3_VARIANTS: dict[str, tuple[bool, int]] = {
    "base": (False, 1),
    "attr": (True, 1),
    "wash2": (False, 2),
    "attr+wash2": (True, 2),
}
WASH_SID = 100501
WASH2_NAME = "rule.100501.rev2"


def m1_regime_feature(cfg: Settings) -> str:
    """Regime variable of the Mondrian variant: `jump_<longest window>` (e.g. `jump_1800`).

    The registry has no return-dispersion feature; `jump` (highest price of the window against
    the close before it, minus 1) is an upward price range over the window, not an absolute
    volatility measure ("ช่วงราคาขาขึ้นในหน้าต่าง 1800 วินาที (jump_1800)"). It is used raw (a
    `z_` column is re-centred on each symbol's own 24 h). Causal: past bars only. Caveat: pump
    rows have a high jump by construction, so Mondrian bins may lower AE recall on pumps; the
    selection's recall gate against V0 handles that.
    """
    return f"jump_{max(cfg.features.windows_s)}"


def m1_variants(cfg: Settings) -> dict[str, AiFusion]:
    """V0 `all`, V1 `no_ai`, V2 `ai_mondrian`, V3 `ai_shift1` / `ai_shift2`."""
    return {
        "all": AiFusion(),
        "no_ai": AiFusion("exclude"),
        "ai_mondrian": AiFusion("mondrian", regime_feature=m1_regime_feature(cfg)),
        "ai_shift1": AiFusion("shift", 1.0),
        "ai_shift2": AiFusion("shift", 2.0),
    }


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


def model_version(det: Detector, prefix: str = "eval") -> str:
    """`prefix-` + 12-hex sha256 of the pickled `state_dict` (a content address of the model)."""
    blob = pickle.dumps(det.state_dict(), protocol=pickle.HIGHEST_PROTOCOL)
    return f"{prefix}-" + hashlib.sha256(blob).hexdigest()[:12]


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


def _labels_on_days(labels: pl.DataFrame, days: Iterable[SymDay]) -> pl.DataFrame:
    """Labels whose `(symbol, day_of(t_start))` is one of `days`."""
    keys = {(sd.symbol, sd.day) for sd in days}
    keep = [
        (sym, day_of_us(t0)) in keys
        for sym, t0 in labels.select("symbol", "t_start_us").iter_rows()
    ]
    return labels.filter(pl.Series(keep, dtype=pl.Boolean))


class _Ctx:
    """Per-call state: split, labels, run metadata, and lazily loaded train frames.

    `with_split=False` skips the evaluation split (and its `split_def` in `meta`): enough to fit
    and calibrate on given days (`fit_and_calibrate`), without reading every labeled symbol.
    """

    def __init__(self, cfg: Settings, store: Store, seed: int, with_split: bool = True):
        self.cfg = cfg
        self.store = store
        self.seed = int(seed)
        labels = conform(store.read_labels(), LABELS)
        self.real = labels.filter(pl.col("provenance") == REAL)
        self.split = make_split(self.real, store, cfg) if with_split else None
        self.config_hash = config_hash(cfg)
        self.meta = {
            "git_sha": git_sha(),
            "config_hash": self.config_hash,
            "signatures_sha": file_sha(SIGNATURES_PATH),
        }
        if self.split is not None:
            self.meta["split_def"] = split_def(self.split, cfg)
        self._train: dict[tuple[SymDay, ...], tuple[pl.DataFrame, float, Counter]] = {}

    # ---- day loading

    def load(
        self,
        items: Iterable[tuple[SymDay, Scenario | None]],
        detectors: list[Detector],
        keep_features: bool = False,
        row_filter: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
        keep: tuple[str, ...] = (),
    ) -> _Part:
        """Load, (inject,) score and concatenate symbol-days in `(symbol, day)` order.

        Without `keep_features` only `symbol`, `t_end_us`, the `keep` columns and the scores stay.
        """
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
                w = w.select("symbol", "t_end_us", *keep, *_score_cols(w))
            schema = schema or dict(w.schema)
            frames.append(w)
            lab = r.labels if sc is not None else r.labels.filter(pl.col("provenance") == REAL)
            labels.append(lab)
            days.append(sd)
        if schema is None:
            schema = {"symbol": pl.String, "t_end_us": pl.Int64}
            schema.update({c: pl.Float64 for c in keep})
            schema.update({f"score__{d.name}": pl.Float64 for d in detectors})
        lab = pl.concat(labels) if labels else empty(LABELS)
        lab = lab.unique(subset=["label_id"], keep="first", maintain_order=True)
        return _Part(_dedupe(frames, schema), _labels_on_days(lab, days), days, skipped=skipped)

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

    def scenario(
        self, sd: SymDay, fam_idx: int, ranges: dict, family: str | None = None
    ) -> Scenario | None:
        """One test-region scenario for `sd` (None: a pump or bear raid too small for the price
        tick). `family` defaults to `FAMILIES[fam_idx]`; `fam_idx` always picks the rng stream."""
        family = FAMILIES[fam_idx] if family is None else family
        rng = np.random.default_rng([self.seed, fam_idx, stable_int(sd.symbol, sd.day)])
        d0 = day_start_us(sd.day)
        sc = sample_scenario(family, "test", d0 + SCENARIO_LEAD_S * S_US, rng, ranges)
        hi = 86_400 - math.ceil(sc.duration_s) - SCENARIO_TAIL_S
        if hi < SCENARIO_LEAD_S:
            raise ValueError(f"{family} duration {sc.duration_s} s does not fit in one day")
        sc = replace(sc, t_start_us=d0 + int(rng.integers(SCENARIO_LEAD_S, hi + 1)) * S_US)
        if family.startswith("pump") or family == BEAR_RAID:
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

    def semisyn_parts(self, detectors: list[Detector], **kw: Any) -> list[_Part]:
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
            part = self.load(items, detectors, **kw)
            part.skipped_tick = n_tick
            parts.append(part)
        return parts

    def bear_raid_part(self, detectors: list[Detector], **kw: Any) -> _Part:
        """The `bear_raid` injection over `test_normal` (M4; FA counted on every day)."""
        assert_injection_allowed(self.cfg.mode)  # Red-Team injection: replay only
        ranges = load_scenario_ranges()
        items, n_tick = [], 0
        for sd in self.split.test_normal:
            sc = self.scenario(sd, BEAR_RAID_STREAM, ranges, family=BEAR_RAID)
            if sc is None:
                n_tick += 1
            else:
                items.append((sd, sc))
        part = self.load(items, detectors, **kw)
        part.skipped_tick = n_tick
        return part

    def real_parts(
        self, events: list[SymDay], normal: list[SymDay], detectors: list[Detector], **kw: Any
    ) -> _Part:
        part = self.clean([*events, *normal], detectors, **kw)
        loaded = set(part.days)
        part.fa_days = [sd for sd in normal if sd in loaded]
        part.labels = _labels_on_days(part.labels, events)
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


def _day_key(symbol: str, day: Any) -> str:
    return f"{symbol}|{day.isoformat()}"


def _fa_by_day(onsets: list[tuple[str, int]], days: list[SymDay]) -> dict[str, int]:
    """Unmatched episodes per `"SYM|YYYY-MM-DD"` of `days`, zero-count days included.

    An episode counts on the symbol-day of its onset; an onset outside `days` (a window that
    runs into the next day's first minutes) counts on the latest earlier listed day of that
    symbol, else the earliest one, so the counts always sum to the number of onsets (a symbol
    with no listed day at all, which the callers never produce, gets its own key).
    """
    out = {_day_key(sd.symbol, sd.day): 0 for sd in days}
    by_sym: dict[str, list] = {}
    for sd in days:
        by_sym.setdefault(sd.symbol, []).append(sd.day)
    for sym in by_sym:
        by_sym[sym].sort()
    for sym, first_us in onsets:
        d = day_of_us(first_us)
        key = _day_key(sym, d)
        listed = by_sym.get(sym)
        if key not in out and listed:
            earlier = [x for x in listed if x <= d]
            key = _day_key(sym, earlier[-1] if earlier else listed[0])
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _unmatched(alerts: list, onsets: list[tuple[str, int]]) -> list:
    """The alerts whose `(symbol, window start)` is among `onsets` (`unmatched_onsets`)."""
    keys = set(onsets)
    return [a for a in alerts if (a.symbol, a.window[0]) in keys]


def _evaluate(
    chief: Chief,
    tau: float,
    fused_parts: list[pl.DataFrame],
    parts: list[_Part],
    hard: tuple[pl.DataFrame, _Part] | None,
    before_s: int,
    after_s: int,
    phase: SellPhase | None = None,
    fa_names: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, int]]]:
    """`(metrics, per_event, fa_by_day)`; `fa_by_day` = `{"normal": {...}, "hard": {...}}`,
    plus `"normal_mult"` (parts per day) when a day is loaded by more than one part. With
    `phase` (M4) every event also gets the sell-phase keys and the metrics the per-phase recall.
    With `fa_names` (M3) the metrics count the unmatched normal-day episodes and how many of them
    carry a name (`n_fa_episodes`, `n_fa_named`, `n_fa_named_by_rule`)."""
    det_any: dict[str, bool] = {}
    det_ok: dict[str, bool] = {}
    ttd: dict[str, float] = {}
    techs: dict[str, list[str]] = {}
    fa_normal: dict[str, int] = {}
    fa_mult: dict[str, int] = {}  # parts that loaded each normal day
    n_alerts = n_matched = n_events = 0
    fa_unmatched, fa_days = 0, 0
    Fs, ys = [], []
    info: dict[str, dict[str, Any]] = {}
    phase_pe: dict[str, dict[str, Any]] = {}
    fa_alerts: list = []  # unmatched normal-day episodes (M3 `fa_names`)
    for fused, part in zip(fused_parts, parts, strict=True):
        alerts = _episodes(chief, fused, tau)
        if phase is not None:
            ex = fused.filter((pl.col("F") >= tau) & (pl.col("F") > 0))  # the episodes' rows
            exceed = {
                str(k[0]): g["t_end_us"].to_numpy()
                for k, g in ex.select("symbol", "t_end_us").group_by("symbol")
            }
            phase_pe.update(phase.per_event(alerts, part.labels, exceed))
        mr = match_events(alerts, part.labels, before_s, after_s)
        det_any.update(mr.detected_any)
        det_ok.update(mr.detected_correct)
        ttd.update(mr.ttd_s)
        techs.update(mr.techniques)
        n_alerts += mr.n_alerts
        n_matched += mr.n_alerts_matched
        n_events += mr.n_events
        if part.fa_days is None:
            fa_unmatched += mr.n_alerts - mr.n_alerts_matched
            fa_days += len(part.days)
            onsets = unmatched_onsets(alerts, part.labels, before_s, after_s)
            day_counts = _fa_by_day(onsets, part.days)
            if fa_names:
                fa_alerts += _unmatched(alerts, onsets)
        else:
            normal = _episodes(chief, _normal_rows(fused, part.fa_days), tau)
            mn = match_events(normal, part.labels, before_s, after_s)
            fa_unmatched += mn.n_alerts - mn.n_alerts_matched
            fa_days += len(part.fa_days)
            onsets = unmatched_onsets(normal, part.labels, before_s, after_s)
            day_counts = _fa_by_day(onsets, part.fa_days)
            if fa_names:
                fa_alerts += _unmatched(normal, onsets)
        for k, v in day_counts.items():
            fa_normal[k] = fa_normal.get(k, 0) + v
            fa_mult[k] = fa_mult.get(k, 0) + 1
        y = window_positive(
            fused["symbol"].to_numpy(), fused["t_end_us"].to_numpy(), part.labels, before_s, after_s
        )
        Fs.append(fused["F"].to_numpy())
        ys.append(y)
        for lid, sym, fam, t0 in part.labels.select(
            "label_id", "symbol", "family", "t_start_us"
        ).iter_rows():
            # symbol + day identify the bootstrap cluster of injected (SEMI-SYN) events
            info[lid] = {"symbol": sym, "day": day_of_us(t0).isoformat(), "family": fam}
    mr = MatchResult(n_events, det_any, det_ok, n_alerts, n_matched, ttd)
    y = np.concatenate(ys) if ys else np.zeros(0, dtype=bool)
    F = np.concatenate(Fs) if Fs else np.zeros(0)
    metrics = summarize(mr, fa_days, int(y.sum()), int(y.size))
    # summarize counts every unmatched episode; the FA rate is defined on normal days only
    metrics["fa_per_symbol_day"] = fa_unmatched / fa_days if fa_days else math.nan
    metrics["pr_auc"] = window_pr_auc(F, y)
    metrics["n_alerts"] = n_alerts
    metrics["n_alerts_matched"] = n_matched
    metrics["n_fa_days"] = fa_days
    if fa_names:
        metrics["n_fa_episodes"] = len(fa_alerts)
        metrics["n_fa_named"] = sum(a.technique_id != UNCLASSIFIED for a in fa_alerts)
        metrics["n_fa_named_by_rule"] = sum(a.technique_source == NAMING_SOURCE for a in fa_alerts)
    fa_by_day: dict[str, dict[str, int]] = {"normal": dict(sorted(fa_normal.items())), "hard": {}}
    if any(m != 1 for m in fa_mult.values()):  # several parts per day (SEMI-SYN families)
        fa_by_day["normal_mult"] = dict(sorted(fa_mult.items()))
    if hard is not None:
        h_fused, h_part = hard
        h_alerts = _episodes(chief, h_fused, tau)
        mh = match_events(h_alerts, h_part.labels, before_s, after_s)
        n_h = len(h_part.days)
        metrics["fa_per_symbol_day_hard"] = (
            (mh.n_alerts - mh.n_alerts_matched) / n_h if n_h else math.nan
        )
        metrics["n_hard_days"] = n_h
        fa_by_day["hard"] = _fa_by_day(
            unmatched_onsets(h_alerts, h_part.labels, before_s, after_s), h_part.days
        )
    per_event = {
        lid: {
            **info.get(lid, {}),
            "detected_any": det_any[lid],
            "detected_correct": det_ok[lid],
            "ttd_s": _json_float(ttd.get(lid, math.nan)),
            "techniques": techs.get(lid, []),
            **phase_pe.get(lid, {}),
        }
        for lid in sorted(det_any)
    }
    if phase is not None:
        metrics.update(phase.summarize(per_event))
    return _clean(metrics), per_event, fa_by_day


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


def _describe(detectors: list[Detector], provenance: str) -> dict[str, Any]:
    """Record fields that depend only on the detector set (shared by OK and error records).

    `classifies`: the Chief can name a technique only through a fired rule or `stat.cusum`;
    technique-agnostic sets (baselines alone, the AE alone, the RF) always say MM-T000, so their
    recall_correct is not meaningful.
    """
    names = [d.name for d in detectors]
    return {
        "detectors": names,
        "deterministic": provenance != SEMI and not (SEEDED & set(names)),
        "classifies": any(d.layer == "rule" or d.name == CUSUM_NAME for d in detectors),
    }


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
    ai: AiFusion | None = None,
    phase: SellPhase | None = None,
    naming: list[Detector] | None = None,
    fa_names: bool = False,
) -> list[dict[str, Any]]:
    """Calibrate one Chief on `cal` (with the AI fusion mode `ai` and the M3 naming rules
    `naming`), then one record per budget (`phase`: the M4 per-phase recall, `fa_names`: the M3
    FA-naming counts, see `_evaluate`)."""
    cfg = ctx.cfg
    skips = _skips(cal=cal, eval=parts, hard=hard, **extra.pop("_skips", {}))
    base = {
        **_describe(detectors, provenance),
        "model_versions": dict(model_versions),
        "n_events": int(sum(p.labels.height for p in parts)),
        "n_symbol_days": int(sum(len(p.days) for p in parts)),
        "n_skipped": skips,
        **extra,
    }
    chief = Chief(detectors, cfg.chief, ctx.config_hash, model_versions, ai=ai, naming=naming)
    cols = ["symbol", "t_end_us", *[f"score__{d.name}" for d in chief.score_detectors]]
    if ai is not None and ai.regime_feature is not None:
        cols.append(ai.regime_feature)
    chief.calibrate(cal.frame.select(cols))
    cal_fused = chief.fuse(cal.frame.select(cols))
    fused_parts = [chief.fuse(p.frame.select(cols)) for p in parts]
    hard_pair = (chief.fuse(hard.frame.select(cols)), hard) if hard is not None else None
    out = []
    for budget in cfg.eval.budgets_per_symbol_day:
        rec = ctx.record(experiment, variant, budget, provenance, **base, n_cal=chief.n_cal)
        try:
            with warnings.catch_warnings():
                # saturation is reported through tau_saturated; other warnings stay visible
                warnings.filterwarnings(
                    "ignore", message=SATURATION_WARNING, category=RuntimeWarning
                )
                tau = chief.fit_threshold(cal_fused, budget)
        except CalibrationError as e:
            rec["error"] = str(e)
            out.append(rec)
            continue
        metrics, per_event, fa_by_day = _evaluate(
            chief,
            tau,
            fused_parts,
            parts,
            hard_pair,
            cfg.eval.before_s,
            cfg.eval.after_s,
            phase,
            fa_names,
        )
        rec.update(
            metrics=metrics,
            per_event=per_event,
            fa_by_day=fa_by_day,
            tau=tau,
            tau_saturated=bool(chief.tau_saturated),
        )
        out.append(rec)
    return out


def _error_records(
    ctx: _Ctx,
    experiment: str,
    variant: str,
    provenance: str,
    err: Exception,
    detectors: list[Detector],
    **extra: Any,
) -> list[dict[str, Any]]:
    desc = _describe(detectors, provenance)
    return [
        ctx.record(experiment, variant, b, provenance, error=str(err), **desc, **extra)
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
        return _error_records(ctx, "real_t001", "full", REAL, err, _flatten(groups, groups))
    dets = _flatten(groups, groups)
    cal, real, hard = _real_sets(ctx, dets)
    return _variant(ctx, "real_t001", "full", REAL, dets, mv, cal, [real], hard, info)


def _exp_semisyn(ctx: _Ctx) -> list[dict]:
    groups = ctx.build(ctx.cfg.detectors)
    mv, info, err = _fit_groups(ctx, groups, ctx.split.train_normal)
    if err is not None:
        dets = _flatten(groups, groups)
        return [r for f in FAMILIES for r in _error_records(ctx, "semisyn", f, SEMI, err, dets)]
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
            for r in _error_records(
                ctx, "ablation", f"{s}:{c}", prov, err, _flatten(groups, combos[c])
            )
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
        return [
            r for n in BASELINES for r in _error_records(ctx, "baselines", n, REAL, err, groups[n])
        ]
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
        return _error_records(ctx, "ref_supervised", RF_NAME, REAL, e, [rf], **extra)
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
        dets = _flatten(groups, groups)
        ae = groups[AE]
        if err is not None:
            out += _error_records(ctx, "retrain_drift", variant, REAL, err, dets, **extra)
            out += _error_records(ctx, "retrain_drift", f"{variant}:ai", REAL, err, ae, **extra)
            continue
        cal = ctx.clean(sp.cal_normal, dets)
        acc = ctx.real_parts(sp.accept_events, sp.accept_normal, dets)
        out += _variant(
            ctx, "retrain_drift", variant, REAL, dets, mv, cal, [acc], None, {**info, **extra}
        )
        # the AE alone through its own Chief, reusing the same scores (no extra fit)
        out += _variant(
            ctx,
            "retrain_drift",
            f"{variant}:ai",
            REAL,
            ae,
            _versions_for(mv, ae),
            cal,
            [acc],
            None,
            {**info, **extra},
        )
    return out


def m1_hard_k(cfg: Settings, n_normal: int) -> int:
    """Accept hard negatives: `hard_negative_k`, at most a third of the accept-normal days."""
    return min(cfg.eval.hard_negative_k, n_normal // 3)


def _most_volatile(ctx: _Ctx, days: list[SymDay], k: int) -> list[SymDay]:
    """The `k` days of `days` with the highest `realized_vol_1m`, ties by `(symbol, day)` order
    (the rule `make_split` uses for the test hard negatives); sorted by `(symbol, day)`."""
    days = sorted(days)
    vols = [
        realized_vol_1m(
            ctx.store.read_trades(s.symbol, day_start_us(s.day), day_start_us(s.day) + DAY_US)
        )
        for s in days
    ]
    order = sorted(range(len(days)), key=lambda i: (-vols[i], days[i]))
    pick = set(order[:k])
    return [s for i, s in enumerate(days) if i in pick]


def _subpart(part: _Part, days: list[SymDay]) -> _Part:
    """The rows of `part` on `days` (no re-scoring); `days` limited to those `part` loaded."""
    loaded = set(part.days)
    keep = [sd for sd in days if sd in loaded]
    return _Part(_normal_rows(part.frame, keep), _labels_on_days(part.labels, keep), keep)


def _exp_m1_fusion(ctx: _Ctx) -> list[dict]:
    sp, cfg = ctx.split, ctx.cfg
    variants = m1_variants(cfg)
    regime = m1_regime_feature(cfg)
    groups = ctx.build(cfg.detectors)
    dets = _flatten(groups, groups)
    mv, info, err = _fit_groups(ctx, groups, sp.train_normal)
    if err is not None:
        return [
            r
            for v, ai in variants.items()
            for split in M1_SPLITS
            for r in _error_records(
                ctx,
                M1,
                f"{split}:{v}",
                SEMI if split == "semisyn" else REAL,
                err,
                dets,
                ai_fusion=ai.as_dict(),
            )
        ]
    keep = (regime,)
    # every set is scored once; the variants only re-fuse these scores
    cal = ctx.clean(sp.cal_normal, dets, keep=keep)
    acc = ctx.real_parts(sp.accept_events, sp.accept_normal, dets, keep=keep)
    hard_days = _most_volatile(ctx, sp.accept_normal, m1_hard_k(cfg, len(sp.accept_normal)))
    acc_hard = _subpart(acc, hard_days)
    test = ctx.real_parts(sp.test_events, sp.test_normal, dets, keep=keep)
    test_hard = ctx.clean(sp.test_hard_negative, dets, keep=keep)
    semi = ctx.semisyn_parts(dets, keep=keep)
    hard_keys = [_day_key(sd.symbol, sd.day) for sd in acc_hard.days]
    out = []
    for v, ai in variants.items():
        base = {**info, "ai_fusion": ai.as_dict(), "regime_feature": regime}
        sets = (
            ("accept", REAL, [acc], acc_hard, {"accept_hard_days": hard_keys}),
            ("test", REAL, [test], test_hard, {}),
            ("semisyn", SEMI, semi, None, {"n_skipped_tick": sum(p.skipped_tick for p in semi)}),
        )
        for split, prov, parts, hard, more in sets:
            out += _variant(
                ctx, M1, f"{split}:{v}", prov, dets, mv, cal, parts, hard, {**base, **more}, ai=ai
            )
    return out


def m2_ai_models(cfg: Settings, seed: int) -> dict[str, Detector]:
    """The four unfitted M2 AI models (`M2_AI_NAMES`): today's AE, its grid points (only
    `hidden` / `bottleneck` change, under their own detector names) and `ai.iforest`."""
    from idsai.ai.autoencoder import AutoencoderDetector  # lazy: keeps torch off import path

    out: dict[str, Detector] = {}
    for key, grid in M2_AE_GRID.items():
        if not grid:
            (out[key],) = build_detectors([AE], cfg, seed)
            continue
        ae_cfg = cfg.ae.model_copy(update=grid)
        det = AutoencoderDetector(model_features(cfg.features), ae_cfg, seed)
        det.name = M2_AI_NAMES[key]
        out[key] = det
    (out["iforest"],) = build_detectors([IFOREST_AI], cfg, seed)
    for key, det in out.items():
        if det.name != M2_AI_NAMES[key] or det.layer != "ai":
            raise ValueError(f"M2 model {key}: got {det.name} on layer {det.layer}")
    return out


def m2_variants() -> dict[str, tuple[str | None, AiFusion]]:
    """`variant -> (model or None, AiFusion)`: `no_ai` and every `<model>:<fusion>`."""
    out: dict[str, tuple[str | None, AiFusion]] = {M2_REF: (None, AiFusion("exclude"))}
    for m in M2_MODELS:
        for f, ai in M2_FUSIONS.items():
            out[f"{m}:{f}"] = (m, ai)
    return out


def _m2_grid(cfg: Settings) -> dict[str, dict[str, int]]:
    return {
        k: {"hidden": cfg.ae.hidden, "bottleneck": cfg.ae.bottleneck, **g}
        for k, g in M2_AE_GRID.items()
    }


def _exp_m2_ai(ctx: _Ctx) -> list[dict]:
    sp, cfg = ctx.split, ctx.cfg
    # the rules and stats of the configured set (today's AE is built by m2_ai_models instead)
    groups = ctx.build(n for n in cfg.detectors if n != AE)
    base = [d for d in _flatten(groups, groups) if d.layer != "ai"]
    models = m2_ai_models(cfg, ctx.seed)
    variants = m2_variants()

    def dets_of(model: str | None) -> list[Detector]:
        keys = M2_MODELS[model] if model is not None else ("ae",)  # no_ai: today's AE, excluded
        return [*base, *(models[k] for k in keys)]

    def fields(model: str | None, ai: AiFusion | None) -> dict[str, Any]:
        return {
            "ai_model": model,
            "ai_fusion": ai.as_dict() if ai is not None else None,
            "ae_grid": _m2_grid(cfg),
        }

    every = [*base, *models.values()]
    try:
        mv, info = ctx.fit(every, sp.train_normal)
    except _VariantError as err:
        out = []
        for v, (m, ai) in variants.items():
            for split in M2_SPLITS:
                prov = SEMI if split == "semisyn" else REAL
                out += _error_records(
                    ctx, M2, f"{split}:{v}", prov, err, dets_of(m), **fields(m, ai)
                )
        for m in M2_ALONE:
            out += _error_records(
                ctx, M2, f"test:{m}:alone", REAL, err, [models[m]], **fields(m, None)
            )
        return out
    # every set is scored once with every detector; the variants only re-fuse these scores
    cal = ctx.clean(sp.cal_normal, every)
    acc = ctx.real_parts(sp.accept_events, sp.accept_normal, every)
    hard_days = _most_volatile(ctx, sp.accept_normal, m1_hard_k(cfg, len(sp.accept_normal)))
    acc_hard = _subpart(acc, hard_days)
    test = ctx.real_parts(sp.test_events, sp.test_normal, every)
    test_hard = ctx.clean(sp.test_hard_negative, every)
    semi = ctx.semisyn_parts(every)
    hard_keys = [_day_key(sd.symbol, sd.day) for sd in acc_hard.days]
    sets = (
        ("accept", REAL, [acc], acc_hard, {"accept_hard_days": hard_keys}),
        ("test", REAL, [test], test_hard, {}),
        ("semisyn", SEMI, semi, None, {"n_skipped_tick": sum(p.skipped_tick for p in semi)}),
    )
    out = []
    for v, (m, ai) in variants.items():
        dets = dets_of(m)
        vmv = _versions_for(mv, dets)
        for split, prov, parts, hard, more in sets:
            extra = {**info, **fields(m, ai), **more}
            out += _variant(
                ctx, M2, f"{split}:{v}", prov, dets, vmv, cal, parts, hard, extra, ai=ai
            )
    for m in M2_ALONE:
        dets = [models[m]]
        extra = {**info, **fields(m, None)}
        vmv = _versions_for(mv, dets)
        out += _variant(ctx, M2, f"test:{m}:alone", REAL, dets, vmv, cal, [test], test_hard, extra)
    return out


def m4_sell_cfg(cfg: Settings) -> Settings:
    """`cfg` with `features.sell_side` on (the M4 eval computes every feature this way)."""
    return cfg.model_copy(update={"features": cfg.features.model_copy(update={"sell_side": True})})


def _exp_m4_sell(ctx: _Ctx) -> list[dict]:
    sp = ctx.split
    # every load computes the sell features too; the existing columns are bit-identical to the
    # flag-off ones, so the fitted models and existing scores are those of real_t001
    ctx.cfg = m4_sell_cfg(ctx.cfg)
    ctx.meta["signatures_sell_sha"] = file_sha(SELL_SIGNATURES_PATH)
    groups = ctx.build(ctx.cfg.detectors)
    dets = _flatten(groups, groups)
    sell_names = {d.name for d in sell_rule_detectors()}
    sets = {"base": [d for d in dets if d.name not in sell_names], "sell": dets}
    mv, info, err = _fit_groups(ctx, groups, sp.train_normal)
    if err is not None:
        return [
            r
            for v in M4_VARIANTS
            for split, prov in (("test", REAL), (BEAR_RAID, SEMI))
            for r in _error_records(
                ctx, M4, f"{split}:{v}", prov, err, sets[v], sell_side=v == "sell"
            )
        ]
    cal, real, hard = _real_sets(ctx, dets)
    raid = ctx.bear_raid_part(dets)
    phase = SellPhase(derive_sell_labels(ctx.store, real.labels))  # once, shared by both
    out = []
    for v in M4_VARIANTS:
        vd, vmv = sets[v], _versions_for(mv, sets[v])
        base = {**info, "sell_side": v == "sell"}
        out += _variant(
            ctx, M4, f"test:{v}", REAL, vd, vmv, cal, [real], hard, dict(base), phase=phase
        )
        extra = {**base, "n_skipped_tick": raid.skipped_tick}
        out += _variant(ctx, M4, f"{BEAR_RAID}:{v}", SEMI, vd, vmv, cal, [raid], None, extra)
    return out


def m3_detectors(cfg: Settings, seed: int) -> tuple[list[Detector], Detector, list[Detector]]:
    """`(live, wash2, naming)`: the unfitted live stack (`M3_LIVE`), rule 100501 rev 2 under its
    own name (`WASH2_NAME`) and the naming rules in priority order (always, whatever
    `chief.attr_rules` says: the variants choose)."""
    live = build_detectors(list(M3_LIVE), cfg, seed)
    if not any(d.name == f"rule.{WASH_SID}" for d in live):
        raise ValueError(f"the live rules lack rule.{WASH_SID}")
    (sig,) = load_signatures(WASH2_SIGNATURES_PATH)
    if (sig.sid, sig.rev) != (WASH_SID, 2):
        raise ValueError(f"{WASH2_SIGNATURES_PATH.name} must hold {WASH_SID} rev 2")
    wash2 = SignatureDetector(sig)
    wash2.name = WASH2_NAME
    naming: list[Detector] = [SignatureDetector(s) for s in load_signatures(NAMING_SIGNATURES_PATH)]
    return live, wash2, naming


def m3_variant_detectors(live: list[Detector], wash2: Detector, rev: int) -> list[Detector]:
    """The detectors of a variant: the live stack, with rev 2 in rev 1's place when `rev == 2`."""
    if rev == 1:
        return list(live)
    return [wash2 if d.name == f"rule.{WASH_SID}" else d for d in live]


def _exp_m3_attr(ctx: _Ctx) -> list[dict]:
    sp, cfg = ctx.split, ctx.cfg
    ctx.meta["signatures_attr_sha"] = file_sha(NAMING_SIGNATURES_PATH)
    ctx.meta["signatures_wash2_sha"] = file_sha(WASH2_SIGNATURES_PATH)
    live, wash2, naming = m3_detectors(cfg, ctx.seed)

    def fields(v: str) -> dict[str, Any]:
        on, rev = M3_VARIANTS[v]
        return {"attr_rules": on, "wash_rev": rev, "ai_fusion": M3_FUSION.as_dict()}

    try:
        mv, info = ctx.fit(live, sp.train_normal)
    except _VariantError as err:
        return [
            r
            for v, (_, rev) in M3_VARIANTS.items()
            for split in M3_SPLITS
            for r in _error_records(
                ctx,
                M3,
                f"{split}:{v}",
                SEMI if split == "semisyn" else REAL,
                err,
                m3_variant_detectors(live, wash2, rev),
                **fields(v),
            )
        ]
    # every set is scored once with every detector; the variants only re-fuse these scores
    every = [*live, wash2, *naming]
    cal = ctx.clean(sp.cal_normal, every)
    acc = ctx.real_parts(sp.accept_events, sp.accept_normal, every)
    hard_days = _most_volatile(ctx, sp.accept_normal, m1_hard_k(cfg, len(sp.accept_normal)))
    acc_hard = _subpart(acc, hard_days)
    test = ctx.real_parts(sp.test_events, sp.test_normal, every)
    test_hard = ctx.clean(sp.test_hard_negative, every)
    semi = ctx.semisyn_parts(every)
    hard_keys = [_day_key(sd.symbol, sd.day) for sd in acc_hard.days]
    sets = (
        ("accept", REAL, [acc], acc_hard, {"accept_hard_days": hard_keys}),
        ("test", REAL, [test], test_hard, {}),
        ("semisyn", SEMI, semi, None, {"n_skipped_tick": sum(p.skipped_tick for p in semi)}),
    )
    out = []
    for v, (on, rev) in M3_VARIANTS.items():
        dets = m3_variant_detectors(live, wash2, rev)
        for split, prov, parts, hard, more in sets:
            out += _variant(
                ctx,
                M3,
                f"{split}:{v}",
                prov,
                dets,
                dict(mv),
                cal,
                parts,
                hard,
                {**info, **fields(v), **more},
                ai=M3_FUSION,
                naming=naming if on else None,
                fa_names=True,
            )
    return out


_RUNNERS: dict[str, Callable[[_Ctx], list[dict]]] = {
    "real_t001": _exp_real_t001,
    "semisyn": _exp_semisyn,
    "ablation": _exp_ablation,
    "baselines": _exp_baselines,
    "ref_supervised": _exp_ref_supervised,
    "retrain_drift": _exp_retrain_drift,
    M1: _exp_m1_fusion,
    M2: _exp_m2_ai,
    M3: _exp_m3_attr,
    M4: _exp_m4_sell,
}
EXPERIMENTS = tuple(_RUNNERS)
INJECTING = ("semisyn", "ablation", M1, M2, M3, M4)  # experiments that run Red-Team injection


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


@dataclass
class Fitted:
    """Detectors fitted on normal days and a Chief calibrated on other normal days (see
    `fit_and_calibrate`)."""

    detectors: list[Detector]
    chief: Chief
    model_versions: dict[str, str]
    info: dict[str, Any]


def fit_and_calibrate(
    cfg: Settings,
    store: Store,
    seed: int,
    train_days: list[SymDay],
    cal_days: list[SymDay],
    budget_per_symbol_day: float,
    names: list[str] | None = None,
    version_prefix: str = "model",
) -> Fitted:
    """The experiments' protocol for one deployable model set: build `names` (default
    `cfg.detectors`), fit them on the masked, grid-thinned rows of `train_days` (`_Ctx.fit`),
    calibrate the Chief (with the configured AI fusion, `AiFusion.from_cfg`) on `cal_days` and fit
    `tau` for the budget.

    `train_days` and `cal_days` must not share a symbol-day. Raises ValueError when nothing can be
    fitted or calibrated (no rows, a failed fit) and `CalibrationError` when the calibration set
    cannot honour the budget. Model versions are content addresses (`model_version`).
    """
    overlap = {(sd.symbol, sd.day) for sd in train_days} & {(sd.symbol, sd.day) for sd in cal_days}
    if overlap:
        raise ValueError(f"train and calibration days overlap: {sorted(overlap)}")
    ctx = _Ctx(cfg, store, seed, with_split=False)
    groups = ctx.build(names if names is not None else cfg.detectors)
    dets = _flatten(groups, groups)
    try:
        _, info = ctx.fit(dets, train_days)
    except _VariantError as e:
        raise ValueError(str(e)) from e
    mv = {d.name: model_version(d, version_prefix) for d in dets if _needs_fit(d)}
    cal = ctx.clean(cal_days, dets)
    if cal.frame.height == 0:
        raise ValueError(f"no calibration rows from {len(cal_days)} day(s): {dict(cal.skipped)}")
    cols = ["symbol", "t_end_us", *[f"score__{d.name}" for d in dets]]
    # a deployable Chief runs the configured AI fusion (`chief.ai_mode`); the experiments pass
    # their own `AiFusion` per variant instead
    chief = Chief(dets, cfg.chief, ctx.config_hash, mv, ai=AiFusion.from_cfg(cfg.chief))
    chief.calibrate(cal.frame.select(cols))
    chief.fit_threshold(chief.fuse(cal.frame.select(cols)), budget_per_symbol_day)
    info = {k: v for k, v in info.items() if not k.startswith("_")}
    info |= {
        "train_skipped": dict(ctx.train_frame(train_days)[2]) if mv else {},
        "cal_rows": cal.frame.height,
        "cal_days": len(cal.days),
        "cal_skipped": dict(cal.skipped),
        "tau": chief.tau,
        "tau_saturated": chief.tau_saturated,
        "n_cal": chief.n_cal,
        "budget_per_symbol_day": float(budget_per_symbol_day),
    }
    return Fitted(dets, chief, mv, info)
