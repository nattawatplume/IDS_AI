"""Time-ordered evaluation splits over (symbol, UTC day) units.

Periods come from `cfg.splits` and are strictly ordered `train < cal < accept < test`.

- Event day: the UTC day of a label's `t_start_us`, with `include_next_minutes = 10` when the
  label starts within 10 minutes of the next midnight (so the `+after_s` matching window is
  covered). Used only when the store has the day and the day before (24 h baseline).
- Normal day: for every labeled symbol, a day of the period with `D` and `D-1` in the store that
  is at least `min_gap_days` from every label of that symbol (any period);
  `normal_days_per_symbol` are sampled per symbol per period from the sorted candidates with one
  `np.random.default_rng(seed)` consumed in a fixed order (periods, then sorted symbols).
- Hard negatives: the `hard_negative_k` test-normal days with the highest realized 1-minute
  volatility move out of `test_normal` into `test_hard_negative`.

Every list is sorted by `(symbol, day)`. Labels that give no event day are counted per period and
reason in `Split.labels_dropped` (out of every period, in the cal period, day or D-1 missing).
"""

import hashlib
import json
from dataclasses import dataclass, field, fields
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl

from idsai.config import DateRange, Settings, SplitsCfg
from idsai.contracts.frames import LABELS, conform
from idsai.data.store import DAY_US, Store

NEXT_MINUTES = 10
MIN_VOL_MINUTES = 30
PERIODS = ("train", "cal", "accept", "test")
LIST_NAMES = (
    "train_normal",
    "train_events",
    "cal_normal",
    "accept_events",
    "accept_normal",
    "test_events",
    "test_normal",
    "test_hard_negative",
)
DROP_REASONS = ("out_of_period", "cal_period", "missing_day", "no_baseline")
NORMAL_LISTS = ("train_normal", "cal_normal", "accept_normal", "test_normal", "test_hard_negative")
_EPOCH = date(1970, 1, 1)
_MINUTE_US = 60_000_000


@dataclass(frozen=True, order=True)
class SymDay:
    symbol: str
    day: date
    include_next_minutes: int = 0


@dataclass(frozen=True)
class Split:
    train_normal: list[SymDay]
    train_events: list[SymDay]
    cal_normal: list[SymDay]
    accept_events: list[SymDay]
    accept_normal: list[SymDay]
    test_events: list[SymDay]
    test_normal: list[SymDay]
    test_hard_negative: list[SymDay]
    # REAL labels considered, and those not usable as events:
    # {period or "none": {reason in DROP_REASONS: n}}
    n_labels: int = 0
    labels_dropped: dict[str, dict[str, int]] = field(default_factory=dict)

    def lists(self) -> dict[str, list[SymDay]]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name in LIST_NAMES}

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.lists().items()}

    def sha12(self) -> str:
        """12-hex sha256 of every list's sorted `(symbol, day, include_next_minutes)` entries."""
        blob = json.dumps(
            {
                k: [[s.symbol, s.day.isoformat(), s.include_next_minutes] for s in sorted(v)]
                for k, v in self.lists().items()
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def day_of_us(ts_us: int) -> date:
    return _EPOCH + timedelta(days=int(ts_us) // DAY_US)


def day_start_us(day: date) -> int:
    return (day - _EPOCH).days * DAY_US


def _in(day: date, rng: DateRange) -> bool:
    return rng[0] <= day <= rng[1]


def period_of(day: date, splits: SplitsCfg) -> str | None:
    for name in PERIODS:
        if _in(day, getattr(splits, name)):
            return name
    return None


def drift_ranges(splits: SplitsCfg) -> tuple[DateRange, DateRange]:
    """`(v1, v2)` training ranges of `retrain_drift`: configured, or the first / last two thirds
    of the train period (whole days)."""
    lo, hi = splits.train
    n = (hi - lo).days + 1
    k = max(1, round(2 * n / 3))
    v1 = splits.drift_v1 or (lo, lo + timedelta(days=k - 1))
    v2 = splits.drift_v2 or (hi - timedelta(days=k - 1), hi)
    for name, (a, b) in (("drift_v1", v1), ("drift_v2", v2)):
        if not (lo <= a <= b <= hi):
            raise ValueError(f"{name} ({a}, {b}) must lie inside train ({lo}, {hi})")
    return v1, v2


def split_def(split: Split, cfg: Settings) -> dict[str, Any]:
    """Split description stored in every run record."""
    return {
        "splits": cfg.splits.model_dump(mode="json"),
        "split_seed": cfg.eval.split_seed,
        "counts": split.counts(),
        "sha12": split.sha12(),
        "n_labels": split.n_labels,
        "labels_dropped": split.labels_dropped,
    }


def realized_vol_1m(trades: pl.DataFrame) -> float:
    """Std of 1-minute log returns of the last trade price per minute (minutes with trades only).

    Fewer than `MIN_VOL_MINUTES` minutes with trades gives 0.0.
    """
    if trades.height == 0:
        return 0.0
    last = (
        trades.sort(["ts_us", "trade_id"])
        .group_by((pl.col("ts_us") // _MINUTE_US).alias("m"), maintain_order=True)
        .agg(pl.col("price").last())
        .sort("m")
    )
    if last.height < MIN_VOL_MINUTES:
        return 0.0
    p = last["price"].to_numpy().astype(np.float64)
    r = np.diff(np.log(p))
    return float(np.std(r)) if r.size else 0.0


def _check_order(splits: SplitsCfg) -> None:
    rngs = [getattr(splits, n) for n in PERIODS]
    for (a_lo, a_hi), (b_lo, _) in zip(rngs, rngs[1:], strict=False):
        assert a_lo <= a_hi < b_lo, "split periods must be ordered train < cal < accept < test"


def make_split(labels: pl.DataFrame, store: Store, cfg: Settings, seed: int | None = None) -> Split:
    """Build the evaluation split (see the module docstring); `seed` defaults to split_seed."""
    sp = cfg.splits
    _check_order(sp)
    seed = cfg.eval.split_seed if seed is None else seed
    labels = conform(labels, LABELS)
    gap = sp.min_gap_days

    label_days: dict[str, list[date]] = {}
    events: dict[str, dict[tuple[str, date], int]] = {p: {} for p in PERIODS}
    days_cache: dict[str, set[date]] = {}
    dropped: dict[str, dict[str, int]] = {}

    def store_days(symbol: str) -> set[date]:
        if symbol not in days_cache:
            days_cache[symbol] = set(store.days(symbol))
        return days_cache[symbol]

    for sym, t0 in labels.select("symbol", "t_start_us").iter_rows():
        d = day_of_us(t0)
        label_days.setdefault(sym, []).append(d)
        period = period_of(d, sp)
        have = store_days(sym)
        reason = None
        if period is None:
            reason = "out_of_period"
        elif period == "cal":  # the cal period has no event set
            reason = "cal_period"
        elif d not in have:
            reason = "missing_day"
        elif d - timedelta(days=1) not in have:
            reason = "no_baseline"
        if reason is not None:
            by = dropped.setdefault(period or "none", {})
            by[reason] = by.get(reason, 0) + 1
            continue
        near_midnight = day_start_us(d) + DAY_US - t0 <= NEXT_MINUTES * _MINUTE_US
        n_next = NEXT_MINUTES if near_midnight else 0
        key = (sym, d)
        events[period][key] = max(events[period].get(key, 0), n_next)

    rng = np.random.default_rng(seed)
    normal: dict[str, list[SymDay]] = {p: [] for p in PERIODS}
    for period in PERIODS:
        lo, hi = getattr(sp, period)
        for sym in sorted(label_days):
            have = store_days(sym)
            lab = label_days[sym]
            cand = sorted(
                d
                for d in have
                if lo <= d <= hi
                and d - timedelta(days=1) in have
                and all(abs((d - ld).days) >= gap for ld in lab)
            )
            k = min(sp.normal_days_per_symbol, len(cand))
            if k:
                picks = rng.choice(len(cand), size=k, replace=False)
                normal[period].extend(SymDay(sym, cand[i]) for i in sorted(picks))

    # hard negatives: most volatile test-normal days (ties by (symbol, day) order)
    test_normal = sorted(normal["test"])
    vols = [
        realized_vol_1m(
            store.read_trades(s.symbol, day_start_us(s.day), day_start_us(s.day) + DAY_US)
        )
        for s in test_normal
    ]
    order = sorted(range(len(test_normal)), key=lambda i: (-vols[i], test_normal[i]))
    hard_idx = set(order[: cfg.eval.hard_negative_k])
    hard = [s for i, s in enumerate(test_normal) if i in hard_idx]
    test_normal = [s for i, s in enumerate(test_normal) if i not in hard_idx]

    def ev(period: str) -> list[SymDay]:
        return sorted(SymDay(s, d, n) for (s, d), n in events[period].items())

    split = Split(
        train_normal=sorted(normal["train"]),
        train_events=ev("train"),
        cal_normal=sorted(normal["cal"]),
        accept_events=ev("accept"),
        accept_normal=sorted(normal["accept"]),
        test_events=ev("test"),
        test_normal=test_normal,
        test_hard_negative=hard,
        n_labels=labels.height,
        labels_dropped={k: dict(sorted(v.items())) for k, v in sorted(dropped.items())},
    )
    for name in NORMAL_LISTS:
        for s in getattr(split, name):
            assert all(abs((s.day - ld).days) >= gap for ld in label_days.get(s.symbol, [])), (
                f"{name} day {s} is within {gap} days of a label"
            )
    return split
