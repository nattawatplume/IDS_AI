"""Batch pipeline for one symbol-day: store trades -> (optional injection) -> windows -> scores.

Day boundaries are UTC midnight. A symbol-day `D` reads trades from `[D-1, D+1 + n minutes)` so
the 24 h baseline of every kept window is available, builds 1 s bars over that whole range and
keeps the window rows with `t_end_us` in `[D, D+1 + n minutes)`.

Windows returned here are always rounded through float32 (and returned as float64), whether they
were just computed or read from the cache, so a score never depends on the cache state. Clean
windows are cached as float32 Parquet; scenario (injected) windows are never cached.
"""

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from idsai.config import FeatureCfg, Settings, quote_unit_for
from idsai.contracts.detector import Detector
from idsai.contracts.frames import LABELS, conform, empty
from idsai.data.store import DAY_US, Store
from idsai.detection.features.bars import S_US, trades_to_bars
from idsai.detection.features.registry import feature_columns
from idsai.detection.features.windows import compute_windows, windows_schema
from idsai.detection.simulator.inject import inject
from idsai.detection.simulator.scenarios import Scenario, base_quote_rate

# Bump when feature math changes: it is part of the cache directory name.
FEATURE_SET_V = "fs1"

_EPOCH = date(1970, 1, 1)
Skipped = Literal["missing_day", "no_baseline", "no_trades"]


@dataclass(frozen=True)
class DayResult:
    symbol: str
    day: date
    windows: pl.DataFrame
    labels: pl.DataFrame
    skipped: Skipped | None = None


def _day_start_us(day: date) -> int:
    return (day - _EPOCH).days * DAY_US


def _features_hash(cfg: FeatureCfg) -> str:
    """12-hex sha256 of the canonical JSON dump of the features config."""
    blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _cache_path(
    cfg: Settings, symbol: str, day: date, n_next: int, shas: tuple[str | None, ...]
) -> Path:
    root = cfg.paths.data_dir / "features" / f"{FEATURE_SET_V}-{_features_hash(cfg.features)}"
    key = "".join(s or "none" for s in shas)
    tag = f"{day.isoformat()}-n{n_next}-{key}" if n_next else f"{day.isoformat()}-{key}"
    return root / symbol / f"{tag}.parquet"


def _round_f32(windows: pl.DataFrame, cfg: FeatureCfg) -> pl.DataFrame:
    """Feature columns rounded through float32 and returned as float64."""
    cols = feature_columns(cfg)
    return windows.with_columns(pl.col(c).cast(pl.Float32).cast(pl.Float64) for c in cols)


def _read_cache(path: Path, cfg: FeatureCfg) -> pl.DataFrame | None:
    """The cached windows, or None when the file is missing, unreadable or has the wrong shape."""
    if not path.exists():
        return None
    try:
        df = pl.read_parquet(path)
        if df.columns != windows_schema(cfg).names():
            return None
        return df.with_columns(
            pl.col("symbol").cast(pl.String),
            pl.col("t_end_us").cast(pl.Int64),
            *(pl.col(c).cast(pl.Float64) for c in feature_columns(cfg)),
        )
    except Exception:  # noqa: BLE001 - any unreadable cache file is simply recomputed
        return None


def _write_cache(path: Path, windows: pl.DataFrame, cfg: FeatureCfg) -> None:
    """Atomic write (temp file, then rename) of float32 Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        windows.with_columns(
            pl.col(c).cast(pl.Float32) for c in feature_columns(cfg)
        ).write_parquet(tmp)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _skip(symbol: str, day: date, cfg: FeatureCfg, reason: Skipped) -> DayResult:
    return DayResult(
        symbol, day, pl.DataFrame(schema=windows_schema(cfg)), empty(LABELS), skipped=reason
    )


def load_day_windows(
    store: Store,
    symbol: str,
    day: date,
    cfg: Settings,
    scenario: Scenario | None = None,
    include_next_minutes: int = 0,
) -> DayResult:
    """Window features for one symbol-day, optionally with a red-team scenario injected.

    Skips (empty windows with the right schema, `skipped` set), checked in this order:
    `missing_day` (no partition for `day`), `no_baseline` (none for `day - 1`), and with a
    scenario `no_trades` (the previous day has no volume to size the injection from).

    Returned windows have `t_end_us` in `[day, day + 1 d + include_next_minutes)` and are rounded
    through float32. `labels` are the store's labels for `symbol` overlapping that range (clean
    run) or just the injected label (scenario run).
    """
    fcfg = cfg.features
    prev = day - timedelta(days=1)
    if not store.has_day(symbol, day):
        return _skip(symbol, day, fcfg, "missing_day")
    if not store.has_day(symbol, prev):
        return _skip(symbol, day, fcfg, "no_baseline")

    d0 = _day_start_us(day)
    lo, hi = d0, d0 + DAY_US + include_next_minutes * 60 * S_US
    start_us = d0 - DAY_US

    path = None
    if scenario is None:
        nxt = day + timedelta(days=1)
        shas = (store.partition_sha(symbol, prev), store.partition_sha(symbol, day))
        if include_next_minutes:
            shas += (store.partition_sha(symbol, nxt),)
        path = _cache_path(cfg, symbol, day, include_next_minutes, shas)
        cached = _read_cache(path, fcfg)
        if cached is not None:
            return DayResult(symbol, day, cached, _clean_labels(store, symbol, lo, hi))

    trades = store.read_trades(symbol, start_us, hi)
    if scenario is None:
        labels = _clean_labels(store, symbol, lo, hi)
    else:
        prev_trades = trades.filter(pl.col("ts_us") < d0)
        try:
            base = base_quote_rate(prev_trades)
        except ValueError:
            return _skip(symbol, day, fcfg, "no_trades")
        trades, labels = inject(trades, scenario, base, mode="replay")

    bars = trades_to_bars(trades, symbol, start_us, hi)
    windows = compute_windows(bars, fcfg, quote_unit_for(symbol, fcfg))
    windows = windows.filter((pl.col("t_end_us") >= lo) & (pl.col("t_end_us") < hi))
    windows = _round_f32(windows, fcfg)
    if path is not None:
        _write_cache(path, windows, fcfg)
    return DayResult(symbol, day, windows, labels)


def _clean_labels(store: Store, symbol: str, lo: int, hi: int) -> pl.DataFrame:
    """The store's labels for `symbol` overlapping `[lo, hi)`."""
    labels = conform(store.read_labels(symbol), LABELS)
    return labels.filter((pl.col("t_end_us") >= lo) & (pl.col("t_start_us") < hi))


def score_windows(windows: pl.DataFrame, detectors: list[Detector]) -> pl.DataFrame:
    """`windows` plus one float64 `score__<detector name>` column per detector (not fitted here)."""
    names = [d.name for d in detectors]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"duplicate detector names: {dups}")
    cols = []
    for det in detectors:
        if windows.height == 0:
            scores = np.empty(0, dtype=np.float64)
        else:
            scores = np.asarray(det.score(windows), dtype=np.float64)
            if scores.shape != (windows.height,):
                raise ValueError(
                    f"detector {det.name!r} returned {scores.shape} scores for "
                    f"{windows.height} rows"
                )
        cols.append(pl.Series(f"score__{det.name}", scores, dtype=pl.Float64))
    return windows.with_columns(cols)
