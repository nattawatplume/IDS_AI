"""Measure one real pump label from trades (peak, impact, duration, volume, rush size, dump tau).

Moved here from `scripts/measure_pumps.py` (which re-exports it) so library code, the M4 sell-phase
labels (`idsai.eval.sell_labels`), can use it without importing from `scripts/`. The method is
documented in that script's docstring; nothing here changed. `measure_trades` is the shared core:
the script feeds it trades read straight from Parquet, `measure_label_store` reads a `Store`.
"""

import math
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl

from idsai.detection.simulator.scenarios import base_quote_rate

S_US = 1_000_000
DAY_US = 86_400 * S_US
_EPOCH = date(1970, 1, 1)

SEARCH_BEFORE_S = 300  # the peak is searched from 5 min before the label time
SEARCH_AFTER_S = 1800  # ... to 30 min after it
REF_S = 3600  # reference window: the hour before the search window
POST_S = 3600  # dump search after the peak
DUMP_BUCKET_S = 10
RAMP_FRAC = 0.1  # price level (fraction of the impact) that marks the ramp start
MIN_IMPACT = 0.03  # below this the label shows no pump in the price
PEAK_MAX_AFTER_S = 600  # a peak later than this after the label is not the announced pump
MIN_DURATION_S = 1.0


def _day_of_us(ts_us: int) -> date:
    return _EPOCH + timedelta(days=int(ts_us) // DAY_US)


def _day_start_us(day: date) -> int:
    return (day - _EPOCH).days * DAY_US


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Smallest value whose cumulative weight (values ascending) reaches half the total."""
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1], side="left")])


def rush_cluster_size(buys: pl.DataFrame) -> float | None:
    """Quote-weighted median number of taker-buy fills per millisecond (None without buys)."""
    if buys.height == 0:
        return None
    groups = buys.group_by(pl.col("ts_us") // 1000).agg(
        pl.len().alias("n"), pl.col("quote_qty").sum().alias("q")
    )
    n = groups["n"].to_numpy().astype(np.float64)
    q = groups["q"].to_numpy().astype(np.float64)
    if not q.sum() > 0:
        return float(np.median(n))
    return weighted_median(n, q)


def measure_window(
    trades: pl.DataFrame, t0: int, base: float, p_ref: float, buy_bg: float
) -> dict[str, Any]:
    """Measure one pump in `trades` (which must cover `[t0 - SEARCH_BEFORE_S, peak + POST_S]`).

    `buy_bg` is the real taker-buy quote per second the pump sits on. Returns the measured
    fields with `status == "measured"`, or a status naming the exclusion reason.
    """
    ts = trades["ts_us"].to_numpy()
    price = trades["price"].to_numpy().astype(np.float64)
    ws = t0 - SEARCH_BEFORE_S * S_US
    we = t0 + SEARCH_AFTER_S * S_US
    lo, hi = np.searchsorted(ts, ws, "left"), np.searchsorted(ts, we, "left")
    if hi <= lo:
        return {"status": "no_window_trades"}
    i_peak = lo + int(np.argmax(price[lo:hi]))  # earliest trade at the maximum price
    t_peak = int(ts[i_peak])
    impact = float(price[i_peak] / p_ref - 1.0)
    out: dict[str, Any] = {"t_peak_us": t_peak, "impact": impact}
    if impact < MIN_IMPACT:
        return {"status": "no_rise", **out}
    if t_peak - t0 > PEAK_MAX_AFTER_S * S_US:
        return {"status": "late_peak", **out}

    level = p_ref * (1.0 + RAMP_FRAC * impact)
    below = np.nonzero(price[lo:i_peak] <= level)[0]
    censored = below.size == 0
    t10 = ws if censored else int(ts[lo + below[-1]])
    duration_s = max((t_peak - t10) / S_US / (1.0 - RAMP_FRAC), MIN_DURATION_S)
    t_s = t_peak - int(round(duration_s * S_US))
    out.update(t_start_est_us=t_s, duration_s=duration_s, duration_censored=bool(censored))

    buys = trades.filter(
        (pl.col("ts_us") >= t_s) & (pl.col("ts_us") <= t_peak) & (pl.col("taker_side") == "buy")
    )
    buy_rate = float(buys["quote_qty"].sum()) / duration_s
    vol_mult = (buy_rate - buy_bg) / base
    out["vol_mult"] = vol_mult
    if not vol_mult > 0:
        return {"status": "no_volume", **out}
    rush_size = rush_cluster_size(buys)

    post = trades.filter((pl.col("ts_us") > t_peak) & (pl.col("ts_us") <= t_peak + POST_S * S_US))
    dump_tau_s = None
    if post.height:
        bucket_us = DUMP_BUCKET_S * S_US
        med = (
            post.group_by(((pl.col("ts_us") - t_peak - 1) // bucket_us).alias("b"))
            .agg(pl.col("price").median())
            .sort("b")
        )
        hit = med.filter(pl.col("price") <= p_ref * (1.0 + impact / math.e))
        if hit.height:
            dump_tau_s = (int(hit["b"][0]) + 0.5) * DUMP_BUCKET_S

    return {"status": "measured", **out, "rush_size": rush_size, "dump_tau_s": dump_tau_s}


def read_span(t0: int) -> tuple[int, int]:
    """`[lo, hi)` of the trades `measure_trades` needs around a label at `t0` (day D's part)."""
    ws = t0 - SEARCH_BEFORE_S * S_US
    return ws - REF_S * S_US, t0 + (SEARCH_AFTER_S + POST_S) * S_US


def measure_trades(prev_trades: pl.DataFrame, trades: pl.DataFrame, t0: int) -> dict[str, Any]:
    """Measure the label at `t0` from day D-1's trades and the trades of `read_span(t0)`."""
    try:
        base = base_quote_rate(prev_trades)
    except ValueError:
        return {"status": "no_base_volume"}
    ws = t0 - SEARCH_BEFORE_S * S_US
    ref = trades.filter(pl.col("ts_us") < ws)
    if ref.height == 0:
        return {"status": "no_reference_trades", "base_quote_per_s": base}
    p_ref = float(ref["price"].median())
    is_buy = pl.col("taker_side") == "buy"
    ref_buy = float(ref.filter(is_buy)["quote_qty"].sum()) / REF_S
    prev_buy = float(prev_trades.filter(is_buy)["quote_qty"].sum()) / 86_400
    buy_bg = min(ref_buy, prev_buy)
    res = measure_window(trades.filter(pl.col("ts_us") >= ws), t0, base, p_ref, buy_bg)
    return {
        "base_quote_per_s": base,
        "p_ref": p_ref,
        "ref_buy_quote_per_s": ref_buy,
        "prev_day_buy_quote_per_s": prev_buy,
        "buy_bg_quote_per_s": buy_bg,
        **res,
    }


def measure_label_store(store: Any, symbol: str, t0: int) -> dict[str, Any]:
    """`measure_trades` for the label at `t0`, reading a `Store` (status `measured` or an
    exclusion: `missing_day`, `no_baseline`, or one of `measure_trades`)."""
    day = _day_of_us(t0)
    if not store.has_day(symbol, day):
        return {"status": "missing_day"}
    prev = day - timedelta(days=1)
    if not store.has_day(symbol, prev):
        return {"status": "no_baseline"}
    prev_trades = store.read_trades(symbol, _day_start_us(prev), _day_start_us(day))
    lo, hi = read_span(t0)
    return measure_trades(prev_trades, store.read_trades(symbol, lo, hi), t0)
