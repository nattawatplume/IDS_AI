"""Trades -> 1 s bars (batch).

A bar is indexed by its start `t_us` (a multiple of 1 s) and aggregates the trades with
`t_us <= ts_us < t_us + 1 s`. Float sums are accumulated sequentially in `(ts_us, trade_id)`
order starting from 0.0 (`np.add.at`), so a streaming builder that does `acc += quote_qty` per
trade produces bit-identical bars.
"""

import numpy as np
import polars as pl

S_US = 1_000_000
PINGPONG_MAX_GAP_US = 50_000
ROUND_TOL = 1e-9

BARS: pl.Schema = pl.Schema(
    {
        "symbol": pl.String,
        "t_us": pl.Int64,
        "n_trades": pl.Int64,
        "buy_quote": pl.Float64,
        "sell_quote": pl.Float64,
        "quote_vol": pl.Float64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "rush_orders": pl.Int64,
        "max_trade_quote": pl.Float64,
        "pingpong": pl.Int64,
        "n_round": pl.Int64,
    }
)


def is_round_qty(qty: np.ndarray) -> np.ndarray:
    """`qty / 10**floor(log10(qty))` within 1e-9 of an integer (one significant digit)."""
    qty = np.asarray(qty, dtype=np.float64)
    out = np.zeros(qty.shape, dtype=bool)
    pos = qty > 0
    q = qty[pos]
    mant = q / 10.0 ** np.floor(np.log10(q))
    out[pos] = np.abs(mant - np.rint(mant)) <= ROUND_TOL
    return out


def trades_to_bars(trades: pl.DataFrame, symbol: str, start_us: int, end_us: int) -> pl.DataFrame:
    """One bar per second `t_us` in `[start_us, end_us)` from `trades` of `symbol` (BARS schema).

    Empty seconds: counts 0, OHLC = previous close; before the first trade OHLC = NaN.
    `rush_orders` = distinct `ts_us // 1000` in the second with >= 2 buy-taker trades;
    `pingpong` = consecutive trade pairs (counted in the second of the later trade) with equal
    qty and price, opposite taker side and `dts <= 50 ms`; `n_round` = trades with a round qty.
    """
    if start_us % S_US or end_us % S_US:
        raise ValueError("start_us and end_us must be multiples of 1 s")
    if end_us < start_us:
        raise ValueError("end_us must be >= start_us")
    n_sec = (end_us - start_us) // S_US
    if n_sec == 0:
        return pl.DataFrame(schema=BARS)
    df = trades.filter(
        (pl.col("symbol") == symbol) & (pl.col("ts_us") >= start_us) & (pl.col("ts_us") < end_us)
    ).sort(["ts_us", "trade_id"])
    side = df["taker_side"]
    bad = side.filter(~side.is_in(["buy", "sell"]))
    if bad.len():
        raise ValueError(f"unknown taker_side values: {bad.unique().to_list()}")

    ts = df["ts_us"].to_numpy().astype(np.int64, copy=False)
    price = df["price"].to_numpy().astype(np.float64, copy=False)
    qty = df["qty"].to_numpy().astype(np.float64, copy=False)
    quote = df["quote_qty"].to_numpy().astype(np.float64, copy=False)
    is_buy = (side == "buy").to_numpy()
    sec = (ts - start_us) // S_US

    n_trades = np.bincount(sec, minlength=n_sec).astype(np.int64)
    buy_quote = np.zeros(n_sec)
    sell_quote = np.zeros(n_sec)
    quote_vol = np.zeros(n_sec)
    max_q = np.zeros(n_sec)
    np.add.at(buy_quote, sec[is_buy], quote[is_buy])
    np.add.at(sell_quote, sec[~is_buy], quote[~is_buy])
    np.add.at(quote_vol, sec, quote)
    np.maximum.at(max_q, sec, quote)

    nan = np.full(n_sec, np.nan)
    o, h, lo, c = nan.copy(), nan.copy(), nan.copy(), nan.copy()
    if len(ts):
        first = np.flatnonzero(np.r_[True, sec[1:] != sec[:-1]])
        last = np.r_[first[1:] - 1, len(ts) - 1]
        s_u = sec[first]
        o[s_u] = price[first]
        c[s_u] = price[last]
        h[s_u] = np.maximum.reduceat(price, first)
        lo[s_u] = np.minimum.reduceat(price, first)
    has = n_trades > 0
    last_idx = np.maximum.accumulate(np.where(has, np.arange(n_sec), -1))
    filled = np.where(last_idx >= 0, c[np.maximum(last_idx, 0)], np.nan)
    o = np.where(has, o, filled)
    h = np.where(has, h, filled)
    lo = np.where(has, lo, filled)
    c = filled

    # rush orders: distinct buy-taker milliseconds with >= 2 fills
    ms = ts[is_buy] // 1000
    if len(ms):
        u_ms, cnt = np.unique(ms, return_counts=True)
        rush_ms = u_ms[cnt >= 2]
        rush = np.bincount((rush_ms * 1000 - start_us) // S_US, minlength=n_sec)
    else:
        rush = np.zeros(n_sec, dtype=np.int64)

    if len(ts) >= 2:
        pp = (
            (qty[1:] == qty[:-1])
            & (price[1:] == price[:-1])
            & (is_buy[1:] != is_buy[:-1])
            & (ts[1:] - ts[:-1] <= PINGPONG_MAX_GAP_US)
        )
        pingpong = np.bincount(sec[1:][pp], minlength=n_sec)
    else:
        pingpong = np.zeros(n_sec, dtype=np.int64)
    n_round = np.bincount(sec[is_round_qty(qty)], minlength=n_sec)

    return pl.DataFrame(
        {
            "symbol": [symbol] * n_sec,
            "t_us": start_us + np.arange(n_sec, dtype=np.int64) * S_US,
            "n_trades": n_trades,
            "buy_quote": buy_quote,
            "sell_quote": sell_quote,
            "quote_vol": quote_vol,
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "rush_orders": rush.astype(np.int64),
            "max_trade_quote": max_q,
            "pingpong": pingpong.astype(np.int64),
            "n_round": n_round.astype(np.int64),
        },
        schema=BARS,
    )
