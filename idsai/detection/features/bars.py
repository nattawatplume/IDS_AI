"""Trades -> 1 s bars: `trades_to_bars` (batch) and `BarBuilder` (stream), identical rows.

A bar is indexed by its start `t_us` (a multiple of 1 s) and aggregates the trades with
`t_us <= ts_us < t_us + 1 s`. Float sums are accumulated sequentially in `(ts_us, trade_id)`
order starting from 0.0 (`np.add.at`), so a streaming builder that does `acc += quote_qty` per
trade produces bit-identical bars.
"""

from collections import Counter
from typing import Any

import numpy as np
import polars as pl

from idsai.contracts.events import Trade

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


class BarBuilder:
    """Streaming `trades_to_bars` for one symbol: `on_trade` buffers, `on_tick` closes seconds.

    Second `s` closes when a tick reaches `s + 1 s + grace_s`; its trades are then folded in
    `(ts_us, trade_id)` order whatever order they arrived in, so a row equals `trades_to_bars` over
    the same seconds, including empty ones. `start_us` is the first second to close (None = the
    second of the first trade or tick seen); use the batch run's start to reproduce it.

    Input that cannot be applied is counted, never silently mixed in: `late` = trades of a closed
    second or older than `start_us`, `duplicates` = a repeated `trade_id` among the open seconds.
    A tick going backward raises ValueError.
    """

    def __init__(self, symbol: str, grace_s: int, start_us: int | None = None):
        if grace_s < 0:
            raise ValueError(f"grace_s must be >= 0, got {grace_s}")
        if start_us is not None and start_us % S_US:
            raise ValueError("start_us must be a multiple of 1 s")
        self.symbol = symbol
        self.grace_us = grace_s * S_US
        self.late = 0
        self.duplicates = 0
        self._next_s = start_us  # the oldest second not closed yet
        self._open: dict[int, list[Trade]] = {}
        self._ids: set[int] = set()
        self._last_tick: int | None = None
        self._close = float("nan")
        self._prev: Trade | None = None  # last folded trade, for pingpong across seconds

    def on_trade(self, t: Trade) -> None:
        if t.symbol != self.symbol:
            raise ValueError(f"trade of {t.symbol!r} given to the bars of {self.symbol!r}")
        if self._next_s is None:
            self._next_s = t.ts_us // S_US * S_US
        if t.ts_us < self._next_s:
            self.late += 1
        elif t.trade_id in self._ids:
            self.duplicates += 1
        else:
            self._ids.add(t.trade_id)
            self._open.setdefault(t.ts_us // S_US * S_US, []).append(t)

    def on_tick(self, t_us: int) -> list[dict[str, Any]]:
        """The bars of every second closed by this tick, in time order."""
        if self._last_tick is not None and t_us < self._last_tick:
            raise ValueError(f"tick went backward: {t_us} after {self._last_tick}")
        self._last_tick = t_us
        if self._next_s is None:
            self._next_s = t_us // S_US * S_US
        out = []
        while t_us >= self._next_s + S_US + self.grace_us:
            out.append(self._close_second(self._next_s))
            self._next_s += S_US
        return out

    def _close_second(self, s: int) -> dict[str, Any]:
        trades = self._open.pop(s, [])
        bar: dict[str, Any] = {
            "symbol": self.symbol,
            "t_us": s,
            "n_trades": len(trades),
            "buy_quote": 0.0,
            "sell_quote": 0.0,
            "quote_vol": 0.0,
            "open": self._close,
            "high": self._close,
            "low": self._close,
            "close": self._close,
            "rush_orders": 0,
            "max_trade_quote": 0.0,
            "pingpong": 0,
            "n_round": 0,
        }
        if not trades:
            return bar
        trades.sort(key=lambda t: (t.ts_us, t.trade_id))
        prices = [t.price for t in trades]
        self._close = prices[-1]
        bar.update(open=prices[0], high=max(prices), low=min(prices), close=prices[-1])
        buy_ms: Counter[int] = Counter()
        for t in trades:
            self._ids.discard(t.trade_id)
            if t.taker_side == "buy":
                bar["buy_quote"] += t.quote_qty
                buy_ms[t.ts_us // 1000] += 1
            else:
                bar["sell_quote"] += t.quote_qty
            bar["quote_vol"] += t.quote_qty
            bar["max_trade_quote"] = max(bar["max_trade_quote"], t.quote_qty)
            p = self._prev
            if (
                p is not None
                and t.qty == p.qty
                and t.price == p.price
                and t.taker_side != p.taker_side
                and t.ts_us - p.ts_us <= PINGPONG_MAX_GAP_US
            ):
                bar["pingpong"] += 1
            self._prev = t
        bar["rush_orders"] = sum(c >= 2 for c in buy_ms.values())
        bar["n_round"] = int(np.count_nonzero(is_round_qty(np.array([t.qty for t in trades]))))
        return bar
