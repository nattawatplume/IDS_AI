"""Synthetic trade streams for tests (no network)."""

from datetime import date

import numpy as np
import polars as pl

from idsai.contracts.frames import TRADES
from idsai.data.store import DAY_US

S = 1_000_000


def day_us(day: date) -> int:
    return (day - date(1970, 1, 1)).days * DAY_US


def make_trades(
    first_day: date,
    n_days: int,
    symbol: str = "TESTBTC",
    seed: int = 5,
    rate: float = 1.0,
    step: float = 1e-7,
    first_id: int = 1,
) -> pl.DataFrame:
    """About `rate` trades/s of a BTC-quoted pair: price ~ 0.0005 moving in `step` ticks."""
    rng = np.random.default_rng(seed)
    t0 = day_us(first_day)
    dur = n_days * DAY_US
    n = rng.poisson(rate * dur / S)
    ts = np.sort(t0 + rng.integers(0, dur, n))
    price = np.round(5e-4 + step * np.cumsum(rng.choice([-1, 0, 0, 1], n)), 7)
    price = np.maximum(price, 1e-7)
    qty = np.round(rng.lognormal(3.0, 1.0, n), 2) + 0.01
    side = np.where(rng.random(n) < 0.5, "buy", "sell")
    return pl.DataFrame(
        {
            "ts_us": ts,
            "recv_us": ts,
            "symbol": [symbol] * n,
            "price": price,
            "qty": qty,
            "quote_qty": price * qty,
            "taker_side": side.tolist(),
            "trade_id": np.arange(first_id, first_id + n, dtype=np.int64),
            "n_fills": np.ones(n, dtype=np.int32),
            "source": ["archive"] * n,
        },
        schema=TRADES,
    )
