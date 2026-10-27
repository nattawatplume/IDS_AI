import math

import numpy as np
import polars as pl
import pytest

from idsai.contracts.frames import TRADES
from idsai.detection.features.bars import BARS, trades_to_bars

S = 1_000_000
T0 = 1_740_787_200_000_000  # 2025-03-01T00:00:00Z


def mk_trades(rows, symbol="ABCUSDT") -> pl.DataFrame:
    """rows: (ts_us, price, qty, side) tuples; quote_qty = price * qty."""
    n = len(rows)
    return pl.DataFrame(
        {
            "ts_us": [r[0] for r in rows],
            "recv_us": [r[0] for r in rows],
            "symbol": [symbol] * n,
            "price": [float(r[1]) for r in rows],
            "qty": [float(r[2]) for r in rows],
            "quote_qty": [float(r[1]) * float(r[2]) for r in rows],
            "taker_side": [r[3] for r in rows],
            "trade_id": list(range(1, n + 1)),
            "n_fills": [1] * n,
            "source": ["archive"] * n,
        },
        schema=TRADES,
    )


def test_schema_and_one_row_per_second():
    b = trades_to_bars(mk_trades([(T0 + 2 * S, 10.0, 1.0, "buy")]), "ABCUSDT", T0, T0 + 5 * S)
    assert b.schema == BARS
    assert b["t_us"].to_list() == [T0 + k * S for k in range(5)]
    assert b["symbol"].to_list() == ["ABCUSDT"] * 5


def test_empty_seconds_filled_and_leading_nan():
    tr = mk_trades(
        [
            (T0 + 2 * S + 100, 10.0, 1.0, "buy"),
            (T0 + 2 * S + 900_000, 12.0, 2.0, "sell"),
            (T0 + 5 * S + 1, 11.0, 1.0, "buy"),
        ]
    )
    b = trades_to_bars(tr, "ABCUSDT", T0, T0 + 7 * S)
    # before the first trade: OHLC NaN, counts 0
    for col in ("open", "high", "low", "close"):
        assert all(math.isnan(v) for v in b[col][:2].to_list())
    assert b["n_trades"].to_list() == [0, 0, 2, 0, 0, 1, 0]
    # the trading second
    r2 = b.row(2, named=True)
    assert (r2["open"], r2["high"], r2["low"], r2["close"]) == (10.0, 12.0, 10.0, 12.0)
    assert r2["buy_quote"] == 10.0 and r2["sell_quote"] == 24.0 and r2["quote_vol"] == 34.0
    assert r2["max_trade_quote"] == 24.0
    # empty seconds: counts 0, OHLC = previous close
    for i in (3, 4):
        r = b.row(i, named=True)
        assert (r["open"], r["high"], r["low"], r["close"]) == (12.0, 12.0, 12.0, 12.0)
        assert r["quote_vol"] == 0.0 and r["buy_quote"] == 0.0 and r["max_trade_quote"] == 0.0
    assert b.row(6, named=True)["close"] == 11.0


def test_other_symbols_and_out_of_range_ignored():
    tr = pl.concat(
        [
            mk_trades([(T0 + S, 10.0, 1.0, "buy")]),
            mk_trades([(T0 + S, 99.0, 1.0, "buy")], symbol="XYZUSDT"),
            mk_trades([(T0 - 1, 50.0, 1.0, "buy"), (T0 + 3 * S, 50.0, 1.0, "buy")]),
        ]
    )
    b = trades_to_bars(tr, "ABCUSDT", T0, T0 + 3 * S)
    assert b["n_trades"].to_list() == [0, 1, 0]
    assert b["close"][1] == 10.0


def test_rush_orders_buy_ms_groups_only():
    tr = mk_trades(
        [
            (T0 + 1000, 10.0, 1.0, "buy"),  # ms 1: two buys -> one rush order
            (T0 + 1500, 10.0, 1.0, "buy"),
            (T0 + 5000, 10.0, 1.0, "sell"),  # ms 5: two sells -> not a rush order
            (T0 + 5100, 10.0, 1.0, "sell"),
            (T0 + 9000, 10.0, 1.0, "buy"),  # ms 9: buy + sell -> not a rush order
            (T0 + 9200, 10.0, 1.0, "sell"),
            (T0 + 20_000, 10.0, 1.0, "buy"),  # ms 20: single buy
        ]
    )
    b = trades_to_bars(tr, "ABCUSDT", T0, T0 + S)
    assert b["rush_orders"].to_list() == [1]


def test_rush_orders_counts_distinct_ms():
    tr = mk_trades(
        [(T0 + 1000 + k, 10.0, 1.0, "buy") for k in range(3)]
        + [(T0 + 7000 + k, 10.0, 1.0, "buy") for k in range(2)]
    )
    assert trades_to_bars(tr, "ABCUSDT", T0, T0 + S)["rush_orders"].to_list() == [2]


@pytest.mark.parametrize(("gap_us", "expected"), [(10_000, 1), (50_000, 1), (60_000, 0)])
def test_pingpong_gap(gap_us, expected):
    tr = mk_trades([(T0 + 100, 10.0, 3.0, "buy"), (T0 + 100 + gap_us, 10.0, 3.0, "sell")])
    assert trades_to_bars(tr, "ABCUSDT", T0, T0 + S)["pingpong"].to_list() == [expected]


def test_pingpong_needs_same_price_qty_and_opposite_side():
    tr = mk_trades(
        [
            (T0 + 100, 10.0, 3.0, "buy"),
            (T0 + 200, 10.0, 3.0, "buy"),  # same side
            (T0 + 300, 10.5, 3.0, "sell"),  # different price
            (T0 + 400, 10.5, 2.0, "buy"),  # different qty
        ]
    )
    assert trades_to_bars(tr, "ABCUSDT", T0, T0 + S)["pingpong"].to_list() == [0]


def test_pingpong_across_second_boundary_counted_at_second_trade():
    tr = mk_trades([(T0 + S - 10_000, 10.0, 3.0, "buy"), (T0 + S + 10_000, 10.0, 3.0, "sell")])
    assert trades_to_bars(tr, "ABCUSDT", T0, T0 + 2 * S)["pingpong"].to_list() == [0, 1]


def test_n_round():
    qtys = [0.5, 20.0, 300.0, 0.37, 1.5, 0.001, 7.0, 0.3]
    tr = mk_trades([(T0 + 1000 * (k + 1), 10.0, q, "buy") for k, q in enumerate(qtys)])
    # round: 0.5, 20, 300, 0.001, 7, 0.3 -> 6 ; not round: 0.37, 1.5
    assert trades_to_bars(tr, "ABCUSDT", T0, T0 + S)["n_round"].to_list() == [6]


def test_sums_are_sequential_in_trade_order():
    # a streaming builder adds quote_qty one by one in (ts_us, trade_id) order from 0.0
    rng = np.random.default_rng(1)
    rows = [
        (T0 + int(t), 1.0, float(q), "buy")
        for t, q in zip(np.sort(rng.integers(0, S, 200)), rng.random(200) * 1e3, strict=True)
    ]
    tr = mk_trades(rows)
    acc = 0.0
    for q in tr["quote_qty"].to_list():
        acc += q
    b = trades_to_bars(tr.reverse(), "ABCUSDT", T0, T0 + S)  # input order must not matter
    assert b["quote_vol"][0] == acc and b["buy_quote"][0] == acc


def test_bad_range_rejected():
    with pytest.raises(ValueError):
        trades_to_bars(mk_trades([]), "ABCUSDT", T0 + 1, T0 + S)
    with pytest.raises(ValueError):
        trades_to_bars(mk_trades([]), "ABCUSDT", T0 + S, T0)


def test_empty_range_gives_nan_bars():
    b = trades_to_bars(mk_trades([]), "ABCUSDT", T0, T0 + 3 * S)
    assert b.height == 3 and b["n_trades"].to_list() == [0, 0, 0]
    assert all(math.isnan(v) for v in b["close"].to_list())
