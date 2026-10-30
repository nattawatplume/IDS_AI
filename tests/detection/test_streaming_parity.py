"""Streaming features reproduce the batch path bit for bit (spec: batch == stream)."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from idsai.config import load_settings
from idsai.contracts.events import Trade
from idsai.contracts.frames import TRADES
from idsai.detection.features.bars import BARS, BarBuilder, trades_to_bars
from idsai.detection.features.registry import feature_columns
from idsai.detection.features.streaming import StreamingWindows
from idsai.detection.features.windows import compute_windows
from tests.synth import day_us, make_trades

S = 1_000_000
SYM = "TESTBTC"
D = date(2025, 3, 1)
T0 = day_us(D)
CFG = load_settings().features
UNIT = 1e-4


def _as_trades(df: pl.DataFrame) -> list[Trade]:
    return [Trade(**r) for r in df.iter_rows(named=True)]


def _raw(ts, price, qty, side, ids=None, symbol=SYM) -> pl.DataFrame:
    n = len(ts)
    price = np.broadcast_to(np.asarray(price, dtype=float), (n,))
    qty = np.broadcast_to(np.asarray(qty, dtype=float), (n,))
    return pl.DataFrame(
        {
            "ts_us": np.asarray(ts, dtype=np.int64),
            "recv_us": np.asarray(ts, dtype=np.int64),
            "symbol": [symbol] * n,
            "price": price,
            "qty": qty,
            "quote_qty": price * qty,
            "taker_side": list(side),
            "trade_id": np.arange(1, n + 1) if ids is None else np.asarray(ids),
            "n_fills": np.ones(n, dtype=np.int32),
            "source": ["archive"] * n,
        },
        schema=TRADES,
    )


@pytest.fixture(scope="module")
def trades_30h() -> pl.DataFrame:
    """30 h of random sparse trades, with rush clusters, pingpong pairs and round quantities."""
    base = make_trades(D, 2, symbol=SYM, seed=21, rate=0.2).filter(
        pl.col("ts_us") < T0 + 30 * 3600 * S
    )
    rng = np.random.default_rng(22)
    n = 400
    ms = rng.integers(0, 30 * 3600 * 1000 - 1, n) * 1000 + T0
    rush = _raw(
        np.repeat(ms, 3) + np.tile([0, 100, 200], n),
        5e-4,
        np.repeat(rng.integers(1, 9, n) * 10.0, 3),
        ["buy"] * 3 * n,
        ids=np.arange(10_000_000, 10_000_000 + 3 * n),
    )
    # pairs straddling a second boundary: the later trade's second counts the pingpong
    s = T0 + rng.integers(1, 30 * 3600, n) * S
    t = np.column_stack([s - 10_000, s + 5_000]).ravel()
    pp = _raw(
        t,
        5.5e-4,
        50.0,
        ["buy", "sell"] * n,
        ids=np.arange(20_000_000, 20_000_000 + 2 * n),
    )
    return pl.concat([base, rush, pp]).sort(["ts_us", "trade_id"])


def _stream(trades, start, end):
    bb = BarBuilder(SYM, 0, start_us=start)
    sw = StreamingWindows(SYM, CFG, UNIT)
    bars, rows = [], []
    it = iter(_as_trades(trades))
    tr = next(it, None)
    for tick in range(start + S, end + S, S):
        while tr is not None and tr.ts_us < tick:
            bb.on_trade(tr)
            tr = next(it, None)
        for b in bb.on_tick(tick):
            bars.append(b)
            r = sw.push_bar(b)
            if r is not None:
                rows.append(r)
    return bars, rows


def _assert_same(frame: pl.DataFrame, rows: list[dict], cols: list[str]):
    assert frame.height == len(rows)
    for c in cols:
        a = frame[c].to_numpy()
        b = np.array([r[c] for r in rows], dtype=a.dtype)
        if a.dtype.kind == "f":
            assert np.array_equal(a, b, equal_nan=True), c
        else:
            assert np.array_equal(a, b), c


def test_30h_stream_equals_batch_on_every_column(trades_30h):
    start, end = T0, T0 + 30 * 3600 * S
    bars_b = trades_to_bars(trades_30h, SYM, start, end)
    win_b = compute_windows(bars_b, CFG, UNIT)
    bars_s, rows_s = _stream(trades_30h, start, end)

    _assert_same(bars_b, bars_s, [c for c in BARS.names() if c != "symbol"])
    assert {b["symbol"] for b in bars_s} == {SYM}
    assert list(bars_s[0]) == BARS.names()  # same columns, same order
    assert bars_b["pingpong"].sum() > 0 and bars_b["rush_orders"].sum() > 0
    assert bars_b["n_round"].sum() > 0

    assert win_b.height > 0
    _assert_same(win_b, rows_s, ["t_end_us", *feature_columns(CFG)])
    assert {r["symbol"] for r in rows_s} == {SYM}
    assert list(rows_s[0]) == win_b.columns


def test_grace_does_not_change_bars_and_shuffle_inside_grace_is_sorted(trades_30h):
    """Trades arriving out of order inside the grace period sort exactly as batch."""
    start, end = T0, T0 + 1800 * S
    sub = trades_30h.filter(pl.col("ts_us") < end)
    want = trades_to_bars(sub, SYM, start, end)

    bb = BarBuilder(SYM, 2, start_us=start)
    order = np.random.default_rng(3)
    trades = _as_trades(sub)
    got: list[dict] = []
    i = 0
    for tick in range(start + S, end + 3 * S, S):
        chunk = []
        while i < len(trades) and trades[i].ts_us < tick:
            chunk.append(trades[i])
            i += 1
        order.shuffle(chunk)
        for t in chunk:
            bb.on_trade(t)
        got.extend(bb.on_tick(tick))
    assert bb.late == 0
    got = [b for b in got if b["t_us"] < end]
    _assert_same(want, got, [c for c in BARS.names() if c != "symbol"])


def test_second_closes_exactly_at_s_plus_1s_plus_grace():
    bb = BarBuilder(SYM, 2, start_us=T0)
    assert bb.on_tick(T0 + 3 * S - 1) == []
    out = bb.on_tick(T0 + 3 * S)
    assert [b["t_us"] for b in out] == [T0]
    assert [b["t_us"] for b in bb.on_tick(T0 + 5 * S)] == [T0 + S, T0 + 2 * S]


def test_late_trade_is_counted_not_applied():
    t = _as_trades(_raw([T0 + 100, T0 + 200, T0 + S + 1], 2.0, 3.0, ["buy"] * 3))
    bb = BarBuilder(SYM, 0, start_us=T0)
    bb.on_trade(t[0])
    (b0,) = bb.on_tick(T0 + S)
    bb.on_trade(t[1])  # its second already closed
    assert bb.late == 1
    bb.on_trade(t[2])
    (b1,) = bb.on_tick(T0 + 2 * S)
    assert b0["n_trades"] == 1 and b1["n_trades"] == 1


def test_trade_older_than_first_second_is_late():
    t = _as_trades(_raw([T0 - 1], 2.0, 3.0, ["buy"]))
    bb = BarBuilder(SYM, 0, start_us=T0)
    bb.on_trade(t[0])
    assert bb.late == 1
    (b,) = bb.on_tick(T0 + S)
    assert b["n_trades"] == 0 and np.isnan(b["close"])


def test_duplicate_trade_is_counted_not_applied():
    t = _as_trades(_raw([T0 + 5, T0 + 5], 2.0, 3.0, ["buy", "buy"], ids=[7, 7]))
    bb = BarBuilder(SYM, 0, start_us=T0)
    bb.on_trade(t[0])
    bb.on_trade(t[1])
    (b,) = bb.on_tick(T0 + S)
    assert b["n_trades"] == 1 and bb.duplicates == 1


def test_tick_going_backward_raises():
    bb = BarBuilder(SYM, 0, start_us=T0)
    bb.on_tick(T0 + 5 * S)
    bb.on_tick(T0 + 5 * S)  # equal is fine
    with pytest.raises(ValueError):
        bb.on_tick(T0 + 5 * S - 1)


def test_default_start_is_the_second_of_the_first_trade_or_tick():
    t = _as_trades(_raw([T0 + 7 * S + 5], 2.0, 3.0, ["buy"]))
    bb = BarBuilder(SYM, 0)
    bb.on_trade(t[0])
    assert [b["t_us"] for b in bb.on_tick(T0 + 8 * S)] == [T0 + 7 * S]
    bb2 = BarBuilder(SYM, 0)
    assert bb2.on_tick(T0 + 7 * S + 5) == []
    assert [b["t_us"] for b in bb2.on_tick(T0 + 8 * S)] == [T0 + 7 * S]


def test_start_must_be_on_a_second_and_grace_not_negative_and_symbol_must_match():
    with pytest.raises(ValueError):
        BarBuilder(SYM, 0, start_us=T0 + 1)
    with pytest.raises(ValueError):
        BarBuilder(SYM, -1)
    t = _as_trades(_raw([T0], 2.0, 3.0, ["buy"], symbol="OTHERBTC"))
    with pytest.raises(ValueError):
        BarBuilder(SYM, 0, start_us=T0).on_trade(t[0])


def test_streaming_windows_rejects_other_symbol_and_gap():
    sw = StreamingWindows(SYM, CFG, UNIT)
    bar = trades_to_bars(_raw([T0], 2.0, 3.0, ["buy"]), SYM, T0, T0 + S).row(0, named=True)
    sw.push_bar(bar)
    with pytest.raises(ValueError):
        sw.push_bar({**bar, "symbol": "OTHERBTC", "t_us": T0 + S})
    with pytest.raises(ValueError):
        sw.push_bar({**bar, "t_us": T0 + 5 * S})  # not contiguous
