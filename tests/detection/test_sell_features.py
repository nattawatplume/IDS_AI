"""M4 sell-side features: `sell_rush_orders`, `sell_rush_*`, `drop_*`, `cusum_sell`."""

import math
from datetime import date

import numpy as np
import polars as pl
import pytest

from idsai.config import load_settings
from idsai.contracts.events import Trade
from idsai.contracts.frames import TRADES
from idsai.detection.features.bars import BARS, BarBuilder, trades_to_bars
from idsai.detection.features.kernel import BarArrays, Kernel, raw_row
from idsai.detection.features.registry import (
    feature_columns,
    model_features,
    raw_columns,
    spec_for,
)
from idsai.detection.features.streaming import StreamingWindows
from idsai.detection.features.windows import compute_windows
from tests.synth import day_us, make_trades

S = 1_000_000
SYM = "TESTBTC"
D = date(2025, 3, 1)
T0 = day_us(D)
UNIT = 1e-4
OFF = load_settings().features
ON = OFF.model_copy(update={"sell_side": True})


def _raw(ts, price, qty, side, ids=None) -> pl.DataFrame:
    n = len(ts)
    price = np.broadcast_to(np.asarray(price, dtype=float), (n,))
    qty = np.broadcast_to(np.asarray(qty, dtype=float), (n,))
    return pl.DataFrame(
        {
            "ts_us": np.asarray(ts, dtype=np.int64),
            "recv_us": np.asarray(ts, dtype=np.int64),
            "symbol": [SYM] * n,
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


def test_sell_rush_orders_counts_sell_taker_milliseconds_with_two_fills():
    ts = [
        T0 + 1000,  # ms 1: two sells -> 1
        T0 + 1500,
        T0 + 3000,  # ms 3: sell + buy -> not a sell rush
        T0 + 3100,
        T0 + 5000,  # ms 5: three sells -> 1 (distinct ms, not fills)
        T0 + 5001,
        T0 + 5002,
        T0 + 7000,  # ms 7: two buys -> rush_orders, not sell
        T0 + 7001,
        T0 + S + 10,  # second 1: two sells in one ms -> 1
        T0 + S + 20,
    ]
    side = ["sell", "sell", "sell", "buy", "sell", "sell", "sell", "buy", "buy", "sell", "sell"]
    bars = trades_to_bars(_raw(ts, 2.0, 1.0, side), SYM, T0, T0 + 2 * S)
    assert bars.columns == BARS.names()
    assert bars["sell_rush_orders"].to_list() == [2, 1]
    assert bars["rush_orders"].to_list() == [1, 0]

    bb = BarBuilder(SYM, 0, start_us=T0)
    for r in _raw(ts, 2.0, 1.0, side).iter_rows(named=True):
        bb.on_trade(Trade(**r))
    got = bb.on_tick(T0 + 2 * S)
    assert [b["sell_rush_orders"] for b in got] == [2, 1]
    assert list(got[0]) == BARS.names()


def test_drop_and_sell_rush_values_on_hand_built_bars():
    # close 100 for 20 s, then lows dip to 90 in the last 10 s; 2 sell rushes per second there
    n = 40
    close = np.full(n, 100.0)
    low = np.full(n, 100.0)
    low[30:40] = np.linspace(99.0, 90.0, 10)
    srush = np.zeros(n, dtype=np.int64)
    srush[30:40] = 2
    a = BarArrays(
        t_us=T0 + np.arange(n, dtype=np.int64) * S,
        n_trades=np.ones(n, dtype=np.int64),
        buy_quote=np.ones(n),
        sell_quote=np.ones(n),
        quote_vol=np.full(n, 2.0),
        high=close.copy(),
        close=close,
        rush_orders=np.zeros(n, dtype=np.int64),
        pingpong=np.zeros(n, dtype=np.int64),
        n_round=np.zeros(n, dtype=np.int64),
        low=low,
        sell_rush_orders=srush,
    )
    cfg = ON.model_copy(update={"windows_s": [10, 20]})
    r = raw_row(a, 40, cfg, UNIT)
    assert r["sell_rush_10"] == 20 and r["sell_rush_20"] == 20
    assert math.isclose(r["drop_10"], 90.0 / 100.0 - 1.0)
    assert math.isclose(r["drop_20"], -0.1)
    r30 = raw_row(a, 30, cfg, UNIT)
    assert r30["drop_10"] == 0.0 and r30["sell_rush_10"] == 0
    assert math.isnan(raw_row(a, 15, cfg, UNIT)["drop_20"])  # window longer than the data


def test_flag_registry_columns_and_model_features():
    assert raw_columns(OFF) == [c for c in raw_columns(ON) if not c.startswith(("sell_", "drop"))]
    off, on = feature_columns(OFF), feature_columns(ON)
    assert on[: len(off)] == off  # existing columns never move
    assert on[len(off) :] == [
        *(f"{b}_{w}" for b in ("sell_rush", "drop") for w in OFF.windows_s),
        *(f"z_{b}_{w}" for b in ("sell_rush", "drop") for w in OFF.windows_s),
        "cusum_sell",
    ]
    assert model_features(ON) == model_features(OFF)
    assert not any("sell" in c or "drop" in c for c in model_features(ON))
    for c in ("sell_rush_60", "z_drop_1800", "cusum_sell"):
        assert spec_for(c).desc_th
    assert spec_for("drop_60").desc_th == "ราคาต่ำสุดในหน้าต่างเทียบราคาปิดก่อนหน้าต่าง (−1)"


@pytest.fixture(scope="module")
def trades_30h() -> pl.DataFrame:
    """30 h of sparse trades with buy and sell rush clusters and a crash."""
    base = make_trades(D, 2, symbol=SYM, seed=31, rate=0.2).filter(
        pl.col("ts_us") < T0 + 30 * 3600 * S
    )
    rng = np.random.default_rng(32)
    n = 400
    ms = rng.integers(0, 30 * 3600 * 1000 - 1, n) * 1000 + T0
    side = np.repeat(np.where(rng.random(n) < 0.5, "buy", "sell"), 3)
    rush = _raw(
        np.repeat(ms, 3) + np.tile([0, 100, 200], n),
        5e-4,
        np.repeat(rng.integers(1, 9, n) * 10.0, 3),
        side,
        ids=np.arange(10_000_000, 10_000_000 + 3 * n),
    )
    # a crash at 28 h: 60 s of sell clusters walking the price down 10 %
    t = T0 + 28 * 3600 * S + np.arange(60) * S
    crash = _raw(
        np.repeat(t, 2) + np.tile([0, 300], 60),
        np.repeat(np.linspace(5e-4, 4.5e-4, 60), 2),
        100.0,
        ["sell"] * 120,
        ids=np.arange(30_000_000, 30_000_120),
    )
    return pl.concat([base, rush, crash]).sort(["ts_us", "trade_id"])


def _stream(trades, start, end, cfg):
    bb = BarBuilder(SYM, 0, start_us=start)
    sw = StreamingWindows(SYM, cfg, UNIT)
    rows = []
    it = iter([Trade(**r) for r in trades.iter_rows(named=True)])
    tr = next(it, None)
    for tick in range(start + S, end + S, S):
        while tr is not None and tr.ts_us < tick:
            bb.on_trade(tr)
            tr = next(it, None)
        for b in bb.on_tick(tick):
            r = sw.push_bar(b)
            if r is not None:
                rows.append(r)
    return rows


def test_sell_features_batch_equals_stream_and_existing_columns_are_unchanged(trades_30h):
    start, end = T0, T0 + 30 * 3600 * S
    bars = trades_to_bars(trades_30h, SYM, start, end)
    assert bars["sell_rush_orders"].sum() > 0
    on = compute_windows(bars, ON, UNIT)
    off = compute_windows(bars, OFF, UNIT)
    # flag on never changes an existing column
    for c in off.columns:
        a, b = off[c].to_numpy(), on[c].to_numpy()
        assert np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else (a == b).all(), c
    rows = _stream(trades_30h, start, end, ON)
    assert len(rows) == on.height and list(rows[0]) == on.columns
    for c in ["t_end_us", *feature_columns(ON)]:
        a = on[c].to_numpy()
        b = np.array([r[c] for r in rows], dtype=a.dtype)
        assert np.array_equal(a, b, equal_nan=True), c
    # the crash shows on the sell side
    crash = on.filter(pl.col("t_end_us") == T0 + (28 * 3600 + 60) * S).row(0, named=True)
    assert crash["drop_60"] < -0.05 and crash["sell_rush_60"] >= 60
    assert crash["z_sell_rush_60"] > 3 and crash["z_drop_60"] < -3
    assert on["cusum_sell"].max() > 0 and (on["cusum_sell"] >= 0).all()


def test_sell_side_needs_low_and_sell_rush():
    k = Kernel(ON, UNIT)
    with pytest.raises(ValueError, match="sell_side"):
        k.push_bar(T0, 1, 1.0, 1.0, 2.0, 1.0, 1.0, 0, 0, 0)
    Kernel(OFF, UNIT).push_bar(T0, 1, 1.0, 1.0, 2.0, 1.0, 1.0, 0, 0, 0)  # off: not needed
