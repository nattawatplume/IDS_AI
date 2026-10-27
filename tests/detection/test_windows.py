import math
import time

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from idsai.config import FeatureCfg
from idsai.contracts.frames import TRADES
from idsai.detection.features.bars import trades_to_bars
from idsai.detection.features.kernel import (
    FLOOR_LOG,
    BarArrays,
    Baseline,
    Kernel,
    cusum_step,
    prefix,
    window_sum,
)
from idsai.detection.features.registry import REGISTRY, feature_columns, model_features
from idsai.detection.features.windows import compute_windows

S = 1_000_000
H = 3600 * S
T0 = 1_740_787_200_000_000  # 2025-03-01T00:00:00Z (UTC midnight)
cfg = FeatureCfg()


# ---------------------------------------------------------------- helpers
def mk_trades(ts, price, qty, side, symbol="ABCUSDT") -> pl.DataFrame:
    ts = np.asarray(ts, dtype=np.int64)
    n = len(ts)
    price = np.broadcast_to(np.asarray(price, dtype=np.float64), (n,))
    qty = np.broadcast_to(np.asarray(qty, dtype=np.float64), (n,))
    side = np.broadcast_to(np.asarray(side, dtype=object), (n,))
    return pl.DataFrame(
        {
            "ts_us": ts,
            "recv_us": ts,
            "symbol": [symbol] * n,
            "price": price,
            "qty": qty,
            "quote_qty": price * qty,
            "taker_side": list(side),
            "trade_id": np.arange(1, n + 1, dtype=np.int64),
            "n_fills": np.ones(n, dtype=np.int32),
            "source": ["archive"] * n,
        },
        schema=TRADES,
    )


def bars_of(trades, start_us, end_us, symbol="ABCUSDT"):
    return trades_to_bars(trades, symbol, start_us, end_us)


def row(w: pl.DataFrame, t_end_us) -> dict:
    hit = w.filter(pl.col("t_end_us") == int(t_end_us))
    assert hit.height == 1, f"no row at {t_end_us}"
    return hit.row(0, named=True)


def bars_const(hours: int):
    """1 trade per second (mid-second), price 100, quote 2.0, alternating side."""
    n = hours * 3600
    ts = T0 + np.arange(n, dtype=np.int64) * S + S // 2
    side = np.where(np.arange(n) % 2 == 0, "buy", "sell")
    return bars_of(mk_trades(ts, 100.0, 0.02, side), T0, T0 + n * S)


def random_sparse_trades(
    hours: float, seed: int, rate_per_s: float = 0.2, start_us: int = T0
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dur = int(hours * H)
    n = rng.poisson(rate_per_s * dur / S)
    ts = np.sort(start_us + rng.integers(0, dur, n))
    steps = rng.choice([-1, 0, 0, 1], n)
    price = 100.0 + 0.01 * np.cumsum(steps)
    qty = np.round(rng.lognormal(0.0, 1.0, n), 3) + 0.001
    side = np.where(rng.random(n) < 0.5, "buy", "sell")
    return mk_trades(ts, price, qty, side)


def values(w: pl.DataFrame) -> np.ndarray:
    return w.select(feature_columns(cfg)).to_numpy()


# ---------------------------------------------------------------- registry
def test_registry_and_columns():
    cols = feature_columns(cfg)
    assert len(cols) == len(set(cols))
    for base in (
        "n_trades",
        "quote_vol",
        "buy_quote",
        "rush",
        "pingpong",
        "round_frac",
        "buy_imb",
        "ret",
        "jump",
    ):
        for w in cfg.windows_s:
            assert f"{base}_{w}" in cols and f"z_{base}_{w}" in cols
    assert cols[-4:] == ["cusum_ret", "cusum_imb", "kk_vol_ratio", "kk_price_ratio"]
    assert REGISTRY["quote_vol"].transform == "log1p_quote"
    assert REGISTRY["ret"].floor == "ret" and REGISTRY["buy_imb"].floor == "ratio"
    mf = model_features(cfg)
    assert mf == model_features(cfg)  # deterministic
    assert all(c.startswith(("z_", "buy_imb_", "round_frac_", "cusum_")) for c in mf)
    assert [c for c in cols if c.startswith("z_")] == mf[: sum(c.startswith("z_") for c in cols)]
    assert "cusum_ret" in mf and "kk_vol_ratio" not in mf and "quote_vol_60" not in mf


def test_prefix_and_window_sum():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    p = prefix(x)
    assert p.dtype == np.float64 and p.tolist() == [0.0, 1.0, 3.0, 6.0, 10.0]
    assert window_sum(p, 4, 2) == 7.0
    c = prefix(np.array([1, 2, 3], dtype=np.int64))
    assert c.dtype == np.int64 and c.tolist() == [0, 1, 3, 6]


def test_baseline_window_excludes_t_and_old_samples():
    small = FeatureCfg(baseline_s=600, min_baseline_samples=3)
    floors = np.array([FLOOR_LOG, np.nan])  # a log feature and a ret feature
    bl = Baseline(small, floors)
    assert np.isnan(bl.z(np.array([1.0, 1.0]))).all()  # no refresh yet
    t = T0 + 3600 * S
    bl.add(t - 700 * S, np.array([1000.0, 1000.0]))  # older than baseline_s: ignored
    for k, v in enumerate([1.0, 2.0, 3.0, 4.0]):
        bl.add(t - (400 - 60 * k) * S, np.array([v, np.nan if k == 0 else v]))
    bl.refresh(t, np.array([np.nan, 0.0, 0.02, 0.04]))
    assert bl.med[0] == 2.5 and bl.valid.tolist() == [True, True]
    assert bl.tick_rel == pytest.approx(0.03)  # median of positive |dclose|/close only
    assert bl.scale[0] == max((3.25 - 1.75) / 1.349, FLOOR_LOG)
    assert bl.med[1] == 3.0  # NaN sample excluded
    bl.add(t, np.array([1e6, 1e6]))  # a sample at the refresh time is never in that refresh
    bl.refresh(t, np.array([0.0]))
    assert bl.med[0] == 2.5 and bl.tick_rel == 1e-4 and bl.scale[1] == pytest.approx(1 / 1.349)
    z = bl.z(np.array([1e9, np.nan]))
    assert z[0] == 50.0 and np.isnan(z[1])
    strict = Baseline(FeatureCfg(baseline_s=600, min_baseline_samples=4), floors)
    for k in range(4):
        strict.add(t - (400 - 60 * k) * S, np.array([1.0, np.nan if k == 0 else 1.0]))
    strict.refresh(t, np.array([0.01]))
    assert strict.valid.tolist() == [True, False]  # validity counts non-NaN samples per feature
    assert strict.scale[0] == FLOOR_LOG and np.isnan(strict.z(np.array([1.0, 1.0]))[1])


def test_baseline_tail_scale_for_zero_inflated_samples():
    small = FeatureCfg(baseline_s=86_400, min_baseline_samples=100)
    bl = Baseline(small, np.array([FLOOR_LOG, 0.05]), is_log=np.array([True, False]))
    t = T0 + 86_400 * S
    for k in range(200):
        log_x = 1.0 if k % 20 == 0 else 0.0  # 5 % ones: median = IQR = 0
        ratio_x = (1.0 if k % 100 == 0 else -1.0) if k % 50 == 0 else 0.0  # 1 % at +1, 1 % at -1
        bl.add(t - (200 - k) * 60 * S, np.array([log_x, ratio_x]))
    bl.refresh(t, np.array([0.01]))
    X = np.array([[1.0 if k % 20 == 0 else 0.0] for k in range(200)])
    assert bl.med.tolist() == [0.0, 0.0]
    assert bl.scale[0] == max((np.quantile(X, 0.99) - 0.0) / 2.326, FLOOR_LOG)
    assert bl.scale[0] > FLOOR_LOG  # the tail term, not the floor, sets the scale
    assert bl.scale[1] == pytest.approx(2.0 / (2 * 2.576), rel=0.05)
    assert bl.z(np.array([1.0, 1.0])).max() < 3  # one ordinary event is not extreme


def test_cusum_step():
    assert cusum_step(0.0, 3.0) == 2.5
    assert cusum_step(1.0, float("nan")) == 0.5
    assert cusum_step(0.2, -5.0) == 0.0
    assert cusum_step(0.0, 1e9) == 9.5  # z clipped to 10
    assert cusum_step(99.0, 10.0) == 100.0


# ---------------------------------------------------------------- windows
def test_window_sums_exact():  # 1 trade/s, quote 2.0
    w = compute_windows(bars_const(hours=8), cfg, 1.0)
    assert w["n_trades_60"][-1] == 60 and w["quote_vol_300"][-1] == 600.0
    assert w["n_trades_1800"][-1] == 1800 and w["buy_imb_10"][-1] == 0.0


def test_row_grid():
    w = compute_windows(bars_const(hours=8), cfg, 1.0)
    t = w["t_end_us"].to_numpy()
    assert t[0] == T0 + 1800 * S  # largest window fully inside the bars
    assert t[-1] == T0 + 8 * H  # = last bar start + 1 s
    assert np.all(np.diff(t) == 10 * S) and np.all(t % (10 * S) == 0)
    assert w.columns == ["symbol", "t_end_us", *feature_columns(cfg)]
    assert w.schema["symbol"] == pl.String and w.schema["t_end_us"] == pl.Int64
    assert all(w.schema[c] == pl.Float64 for c in feature_columns(cfg))


def test_row_grid_unaligned_start():
    start = T0 + 7 * S  # bars do not start on a stride boundary
    tr = mk_trades([start + 100], 1.0, 1.0, "buy")
    w = compute_windows(bars_of(tr, start, start + 3600 * S), cfg, 1.0)
    t = w["t_end_us"].to_numpy()
    assert t[0] == T0 + 1810 * S and t[-1] == start + 3600 * S - 7 * S


def test_trade_at_t_end_excluded():
    t_end = T0 + 2400 * S
    bars = bars_of(mk_trades([t_end], 100.0, 1.0, "buy"), T0, T0 + H)
    w = compute_windows(bars, cfg, 1.0)
    assert row(w, t_end)["n_trades_10"] == 0 and row(w, t_end + 10 * S)["n_trades_10"] == 1


def test_close_semantics_at_window_edges():
    t_end = T0 + 2400 * S
    tr = mk_trades([T0 + 100 * S, t_end - 1, t_end], [100.0, 101.0, 105.0], 1.0, "buy")
    w = compute_windows(bars_of(tr, T0, T0 + H), cfg, 1.0)
    r = row(w, t_end)
    assert r["n_trades_10"] == 1  # the trade 1 µs before t_end is inside
    assert r["ret_10"] == pytest.approx(math.log(101.0 / 100.0), rel=1e-12)
    assert r["jump_10"] == 101.0 / 100.0 - 1
    r2 = row(w, t_end + 10 * S)
    assert r2["ret_10"] == pytest.approx(math.log(105.0 / 101.0), rel=1e-12)
    # first row: close_{t-1800} would be bar -1 -> NaN, never wraps around
    first = w.row(0, named=True)
    assert math.isnan(first["ret_1800"]) and math.isnan(first["jump_1800"])


def test_z_nan_until_6h_of_samples():
    zc = [c for c in feature_columns(cfg) if c.startswith("z_")]
    w5 = compute_windows(bars_const(hours=5), cfg, 1.0)
    assert w5.select(zc).to_numpy().size and np.isnan(w5.select(zc).to_numpy()).all()
    w7 = compute_windows(bars_const(hours=7), cfg, 1.0)
    assert np.isfinite(w7.select(zc).to_numpy()[-1]).all()
    # samples every 60 s from the first row (T0 + 1800 s): the refresh at T0 + 6.5 h is the first
    # with 360 samples; z uses the latest refresh, so the row before it (refresh with 355) is NaN
    t_first = T0 + 1800 * S + 360 * 60 * S
    assert (t_first - T0) % (300 * S) == 0
    assert math.isnan(row(w7, t_first - 10 * S)["z_n_trades_10"])
    assert math.isnan(row(w7, t_first - 300 * S)["z_n_trades_10"])
    assert math.isfinite(row(w7, t_first)["z_n_trades_10"])
    # validity is per feature: ret_1800 of the first row (a sample) is NaN -> 359 valid at 6.5 h
    assert math.isnan(row(w7, t_first)["z_ret_1800"])
    assert math.isfinite(row(w7, t_first + 300 * S)["z_ret_1800"])


_SAT: dict = {}


def satoshi_windows() -> pl.DataFrame:
    """Clean sparse BTC pair: price in {100, 101} sat, 1 trade / 5 min, ~0.001 BTC, 30 h."""
    if not _SAT:
        rng = np.random.default_rng(7)
        n = 30 * 12
        ts = T0 + np.arange(n, dtype=np.int64) * 300 * S + rng.integers(0, 300 * S, n)
        price = rng.choice([100e-8, 101e-8], n)
        side = np.where(rng.random(n) < 0.5, "buy", "sell")
        tr = mk_trades(ts, price, 0.001 / price, side, symbol="ABCBTC")
        _SAT["w"] = compute_windows(bars_of(tr, T0, T0 + 30 * H, "ABCBTC"), cfg, 1e-4)
    return _SAT["w"]


def _hot_fraction(w: pl.DataFrame, side: str) -> float:
    z = np.nan_to_num(w.select(pl.col("^z_.*$")).to_numpy())  # NaN z -> 0 (not hot)
    scored = np.isfinite(w.select(pl.col("^z_.*$")).to_numpy()).any(axis=1)
    assert scored.sum() > 1000
    hot = {"high": z >= 6, "abs": np.abs(z) >= 6}[side].any(axis=1)
    return float(hot[scored].mean())


def test_satoshi_grid_sparse_pair():  # price in {100, 101} sat, 1 trade / 5 min, ~0.001 BTC
    w = satoshi_windows()
    z = w.select(pl.col("^z_(ret|jump)_.*$")).to_numpy()
    assert np.isfinite(z).sum() > 1000
    assert np.nanmax(np.abs(z)) < 50  # not pinned at clip by tick bounce
    assert np.nanmax(np.abs(w.select(pl.col("^z_ret_.*$")).to_numpy())) < 5
    # zero-inflated baselines (median = IQR = 0): one ordinary trade must not look extreme
    assert _hot_fraction(w, "high") < 0.01
    assert w["cusum_imb"].max() < 8 and w["cusum_ret"].max() < 8  # stealth rule threshold


@pytest.mark.xfail(
    strict=True,
    reason="fix round 1: the prescribed one-sided upper tail for log features leaves EMPTY "
    "windows of a regular sparse pair at z <= -6 (measured 18.5% of clean rows, mainly "
    "z_quote_vol_300 / z_buy_quote_1800); needs a controller ruling on a lower-tail scale",
)
def test_satoshi_no_extreme_abs_z():
    assert _hot_fraction(satoshi_windows(), "abs") < 0.01


def bars_btc_burst():
    """0.01 BTC/min pair (Poisson, 4 trades/min) for 8 h, then 2 BTC in one minute."""
    rng = np.random.default_rng(3)
    dur_s = 9 * 3600
    n = rng.poisson(4 * dur_s / 60)
    ts = np.sort(T0 + rng.integers(0, dur_s * S, n))
    t_burst = T0 + 8 * H
    ts = ts[(ts < t_burst) | (ts >= t_burst + 60 * S)]
    burst = t_burst + np.sort(rng.integers(0, 60 * S, 40))
    all_ts = np.sort(np.concatenate([ts, burst]))
    is_burst = (all_ts >= t_burst) & (all_ts < t_burst + 60 * S)
    price = 1e-6
    quote = np.where(is_burst, 0.05, 0.0025 * rng.uniform(0.8, 1.2, len(all_ts)))
    side = np.where(is_burst | (rng.random(len(all_ts)) < 0.5), "buy", "sell")
    tr = mk_trades(all_ts, price, quote / price, side, symbol="ABCBTC")
    return bars_of(tr, T0, T0 + dur_s * S, "ABCBTC"), t_burst + 60 * S


def test_btc_pump_z_reaches_rule_threshold():  # 2 BTC/min burst on a 0.01 BTC/min pair
    bars, t_peak = bars_btc_burst()
    w = compute_windows(bars, cfg, 1e-4)
    assert row(w, t_peak)["z_quote_vol_60"] >= 6
    # and the same pair's normal minutes are not flagged
    pre = w.filter(pl.col("t_end_us") < t_peak - 60 * S)["z_quote_vol_60"].drop_nans()
    assert pre.len() > 100 and pre.max() < 6


def test_cusum_on_10s_rows():  # single +3 sigma 10 s move adds <= 2.5, not x6
    n = 8 * 3600
    t_move = T0 + 7 * H + 5 * S  # bar where the price steps up once and stays
    ts = T0 + np.arange(n, dtype=np.int64) * S + S // 2
    price = np.where(ts >= t_move, 100.0 * math.exp(3e-4), 100.0)
    w = compute_windows(bars_of(mk_trades(ts, price, 0.02, "buy"), T0, T0 + n * S), cfg, 1.0)
    z = w["z_ret_10"].to_numpy()
    jump_rows = np.flatnonzero(np.abs(z) > 0.5)
    assert len(jump_rows) == 1  # the step is seen by exactly one 10 s row
    assert 2.9 < z[jump_rows[0]] < 3.1
    c = w["cusum_ret"].to_numpy()
    assert c.max() <= 2.5 + 1e-6 and c.max() > 2.4
    assert np.max(np.diff(c)) <= 2.5 + 1e-6


@settings(deadline=None, max_examples=10)
@given(st.integers(min_value=1, max_value=3600))
def test_causal(extra_s):  # appending future bars never changes existing rows
    t_cut = T0 + 13 * H
    base, full_tr = _causal_data()
    future_end = t_cut + extra_s * S
    full = compute_windows(bars_of(full_tr, T0, future_end), cfg, 1.0)
    old = full.filter(pl.col("t_end_us") <= t_cut)
    assert old["t_end_us"].to_list() == base["t_end_us"].to_list()
    assert np.array_equal(values(old), values(base), equal_nan=True)


_CAUSAL: dict = {}


def _causal_data():
    if not _CAUSAL:
        t_cut = T0 + 13 * H
        tr = random_sparse_trades(14, seed=11)
        # an adversarial pump starting exactly at the cut: any look-ahead shows up
        pump = mk_trades(t_cut + np.arange(0, 600 * S, 50_000), 150.0, 50.0, "buy")
        tr = pl.concat(
            [
                tr.filter(pl.col("ts_us") < t_cut),
                pump,
                tr.filter(pl.col("ts_us") >= t_cut + 600 * S),
            ]
        )
        tr = tr.with_columns(pl.int_range(1, tr.height + 1).cast(pl.Int64).alias("trade_id"))
        base = compute_windows(bars_of(tr, T0, t_cut), cfg, 1.0)
        v = values(base)
        assert np.isfinite(base["z_quote_vol_60"].to_numpy()).any()
        assert np.isfinite(base["kk_vol_ratio"].to_numpy()).any()
        assert np.isfinite(v).any()
        _CAUSAL["v"] = (base, tr)
    return _CAUSAL["v"]


def test_no_time_of_day_columns():
    cols = feature_columns(cfg)
    bad = ("hour", "minute", "tod", "time_of_day", "weekday", "dow", "second")
    assert not [c for c in cols if any(b in c for b in bad)]
    # features are invariant to shifting the clock by a multiple of the refresh period
    tr = random_sparse_trades(8, seed=5)
    shift = 3 * H + 5 * 60 * S
    shifted = tr.with_columns(pl.col("ts_us") + shift, pl.col("recv_us") + shift)
    a = compute_windows(bars_of(tr, T0, T0 + 8 * H), cfg, 1.0)
    b = compute_windows(bars_of(shifted, T0 + shift, T0 + shift + 8 * H), cfg, 1.0)
    assert (b["t_end_us"] - a["t_end_us"]).unique().to_list() == [shift]
    assert np.isfinite(values(a)).any()
    assert np.array_equal(values(a), values(b), equal_nan=True)


def test_incremental_kernel_equals_batch():
    """The stream contract: bars fed one by one (or in chunks) give bit-identical rows."""
    tr = random_sparse_trades(13, seed=21, rate_per_s=0.5)
    bars = bars_of(tr, T0 + 3 * S, T0 + 13 * H)
    batch = compute_windows(bars, cfg, 1.0)
    arr = BarArrays.from_frame(bars)

    k = Kernel(cfg, 1.0)
    rows, t_ends = [], []
    for i in range(len(arr.t_us)):
        out = k.push_bar(
            int(arr.t_us[i]),
            int(arr.n_trades[i]),
            float(arr.buy_quote[i]),
            float(arr.sell_quote[i]),
            float(arr.quote_vol[i]),
            float(arr.high[i]),
            float(arr.close[i]),
            int(arr.rush_orders[i]),
            int(arr.pingpong[i]),
            int(arr.n_round[i]),
        )
        if out is not None:
            t_ends.append(out[0])
            rows.append(out[1])
    assert t_ends == batch["t_end_us"].to_list()
    assert np.array_equal(np.vstack(rows), values(batch), equal_nan=True)
    assert np.isfinite(batch["z_ret_10"].to_numpy()).any()
    assert np.isfinite(batch["kk_price_ratio"].to_numpy()).any()

    # chunked extend (random chunk sizes) gives the same rows too
    k2 = Kernel(cfg, 1.0)
    rng = np.random.default_rng(0)
    i, got_t, got = 0, [], []
    while i < len(arr.t_us):
        j = min(len(arr.t_us), i + int(rng.integers(1, 5000)))
        t, v = k2.extend(arr.slice(i, j))
        got_t.extend(t.tolist())
        got.append(v)
        i = j
    assert got_t == t_ends
    assert np.array_equal(np.vstack(got), values(batch), equal_nan=True)
    assert k2.row_dict(got[-1][-1])["n_trades_10"] == batch["n_trades_10"][-1]


def test_kernel_rejects_gaps():
    k = Kernel(cfg, 1.0)
    k.push_bar(T0, 0, 0.0, 0.0, 0.0, math.nan, math.nan, 0, 0, 0)
    with pytest.raises(ValueError):
        k.push_bar(T0 + 2 * S, 0, 0.0, 0.0, 0.0, math.nan, math.nan, 0, 0, 0)


def test_short_or_empty_bars():
    bars = bars_const(hours=1).head(1000)  # shorter than the largest window
    w = compute_windows(bars, cfg, 1.0)
    assert w.height == 0 and w.columns == ["symbol", "t_end_us", *feature_columns(cfg)]
    assert compute_windows(bars.head(0), cfg, 1.0).height == 0


def test_kk_ratio():
    # steady 1 trade/s for 14 h, then the 30 min before the last row trade 5x the quote
    n = 14 * 3600
    ts = T0 + np.arange(n, dtype=np.int64) * S + S // 2
    hot = ts >= T0 + 14 * H - 1800 * S
    tr = mk_trades(
        ts,
        np.where(hot, 110.0, 100.0),
        np.where(hot, 5.0, 1.0) / np.where(hot, 110.0, 100.0) * 100.0,
        "buy",
    )
    w = compute_windows(bars_of(tr, T0, T0 + n * S), cfg, 1.0)
    last = w.row(-1, named=True)
    assert last["kk_vol_ratio"] == pytest.approx(5.0)
    assert last["kk_price_ratio"] == pytest.approx(1.1)
    early = w.filter(pl.col("t_end_us") < T0 + 1800 * S + 12 * H)
    assert early["kk_vol_ratio"].is_nan().all()  # 24 x 30 min of history not yet there


def test_kk_zero_history_mean_is_nan():
    # one trade at the start, silence, then trades in the last 30 min: 12 h history mean = 0
    n = 14 * 3600
    ts = np.concatenate([[T0 + S], T0 + np.arange(n - 1800, n, dtype=np.int64) * S])
    w = compute_windows(bars_of(mk_trades(ts, 100.0, 1.0, "buy"), T0, T0 + n * S), cfg, 1.0)
    last = w.row(-1, named=True)
    assert last["quote_vol_1800"] > 0 and math.isnan(last["kk_vol_ratio"])
    assert last["kk_price_ratio"] == pytest.approx(1.0)


def test_kernel_rejects_bad_grid():
    for bad in (
        {"stride_s": 7},  # 1800 % 7 != 0
        {"baseline_sample_s": 65},
        {"baseline_refresh_s": 305},
    ):
        with pytest.raises(ValueError):
            Kernel(FeatureCfg.model_construct(**{**FeatureCfg().model_dump(), **bad}), 1.0)


@pytest.mark.slow
def test_one_symbol_day_under_5s():  # 2 days of bars, rows for D only
    day = T0 + 86_400 * S
    tr = random_sparse_trades(48, seed=2, rate_per_s=3.0)
    bars = bars_of(tr, T0, T0 + 48 * H)
    assert bars.height == 172_800
    t = time.perf_counter()
    w = compute_windows(bars, cfg, 1.0)
    dt = time.perf_counter() - t
    d = w.filter((pl.col("t_end_us") >= day) & (pl.col("t_end_us") < day + 86_400 * S))
    assert d.height == 8640
    print(f"compute_windows 2 days: {dt:.2f} s")
    assert dt < 5.0
