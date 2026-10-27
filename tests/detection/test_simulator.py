import dataclasses
import hashlib

import numpy as np
import polars as pl
import pytest

from idsai.config import FeatureCfg
from idsai.contracts.frames import LABELS, TRADES
from idsai.detection.features.bars import is_round_qty, trades_to_bars
from idsai.detection.features.windows import compute_windows
from idsai.detection.simulator import (
    EthicsError,
    Scenario,
    assert_injection_allowed,
    base_quote_rate,
    inject,
    load_scenario_ranges,
    sample_scenario,
)

S = 1_000_000
H = 3600 * S
T0 = 1_740_787_200_000_000  # 2025-03-01T00:00:00Z
SYM = "ABCUSDT"
RANGES = load_scenario_ranges()
FAMILIES = ("pump_burst", "pump_stealth", "wash_pingpong")


def mk_trades(ts, price, qty, side="buy", symbol=SYM) -> pl.DataFrame:
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


def steady(hours=6, step_s=2, price=100.0, qty=0.02):
    """One trade every `step_s` seconds at a constant price, alternating taker side."""
    n = hours * 3600 // step_s
    ts = T0 + np.arange(n, dtype=np.int64) * step_s * S + S // 2
    side = np.where(np.arange(n) % 2 == 0, "buy", "sell")
    return mk_trades(ts, price, qty, side)


T_START = T0 + 4 * H


def scenario(family="pump_burst", **kw) -> Scenario:
    base = {
        "pump_burst": dict(
            technique_id="MM-T001",
            sub_technique="MM-T001.001",
            duration_s=60.0,
            impact=0.2,
            vol_mult=10.0,
            rush_size=4,
            dump_tau_s=300.0,
        ),
        "pump_stealth": dict(
            technique_id="MM-T001",
            sub_technique="MM-T001.002",
            duration_s=1800.0,
            impact=0.05,
            vol_mult=1.5,
            rush_size=1,
            dump_tau_s=0.0,
        ),
        "wash_pingpong": dict(
            technique_id="MM-T005",
            sub_technique="MM-T005.001",
            duration_s=600.0,
            impact=0.0,
            vol_mult=3.0,
            rush_size=1,
            dump_tau_s=0.0,
        ),
    }[family]
    base.update(kw)
    return Scenario(family=family, t_start_us=T_START, seed=7, region="train", **base)


def sim_only(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("source") == "sim")


# ------------------------------------------------------------------ guard
def test_ethics_guard_modes():
    assert_injection_allowed("replay")
    assert_injection_allowed("sim")
    for bad in ("live", "LIVE", "", None, "paper"):
        with pytest.raises(EthicsError) as ei:
            assert_injection_allowed(bad)  # type: ignore[arg-type]
        msg = str(ei.value)
        assert "ห้าม" in msg and "forbidden" in msg.lower()
    assert issubclass(EthicsError, RuntimeError)


def test_inject_live_raises_before_anything_else():
    with pytest.raises(EthicsError):
        inject(steady(), scenario(), 1.0, "live")
    # the guard is the first statement: garbage inputs still give EthicsError
    with pytest.raises(EthicsError):
        inject(None, None, -1.0, "live")  # type: ignore[arg-type]


# ------------------------------------------------------------------ rates
def test_base_quote_rate_max_of_two_terms():
    ts = T0 + np.arange(100, dtype=np.int64) * S
    busy = mk_trades(ts, 100.0, 1.0)  # total 10 000 quote, median 100
    assert base_quote_rate(busy) == pytest.approx(max(10_000 / 86400, 100 / 60))
    sparse = mk_trades(T0 + np.arange(288, dtype=np.int64) * 300 * S, 100.0, 1.0)  # 1 per 5 min
    assert base_quote_rate(sparse) == pytest.approx(100 / 60)  # median term dominates
    big = mk_trades(ts, 100.0, [1.0] * 99 + [1e6])  # total term dominates
    assert base_quote_rate(big) == pytest.approx(big["quote_qty"].sum() / 86400)


def test_base_quote_rate_no_trades_raises():
    with pytest.raises(ValueError):
        base_quote_rate(pl.DataFrame(schema=TRADES))
    with pytest.raises(ValueError):
        base_quote_rate(mk_trades([T0], 100.0, 0.0))


# ------------------------------------------------------------------ sampling
def test_ranges_file_values():
    r = RANGES
    assert r["pump_burst"]["train"]["impact"] == [0.05, 0.30]
    assert r["pump_burst"]["test"]["impact"] == [0.03, 0.05]
    assert r["pump_stealth"]["test"]["duration_s"] == [5400, 7200]
    assert r["wash_pingpong"]["test"]["vol_mult"] == [1.5, 2]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("region", ["train", "test"])
def test_sampled_scenarios_inside_ranges(family, region):
    rng = np.random.default_rng(0)
    rg = RANGES[family][region]
    for _ in range(50):
        sc = sample_scenario(family, region, T_START, rng, RANGES)
        assert sc.family == family and sc.region == region and sc.t_start_us == T_START
        lo, hi = rg["duration_s"]
        assert lo <= sc.duration_s <= hi
        lo, hi = rg["vol_mult"]
        assert lo <= sc.vol_mult <= hi
        if "impact" in rg:
            lo, hi = rg["impact"]
            assert lo <= sc.impact <= hi
        else:
            assert sc.impact == 0
        if "rush_size" in rg:
            lo, hi = rg["rush_size"]
            assert isinstance(sc.rush_size, int) and lo <= sc.rush_size <= hi
        else:
            assert sc.rush_size == 1
        if "dump_tau_s" in rg:
            lo, hi = rg["dump_tau_s"]
            assert lo <= sc.dump_tau_s <= hi
        else:
            assert sc.dump_tau_s == 0
        assert isinstance(sc.seed, int)


def test_sampling_deterministic_and_technique_ids():
    a = sample_scenario("pump_burst", "train", T_START, np.random.default_rng(5), RANGES)
    b = sample_scenario("pump_burst", "train", T_START, np.random.default_rng(5), RANGES)
    c = sample_scenario("pump_burst", "train", T_START, np.random.default_rng(6), RANGES)
    assert a == b and a != c
    got = {
        f: sample_scenario(f, "train", T_START, np.random.default_rng(0), RANGES) for f in FAMILIES
    }
    ids = {f: (s.technique_id, s.sub_technique) for f, s in got.items()}
    assert ids == {
        "pump_burst": ("MM-T001", "MM-T001.001"),
        "pump_stealth": ("MM-T001", "MM-T001.002"),
        "wash_pingpong": ("MM-T005", "MM-T005.001"),
    }


# ------------------------------------------------------------------ injection: general
@pytest.mark.parametrize("family", FAMILIES)
def test_deterministic_by_seed(family):
    tr = steady()
    sc = scenario(family)
    a, la = inject(tr, sc, 1.0, "replay")
    b, lb = inject(tr, sc, 1.0, "replay")
    assert a.equals(b) and la.equals(lb)
    c, _ = inject(tr, dataclasses.replace(sc, seed=8), 1.0, "replay")
    assert not sim_only(a).equals(sim_only(c))


@pytest.mark.parametrize("family", FAMILIES)
def test_schema_sorting_ids_and_no_mutation(family):
    tr = steady()
    before = tr.clone()
    sc = scenario(family)
    out, _ = inject(tr, sc, 1.0, "sim")
    assert tr.equals(before)  # input frame untouched
    assert out.schema == TRADES
    keys = list(zip(out["ts_us"].to_list(), out["trade_id"].to_list(), strict=True))
    assert keys == sorted(keys)
    sim = sim_only(out)
    assert sim.height > 0
    assert (sim["recv_us"] == sim["ts_us"]).all() and (sim["n_fills"] == 1).all()
    assert (sim["symbol"] == SYM).all()
    ids = sim["trade_id"].to_list()  # negative, unique, -1, -2, ... in time order
    assert ids == list(range(-1, -len(ids) - 1, -1))
    assert (sim["ts_us"].diff().drop_nulls() >= 0).all()
    t_peak = T_START + int(sc.duration_s * S)
    assert sim["ts_us"].min() >= T_START and sim["ts_us"].max() < t_peak
    # real trades keep their ids and count
    real = out.filter(pl.col("source") != "sim")
    assert real["trade_id"].to_list() == tr["trade_id"].to_list()
    assert (sim["quote_qty"] == sim["price"] * sim["qty"]).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_pre_t_start_trades_unchanged(family):
    tr = steady()
    out, _ = inject(tr, scenario(family), 1.0, "replay")
    pre_in = tr.filter(pl.col("ts_us") < T_START)
    pre_out = out.filter(pl.col("ts_us") < T_START)
    assert pre_in.equals(pre_out)


def test_label_row():
    sc = scenario("pump_stealth")
    _, lab = inject(steady(), sc, 1.0, "replay")
    assert lab.schema == LABELS and lab.height == 1
    row = lab.row(0, named=True)
    key = f"{SYM}|{T_START}|pump_stealth|{sc.seed}"
    assert row["label_id"] == hashlib.sha1(key.encode()).hexdigest()[:12]
    assert row["source"] == "sim" and row["symbol"] == SYM
    assert row["technique_id"] == "MM-T001" and row["family"] == "pump_stealth"
    assert row["scenario_seed"] == sc.seed and row["provenance"] == "SEMI-SYN"
    assert row["t_start_us"] == T_START
    assert row["t_end_us"] == T_START + int(sc.duration_s * 1e6)
    _, lab2 = inject(steady(), scenario("wash_pingpong"), 1.0, "replay")
    assert lab2["technique_id"][0] == "MM-T005" and lab2["family"][0] == "wash_pingpong"
    assert lab2["t_end_us"][0] == T_START + 600 * S


# ------------------------------------------------------------------ pump
def test_peak_price_reaches_impact():
    sc = scenario("pump_burst", impact=0.2)
    out, _ = inject(steady(), sc, 1.0, "replay")  # constant real price 100.0
    pre_price = 100.0
    t_peak = T_START + int(sc.duration_s * S)
    near = out.filter((pl.col("ts_us") >= t_peak - 5 * S) & (pl.col("ts_us") < t_peak + S))
    assert near["price"].max() >= (1 + 0.9 * sc.impact) * pre_price
    # halfway up the ramp the price sits around +half the impact
    mid = out.filter((pl.col("ts_us") >= T_START + 28 * S) & (pl.col("ts_us") < T_START + 32 * S))
    assert 100 * (1 + 0.35 * sc.impact) < mid["price"].mean() < 100 * (1 + 0.65 * sc.impact)


def test_dump_decay_after_peak():
    sc = scenario("pump_burst", impact=0.2, dump_tau_s=300.0)
    out, _ = inject(steady(), sc, 1.0, "replay")
    t_peak = T_START + 60 * S
    at_tau = out.filter(
        (pl.col("source") == "archive")
        & (pl.col("ts_us") >= t_peak + 298 * S)
        & (pl.col("ts_us") < t_peak + 302 * S)
    )
    assert at_tau["price"].mean() == pytest.approx(100 * (1 + 0.2 / np.e), rel=1e-3)
    late = out.filter((pl.col("source") == "archive") & (pl.col("ts_us") >= t_peak + 3000 * S))
    assert late["price"].max() == pytest.approx(100.0, abs=0.01)


@pytest.mark.parametrize("rush", [3, 8])
def test_burst_clusters_share_one_ms(rush):
    out, _ = inject(steady(), scenario("pump_burst", rush_size=rush), 1.0, "replay")
    sim = sim_only(out)
    assert (sim["taker_side"] == "buy").all()
    counts = sim.group_by(pl.col("ts_us") // 1000).len()["len"]
    assert (counts == rush).all()
    assert sim["ts_us"].n_unique() == sim.height  # distinct us inside the ms
    bars = trades_to_bars(sim, SYM, T_START, T_START + 60 * S)
    assert bars["rush_orders"].sum() == sim.height // rush


def test_stealth_has_no_clusters_and_volume_matches():
    sc = scenario("pump_stealth", vol_mult=1.5, duration_s=1800.0)
    base = 2.0
    out, _ = inject(steady(), sc, base, "replay")
    sim = sim_only(out)
    assert sim.group_by(pl.col("ts_us") // 1000).len()["len"].max() == 1
    expected = sc.vol_mult * base * sc.duration_s
    assert 0.8 * expected <= sim["quote_qty"].sum() <= 1.2 * expected


def test_burst_volume_matches_expected():
    sc = scenario("pump_burst", vol_mult=10.0, duration_s=120.0, rush_size=5)
    out, _ = inject(steady(), sc, 2.0, "replay")
    expected = 10.0 * 2.0 * 120.0
    assert 0.8 * expected <= sim_only(out)["quote_qty"].sum() <= 1.2 * expected


def test_prices_snap_to_observed_tick():
    rng = np.random.default_rng(0)
    n = 4000
    ts = T0 + np.arange(n, dtype=np.int64) * 2 * S + 1
    price = np.round(100.0 + 0.01 * np.cumsum(rng.choice([-1, 0, 1], n)), 2)
    tr = mk_trades(ts, price, 0.02, np.where(np.arange(n) % 2 == 0, "buy", "sell"))
    sc = dataclasses.replace(
        scenario("pump_burst", impact=0.1, dump_tau_s=100.0), t_start_us=T0 + 3000 * S
    )
    out, _ = inject(tr, sc, 1.0, "replay")
    after = out.filter(pl.col("ts_us") >= sc.t_start_us)
    k = after["price"].to_numpy() / 0.01
    assert np.allclose(k, np.rint(k), atol=1e-6)
    assert after["price"].min() >= 0.01
    # modified real trades keep quote_qty consistent with price * qty
    assert np.allclose(out["quote_qty"].to_numpy(), out["price"].to_numpy() * out["qty"].to_numpy())


def test_prices_never_below_one_tick():
    ts = T0 + np.arange(200, dtype=np.int64) * S
    price = np.where(np.arange(200) % 2 == 0, 0.01, 0.02)  # tick 0.01
    tr = mk_trades(ts, price, 5.0, "sell")
    sc = dataclasses.replace(scenario("pump_stealth", impact=0.0), t_start_us=T0 + 50 * S)
    out, _ = inject(tr, sc, 1.0, "replay")
    assert out["price"].min() >= 0.01 - 1e-12


# ------------------------------------------------------------------ wash pingpong
def test_pingpong_pairs_equal_qty_opposite_sides_same_price():
    tr = steady(price=100.0, qty=0.02)
    out, _ = inject(tr, scenario("wash_pingpong"), 1.0, "replay")
    sim = sim_only(out)
    assert sim.height >= 2 and sim.height % 2 == 0
    p = sim["price"].to_numpy()
    q = sim["qty"].to_numpy()
    ts = sim["ts_us"].to_numpy()
    sd = sim["taker_side"].to_list()
    half = len(sd) // 2
    assert (p == 100.0).all()  # current real price, no impact
    assert is_round_qty(q).all()
    assert (q[0::2] == q[1::2]).all() and len(set(q.tolist())) == 1
    assert sd[0::2] == ["buy"] * half and sd[1::2] == ["sell"] * half
    gap = ts[1::2] - ts[0::2]
    assert (gap >= 1000).all() and (gap <= 20_000).all()
    assert q[0] == 0.02  # nearest {1,2,5} x 10^k to the typical 0.02
    # real trades are unchanged (no price modification at all)
    assert out.filter(pl.col("source") == "archive").equals(tr)


def test_pingpong_feature_pairs_visible_to_bars():
    out, _ = inject(steady(), scenario("wash_pingpong"), 1.0, "replay")
    n_pairs = sim_only(out).height // 2
    bars = trades_to_bars(out, SYM, T_START, T_START + 600 * S)
    assert bars["pingpong"].sum() >= n_pairs


def test_sparse_pair_injection_adds_quote_volume():
    # one trade per 5 min for two days on a sparse pair
    n = 2 * 288
    ts = T0 + np.arange(n, dtype=np.int64) * 300 * S + S // 2
    tr = mk_trades(ts, 50.0, 0.37, np.where(np.arange(n) % 2 == 0, "buy", "sell"))
    base = base_quote_rate(tr.filter(pl.col("ts_us") < T0 + 24 * H))
    for family in FAMILIES:
        sc = dataclasses.replace(scenario(family), t_start_us=T0 + 30 * H)
        out, _ = inject(tr, sc, base, "replay")
        expected = sc.vol_mult * base * sc.duration_s
        added = sim_only(out)["quote_qty"].sum()
        assert added >= 0.5 * expected, (family, added, expected)
        assert added <= 2.5 * expected, (family, added, expected)


# ------------------------------------------------------------------ end-to-end visibility
@pytest.mark.slow
def test_injected_pump_visible_in_features_z_quote_vol():
    rng = np.random.default_rng(11)
    dur = 48 * H
    n = rng.poisson(0.5 * dur / S)
    ts = np.sort(T0 + rng.integers(0, dur, n))
    price = np.round(100.0 + 0.01 * np.cumsum(rng.choice([-1, 0, 0, 1], n)), 2)
    qty = np.round(rng.lognormal(0.0, 1.0, n), 3) + 0.001
    side = np.where(rng.random(n) < 0.5, "buy", "sell")
    real = mk_trades(ts, price, qty, side)
    t_start = T0 + 36 * H
    day = real.filter((pl.col("ts_us") >= t_start - 24 * H) & (pl.col("ts_us") < t_start))
    base = base_quote_rate(day)
    sc = sample_scenario("pump_burst", "train", t_start, np.random.default_rng(3), RANGES)
    sc = dataclasses.replace(sc, duration_s=60.0, vol_mult=20.0)
    out, _ = inject(real, sc, base, "replay")
    cfg = FeatureCfg()
    w = compute_windows(trades_to_bars(out, SYM, T0, T0 + dur), cfg, 1.0)
    t_peak = t_start + 60 * S
    assert w.filter(pl.col("t_end_us") == t_peak)["z_quote_vol_60"][0] > 6
    clean = compute_windows(trades_to_bars(real, SYM, T0, T0 + dur), cfg, 1.0)
    assert clean.filter(pl.col("t_end_us") == t_peak)["z_quote_vol_60"][0] < 6
