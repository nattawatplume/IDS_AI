"""M4 `bear_raid` injection: price down, sell-taker rush clusters, deterministic (SEMI-SYN)."""

import dataclasses

import numpy as np
import polars as pl

from idsai.detection.simulator import inject, load_scenario_ranges, sample_scenario
from tests.detection.test_simulator import T0, H, S, steady

RANGES = load_scenario_ranges()


def _scenario(seed=11, **kw):
    sc = sample_scenario("bear_raid", "test", T0 + 3 * H, np.random.default_rng(seed), RANGES)
    return dataclasses.replace(sc, **kw)


def test_bear_raid_scenario_fields():
    sc = _scenario()
    assert (sc.family, sc.technique_id, sc.sub_technique) == ("bear_raid", "MM-T006", "MM-T006.001")
    rg = RANGES["bear_raid"]["test"]
    assert rg["impact"][0] <= sc.impact <= rg["impact"][1] and sc.rush_size >= 3


def test_bear_raid_pushes_price_down_with_sell_taker_rush_fills():
    real = steady(hours=6)
    sc = _scenario(duration_s=120.0, impact=0.05, dump_tau_s=300.0, rush_size=4)
    out, lab = inject(real, sc, base_quote_per_s=1.0, mode="replay")
    sim = out.filter(pl.col("source") == "sim")
    assert sim.height > 0 and (sim["taker_side"] == "sell").all()
    t0, t_peak = sc.t_start_us, sc.t_start_us + 120 * S
    assert sim["ts_us"].min() >= t0 and sim["ts_us"].max() < t_peak
    # clusters of exactly rush_size fills per millisecond
    sizes = sim.group_by(pl.col("ts_us") // 1000).len()["len"]
    assert (sizes == 4).all()
    # the real price falls by about `impact` at the end of the raid, then recovers
    real_out = out.filter(pl.col("source") != "sim")
    before = real_out.filter(pl.col("ts_us") < t0)["price"]
    at_low = real_out.filter((pl.col("ts_us") >= t_peak - 4 * S) & (pl.col("ts_us") < t_peak))
    later = real_out.filter(pl.col("ts_us") > t_peak + 3000 * S)["price"]
    assert (before == 100.0).all()
    assert np.allclose(at_low["price"].to_numpy(), 95.0, atol=0.3)
    assert later.max() <= 100.0 and later.min() > 99.0
    assert (out["price"] <= 100.0).all()  # never above the pre-raid price
    bars = out.filter((pl.col("ts_us") >= t0) & (pl.col("ts_us") < t_peak))
    assert bars["price"].min() < 96.0
    assert lab["technique_id"].to_list() == ["MM-T006"]
    assert lab["family"].to_list() == ["bear_raid"]
    assert lab["provenance"].to_list() == ["SEMI-SYN"]


def test_bear_raid_is_deterministic_for_a_seed():
    real = steady(hours=6)
    a, la = inject(real, _scenario(seed=3), 1.0, "replay")
    b, lb = inject(real, _scenario(seed=3), 1.0, "replay")
    c, _ = inject(real, _scenario(seed=4), 1.0, "replay")
    assert a.equals(b) and la.equals(lb) and not a.equals(c)


def test_pump_burst_injection_unchanged():
    """The pump path still injects taker buys and pushes the price up."""
    real = steady(hours=6)
    sc = sample_scenario("pump_burst", "test", T0 + 3 * H, np.random.default_rng(5), RANGES)
    out, _ = inject(real, sc, 1.0, "replay")
    sim = out.filter(pl.col("source") == "sim")
    assert (sim["taker_side"] == "buy").all()
    assert out["price"].max() > 100.0 and out["price"].min() == 100.0
