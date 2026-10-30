"""An outage must not enter the baseline as zero activity; see `StreamPipeline._on_gap`."""

import asyncio
from datetime import date
from typing import Any

import numpy as np
import polars as pl
import pytest

from idsai.ai.chief import Chief
from idsai.bus import Bus
from idsai.config import load_settings
from idsai.contracts.detector import Detector
from idsai.data.replay import Replayer
from idsai.data.store import Store
from idsai.detection.features.kernel import QUANTILES, Baseline, Kernel
from idsai.detection.stats import CusumDetector
from idsai.pipeline.stream import StreamPipeline
from tests.pipeline.test_stream import NTrades
from tests.synth import day_us, make_trades

S = 1_000_000
H = 3600 * S
SYM = "TESTBTC"
T0 = day_us(date(2025, 3, 2))
GAP = 6 * H
RESUME = T0 + 7 * H + 30 * S  # + 1800 s is not a sampling time, so every sample has a full window
MEDIAN = list(QUANTILES).index(0.5)


def cheap_chief(cfg: Any, symbol: str = SYM) -> Chief:
    """A Chief that alerts on a burst of trades in 10 s, with no fitting."""
    dets: list[Detector] = [NTrades(), CusumDetector()]
    chief = Chief(dets, cfg.chief, "h", {})
    cal = pl.DataFrame(
        {
            "symbol": [symbol] * 2000,
            "t_end_us": T0 + np.arange(2000, dtype=np.int64) * 10 * S,
            "score__stat.ntrades": np.resize(np.arange(10.0), 2000),
            "score__stat.cusum": np.zeros(2000),
        }
    )
    chief.calibrate(cal)
    chief.tau = 2.0
    return chief


@pytest.fixture(scope="module")
def cfg():
    return load_settings()


@pytest.fixture(scope="module")
def chief(cfg):
    return cheap_chief(cfg)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "data", tmp_path / "data" / "a.duckdb") as s:
        yield s


@pytest.fixture(scope="module")
def trades():
    return make_trades(date(2025, 3, 2), 1, symbol=SYM, seed=4, rate=0.5).filter(
        pl.col("ts_us") < RESUME + H
    )


async def play(df: pl.DataFrame, start_us: int | None = None):
    """Trades and ticks of `df` at max speed, without its preroll_done."""
    async for ev in Replayer(df, speed=0, start_us=start_us):
        if ev[0] != "preroll_done":
            yield ev


def samples(pipe: StreamPipeline) -> tuple[np.ndarray, np.ndarray]:
    b = pipe._windows[SYM].kernel.baseline
    return b._t.view()[b._start :].copy(), b._x.view()[b._start :].copy()


# ------------------------------------------------------------------ Kernel(baseline=)
def test_kernel_can_continue_an_existing_baseline(cfg):
    first = Kernel(cfg.features, 1e-4)
    second = Kernel(cfg.features, 1e-4, baseline=first.baseline)
    assert second.baseline is first.baseline


def test_kernel_refuses_a_baseline_of_another_shape(cfg):
    with pytest.raises(ValueError):
        Kernel(cfg.features, 1e-4, baseline=Baseline(cfg.features, np.array([0.25, 0.25])))


# ------------------------------------------------------------------ the baseline across a gap
@pytest.fixture(scope="module")
def gap_runs(cfg, chief, trades, tmp_path_factory):
    """The same data three ways, all unscored: with a 6 h gap, without any outage, and with the
    outage filled by empty bars (what happens if nobody says "gap")."""
    a = trades.filter(pl.col("ts_us") < RESUME)
    b = trades.filter(pl.col("ts_us") >= RESUME)
    later = b.with_columns(pl.col("ts_us") + GAP, pl.col("recv_us") + GAP)

    async def with_gap():
        last = 0
        async for ev in play(a):
            last = ev[1] if ev[0] == "tick" else last
            yield ev
        yield "gap", (last, RESUME + GAP)
        async for ev in play(later, start_us=RESUME + GAP):
            yield ev

    tmp = tmp_path_factory.mktemp("gap")
    out: dict[str, Any] = {}
    sources = (
        ("gap", with_gap()),
        ("whole", play(trades)),
        ("filled", play(pl.concat([a, later]))),
    )
    with Store(tmp / "d", tmp / "d" / "a.duckdb") as st:
        for name, src in sources:
            pipe = StreamPipeline(cfg, st, [], chief, Bus(), "replay")
            asyncio.run(pipe.run(src))
            out[name] = pipe
    out["a_last_tick"] = int(a["ts_us"].max()) // S * S + S
    return out


def test_no_baseline_sample_inside_the_gap(gap_runs):
    t, _ = samples(gap_runs["gap"])
    inside = t[(t > gap_runs["a_last_tick"]) & (t < RESUME + GAP + 1800 * S)]
    assert inside.size == 0
    assert t[0] < RESUME and t[-1] > RESUME + GAP  # both sides are in there


def test_filling_the_outage_would_pollute_the_baseline(gap_runs):
    """The control: without the event the same data gets hours of zero-activity samples."""
    ft, fx = samples(gap_runs["filled"])
    t, x = samples(gap_runs["gap"])
    i = gap_runs["gap"]._windows[SYM].kernel.raw_cols.index("n_trades_10")
    in_gap = (ft > gap_runs["a_last_tick"] + 1800 * S) & (ft < RESUME + GAP)
    assert in_gap.sum() >= 300
    assert np.all(fx[in_gap, i] == 0.0)
    assert np.nanmedian(fx[:, i]) < np.nanmedian(x[:, i])


def test_baseline_after_the_gap_equals_the_data_with_the_gap_never_played(gap_runs):
    """The reference is one contiguous run. Its samples of the first 1800 s after the resume are
    the only ones a restarted kernel cannot have measured: they read bars before the resume."""
    t, x = samples(gap_runs["gap"])
    rt, rx = samples(gap_runs["whole"])
    keep = (rt <= gap_runs["a_last_tick"]) | (rt >= RESUME + 1800 * S)
    rt, rx = rt[keep], rx[keep]
    shifted = np.where(t > RESUME, t - GAP, t)
    assert shifted.tolist() == rt.tolist()
    assert np.allclose(x, rx, atol=1e-9, equal_nan=True)

    b = gap_runs["gap"]._windows[SYM].kernel.baseline
    assert b.refreshed_at > RESUME + GAP + 1800 * S
    before = t < b.refreshed_at
    want = np.nanquantile(rx[before], QUANTILES[MEDIAN], axis=0, method="linear")
    assert b.valid.any()
    assert np.allclose(b.med[b.valid], want[b.valid], atol=1e-9)


# ------------------------------------------------------------------ rows and alerts across a gap
def burst_at(df: pl.DataFrame, t_us: int, first_id: int) -> pl.DataFrame:
    ts = t_us + np.arange(60, dtype=np.int64) * 100_000
    return df.head(60).with_columns(
        pl.Series("ts_us", ts),
        pl.Series("recv_us", ts),
        pl.Series("trade_id", first_id + np.arange(60, dtype=np.int64)),
    )


@pytest.fixture
def gap_with_bursts(cfg, chief, trades, store):
    """1 h, a 6 h gap, 1 h: a burst while the new kernel warms up, and another one after it."""
    a = trades.filter(pl.col("ts_us") < T0 + H)
    resume = T0 + H + GAP
    base = trades.filter((pl.col("ts_us") >= T0 + H) & (pl.col("ts_us") < T0 + 2 * H)).with_columns(
        pl.col("ts_us") + GAP, pl.col("recv_us") + GAP
    )
    cold = burst_at(base, resume + 600 * S, 10_000_000)
    warm = burst_at(base, resume + 2400 * S, 20_000_000)
    b = pl.concat([base, cold, warm]).sort(["ts_us", "trade_id"])
    last = int(a["ts_us"].max()) // S * S + S
    seen: dict[str, Any] = {}
    pipe = StreamPipeline(cfg, store, chief.detectors, chief, Bus(maxsize=100_000), "live")

    async def source():
        yield "preroll_done", T0
        async for ev in play(a):
            yield ev
        seen["baseline"] = pipe._windows[SYM].kernel.baseline
        seen["samples"] = pipe.baseline_samples(SYM)
        yield "gap", (last, resume)
        assert pipe._windows[SYM].kernel.baseline is seen["baseline"]
        assert pipe._windows[SYM].kernel.n_bars == 0
        async for ev in play(b, start_us=resume):
            yield ev

    asyncio.run(pipe.run(source()))
    return {"pipe": pipe, "last": last, "resume": resume, "seen": seen}


def test_no_rows_until_the_new_kernel_is_warm(gap_with_bursts):
    g = gap_with_bursts
    ts = [p["t_end_us"] for p in g["pipe"].series(SYM)]
    assert ts == sorted(ts)
    assert not [t for t in ts if g["last"] < t < g["resume"] + 1800 * S]
    assert ts[-1] > g["resume"] + 1800 * S
    assert g["seen"]["samples"] > 0


def test_no_alert_before_the_new_kernel_is_warm(gap_with_bursts, store):
    g = gap_with_bursts
    alerts = store.list_alerts()
    assert alerts and {a.provenance for a in alerts} == {"live"}
    firsts = sorted(a.window[0] for a in alerts)
    assert not [t for t in firsts if g["last"] < t < g["resume"] + 1800 * S]
    assert any(t >= g["resume"] + 2400 * S for t in firsts)  # the later burst is still found


def test_open_episode_survives_the_gap_and_closes_with_its_own_window(cfg, chief, store):
    """ChiefStream continues: the episode stays open over the gap, the next row closes it."""
    quiet = make_trades(date(2025, 3, 2), 1, symbol=SYM, seed=4, rate=0.5)
    end = T0 + 2000 * S
    a = quiet.filter(pl.col("ts_us") < end)
    a = pl.concat([a, burst_at(a, end - 15 * S, 10_000_000)]).sort(["ts_us", "trade_id"])
    resume = end + GAP
    b = quiet.filter((pl.col("ts_us") >= end) & (pl.col("ts_us") < end + 1810 * S)).with_columns(
        pl.col("ts_us") + GAP, pl.col("recv_us") + GAP
    )
    bus = Bus(maxsize=100_000)
    q = bus.subscribe("alerts")
    pipe = StreamPipeline(cfg, store, chief.detectors, chief, bus, "live")
    open_at_gap = []

    async def source():
        yield "preroll_done", T0
        async for ev in play(a):
            yield ev
        yield "gap", (end, resume)
        open_at_gap.append(not q.empty() and not any(x.status == "closed" for x in q._queue))
        async for ev in play(b, start_us=resume):
            yield ev

    asyncio.run(pipe.run(source()))
    assert open_at_gap == [True]
    got = []
    while not q.empty():
        got.append(q.get_nowait())
    before_gap = [x for x in got if x.status == "closed" and x.window[0] < end]
    assert len(before_gap) == 1 and before_gap[0].window[1] <= end
    assert before_gap[0].ts_event_us < end  # still that episode, not one merged across the gap


def test_trades_in_grace_at_the_outage_are_not_lost(cfg, chief, store):
    """Bars still waiting for the grace period close before the kernel is rebuilt."""
    t = make_trades(date(2025, 3, 2), 1, symbol=SYM, seed=4, rate=0.5).filter(
        pl.col("ts_us") < T0 + 1805 * S
    )
    bus = Bus(maxsize=100_000)
    q = bus.subscribe("bars")
    pipe = StreamPipeline(cfg, store, chief.detectors, chief, bus, "live")
    last = int(t["ts_us"].max()) // S * S + S

    async def source():
        yield "preroll_done", T0
        async for ev in play(t):
            yield ev
        yield "gap", (last, last + GAP)

    asyncio.run(pipe.run(source()))
    bars = []
    while not q.empty():
        bars.append(q.get_nowait())
    assert bars[-1]["t_us"] == last - S
    assert sum(b["n_trades"] for b in bars) == t.height


def test_gap_before_any_data_just_sets_the_origin(cfg, chief, store):
    pipe = StreamPipeline(cfg, store, [], chief, Bus(), "live")

    async def source():
        yield "gap", (T0, T0 + GAP)
        yield "preroll_done", T0 + GAP

    asyncio.run(pipe.run(source()))
    assert pipe._origin_us == T0 + GAP


def test_baseline_samples(cfg, chief, store, trades):
    pipe = StreamPipeline(cfg, store, [], chief, Bus(), "replay")
    assert pipe.baseline_samples(SYM) == 0
    asyncio.run(pipe.run(play(trades.filter(pl.col("ts_us") < T0 + 2 * H))))
    assert pipe.baseline_samples("NOSUCH") == 0
    assert pipe.baseline_samples(SYM) > 80  # one per 60 s from the first row at 1800 s, up to 2 h


def test_a_replay_never_emits_gap(trades):
    async def kinds():
        return {ev[0] async for ev in Replayer(trades.head(2000), speed=0)}

    assert "gap" not in asyncio.run(kinds())
