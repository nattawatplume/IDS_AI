"""Whole stream path against the batch path: same alerts, same scored rows, exact floats."""

import asyncio
import json
import math
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from idsai.ai.chief import Chief
from idsai.bus import Bus
from idsai.config import config_hash, load_settings
from idsai.contracts.detector import Detector, Evidence
from idsai.data.replay import Replayer
from idsai.data.store import Store
from idsai.detection.registry import build_detectors
from idsai.detection.simulator import Scenario, base_quote_rate
from idsai.detection.stats import CusumDetector
from idsai.pipeline import stream as stream_mod
from idsai.pipeline.batch import load_day_windows, score_windows
from idsai.pipeline.stream import StreamPipeline
from tests.synth import day_us, make_trades

S = 1_000_000
SYM = "TESTBTC"
D = date(2025, 3, 2)
D_US = day_us(D)
BURST_US = D_US + 15 * 3600 * S


def settings(tmp) -> Any:
    return load_settings(
        overrides={
            "paths": {"data_dir": str(tmp / "data"), "app_db": str(tmp / "data" / "app.duckdb")},
            "ae": {"epochs": 2, "hidden": 8, "bottleneck": 4},
        }
    )


def burst() -> Scenario:
    return Scenario(
        family="pump_burst",
        technique_id="MM-T001",
        sub_technique="MM-T001.001",
        t_start_us=BURST_US,
        duration_s=120.0,
        impact=0.15,
        vol_mult=30.0,
        rush_size=5,
        dump_tau_s=300.0,
        seed=7,
        region="test",
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Fitted default detectors and a calibrated Chief, from the clean day of the fixture."""
    tmp = tmp_path_factory.mktemp("stream")
    cfg = settings(tmp)
    # the last trade stays inside the final 10 s of the day, so the stream ends on the same row
    # as the batch path (which reads to the end of the day and keeps `t_end_us < D + 1`)
    trades = make_trades(D - timedelta(days=1), 2, symbol=SYM, seed=9, rate=0.25).filter(
        pl.col("ts_us") < D_US + 86_400 * S - 2 * S
    )
    store = Store(cfg.paths.data_dir, cfg.paths.app_db)
    store.write_trades(trades)
    clean = load_day_windows(store, SYM, D, cfg)
    dets = build_detectors(cfg.detectors, cfg)
    for d in dets:
        d.fit(clean.windows)
    scored = score_windows(clean.windows, dets)
    chief = Chief(dets, cfg.chief, config_hash(cfg), {"ai.autoencoder_trade": "test"})
    chief.calibrate(scored)
    chief.fit_threshold(chief.fuse(scored), budget_per_symbol_day=10.0)
    yield {"cfg": cfg, "trades": trades, "store": store, "dets": dets, "chief": chief}
    store.close()


def replay_run(world, monkeypatch=None):
    """Replay day D (pre-roll D - 1) with the burst injected; capture what got scored."""
    bus = Bus(maxsize=100_000)
    qa, qb, qs = bus.subscribe("alerts"), bus.subscribe("bars"), bus.subscribe("scores")
    cap: list[pl.DataFrame] = []
    real = stream_mod.score_windows

    def spy(frame, dets):
        out = real(frame, dets)
        cap.append(out)
        return out

    if monkeypatch is not None:
        monkeypatch.setattr(stream_mod, "score_windows", spy)
    prev = world["trades"].filter(pl.col("ts_us") < D_US)
    r = Replayer(world["trades"], speed=0, start_us=D_US, preroll_from_us=D_US - 86_400 * S)
    label = r.inject(burst(), base_quote_rate(prev), "replay")
    pipe = StreamPipeline(
        world["cfg"], world["store"], world["dets"], world["chief"], bus, "replay"
    )
    asyncio.run(pipe.run(r))
    return {"pipe": pipe, "bus": bus, "qa": qa, "qb": qb, "qs": qs, "cap": cap, "label": label}


@pytest.fixture(scope="module")
def run(world):
    mp = pytest.MonkeyPatch()
    try:
        yield replay_run(world, mp)
    finally:
        mp.undo()


def drain(q) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


@pytest.fixture(scope="module")
def alerts(run):
    return drain(run["qa"])


@pytest.fixture(scope="module")
def batch_inj(world):
    r = load_day_windows(world["store"], SYM, D, world["cfg"], scenario=burst())
    scored = score_windows(r.windows, world["dets"])
    return {"scored": scored, "fused": world["chief"].fuse(scored), "label": r.labels}


# ------------------------------------------------------------------ the brief's pipeline test
def test_burst_gives_an_mm_t001_alert_overlapping_the_label(run, alerts, world):
    lab = run["label"]
    lo, hi = lab["t_start_us"][0], lab["t_end_us"][0]
    hits = [
        a
        for a in alerts
        if a.technique_id.startswith("MM-T001") and a.window[0] <= hi and a.window[1] >= lo
    ]
    assert hits, [(a.technique_id, a.window) for a in alerts]
    stored = world["store"].get_alert(hits[0].alert_id)
    assert stored is not None and stored.status == "closed"
    assert stored.provenance == "replay"


def test_no_publication_before_preroll_done(run, alerts):
    assert all(a.ts_event_us >= D_US and a.window[0] >= D_US for a in alerts)
    bars = drain(run["qb"])
    assert bars and min(b["t_us"] for b in bars) >= D_US - S
    scores = drain(run["qs"])
    assert scores and min(s["t_end_us"] for s in scores) >= D_US
    assert all(f["t_end_us"] >= D_US for f in run["cap"][0].iter_rows(named=True))


def test_p99_latency_under_2s_and_one_sample_per_row(run):
    lat = np.array(run["pipe"].latency_ms)
    assert run["pipe"].latency_ms.maxlen == 10_000
    assert len(lat) == len(run["cap"])
    p99 = float(np.percentile(lat, 99))
    print(f"\nstream p99 latency: {p99:.2f} ms over {len(lat)} rows")
    assert p99 < 2000.0


# ------------------------------------------------------------------ F2: stream == batch
def test_scored_rows_equal_the_batch_path_exactly(run, batch_inj):
    got = pl.concat(run["cap"])
    want = batch_inj["scored"]
    assert got.columns == want.columns
    assert got.height == want.height
    for c in want.columns:
        if want[c].dtype == pl.Float64:
            assert np.array_equal(got[c].to_numpy(), want[c].to_numpy(), equal_nan=True), c
        else:
            assert got[c].to_list() == want[c].to_list(), c


def test_final_alerts_equal_chief_episodes_exactly(run, alerts, world, batch_inj):
    want = world["chief"].episodes(batch_inj["fused"], provenance="replay")
    assert want, "the fixture must produce at least one episode"
    closed = [a for a in alerts if a.status == "closed"]
    key = lambda a: (a.symbol, a.ts_event_us)  # noqa: E731
    assert sorted(closed, key=key) == sorted(want, key=key)
    for a, b in zip(sorted(closed, key=key), sorted(want, key=key), strict=True):
        assert a.alert_id == b.alert_id and a.window == b.window
        assert a.technique_id == b.technique_id and a.fused_p == b.fused_p


def test_series_f_and_lanes_equal_the_batch_fusion(run, world, batch_inj):
    series = run["pipe"].series(SYM)
    fused = batch_inj["fused"]
    assert [p["t_end_us"] for p in series] == fused["t_end_us"].to_list()
    assert [p["F"] for p in series] == fused["F"].to_list()
    for layer in ("rule", "stat", "ai"):
        names = [d.name for d in world["dets"] if d.layer == layer]
        ps = fused.select([f"p__{n}" for n in names]).to_numpy()
        want = np.max(-np.log10(ps), axis=1) + 0.0
        assert np.array_equal([p[f"lane_{layer}"] for p in series], want), layer
    assert {*series[0]} == {"t_end_us", "close", "F", "lane_rule", "lane_stat", "lane_ai"}


# ------------------------------------------------------------------ F5
def test_every_default_detector_is_row_independent(world, batch_inj):
    w = batch_inj["scored"].select(
        [c for c in batch_inj["scored"].columns if not c.startswith("score__")]
    )
    idx = np.linspace(0, w.height - 1, 200).astype(int).tolist()
    for d in world["dets"]:
        batch = np.asarray(d.score(w), dtype=np.float64)
        # out of order and alone: a stateful detector would differ
        alone = np.array([d.score(w[i : i + 1])[0] for i in reversed(idx)])[::-1]
        assert np.array_equal(batch[idx], alone, equal_nan=True), d.name


def test_default_detectors_include_every_layer(world):
    assert {d.layer for d in world["dets"]} == {"rule", "stat", "ai"}
    assert "stat.cusum" in {d.name for d in world["dets"]}


# ------------------------------------------------------------------ fast fixtures
class NTrades(Detector):
    """Cheap detector so seek tests can alert within a few hours of data."""

    name = "stat.ntrades"
    layer = "stat"
    technique_ids = ["ANY"]
    requires = ["n_trades_10"]

    def score(self, windows: pl.DataFrame) -> np.ndarray:
        return windows["n_trades_10"].to_numpy().astype(np.float64)

    def evidence(self, row: dict[str, Any]) -> Evidence:
        return Evidence.from_items([("n_trades_10", float(row["n_trades_10"]), 1.0)])


@pytest.fixture(scope="module")
def small():
    """4 h of trades with bursts, and a Chief alerting on them, without touching disk data."""
    cfg = load_settings()
    t0 = day_us(D)
    base = make_trades(D, 1, symbol=SYM, seed=4, rate=0.5).filter(
        pl.col("ts_us") < t0 + 4 * 3600 * S
    )
    extra = []
    for k, at in enumerate((3600, 7200, 10800)):
        ts = t0 + at * S + np.arange(60, dtype=np.int64) * 100_000
        one = base.head(60).with_columns(
            pl.Series("ts_us", ts),
            pl.Series("recv_us", ts),
            pl.Series("trade_id", 10_000_000 * (k + 1) + np.arange(60, dtype=np.int64)),
        )
        extra.append(one)
    trades = pl.concat([base, *extra]).sort(["ts_us", "trade_id"])
    dets: list[Detector] = [NTrades(), CusumDetector()]
    chief = Chief(dets, cfg.chief, "h", {})
    cal = pl.DataFrame(
        {
            "symbol": [SYM] * 2000,
            "t_end_us": t0 + np.arange(2000, dtype=np.int64) * 10 * S,
            "score__stat.ntrades": np.resize(np.arange(10.0), 2000),
            "score__stat.cusum": np.zeros(2000),
        }
    )
    chief.calibrate(cal)
    chief.tau = 2.0
    return {"cfg": cfg, "trades": trades, "dets": dets, "chief": chief, "t0": t0}


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "data", tmp_path / "data" / "a.duckdb") as s:
        yield s


def pipe_for(small, store, mode="replay", bus=None):
    return StreamPipeline(small["cfg"], store, small["dets"], small["chief"], bus or Bus(), mode)


def test_mode_must_be_known(small, store):
    with pytest.raises(ValueError):
        pipe_for(small, store, mode="paper")
    assert pipe_for(small, store, mode="sim").mode == "sim"


def test_alert_provenance_is_the_mode(small, store):
    p = pipe_for(small, store, mode="sim")
    asyncio.run(p.run(Replayer(small["trades"], speed=0)))
    al = store.list_alerts()
    assert al and {a.provenance for a in al} == {"sim"}
    assert {a.status for a in al} == {"closed"}


def test_preroll_is_features_only(small, store, monkeypatch):
    calls = []
    real = stream_mod.score_windows
    monkeypatch.setattr(stream_mod, "score_windows", lambda f, d: calls.append(1) or real(f, d))
    bus = Bus()
    qs = [bus.subscribe(t) for t in ("bars", "scores", "alerts")]
    t0 = small["t0"]

    async def source():
        r = Replayer(small["trades"], speed=0, start_us=t0 + 4 * 3600 * S, preroll_from_us=t0)
        async for k, v in r:
            if k == "preroll_done":
                return  # stop right at the marker: all of it was pre-roll
            yield k, v

    p = pipe_for(small, store, bus=bus)
    asyncio.run(p.run(source()))
    assert calls == [] and not p.latency_ms and not store.list_alerts()
    assert all(q.empty() for q in qs) and p.series(SYM) == []


def fresh_rows(small, store, start, preroll_from):
    r = Replayer(small["trades"], speed=0, start_us=start, preroll_from_us=preroll_from)
    p = pipe_for(small, store)
    asyncio.run(p.run(r))
    return p.series(SYM)


@pytest.mark.parametrize("target_h, at_h", [(1.5, 3.0), (3.0, 1.5)], ids=["backward", "forward"])
def test_seek_through_the_pipeline_rebuilds_state(small, tmp_path, target_h, at_h):
    t0, target, at = (
        small["t0"],
        small["t0"] + int(target_h * 3600) * S,
        small["t0"] + int(at_h * 3600) * S,
    )
    bus = Bus(maxsize=200_000)
    qb = bus.subscribe("bars")
    seen = []

    with Store(tmp_path / "a", tmp_path / "a" / "a.duckdb") as st:
        r = Replayer(small["trades"], speed=0)
        p = pipe_for(small, st, bus=bus)

        async def source():
            sought = False
            async for k, v in r:
                yield k, v
                if k == "tick" and v == at and not sought:
                    sought = True
                    r.seek(target)
                if k == "seek":
                    seen.append(len(drain(qb)))
                    assert p.series(SYM) == []  # the ring buffer restarts with the new timeline
                if k == "preroll_done" and sought:
                    assert drain(qb) == []  # nothing was published during the re-pre-roll

        asyncio.run(p.run(source()))
        after_seek = p.series(SYM)

    with Store(tmp_path / "b", tmp_path / "b" / "b.duckdb") as st:
        fresh = fresh_rows(small, st, target, t0)
    assert seen and after_seek == fresh  # exactly like a fresh pipeline started at the target
    assert after_seek[0]["t_end_us"] >= target
    ts = [q["t_end_us"] for q in after_seek]
    assert ts == sorted(ts)


def test_seek_closes_the_open_episode_so_no_stale_open_alert_remains(small, store):
    t0 = small["t0"]

    async def source():
        r = Replayer(small["trades"], speed=0)
        sought = False
        async for k, v in r:
            yield k, v
            if k == "tick" and v == t0 + 3600 * S + 5 * S and not sought:
                sought = True
                r.seek(t0 + 3 * 3600 * S)

    asyncio.run(pipe_for(small, store).run(source()))
    al = store.list_alerts(limit=1000)
    assert al and {a.status for a in al} == {"closed"}


def test_series_is_bounded_and_strict_json(small, store):
    p = pipe_for(small, store)
    assert p._ring_maxlen == 86_400 // small["cfg"].features.stride_s
    asyncio.run(p.run(Replayer(small["trades"], speed=0)))
    s = p.series(SYM)
    assert s
    json.dumps(s, allow_nan=False)
    assert p.series("NOSUCH") == []
    assert all(q["lane_ai"] is None for q in s)  # no AI detector in this Chief
    assert all(q["F"] is None or math.isfinite(q["F"]) for q in s)


def test_multi_symbol_rows_and_alerts_match_per_symbol_runs(small, tmp_path):
    other = small["trades"].with_columns(pl.lit("OTHERBTC").alias("symbol"))
    both = pl.concat([small["trades"], other])
    with Store(tmp_path / "a", tmp_path / "a" / "a.duckdb") as st:
        p = pipe_for(small, st)
        asyncio.run(p.run(Replayer(both, speed=0)))
        ser_a, ser_b = p.series(SYM), p.series("OTHERBTC")
    with Store(tmp_path / "b", tmp_path / "b" / "b.duckdb") as st:
        one = pipe_for(small, st)
        asyncio.run(one.run(Replayer(small["trades"], speed=0)))
        want = one.series(SYM)
    assert ser_a == want and ser_b == want


def test_live_mode_grace_closes_trailing_bars_at_end(small, store):
    """In live mode bars wait for the grace period; the end of the source drains them."""
    grace = small["cfg"].live.bar_grace_s
    assert grace > 0

    t0 = small["t0"]
    sub = small["trades"].filter(pl.col("ts_us") < t0 + 1900 * S + 500_000)

    async def source():
        r = Replayer(sub, speed=0)
        async for ev in r:
            yield ev

    p = pipe_for(small, store, mode="live")
    asyncio.run(p.run(source()))
    last_tick = sub["ts_us"].max() // S * S + S
    assert p.series(SYM)[-1]["t_end_us"] == last_tick // (10 * S) * (10 * S)


def test_unknown_event_kind_raises(small, store):
    async def source():
        yield ("nonsense", 1)

    with pytest.raises(ValueError):
        asyncio.run(pipe_for(small, store).run(source()))
