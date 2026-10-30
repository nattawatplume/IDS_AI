"""Seek cleanup, late-symbol backlog and the multi-day batch/stream property of the stream path."""

import asyncio
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from idsai.ai.chief import Chief
from idsai.bus import Bus
from idsai.config import config_hash, load_settings
from idsai.contracts.alert import Alert
from idsai.contracts.detector import Detector, Evidence
from idsai.contracts.events import Trade
from idsai.contracts.frames import TRADES
from idsai.data.replay import Replayer
from idsai.data.store import Store
from idsai.detection.features.registry import feature_columns
from idsai.detection.registry import build_detectors
from idsai.detection.simulator import Scenario, base_quote_rate
from idsai.detection.stats import CusumDetector
from idsai.pipeline import stream as stream_mod
from idsai.pipeline.batch import load_day_windows, score_windows
from idsai.pipeline.stream import StreamPipeline
from tests.synth import day_us, make_trades

S = 1_000_000
H = 3600 * S
SYM = "TESTBTC"
D = date(2025, 3, 2)
T0 = day_us(D)


class NTrades(Detector):
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
    """4 h of trades with a 60 s burst at 1 h, 2 h and 3 h: one long-window episode each."""
    cfg = load_settings()
    base = make_trades(D, 1, symbol=SYM, seed=4, rate=0.5).filter(pl.col("ts_us") < T0 + 4 * H)
    extra = []
    for k, at in enumerate((1, 2, 3)):
        ts = T0 + at * H + np.arange(600, dtype=np.int64) * 100_000
        ids = 10_000_000 * (k + 1) + np.arange(600, dtype=np.int64)
        extra.append(
            pl.concat([base.head(100)] * 6).with_columns(
                pl.Series("ts_us", ts), pl.Series("recv_us", ts), pl.Series("trade_id", ids)
            )
        )
    trades = pl.concat([base, *extra]).sort(["ts_us", "trade_id"])
    dets: list[Detector] = [NTrades(), CusumDetector()]
    chief = Chief(dets, cfg.chief, "h", {})
    n = 2000
    chief.calibrate(
        pl.DataFrame(
            {
                "symbol": [SYM] * n,
                "t_end_us": T0 + np.arange(n, dtype=np.int64) * 10 * S,
                "score__stat.ntrades": np.resize(np.arange(40.0), n),
                "score__stat.cusum": np.zeros(n),
            }
        )
    )
    chief.tau = 2.0
    return {"cfg": cfg, "trades": trades, "dets": dets, "chief": chief}


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "s", tmp_path / "s" / "a.duckdb") as s:
        yield s


def pipe(small, store, bus=None, mode="replay"):
    return StreamPipeline(
        small["cfg"], store, small["dets"], small["chief"], bus or Bus(100_000), mode
    )


def drain(q) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def all_alerts(store) -> list[Alert]:
    return sorted(store.list_alerts(limit=10_000), key=lambda a: a.alert_id)


def clean_run(small, tmp_path, mode="replay") -> list[Alert]:
    with Store(tmp_path / "clean", tmp_path / "clean" / "c.duckdb") as st:
        asyncio.run(pipe(small, st, mode=mode).run(Replayer(small["trades"], speed=0)))
        return all_alerts(st)


def seek_when(replayer, should_seek, target):
    """Pass `replayer` through, seeking once to `target` right after `should_seek(event)`."""

    async def source():
        done = False
        async for ev in replayer:
            yield ev
            if not done and should_seek(ev):
                done = True
                replayer.seek(target)

    return source()


def test_forward_seek_mid_episode_leaves_no_row_for_that_episode(small, store):
    bus = Bus(100_000)
    qa = bus.subscribe("alerts")
    seen_open: list[Alert] = []
    reset: list[dict] = []
    r = Replayer(small["trades"], speed=0)

    async def source():
        done = False
        async for ev in r:
            yield ev
            for m in drain(qa):
                (reset if isinstance(m, dict) else seen_open).append(m)
            if not done and seen_open:
                done = True
                assert seen_open[0].status == "open"
                r.seek(T0 + int(2.5 * H))

    asyncio.run(pipe(small, store, bus).run(source()))
    victim = seen_open[0]
    assert victim.window[0] < T0 + 2 * H  # the 1 h burst
    assert store.get_alert(victim.alert_id) is None
    assert not [
        a
        for a in drain(qa) + seen_open
        if isinstance(a, Alert) and a.alert_id == victim.alert_id and a.status == "closed"
    ]
    assert reset == [
        {"event": "alerts_reset", "t_us": T0 + int(2.5 * H), "deleted": [victim.alert_id]}
    ]
    assert all(a.window[0] >= T0 + int(2.5 * H) for a in all_alerts(store))


def _closed_windows(small, tmp_path) -> list[Alert]:
    return clean_run(small, tmp_path)


def test_backward_seek_to_before_a_closed_episode_then_full_rerun_equals_clean(
    small, store, tmp_path
):
    clean = _closed_windows(small, tmp_path)
    assert len(clean) == 3
    # past the third episode; back to between the first and the second
    target = T0 + int(1.5 * H)
    last = clean[-1].window[1] + 400 * S
    r = Replayer(small["trades"], speed=0)
    p = pipe(small, store)
    asyncio.run(p.run(seek_when(r, lambda ev: ev == ("tick", last), target)))
    assert all_alerts(store) == clean


def test_backward_seek_to_the_very_start_then_rerun_equals_clean(small, store, tmp_path):
    clean = _closed_windows(small, tmp_path)
    r = Replayer(small["trades"], speed=0)
    last = clean[-1].window[1] + 400 * S
    first = r.cursor_us
    asyncio.run(pipe(small, store).run(seek_when(r, lambda ev: ev == ("tick", last), first)))
    assert all_alerts(store) == clean


def test_backward_seek_into_a_closed_episode_deletes_it_and_reports_it(small, store, tmp_path):
    clean = _closed_windows(small, tmp_path)
    mid = next(a for a in clean if T0 + 2 * H <= a.window[0] < T0 + 3 * H)
    assert mid.window[0] < mid.window[1]
    target = mid.window[1]
    last = clean[-1].window[1] + 400 * S
    bus = Bus(100_000)
    qa = bus.subscribe("alerts")
    r = Replayer(small["trades"], speed=0)
    state: dict[str, Any] = {}

    async def source():
        done = False
        async for ev in r:
            yield ev
            if not done and ev == ("tick", last):
                done = True
                r.seek(target)
            elif done and ev[0] == "seek":
                state["after_seek"] = all_alerts(store)

    asyncio.run(pipe(small, store, bus).run(source()))
    assert [a.alert_id for a in state["after_seek"]] == [clean[0].alert_id]
    (reset,) = [m for m in drain(qa) if isinstance(m, dict)]
    assert reset["event"] == "alerts_reset" and reset["t_us"] == target
    assert set(reset["deleted"]) == {clean[1].alert_id, clean[2].alert_id}
    assert store.get_alert(mid.alert_id) is None  # its re-run starts later, under another id
    assert all(a.window[0] >= target for a in all_alerts(store) if a.alert_id != clean[0].alert_id)


def test_seek_never_touches_live_or_other_provenance_rows(small, store, tmp_path):
    clean = _closed_windows(small, tmp_path)
    live = clean[0].model_copy(update={"alert_id": "live-x", "provenance": "live"})
    other = clean[0].model_copy(update={"alert_id": "sim-x", "provenance": "sim"})
    store.upsert_alert(live)
    store.upsert_alert(other)
    last = clean[-1].window[1] + 400 * S
    r = Replayer(small["trades"], speed=0)
    asyncio.run(pipe(small, store).run(seek_when(r, lambda ev: ev == ("tick", last), r.cursor_us)))
    assert store.get_alert("live-x") == live
    assert store.get_alert("sim-x") == other


def test_seek_in_a_live_pipeline_raises_and_deletes_nothing(small, store, tmp_path):
    a = _closed_windows(small, tmp_path)[0].model_copy(update={"provenance": "live"})
    store.upsert_alert(a)

    async def source():
        yield "seek", T0

    with pytest.raises(ValueError):
        asyncio.run(pipe(small, store, mode="live").run(source()))
    assert store.get_alert(a.alert_id) == a


def test_same_data_in_two_provenances_keeps_both_rows(small, store):
    asyncio.run(pipe(small, store, mode="sim").run(Replayer(small["trades"], speed=0)))
    n_sim = len(all_alerts(store))
    asyncio.run(pipe(small, store, mode="replay").run(Replayer(small["trades"], speed=0)))
    al = all_alerts(store)
    assert n_sim and len(al) == 2 * n_sim
    assert all(a.alert_id.startswith(f"{a.provenance}-{SYM}-") for a in al)


# ------------------------------------------------------------------ late-symbol backlog
def _trade_events(n_gap_s: int):
    row = make_trades(D, 1, symbol="LATEBTC", seed=2, rate=0.01).head(1)
    row = row.with_columns(
        pl.lit(T0 + n_gap_s * S + 5).cast(pl.Int64).alias("ts_us"),
        pl.lit(T0 + n_gap_s * S + 5).cast(pl.Int64).alias("recv_us"),
    )
    return Trade(**row.select(TRADES.names()).row(0, named=True))


async def _no_wait_source(gap: int):
    """Never suspends by itself: only the pipeline can let another coroutine run."""
    yield "preroll_done", T0
    yield "tick", T0 + S
    yield "trade", _trade_events(gap)
    yield "tick", T0 + gap * S + S


def _backlog_run(small, store, gap):
    p = pipe(small, store)

    async def main():
        beats = 0

        async def hb():
            nonlocal beats
            while True:
                beats += 1
                await asyncio.sleep(0)

        t = asyncio.create_task(hb())
        await asyncio.sleep(0)
        start = beats
        await p.run(_no_wait_source(gap))
        t.cancel()
        return beats - start

    return asyncio.run(main()), p.series("LATEBTC")


def test_late_symbol_backlog_yields_and_output_is_unchanged(small, store, tmp_path, monkeypatch):
    gap = 10_000
    beats, series = _backlog_run(small, store, gap)
    assert beats >= 5 and len(series) > 800
    monkeypatch.setattr(stream_mod, "BACKLOG_YIELD_BARS", 10**9)
    with Store(tmp_path / "b", tmp_path / "b" / "b.duckdb") as st:
        beats_off, same = _backlog_run(small, st, gap)
    assert beats_off < beats
    assert same == series


# ------------------------------------------------------------------ multi-day property
def test_multi_day_stream_vs_per_day_batch_property(tmp_path):
    """The stream runs on across midnight; batch restarts its kernel per day (Parity decision)."""
    d1, d2 = D, D + timedelta(days=1)
    d1_us, d2_us = day_us(d1), day_us(d2)
    cfg = load_settings(
        overrides={
            "paths": {"data_dir": str(tmp_path / "d"), "app_db": str(tmp_path / "d" / "a.db")}
        }
    )
    end = d2_us + 24 * H
    trades = make_trades(d1 - timedelta(days=1), 3, symbol=SYM, seed=9, rate=0.25).filter(
        pl.col("ts_us") < end - 2 * S
    )
    sc = Scenario(
        family="pump_burst",
        technique_id="MM-T001",
        sub_technique="MM-T001.001",
        t_start_us=d2_us + 15 * H,
        duration_s=120.0,
        impact=0.15,
        vol_mult=30.0,
        rush_size=5,
        dump_tau_s=300.0,
        seed=7,
        region="test",
    )
    with Store(cfg.paths.data_dir, cfg.paths.app_db) as store:
        store.write_trades(trades)
        clean = load_day_windows(store, SYM, d1, cfg)
        dets = build_detectors(["rules", "stat.robust_z", "stat.cusum"], cfg)
        cal = score_windows(clean.windows, dets)
        chief = Chief(dets, cfg.chief, config_hash(cfg), {})
        chief.calibrate(cal)
        chief.fit_threshold(chief.fuse(cal), budget_per_symbol_day=10.0)

        batch_d2 = load_day_windows(store, SYM, d2, cfg, scenario=sc)
        want_rows = pl.concat([clean.windows, batch_d2.windows])
        want_scored = score_windows(want_rows, dets)
        want = chief.episodes(chief.fuse(want_scored), provenance="replay")

        got_rows: list[pl.DataFrame] = []
        real = stream_mod.score_windows
        stream_mod.score_windows = lambda f, d: got_rows.append(f) or real(f, d)
        try:
            prev = trades.filter((pl.col("ts_us") >= d1_us) & (pl.col("ts_us") < d2_us))
            r = Replayer(trades, speed=0, start_us=d1_us, preroll_from_us=d1_us - 24 * H)
            r.inject(sc, base_quote_rate(prev), "replay")
            bus = Bus(100_000)
            qa = bus.subscribe("alerts")
            asyncio.run(StreamPipeline(cfg, store, dets, chief, bus, "replay").run(r))
        finally:
            stream_mod.score_windows = real
        got = [a for a in drain(qa) if a.status == "closed"]

    key = lambda a: a.ts_event_us  # noqa: E731
    assert want and [
        (a.alert_id, a.window, a.technique_id, a.fused_p) for a in sorted(got, key=key)
    ] == [(a.alert_id, a.window, a.technique_id, a.fused_p) for a in sorted(want, key=key)]
    assert any(a.technique_id == "MM-T001" and a.window[0] >= d2_us for a in got)

    got_w = pl.concat(got_rows)
    assert got_w["t_end_us"].to_list() == want_rows["t_end_us"].to_list()
    cols = feature_columns(cfg.features)
    a, b = got_w.select(cols).to_numpy(), want_rows.select(cols).to_numpy()
    t = want_rows["t_end_us"].to_numpy()
    # the first played day has the same first bar: bit-identical
    day1 = t < d2_us
    assert np.array_equal(a[day1], b[day1], equal_nan=True)
    # later days: differs in the first hour, then at most 1e-7 in every column
    later = t > d2_us + H
    assert np.array_equal(np.isnan(a[later]), np.isnan(b[later]))
    diff = np.nan_to_num(np.abs(a[later] - b[later]), nan=0.0)
    print(f"\nmulti-day max |stream - batch| beyond 1 h: {diff.max():.3e}")
    assert diff.max() <= 1e-7
    first_hour = (t >= d2_us) & (t <= d2_us + H)
    assert not np.array_equal(a[first_hour], b[first_hour], equal_nan=True)
