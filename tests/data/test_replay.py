import asyncio
import math
import time

import numpy as np
import polars as pl
import pytest

from idsai.contracts.events import Trade
from idsai.contracts.frames import TRADES
from idsai.data.replay import Replayer
from idsai.detection.simulator import Scenario
from idsai.detection.simulator.guard import EthicsError

S = 1_000_000
H = 3600 * S
T0 = 1_740_787_200 * S
SYM = "TESTBTC"


def frame(ts, ids=None, symbol=SYM) -> pl.DataFrame:
    n = len(ts)
    price = 100.0 + np.arange(n) * 0.01
    return pl.DataFrame(
        {
            "ts_us": np.asarray(ts, dtype=np.int64),
            "recv_us": np.asarray(ts, dtype=np.int64),
            "symbol": [symbol] * n,
            "price": price,
            "qty": np.full(n, 2.0),
            "quote_qty": price * 2.0,
            "taker_side": ["buy"] * n,
            "trade_id": np.arange(1, n + 1) if ids is None else np.asarray(ids),
            "n_fills": np.ones(n, dtype=np.int32),
            "source": ["archive"] * n,
        },
        schema=TRADES,
    )


def burst(t_start: int, duration_s: float = 30.0) -> Scenario:
    return Scenario(
        family="pump_burst",
        technique_id="MM-T001",
        sub_technique="MM-T001.001",
        t_start_us=t_start,
        duration_s=duration_s,
        impact=0.1,
        vol_mult=10.0,
        rush_size=3,
        dump_tau_s=60.0,
        seed=3,
        region="test",
    )


def collect(r: Replayer) -> list[tuple]:
    async def main():
        return [ev async for ev in r]

    return asyncio.run(main())


def summary(events) -> list[tuple]:
    return [(k, v.trade_id if isinstance(v, Trade) else v) for k, v in events]


def test_trades_ordered_by_ts_then_id_and_tick_follows_its_trades():
    df = frame([T0 + 5, T0 + 5, T0 + 900, T0 + S + 3], ids=[9, 2, 1, 4])
    ev = collect(Replayer(df.sample(fraction=1.0, shuffle=True, seed=1), speed=0))
    assert summary(ev) == [
        ("preroll_done", T0),
        ("trade", 2),
        ("trade", 9),
        ("trade", 1),
        ("tick", T0 + S),
        ("trade", 4),
        ("tick", T0 + 2 * S),
    ]
    assert isinstance(ev[1][1], Trade) and ev[1][1].symbol == SYM


def test_empty_seconds_still_tick():
    ev = collect(Replayer(frame([T0, T0 + 3 * S + 1]), speed=0))
    assert [v for k, v in ev if k == "tick"] == [T0 + i * S for i in range(1, 5)]


def test_preroll_marker_once_after_the_preroll_and_before_the_first_live_tick():
    df = frame([T0 + i * S for i in range(10)])
    ev = summary(collect(Replayer(df, speed=0, start_us=T0 + 5 * S, preroll_from_us=T0 + 2 * S)))
    assert [k for k, _ in ev].count("preroll_done") == 1
    i = ev.index(("preroll_done", T0 + 5 * S))
    assert ev[:i] == [
        ("trade", 3),
        ("tick", T0 + 3 * S),
        ("trade", 4),
        ("tick", T0 + 4 * S),
        ("trade", 5),
    ]
    # every event before the marker is before start; the row ending at start follows it
    assert ev[i + 1] == ("tick", T0 + 5 * S)
    assert ev[i + 2] == ("trade", 6)


def test_no_preroll_requested_still_emits_the_marker_first():
    ev = collect(Replayer(frame([T0 + 3 * S, T0 + 4 * S]), speed=0, start_us=T0 + 3 * S))
    assert ev[0] == ("preroll_done", T0 + 3 * S)
    assert [k for k, _ in ev].count("preroll_done") == 1


def test_empty_input_emits_marker_and_ends():
    assert collect(Replayer(frame([]), speed=0)) == [("preroll_done", 0)]


def test_preroll_after_start_is_rejected():
    with pytest.raises(ValueError):
        Replayer(frame([T0]), start_us=T0, preroll_from_us=T0 + S)


def test_seek_backward_emits_seek_then_a_24h_preroll_then_the_marker():
    df = frame([T0 + i * H for i in range(30)])

    async def main():
        r = Replayer(df, speed=0)
        out, sought = [], False
        async for ev in r:
            out.append(ev)
            if not sought and ev == ("tick", T0 + 29 * H + S):
                r.seek(T0 + 26 * H)
                assert r.cursor_us == T0 + 26 * H
                sought = True
            elif sought and ev[0] == "preroll_done":
                break
        return out

    ev = summary(asyncio.run(main()))
    i = ev.index(("seek", T0 + 26 * H))
    assert ev[i + 1] == ("trade", 3)  # the trade at T0 + 2 h == seek - 24 h comes first
    assert ev[-1] == ("preroll_done", T0 + 26 * H)
    assert ("trade", 2) not in ev[i:]
    assert sum(1 for k, _ in ev if k == "preroll_done") == 2


def test_seek_clamps_to_the_data_range():
    df = frame([T0 + 10 * S, T0 + 20 * S])
    r = Replayer(df, speed=0)
    r.seek(T0 - 100 * S)
    assert r.cursor_us == T0 + 10 * S
    r.seek(T0 + 10_000 * S)
    assert r.cursor_us == T0 + 21 * S


def test_seek_forward_skips_ahead():
    df = frame([T0 + i * S for i in range(5)])

    async def main():
        r = Replayer(df, speed=0)
        out = []
        async for ev in r:
            out.append(ev)
            if ev == ("tick", T0 + S) and ("seek", T0 + 3 * S) not in out:
                r.seek(T0 + 3 * S)
        return out

    ev = summary(asyncio.run(main()))
    i = ev.index(("seek", T0 + 3 * S))
    # re-pre-roll from the clamped data start, so features can be rebuilt from scratch
    assert ev[i + 1] == ("trade", 1)
    assert ("preroll_done", T0 + 3 * S) in ev[i:]
    assert ev[-1] == ("tick", T0 + 5 * S)


def test_pause_stops_events_and_resume_continues():
    df = frame([T0 + i * S for i in range(4)])

    async def main():
        r = Replayer(df, speed=0)
        seen = []

        async def consume():
            async for ev in r:
                seen.append(ev)

        r.pause()
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        assert seen == []
        r.resume()
        await asyncio.wait_for(task, 2)
        return seen

    assert len(asyncio.run(main())) == 1 + 4 + 4


def test_pause_applies_mid_stream():
    df = frame([T0 + i * S for i in range(5)])

    async def main():
        r = Replayer(df, speed=0)
        seen = []

        async def consume():
            async for ev in r:
                seen.append(ev)
                if len(seen) == 3:
                    r.pause()

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        assert len(seen) == 3
        r.resume()
        await asyncio.wait_for(task, 2)
        return seen

    assert len(asyncio.run(main())) == 1 + 5 + 5


def test_inject_changes_only_trades_after_the_cursor_and_returns_the_label():
    df = frame([T0 + i * S for i in range(40)])

    async def main():
        r = Replayer(df, speed=0)
        before, after, lab = [], [], None
        async for k, v in r:
            if k == "tick" and v == T0 + 10 * S:
                assert r.cursor_us == v
                lab = r.inject(burst(v, 5.0), 1.0, "replay")
            elif k == "trade":
                (after if lab is not None else before).append(v)
        return before, after, lab

    before, after, lab = asyncio.run(main())
    orig = {r["trade_id"]: r for r in df.iter_rows(named=True)}
    assert len(before) == 10
    assert all(t.price == orig[t.trade_id]["price"] for t in before)
    assert any(t.trade_id < 0 and t.source == "sim" for t in after)
    assert any(t.price != orig[t.trade_id]["price"] for t in after if t.trade_id > 0)
    assert all(t.ts_us >= T0 + 10 * S for t in after)
    assert [t.ts_us for t in after] == sorted(t.ts_us for t in after)
    assert lab.height == 1 and lab["t_start_us"][0] == T0 + 10 * S
    assert lab["technique_id"][0] == "MM-T001"
    assert df.equals(frame([T0 + i * S for i in range(40)]))  # the caller's frame is untouched


def test_inject_mid_second_keeps_same_timestamp_trades_consistent():
    df = frame([T0, T0 + 5, T0 + 5, T0 + 10 * S], ids=[1, 2, 3, 4])

    async def main():
        r = Replayer(df, speed=0)
        out = []
        async for k, v in r:
            if k == "trade":
                out.append(v.trade_id)
                if v.trade_id == 2:
                    assert r.cursor_us == T0 + 6  # the unsent twin at T0 + 5 keeps its place
                    with pytest.raises(ValueError):
                        r.inject(burst(T0 + 5, 1.0), 1.0, "replay")
                    r.inject(burst(T0 + 6, 1.0), 1.0, "replay")
        return out

    out = asyncio.run(main())
    assert out[:3] == [1, 2, 3]
    assert out[-1] == 4 and min(out) < 0


def test_inject_before_cursor_raises():
    df = frame([T0 + i * S for i in range(10)])

    async def main():
        r = Replayer(df, speed=0)
        async for k, v in r:
            if k == "tick" and v == T0 + 5 * S:
                with pytest.raises(ValueError):
                    r.inject(burst(T0 + 2 * S, 1.0), 1.0, "replay")
                with pytest.raises(ValueError):
                    r.inject(burst(v - 1, 1.0), 1.0, "replay")

    asyncio.run(main())


def test_cursor_during_preroll_is_the_start_so_preroll_stays_untouched():
    df = frame([T0 + i * S for i in range(10)])
    r = Replayer(df, speed=0, start_us=T0 + 5 * S, preroll_from_us=T0)
    assert r.cursor_us == T0 + 5 * S
    with pytest.raises(ValueError):
        r.inject(burst(T0 + 3 * S, 1.0), 1.0, "sim")


@pytest.mark.parametrize("mode", ["live", "LIVE", ""])
def test_inject_refuses_non_replay_modes_and_leaves_everything_untouched(mode):
    df = frame([T0 + i * S for i in range(5)])
    want = collect(Replayer(df, speed=0))
    r = Replayer(df, speed=0)
    c = r.cursor_us
    with pytest.raises(EthicsError):
        r.inject(burst(T0 + 2 * S, 1.0), 1.0, mode)
    assert r.cursor_us == c
    assert collect(r) == want


def test_inject_needs_a_symbol_on_multi_symbol_input():
    df = pl.concat([frame([T0, T0 + S]), frame([T0, T0 + S], symbol="OTHERBTC")])
    r = Replayer(df, speed=0)
    with pytest.raises(ValueError):
        r.inject(burst(T0 + S, 0.5), 1.0, "sim")
    lab = r.inject(burst(T0 + S, 0.5), 1.0, "sim", symbol="OTHERBTC")
    assert lab["symbol"][0] == "OTHERBTC"
    sims = [v for k, v in collect(r) if k == "trade" and v.source == "sim"]
    assert sims and {t.symbol for t in sims} == {"OTHERBTC"}


def test_same_input_gives_the_same_events_at_any_speed():
    rng = np.random.default_rng(1)
    df = frame(np.sort(T0 + rng.integers(0, 3 * S, 500)))
    assert collect(Replayer(df, speed=0)) == collect(Replayer(df, speed=1e4))


def test_finite_speed_paces_by_event_time_and_max_speed_does_not():
    df = frame([T0, T0 + 2 * S])
    t = time.monotonic()
    collect(Replayer(df, speed=20.0))  # 3 s of event time / 20 = 0.15 s
    assert time.monotonic() - t >= 0.1
    t = time.monotonic()
    collect(Replayer(df, speed=0))
    assert time.monotonic() - t < 0.1


def test_preroll_is_not_paced():
    df = frame([T0, T0 + 3600 * S, T0 + 3601 * S])
    t = time.monotonic()
    ev = collect(Replayer(df, speed=1.0, start_us=T0 + 3601 * S, preroll_from_us=T0))
    assert time.monotonic() - t < 3.0  # 1 h of pre-roll is instant; only the tail is paced
    assert ev[-1][0] == "tick"


def test_set_speed_validates():
    r = Replayer(frame([T0]), speed=0)
    r.set_speed(2.5)
    r.set_speed(0)
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            r.set_speed(bad)
    with pytest.raises(ValueError):
        Replayer(frame([T0]), speed=-1)


def test_max_speed_still_yields_to_the_event_loop():
    df = frame(T0 + np.arange(5000) * 1000)

    async def main():
        beats = 0

        async def heartbeat():
            nonlocal beats
            while True:
                beats += 1
                await asyncio.sleep(0)

        hb = asyncio.create_task(heartbeat())
        async for _ in Replayer(df, speed=0):
            pass
        hb.cancel()
        return beats

    assert asyncio.run(main()) >= 5


def test_multi_symbol_rows_stay_ordered_per_symbol():
    a = frame([T0 + 1, T0 + S + 1], ids=[1, 2])
    b = frame([T0 + 2, T0 + S + 2], ids=[1, 2], symbol="OTHERBTC")
    ev = collect(Replayer(pl.concat([b, a]), speed=0))
    assert [(v.symbol, v.ts_us) for k, v in ev if k == "trade"] == [
        (SYM, T0 + 1),
        ("OTHERBTC", T0 + 2),
        (SYM, T0 + S + 1),
        ("OTHERBTC", T0 + S + 2),
    ]


def test_requested_range_before_the_first_trade_ticks_from_the_requested_second():
    """Parity: a batch run from the same first bar has these empty bars too."""
    ev = collect(Replayer(frame([T0 + 3 * S]), speed=0, start_us=T0 + 3 * S, preroll_from_us=T0))
    assert [v for k, v in ev if k == "tick"] == [T0 + i * S for i in range(1, 5)]
    r = Replayer(frame([T0 + 3 * S]), speed=0, start_us=T0)
    r.seek(T0 - 100 * S)
    assert r.cursor_us == T0


def test_seek_after_the_end_raises_with_the_cursor_unchanged():
    r = Replayer(frame([T0, T0 + S]), speed=0)
    collect(r)
    c = r.cursor_us
    with pytest.raises(RuntimeError):
        r.seek(T0)
    assert r.cursor_us == c


def test_seek_after_an_abandoned_and_closed_iterator_raises_with_the_cursor_unchanged():
    async def main():
        r = Replayer(frame([T0 + i * S for i in range(5)]), speed=0)
        it = aiter(r)
        async for ev in it:
            if ev[0] == "tick":
                break
        r.seek(T0)  # break alone does not close the iterator: still accepted (documented)
        await it.aclose()
        c = r.cursor_us
        with pytest.raises(RuntimeError):
            r.seek(T0 + S)
        assert r.cursor_us == c

    asyncio.run(main())
