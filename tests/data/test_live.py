"""Live sensor against a fake server: real sockets on loopback, fake clock and backoff sleep."""

import asyncio
import http
import json
import math
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import numpy as np
import polars as pl
import pytest
import websockets

from idsai.ai.auditor import Auditor
from idsai.bus import Bus
from idsai.config import load_settings
from idsai.contracts.events import Trade
from idsai.data import live
from idsai.data.live import GAP_RESTART_S, LiveTradeSensor, parse_trade, warmup_then_live
from idsai.data.replay import Replayer
from idsai.data.store import Store
from idsai.detection.features.bars import BarBuilder
from idsai.detection.simulator import Scenario
from idsai.detection.simulator.guard import EthicsError, assert_injection_allowed
from idsai.pipeline.stream import StreamPipeline
from tests.pipeline.test_stream_gap import cheap_chief
from tests.synth import day_us, make_trades

S = 1_000_000
T0 = day_us(date(2025, 3, 2))
SYM = "BTCUSDT"
LIMIT_S = 20  # no test should come near this: it turns a hang into a failure

# the documented shape of a spot @trade message in a combined stream, with timeUnit=MICROSECOND
PAYLOAD = (
    '{"stream":"btcusdt@trade","data":{"e":"trade","E":1740873600123456,"s":"BTCUSDT",'
    '"t":4593012345,"p":"86012.45000000","q":"0.00123000","T":1740873600123000,"m":true,"M":true}}'
)


def body(i: int, t_us: int | None = None, symbol: str = SYM, **over: Any) -> dict[str, Any]:
    t = T0 + i * S if t_us is None else t_us
    d = {"e": "trade", "E": t, "s": symbol, "t": i, "p": "2.5", "q": "4", "T": t, "m": False}
    return {**d, **over}


def tmsg(i: int, t_us: int | None = None, symbol: str = SYM, **over: Any) -> str:
    return json.dumps({"stream": f"{symbol.lower()}@trade", "data": body(i, t_us, symbol, **over)})


class Clock:
    """A settable local clock; `skew_us` is how far it is from the fake exchange's time."""

    def __init__(self, now: int = T0, skew_us: int = 0):
        self.exchange = now
        self.skew_us = skew_us

    def __call__(self) -> int:
        return self.exchange + self.skew_us

    def mono(self) -> float:
        """A monotonic clock: exchange time passes on it, a step of the local clock does not."""
        return (self.exchange - T0) / S


class Sleeps:
    """Records backoff delays without waiting them; optionally lets time pass."""

    def __init__(self, clock: Clock | None = None, advance: dict[int, float] | None = None):
        self.delays: list[float] = []
        self.clock = clock
        self.advance = advance or {}

    async def __call__(self, s: float) -> None:
        self.delays.append(s)
        if self.clock is not None:
            self.clock.exchange += int(self.advance.get(len(self.delays), s) * S)
        await asyncio.sleep(0)


@asynccontextmanager
async def server(handler, **kw):
    """A loopback server; handlers still waiting when the test is done are stopped with it."""
    running: set[asyncio.Task] = set()

    async def stoppable(ws):
        task = asyncio.create_task(handler(ws))
        running.add(task)
        await asyncio.gather(task, return_exceptions=True)

    async with websockets.serve(stoppable, "127.0.0.1", 0, **kw) as srv:
        try:
            yield f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}/stream"
        finally:
            for task in running:
                task.cancel()


def run(coro):
    async def limited():
        async with asyncio.timeout(LIMIT_S):
            return await coro

    return asyncio.run(limited())


async def take(sensor: LiveTradeSensor, n: int, kind: str = "trade") -> list[tuple[str, Any]]:
    """Events up to and including the `n`th of `kind`."""
    out: list[tuple[str, Any]] = []
    async for ev in sensor:
        out.append(ev)
        if sum(e[0] == kind for e in out) == n:
            return out
    return out


def ids(events: list[tuple[str, Any]]) -> list[int]:
    return [e[1].trade_id for e in events if e[0] == "trade"]


def sensor_for(url: str, clock=None, sleep=None, symbols=None, **kw) -> LiveTradeSensor:
    """Fake clocks and backoff; the hold before the start is off unless a test is about it."""
    clock = clock or Clock()
    kw.setdefault("start_hold_s", 0)
    return LiveTradeSensor(
        symbols or [SYM], url, clock=clock, monotonic=clock.mono, sleep=sleep or Sleeps(), **kw
    )


# ------------------------------------------------------------------ parse_trade
def test_parse_trade_on_a_recorded_shape_payload():
    t = parse_trade(PAYLOAD, recv_us=1740873600200000)
    assert t == Trade(
        ts_us=1740873600123000,
        recv_us=1740873600200000,
        symbol="BTCUSDT",
        price=86012.45,
        qty=0.00123,
        quote_qty=86012.45 * 0.00123,
        taker_side="sell",
        trade_id=4593012345,
        n_fills=1,
        source="live",
    )


def test_parse_trade_takes_the_bare_payload_and_bytes_and_a_dict():
    bare = json.dumps(json.loads(PAYLOAD)["data"])
    want = parse_trade(PAYLOAD, recv_us=1)
    assert parse_trade(bare, recv_us=1) == want
    assert parse_trade(PAYLOAD.encode(), recv_us=1) == want
    assert parse_trade(json.loads(PAYLOAD), recv_us=1) == want


def test_m_false_is_a_buy_and_recv_defaults_to_the_local_clock():
    t = parse_trade(body(1))
    assert t.taker_side == "buy"
    assert abs(t.recv_us - time.time_ns() // 1000) < 60 * S


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        b"\xff\xfe",
        "[1, 2]",
        "null",
        '{"data": 3}',
        body(1, e="aggTrade"),
        body(1, p="NaN"),
        body(1, p="inf"),
        body(1, p="0"),
        body(1, p="-1"),
        body(1, p="abc"),
        body(1, p=None),
        body(1, p=True),
        body(1, q="0"),
        body(1, q="1e400"),
        body(1, p="1e308", q="1e308"),
        body(1, m="false"),
        body(1, t="5"),
        body(1, t=-5),
        body(1, T=1740873600123),  # milliseconds: the wrong unit, not an old trade
        body(1, T=1.5e15),
        body(1, s="BTC/USDT"),
        body(1, s=5),
        *[{k: v for k, v in body(1).items() if k != gone} for gone in "eEstpqTm"],
    ],
)
def test_parse_trade_refuses_what_it_would_have_to_guess(bad):
    with pytest.raises(ValueError):
        parse_trade(bad)


# ------------------------------------------------------------------ construction
def test_url_format_lowercases_and_dedups():
    s = LiveTradeSensor(
        ["BTCUSDT", "ETHUSDT", "btcusdt"], "wss://data-stream.binance.vision/stream"
    )
    assert s.symbols == ["BTCUSDT", "ETHUSDT"]
    assert s.stream_url == (
        "wss://data-stream.binance.vision/stream"
        "?streams=btcusdt@trade/ethusdt@trade&timeUnit=MICROSECOND"
    )
    assert s.rotate_s == 23 * 3600


def test_the_configured_url_is_allowed():
    cfg = load_settings()
    assert LiveTradeSensor(cfg.live.symbols, cfg.live.ws_url).stream_url.startswith(cfg.live.ws_url)


@pytest.mark.parametrize(
    "url",
    [
        "wss://stream.binance.com:9443/stream",
        "wss://api.binance.com/stream",
        "wss://data-stream.binance.vision.evil.example/stream",
        "wss://localhost.evil.example/stream",
        "ws://data-stream.binance.vision/stream",  # the public host only over TLS
        "https://data-stream.binance.vision/stream",
        "wss://key:secret@data-stream.binance.vision/stream",
        "wss://data-stream.binance.vision/stream?streams=x",
        "wss://data-stream.binance.vision/stream?",
        "wss://data-stream.binance.vision/stream#",
        "ws://127.0.0.1:1/stream?",
        "wss:///stream",
        "",
    ],
)
def test_host_refusal(url):
    with pytest.raises(ValueError):
        LiveTradeSensor([SYM], url)


@pytest.mark.parametrize("url", ["ws://127.0.0.1:1/s", "ws://localhost:1/s", "ws://[::1]:1/s"])
def test_loopback_is_allowed(url):
    assert LiveTradeSensor([SYM], url).url == url


@pytest.mark.parametrize("symbols", [[], "BTCUSDT", ["BTC/USDT"], [""], ["../x"], [5]])
def test_symbols_are_checked(symbols):
    with pytest.raises(ValueError):
        LiveTradeSensor(symbols, "ws://127.0.0.1:1/s")


def test_no_order_or_auth_code_in_the_module():
    src = open(live.__file__, encoding="utf-8").read()
    assert ".send(" not in src
    for word in (
        "api_key",
        "apikey",
        "X-MBX",
        "secret",
        "signature",
        "/order",
        "additional_headers",
    ):
        assert word.lower() not in src.lower(), word


# ------------------------------------------------------------------ reconnect
def test_server_closing_after_3_messages_reconnects_and_continues_without_duplicates():
    """The second connection repeats the last trade: dedup across the reconnect."""
    conns: list[Any] = []

    async def handler(ws):
        conns.append(ws)
        if len(conns) == 1:
            for i in (1, 2, 3):
                await ws.send(tmsg(i))
        else:
            for i in (3, 4, 5):
                await ws.send(tmsg(i))
            await ws.wait_closed()

    async def main():
        sleeps = Sleeps()
        async with server(handler) as url:
            sensor = sensor_for(url, sleep=sleeps)
            got = await take(sensor, 5)
            await sensor.aclose()
            return got, sensor, sleeps

    got, sensor, sleeps = run(main())
    assert ids(got) == [1, 2, 3, 4, 5]
    assert sensor.dupes == 1 and sensor.reconnects == 1 and sensor.rotations == 0
    assert sleeps.delays == [1]
    assert len(conns) == 2


def test_dedup_window_is_the_last_10000():
    async def handler(ws):
        for i in range(10_001):
            await ws.send(tmsg(i, T0))
        await ws.send(tmsg(10_000, T0))  # still inside the window
        await ws.send(tmsg(0, T0))  # just evicted, so delivered again
        await ws.wait_closed()

    async def main():
        async with server(handler) as url:
            sensor = sensor_for(url)
            got = await take(sensor, 10_002)
            await sensor.aclose()
            return ids(got), sensor.dupes

    got, dupes = run(main())
    assert dupes == 1 and got[-1] == 0 and len(got) == 10_002


def test_dedup_key_includes_the_symbol():
    async def handler(ws):
        await ws.send(tmsg(7))
        await ws.send(tmsg(7, symbol="ETHUSDT"))
        await ws.wait_closed()

    async def main():
        async with server(handler) as url:
            sensor = sensor_for(url, symbols=[SYM, "ETHUSDT"])
            got = await take(sensor, 2)
            await sensor.aclose()
            return [e[1].symbol for e in got if e[0] == "trade"], sensor.dupes

    assert run(main()) == ([SYM, "ETHUSDT"], 0)


def test_backoff_sequence_cap_and_reset():
    """Handshake failures (HTTP 429 and 418) double up to 60 s; a delivered message resets it."""
    attempts = {"n": 0}
    failures = 8

    def process_request(conn, request):
        attempts["n"] += 1
        if attempts["n"] <= failures:
            status = http.HTTPStatus.TOO_MANY_REQUESTS if attempts["n"] % 2 else 418
            return conn.respond(status, "go away\n")
        return None

    sent = {"n": 0}

    async def handler(ws):
        sent["n"] += 1
        await ws.send(tmsg(sent["n"]))
        await asyncio.sleep(0.02)  # let it arrive: an abort discards what is still unsent
        ws.transport.abort()

    async def main():
        sleeps = Sleeps()
        async with server(handler, process_request=process_request) as url:
            sensor = sensor_for(url, sleep=sleeps)
            await take(sensor, 2)
            await sensor.aclose()
            return sleeps.delays, sensor.reconnects

    delays, reconnects = run(main())
    assert delays == [1, 2, 4, 8, 16, 32, 60, 60, 1]
    assert reconnects == 2  # connections opened, not attempts made


def test_a_refused_port_backs_off_too():
    async def main():
        probe = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        port = probe.sockets[0].getsockname()[1]
        probe.close()
        await probe.wait_closed()
        done = asyncio.Event()
        sleeps: list[float] = []

        async def sleep(s):
            sleeps.append(s)
            done.set()
            await asyncio.sleep(0)

        sensor = LiveTradeSensor([SYM], f"ws://127.0.0.1:{port}/s", clock=Clock(), sleep=sleep)
        reader = asyncio.create_task(anext(sensor, None))
        await done.wait()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        return sleeps

    assert run(main())[0] == 1


def test_a_redirect_to_another_host_is_a_failed_handshake():
    visited = []

    async def other(ws):
        visited.append(1)
        await ws.send(tmsg(1))

    async def main():
        async with server(other) as other_url:

            def redirect(conn, request):
                resp = conn.respond(http.HTTPStatus.FOUND, "")
                resp.headers["Location"] = other_url
                return resp

            async with server(other, process_request=redirect) as url:
                done = asyncio.Event()

                async def sleep(s):
                    done.set()
                    await asyncio.sleep(0)

                sensor = LiveTradeSensor([SYM], url, clock=Clock(), sleep=sleep)
                reader = asyncio.create_task(anext(sensor, None))
                await done.wait()
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)

    run(main())
    assert visited == []


def test_rotation_reconnects_without_backoff_and_counts_apart():
    clock = Clock()
    conns = []

    async def handler(ws):
        conns.append(ws)
        await ws.send(tmsg(len(conns)))
        await ws.wait_closed()

    async def main():
        sleeps = Sleeps()
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock, sleep=sleeps, rotate_s=30)
            got = await take(sensor, 1)
            clock.exchange += 31 * S  # due: seen at the next look at the clock
            got += await take(sensor, 1)
            await sensor.aclose()
            return got, sensor, sleeps

    got, sensor, sleeps = run(main())
    assert ids(got) == [1, 2]
    assert sensor.rotations == 1 and sensor.reconnects == 0 and sensor.gaps == 0
    assert sleeps.delays == []
    assert len(conns) == 2


# ------------------------------------------------------------------ messages and the wire
def test_bad_messages_are_counted_and_skipped():
    async def handler(ws):
        await ws.send("not json")
        await ws.send(json.dumps(body(1, e="kline")))
        await ws.send(tmsg(2, p="NaN"))
        await ws.send(tmsg(3, symbol="ETHUSDT"))  # valid, but not what was asked for
        await ws.send(tmsg(4))
        await ws.wait_closed()

    async def main():
        async with server(handler) as url:
            sensor = sensor_for(url)
            got = await take(sensor, 1)
            await sensor.aclose()
            return ids(got), sensor.bad

    assert run(main()) == ([4], 4)


def test_nothing_is_sent_to_the_server():
    received: list[Any] = []
    headers: list[Any] = []

    async def handler(ws):
        headers.append(ws.request.headers)
        await ws.send(tmsg(1))
        async for m in ws:
            received.append(m)

    async def main():
        async with server(handler) as url:
            sensor = sensor_for(url)
            await take(sensor, 1)
            await sensor.aclose()
            await asyncio.sleep(0.01)

    run(main())
    assert received == []
    names = {k.lower() for k in headers[0].keys()}
    assert not names & {"authorization", "cookie", "x-mbx-apikey"}


def test_a_full_queue_applies_backpressure_and_loses_nothing():
    async def handler(ws):
        for i in range(300):
            await ws.send(tmsg(i, T0))
        await ws.wait_closed()

    async def main():
        async with server(handler) as url:
            sensor = sensor_for(url, queue_size=4)
            first = await anext(sensor)
            await asyncio.sleep(0.05)
            assert sensor._queue.qsize() <= 4
            got = [first, *await take(sensor, 300)]
            await sensor.aclose()
            return ids(got)

    assert run(main()) == list(range(300))


# ------------------------------------------------------------------ cleanup
def extra_tasks(before: set) -> set:
    return asyncio.all_tasks() - before - {asyncio.current_task()}


def test_aclose_leaves_no_task_or_socket():
    async def main():
        done = asyncio.Event()

        async def handler(ws):
            await ws.send(tmsg(1))
            await ws.wait_closed()
            done.set()

        async with server(handler) as url:
            await asyncio.sleep(0)
            before = asyncio.all_tasks()
            sensor = sensor_for(url)
            await take(sensor, 1)
            await sensor.aclose()
            await sensor.aclose()
            await done.wait()  # the server saw the socket close
            await asyncio.sleep(0.01)
            assert not {t for t in extra_tasks(before) if "live-trade" in t.get_name()}
            assert sensor._task is None
            with pytest.raises(StopAsyncIteration):
                await anext(sensor)

    run(main())


def test_cancelling_the_consumer_leaves_no_task_or_socket():
    async def main():
        done = asyncio.Event()
        got_one = asyncio.Event()

        async def handler(ws):
            await ws.send(tmsg(1))
            await ws.wait_closed()
            done.set()

        async with server(handler) as url:
            sensor = sensor_for(url)

            async def consume():
                async for _ in sensor:
                    got_one.set()

            task = asyncio.create_task(consume())
            await got_one.wait()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await done.wait()
            assert sensor._task is None
            assert not [t for t in asyncio.all_tasks() if t.get_name() == "live-trade-sensor"]

    run(main())


def test_aclose_during_backoff_leaves_no_task():
    async def main():
        waiting = asyncio.Event()

        async def sleep(s):
            waiting.set()
            await asyncio.Event().wait()

        probe = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        port = probe.sockets[0].getsockname()[1]
        probe.close()
        await probe.wait_closed()
        sensor = LiveTradeSensor([SYM], f"ws://127.0.0.1:{port}/s", clock=Clock(), sleep=sleep)
        reader = asyncio.create_task(anext(sensor, None))
        await waiting.wait()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        assert not [t for t in asyncio.all_tasks() if t.get_name() == "live-trade-sensor"]

    run(main())


def test_a_bug_in_the_reader_reaches_the_consumer():
    async def handler(ws):
        await ws.send(tmsg(1))
        await ws.wait_closed()

    async def main():
        def clock():
            raise RuntimeError("boom")

        async with server(handler) as url:
            sensor = LiveTradeSensor([SYM], url, clock=clock, sleep=Sleeps())
            with pytest.raises(RuntimeError, match="boom"):
                await take(sensor, 1)
            assert sensor._task is None

    run(main())


# ------------------------------------------------------------------ start, skew, tick order
class Lockstep:
    """A fake exchange: moves its clock, sends one message, waits until the consumer has it."""

    def __init__(self, clock: Clock):
        self.clock = clock
        self.ack = asyncio.Event()

    async def send(self, ws, msg: str, t_us: int, ack: bool = True) -> None:
        self.clock.exchange = t_us
        self.ack.clear()
        await ws.send(msg)
        if ack:
            await self.ack.wait()
        else:  # a message the sensor holds back: nothing to wait for
            await asyncio.sleep(0.01)


async def acked(sensor, step: Lockstep):
    async for ev in sensor:
        yield ev
        if ev[0] == "trade":
            step.ack.set()


def test_preroll_done_comes_first_once_and_ticks_only_after_the_first_trade():
    clock = Clock()
    step = None

    async def handler(ws):
        for i in range(5):
            await step.send(ws, tmsg(i), T0 + i * S)
        await ws.wait_closed()

    async def main():
        nonlocal step
        step = Lockstep(clock)
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock)
            out = []
            async for ev in acked(sensor, step):
                out.append(ev)
                if ev[0] == "trade" and ev[1].trade_id == 4:
                    break
            await sensor.aclose()
            return out

    out = run(main())
    assert out[0] == ("preroll_done", T0)
    assert [e[0] for e in out].count("preroll_done") == 1
    assert out[1][0] == "trade"
    ticks = [e[1] for e in out if e[0] == "tick"]
    assert ticks == sorted(set(ticks)) and ticks[0] == T0 + S
    assert all(t % S == 0 for t in ticks)
    for i, e in enumerate(out):  # a trade comes before the tick that closes its second
        if e[0] == "trade":
            assert not [t for _, t in (x for x in out[:i] if x[0] == "tick") if t > e[1].ts_us]


def test_a_silent_market_starts_ticking_60_s_after_connecting():
    clock = Clock()
    release = {}

    async def handler(ws):
        release["ws"] = ws
        await ws.wait_closed()

    async def main():
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock)
            reader = asyncio.create_task(anext(sensor))
            while "ws" not in release:
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.02)
            assert not reader.done()  # connected, silent, under 60 s: nothing at all
            clock.exchange += 61 * S
            first = await reader
            clock.exchange += S
            second = await anext(sensor)
            await sensor.aclose()
            return first, second

    first, second = run(main())
    assert first == ("preroll_done", T0 + 61 * S)
    assert second == ("tick", T0 + 62 * S)


@pytest.mark.parametrize("skew_s", [10, -10, 3600, -3600, 0])
def test_a_skewed_local_clock_loses_no_trade_in_a_real_bar_builder(skew_s):
    clock = Clock(skew_us=skew_s * S)
    n = 40
    step = None

    async def handler(ws):
        for i in range(n):
            t = T0 + i * S // 2
            await step.send(ws, tmsg(i, t), t)
        await ws.wait_closed()

    async def main():
        nonlocal step
        step = Lockstep(clock)
        builder = BarBuilder(SYM, grace_s=2)
        bars = []
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock)
            seen = 0
            async for kind, v in acked(sensor, step):
                if kind == "trade":
                    builder.on_trade(v)
                    seen += 1
                    if seen == n:
                        break
                elif kind == "tick":
                    bars += builder.on_tick(v)
            skew = sensor.clock_skew_ms
            await sensor.aclose()
        return builder, bars, skew

    builder, bars, skew = run(main())
    assert builder.late == 0
    assert skew == pytest.approx(-skew_s * 1000, abs=1000)
    assert sum(b["n_trades"] for b in bars) > 0  # the ticks really closed bars
    assert not math.isnan(bars[-1]["close"])


@pytest.mark.parametrize("step_s", [90, -90])
def test_a_wall_clock_step_while_connected_has_no_effect(step_s):
    clock = Clock()
    step = None

    async def handler(ws):
        for i in range(10):
            await step.send(ws, tmsg(i), T0 + i * S)
        clock.skew_us = step_s * S
        await asyncio.sleep(0.02)  # the sensor looks at its clocks without a trade first
        for i in range(10, 20):
            await step.send(ws, tmsg(i), T0 + i * S)
        await ws.wait_closed()

    async def main():
        nonlocal step
        step = Lockstep(clock)
        builder = BarBuilder(SYM, grace_s=2)
        ticks, trades = [], []
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock)
            async for kind, v in acked(sensor, step):
                if kind == "tick":
                    ticks.append(v)
                    builder.on_tick(v)
                elif kind == "trade":
                    builder.on_trade(v)
                    trades.append(v)
                    if v.trade_id == 19:
                        break
            await sensor.aclose()
        return ticks, trades, builder, sensor

    ticks, trades, builder, sensor = run(main())
    assert sensor.gaps == 0 and builder.late == 0
    assert ticks == list(range(ticks[0], ticks[-1] + 1, S))
    assert sensor.clock_skew_ms == pytest.approx(0, abs=1000)
    assert trades[-1].recv_us == trades[-1].ts_us + step_s * S  # recv_us stays the wall clock


# ------------------------------------------------------------------ gaps
def outage_events(outage_s: float) -> tuple[list, LiveTradeSensor]:
    """Two trades, a lost connection for `outage_s` of fake time, two more trades."""
    clock = Clock()
    conns = []
    step = None

    async def handler(ws):
        conns.append(ws)
        if len(conns) == 1:
            await step.send(ws, tmsg(1, T0 + S), T0 + S)
            await step.send(ws, tmsg(2, T0 + 2 * S), T0 + 2 * S)
        else:
            t = clock.exchange
            await step.send(ws, tmsg(3, t), t)
            await step.send(ws, tmsg(4, t + 100_000), t + 100_000)
            await ws.wait_closed()

    async def main():
        nonlocal step
        step = Lockstep(clock)
        sleeps = Sleeps(clock, advance={1: outage_s})
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock, sleep=sleeps)
            out = []
            async for ev in acked(sensor, step):
                out.append(ev)
                if ev[0] == "trade" and ev[1].trade_id == 4:
                    break
            await sensor.aclose()
            return out, sensor

    return run(main())


def test_a_long_outage_emits_gap_first_and_no_ticks_inside_it():
    out, sensor = outage_events(6 * 3600)
    gaps = [e[1] for e in out if e[0] == "gap"]
    assert sensor.gaps == 1 and sensor.reconnects == 1 and len(gaps) == 1
    frm, to = gaps[0]
    assert frm == T0 + 2 * S  # the last tick before the outage
    assert to >= T0 + 2 * S + 6 * 3600 * S
    ticks = [e[1] for e in out if e[0] == "tick"]
    assert not [t for t in ticks if frm < t <= to]
    i_gap = [e[0] for e in out].index("gap")
    assert ids(out[:i_gap]) == [1, 2] and ids(out[i_gap:]) == [3, 4]
    assert [e[0] for e in out].count("preroll_done") == 1


def test_a_short_outage_resumes_ticks_without_a_gap():
    out, sensor = outage_events(GAP_RESTART_S - 2)
    assert sensor.gaps == 0 and sensor.reconnects == 1
    assert "gap" not in [e[0] for e in out]
    ticks = [e[1] for e in out if e[0] == "tick"]
    assert ticks == list(range(T0 + 2 * S, ticks[-1] + 1, S))  # the short empty fill, every second
    assert ticks[-1] >= T0 + (GAP_RESTART_S - 1) * S


@pytest.mark.parametrize("outage_s, gaps", [(60, 0), (61, 1)])
def test_the_exact_gap_boundary(outage_s, gaps):
    out, sensor = outage_events(outage_s)
    assert sensor.gaps == gaps
    assert [e[0] for e in out].count("gap") == gaps


def test_a_stall_while_connected_is_a_gap_too():
    """A sleeping laptop: the socket looks alive, and hours of empty ticks must not come out."""
    clock = Clock()
    step = None

    async def handler(ws):
        await step.send(ws, tmsg(1), T0)
        await step.send(ws, tmsg(2, T0 + S), T0 + S)
        t = T0 + 5 * 3600 * S
        await step.send(ws, tmsg(3, t), t)
        await ws.wait_closed()

    async def main():
        nonlocal step
        step = Lockstep(clock)
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock)
            out = []
            async for ev in acked(sensor, step):
                out.append(ev)
                if ev[0] == "trade" and ev[1].trade_id == 3:
                    break
            for _ in range(3):
                if sensor.gaps:
                    break
                await asyncio.sleep(0.01)
            n, gaps = len([e for e in out if e[0] == "tick"]), sensor.gaps
            await sensor.aclose()
            return n, gaps

    n, gaps = run(main())
    assert gaps == 1 and n < 10


# ------------------------------------------------------------------ warm-up (G6) with a fake sensor
CFG = load_settings()
LIVE_T = T0 + 8 * 3600 * S


class FakeSensor:
    """Events given in advance; records its closing."""

    def __init__(self, events: list[tuple[str, Any]], symbols: list[str] | None = None):
        self.events = events
        self.closed = False
        self._it = iter(events)
        if symbols is not None:
            self.symbols = symbols

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed = True


def live_trades(t_us: int, n_s: int, first_id: int = 10**9) -> list[tuple[str, Any]]:
    """preroll_done, then one trade per second and its tick."""
    out: list[tuple[str, Any]] = [("preroll_done", t_us)]
    for i in range(n_s):
        t = t_us + i * S
        tr = Trade(
            ts_us=t + 1,
            recv_us=t + 1,
            symbol="TESTBTC",
            price=5e-4,
            qty=7.0,
            quote_qty=3.5e-3,
            taker_side="buy",
            trade_id=first_id + i,
            n_fills=1,
            source="live",
        )
        out += [("trade", tr), ("tick", t + S)]
    return out


@pytest.fixture(scope="module")
def archive():
    return make_trades(date(2025, 3, 2), 1, seed=4, rate=0.5).filter(
        pl.col("ts_us") < T0 + 7 * 3600 * S
    )


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "data", tmp_path / "data" / "a.duckdb") as s:
        yield s


def test_warmup_event_order(archive):
    small = archive.filter(pl.col("ts_us") < T0 + 60 * S)
    sensor = FakeSensor(live_trades(LIVE_T, 2))

    async def main():
        return [ev async for ev in warmup_then_live(small, sensor)]

    out = run(main())
    kinds = [e[0] for e in out]
    assert kinds.count("preroll_done") == 1 and kinds.count("gap") == 1
    i_gap = kinds.index("gap")
    assert kinds[i_gap + 1] == "preroll_done"
    assert "preroll_done" not in kinds[:i_gap]
    assert all(e[1].source == "archive" for e in out[:i_gap] if e[0] == "trade")
    assert all(e[1].source == "live" for e in out[i_gap:] if e[0] == "trade")
    end = small["ts_us"].max() // S * S + S
    assert out[i_gap] == ("gap", (end, LIVE_T))
    assert out[i_gap + 1 :] == sensor.events
    assert sensor.closed


@pytest.mark.parametrize("empty", [None, "empty"])
def test_warmup_without_an_archive_is_just_the_sensor(archive, empty):
    frame = None if empty is None else archive.head(0)
    sensor = FakeSensor(live_trades(LIVE_T, 2))

    async def main():
        return [ev async for ev in warmup_then_live(frame, sensor)]

    assert run(main()) == sensor.events
    assert sensor.closed


def test_warmup_closes_the_sensor_when_the_consumer_stops(archive):
    sensor = FakeSensor(live_trades(LIVE_T, 2))

    async def main():
        gen = warmup_then_live(archive.head(10), sensor)
        await anext(gen)
        await gen.aclose()

    run(main())
    assert sensor.closed


def test_warmup_a_clock_before_the_archive_end_keeps_the_gap_ordered(archive):
    small = archive.filter(pl.col("ts_us") < T0 + 60 * S)
    sensor = FakeSensor([("preroll_done", T0)])

    async def main():
        return [ev async for ev in warmup_then_live(small, sensor)]

    frm, to = next(e[1] for e in run(main()) if e[0] == "gap")
    assert to == frm


def test_warmup_publishes_nothing_of_the_archive_and_the_baseline_is_ready(archive, store):
    chief = cheap_chief(CFG, "TESTBTC")
    bus = Bus(maxsize=100_000)
    qb, qs, qa = bus.subscribe("bars"), bus.subscribe("scores"), bus.subscribe("alerts")
    pipe = StreamPipeline(CFG, store, chief.detectors, chief, bus, "live")
    at_first_row: list[int] = []
    real = pipe._on_bar

    def spy(bar, received):
        real(bar, received)
        if not at_first_row and pipe.series("TESTBTC"):
            at_first_row.append(pipe.baseline_samples("TESTBTC"))

    pipe._on_bar = spy
    asyncio.run(pipe.run(warmup_then_live(archive, FakeSensor(live_trades(LIVE_T, 1850)))))

    bars = [qb.get_nowait() for _ in range(qb.qsize())]
    scores = [qs.get_nowait() for _ in range(qs.qsize())]
    alerts = [qa.get_nowait() for _ in range(qa.qsize())]
    assert bars and min(b["t_us"] for b in bars) >= LIVE_T
    assert scores and min(s["t_end_us"] for s in scores) >= LIVE_T + 1800 * S
    assert all(a.window[0] >= LIVE_T for a in alerts)
    assert at_first_row and at_first_row[0] >= CFG.features.min_baseline_samples
    assert all(a.provenance == "live" for a in store.list_alerts())


def test_auditor_warmup_with_an_empty_archive_and_after_enough_samples(archive, store):
    chief = cheap_chief(CFG, "TESTBTC")
    auditor = Auditor(CFG)

    cold = StreamPipeline(CFG, store, chief.detectors, chief, Bus(), "live")
    asyncio.run(cold.run(warmup_then_live(None, FakeSensor(live_trades(LIVE_T, 5)))))
    check = auditor.warmup(cold.baseline_samples("TESTBTC"))
    assert check.status == "warn"
    assert check.value["hours_remaining"] == CFG.features.min_baseline_samples * 60 / 3600

    warm = StreamPipeline(CFG, store, chief.detectors, chief, Bus(), "live")
    asyncio.run(warm.run(warmup_then_live(archive, FakeSensor(live_trades(LIVE_T, 5)))))
    assert warm.baseline_samples("TESTBTC") >= CFG.features.min_baseline_samples
    assert auditor.warmup(warm.baseline_samples("TESTBTC")).status == "ok"


# ------------------------------------------------------------------ end to end (G8)
def test_fake_server_to_a_live_pipeline_to_a_stored_alert(store):
    chief = cheap_chief(CFG, "TESTBTC")
    clock = Clock()
    n_quiet, burst = 1820, 60
    step = None

    async def handler(ws):
        i = 0
        for k in range(n_quiet):
            t = T0 + k * S + 1
            i += 1
            await step.send(ws, tmsg(i, t, "TESTBTC", q=str(7 + k % 5)), t)
        for k in range(burst):
            t = T0 + n_quiet * S + k * 100_000
            i += 1
            await step.send(ws, tmsg(i, t, "TESTBTC", q=str(7 + k % 5)), t)
        for k in range(30):
            t = T0 + (n_quiet + 10 + k) * S
            i += 1
            await step.send(ws, tmsg(i, t, "TESTBTC", q=str(7 + k % 5)), t)
        await ws.wait_closed()

    total = n_quiet + burst + 30

    async def main():
        nonlocal step
        step = Lockstep(clock)
        bus = Bus(maxsize=100_000)
        pipe = StreamPipeline(CFG, store, chief.detectors, chief, bus, "live")
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock, symbols=["TESTBTC"])

            async def source():
                seen = 0
                async for ev in acked(sensor, step):
                    yield ev
                    if ev[0] == "trade":
                        seen += 1
                        if seen == total:
                            return

            await pipe.run(source())
            await sensor.aclose()
        return pipe, sensor

    pipe, sensor = asyncio.run(main())
    assert sensor.bad == 0 and sensor.dupes == 0
    assert pipe._builders["TESTBTC"].late == 0
    alerts = store.list_alerts()
    burst_alerts = [a for a in alerts if a.window[1] >= T0 + n_quiet * S]
    assert burst_alerts
    for a in alerts:
        assert a.provenance == "live"
        assert a.alert_id.startswith("live-")
        assert store.get_alert(a.alert_id) is not None
    with pytest.raises(ValueError):  # a live pipeline cannot seek, and live rows stay
        pipe._on_seek(T0)


# ------------------------------------------------------------------ fix round 1
DAY_S = 86_400


def rogue_run(at: int, rogue_s: int, n_after: int = 30, **rogue_fields: Any):
    """Trades 0.2 s apart; number `at` carries a `T` that is `rogue_s` ahead. Real start hold."""
    clock = Clock()
    total = at + n_after
    step = None

    async def handler(ws):
        for i in range(total):
            t = T0 + i * S // 5
            msg = tmsg(i, t + rogue_s * S if i == at else t, **(rogue_fields if i == at else {}))
            await step.send(ws, msg, t, ack=i != 0 and not (rogue_fields and i == at))
        await ws.wait_closed()

    async def main():
        nonlocal step
        step = Lockstep(clock)
        builder = BarBuilder(SYM, grace_s=2)
        bars = []
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock, start_hold_s=5)
            first = None
            async for kind, v in acked(sensor, step):
                first = first if first is not None else (kind, v)
                if kind == "trade":
                    builder.on_trade(v)
                    if v.trade_id == total - 1:
                        break
                elif kind == "tick":
                    bars += builder.on_tick(v)
            await sensor.aclose()
        return builder, bars, sensor, first

    return run(main())


@pytest.mark.parametrize("rogue_s", [30, 3600, DAY_S])
@pytest.mark.parametrize("at", [0, 1, 1000])
def test_one_rogue_trade_does_not_wreck_the_session(at, rogue_s):
    builder, bars, sensor, first = rogue_run(at, rogue_s)
    assert builder.late == 0 and sensor.gaps == 0
    assert sensor.clock_skew_ms == pytest.approx(0, abs=1000)
    assert first == ("preroll_done", T0)
    assert sum(b["n_trades"] for b in bars) > 0  # the session goes on closing real bars


def test_a_trade_time_far_from_its_event_time_is_refused():
    assert parse_trade(body(1, E=T0 + 1 * S + 60 * S)).trade_id == 1
    with pytest.raises(ValueError):
        parse_trade(body(1, E=T0 + 1 * S + 60 * S + 1))
    with pytest.raises(ValueError):
        parse_trade(body(1, E=T0 + 1 * S - 60 * S - 1))
    with pytest.raises(ValueError):
        parse_trade(body(1, E="soon"))


def test_a_rogue_refused_by_its_event_time_is_only_counted():
    builder, _, sensor, _ = rogue_run(1, DAY_S, E=T0)
    assert sensor.bad == 1 and builder.late == 0 and sensor.gaps == 0


def test_a_lone_first_trade_is_released_when_the_hold_ends():
    clock = Clock()

    async def handler(ws):
        await ws.send(tmsg(0, T0))
        await ws.wait_closed()

    async def main():
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock, start_hold_s=5)
            reader = asyncio.create_task(anext(sensor))
            await asyncio.sleep(0.05)
            assert not reader.done()
            clock.exchange += 5 * S
            first = await reader
            second = await anext(sensor)
            await sensor.aclose()
            return first, second

    first, second = run(main())
    assert first == ("preroll_done", T0) and second[0] == "trade"


@pytest.mark.parametrize(
    "poison",
    [
        json.dumps(body(1)).replace('"2.5"', "1" + "0" * 400),  # p = 10**400: OverflowError
        "[" * 200_000,  # RecursionError
        "[]",
        "null",
        json.dumps(body(1, t=2**70)),  # would not fit the store's int64
        json.dumps(body(1, T=2**70, E=2**70)),
    ],
    ids=["overflow", "recursion", "list", "null", "huge_id", "huge_time"],
)
def test_poison_messages_are_counted_and_the_iterator_goes_on(poison):
    async def handler(ws):
        await ws.send(poison)
        await ws.send(tmsg(2))
        await ws.wait_closed()

    async def main():
        async with server(handler) as url:
            sensor = sensor_for(url)
            got = await take(sensor, 1)
            await sensor.aclose()
            return ids(got), sensor.bad

    assert run(main()) == ([2], 1)


def test_parse_trade_says_valueerror_for_poison_too():
    with pytest.raises(ValueError):
        parse_trade("[" * 200_000)
    with pytest.raises(ValueError):
        parse_trade(body(1, p=10**400))


def test_aclose_from_another_task_ends_a_pending_anext():
    async def handler(ws):
        await ws.wait_closed()

    async def main():
        async with server(handler) as url:
            sensor = sensor_for(url)

            async def consume():
                return [ev async for ev in sensor]

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.05)
            assert not task.done()
            await sensor.aclose()
            async with asyncio.timeout(1):
                assert await task == []
            assert not [t for t in asyncio.all_tasks() if t.get_name() == "live-trade-sensor"]

    run(main())


class SimExchange:
    """Stands in for the connection: 100 trades/s, `recv` gives what the clock has reached. Time
    passes whether or not anyone reads, which a socket cannot show without real waiting."""

    def __init__(self, clock: Clock, rogue: dict[int, int] | None = None, last: int | None = None):
        self.clock = clock
        self.t0 = clock.exchange
        self.sent = 0
        self.rogue = rogue or {}  # trade number -> seconds its `T` is ahead
        self.last = last  # no trades after this many: silence

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    @property
    def backlog(self) -> bool:
        if self.last is not None and self.sent >= self.last:
            return False
        return self.t0 + self.sent * 10_000 <= self.clock.exchange

    async def recv(self) -> str:
        while not self.backlog:
            await asyncio.sleep(0)
        self.sent += 1
        t = self.t0 + (self.sent - 1) * 10_000 + self.rogue.get(self.sent, 0) * S
        return tmsg(self.sent, t)


def sim_run(monkeypatch, blocked_s=0, stall_s=0, queue_size=1000, hold=0, rogue=None, last=None):
    """5 s of normal reading; the consumer then does nothing for `blocked_s`, or the whole event
    loop stands still for `stall_s` (one jump of the clock); the backlog drains; 5 s more."""
    clock = Clock()
    sim = SimExchange(clock, rogue, last)
    monkeypatch.setattr(live, "_NoRedirectConnect", lambda url, **kw: sim)

    async def main():
        frozen = False
        builder = BarBuilder(SYM, grace_s=2)
        got: list[int] = []
        ticks: list[int] = []
        skews: list[float] = []
        sensor = sensor_for(
            "ws://127.0.0.1:1/s", clock=clock, queue_size=queue_size, start_hold_s=hold
        )

        async def exchange_time():
            while True:
                if not frozen:
                    clock.exchange += 10_000
                await asyncio.sleep(0)

        def feed(ev):
            skews.append(sensor.clock_skew_ms)
            if ev[0] == "trade":
                builder.on_trade(ev[1])
                got.append(ev[1].trade_id)
            elif ev[0] == "tick":
                ticks.append(ev[1])
                builder.on_tick(ev[1])

        async def consume_until(t_us):
            while clock.exchange < t_us:
                feed(await anext(sensor))

        passing = asyncio.create_task(exchange_time())
        await consume_until(T0 + 5 * S)
        until = clock.exchange + blocked_s * S
        while clock.exchange < until:  # the consumer does nothing at all
            await asyncio.sleep(0)
        if blocked_s:
            assert sensor._queue.full()  # the reader really was blocked, for most of that time
        frozen = True  # time stands still while the backlog drains, so it does drain
        clock.exchange += stall_s * S
        while sim.backlog or not sensor._queue.empty():
            feed(await anext(sensor))
        if last is None:
            frozen = False
            await consume_until(clock.exchange + 5 * S)
        else:  # silence: only the ticks are left, and fake time must not race ahead of real polls
            clock.exchange += 5 * S
            while not ticks or ticks[-1] < clock.exchange // S * S:
                feed(await anext(sensor))
        passing.cancel()
        await sensor.aclose()
        return builder, got, ticks, skews, sensor

    return run(main())


@pytest.mark.parametrize("blocked_s, queue_size", [(170, 10_000), (90, 1000)])
def test_a_blocked_consumer_causes_no_gap_and_no_late_trade(monkeypatch, blocked_s, queue_size):
    builder, got, ticks, skews, sensor = sim_run(
        monkeypatch, blocked_s=blocked_s, queue_size=queue_size
    )
    assert sensor.gaps == 0 and builder.late == 0
    assert got == list(range(1, len(got) + 1))
    assert len(got) > (blocked_s + 5) * 100
    assert ticks == list(range(ticks[0], ticks[-1] + 1, S))
    assert sensor.clock_skew_ms == pytest.approx(0, abs=1000)


def test_backlog_trades_are_not_skew_samples(monkeypatch):
    """Sampled, the backlog would drag the offset down to how far behind the reader is."""
    *_, skews, sensor = sim_run(monkeypatch, blocked_s=90)
    assert min(skews) > -1000 and sensor.gaps == 0


def test_a_rogue_first_trade_does_not_disable_catch_up(monkeypatch):
    builder, got, _, _, sensor = sim_run(monkeypatch, blocked_s=90, hold=5, rogue={1: DAY_S})
    assert builder.late == 0 and sensor.gaps == 0
    assert sorted(got) == list(range(1, len(got) + 1))


def test_a_rogue_during_the_session_does_not_disable_catch_up(monkeypatch):
    builder, _, _, _, sensor = sim_run(monkeypatch, blocked_s=90, rogue={300: 3600})
    assert builder.late == 0 and sensor.gaps == 0


def test_a_stalled_event_loop_makes_no_trade_late(monkeypatch):
    """`time.sleep` in the consumer: nobody blocks on the queue, the trades wait in the socket."""
    builder, got, ticks, _, sensor = sim_run(monkeypatch, stall_s=5)
    assert builder.late == 0 and sensor.gaps == 0
    assert got == list(range(1, len(got) + 1))
    assert ticks == list(range(ticks[0], ticks[-1] + 1, S))


def test_blocked_time_is_not_spent_before_the_backlog_is_gone(monkeypatch):
    """A 100 s burst, then silence, then a long block: the silent part is blocked time too."""
    builder, got, ticks, _, sensor = sim_run(monkeypatch, blocked_s=170, last=10_000)
    assert sensor.gaps == 0 and builder.late == 0
    assert len(got) == 10_000
    assert ticks == list(range(ticks[0], ticks[-1] + 1, S))


def test_a_blocked_consumer_in_a_quiet_market_causes_no_gap():
    """No trades to follow: only the blocked time keeps this from looking like an outage."""
    clock = Clock()

    async def handler(ws):
        await ws.send(tmsg(0, T0))
        await ws.wait_closed()

    async def main():
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock, queue_size=3)
            await take(sensor, 1)
            clock.exchange += 5 * S  # ticks fill the queue and the reader blocks on the next
            while not sensor._queue.full():  # the reader sees the clock at its next poll
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.01)
            clock.exchange += 90 * S
            ticks = []
            while not ticks or ticks[-1] < T0 + 95 * S:
                kind, v = await anext(sensor)
                assert kind == "tick"
                ticks.append(v)
            await sensor.aclose()
            return ticks, sensor

    ticks, sensor = run(main())
    assert sensor.gaps == 0
    assert ticks == list(range(T0 + S, T0 + 96 * S, S))


@pytest.mark.parametrize("rogue_s", [3600, DAY_S])
def test_a_lone_rogue_first_trade_recovers_with_a_gap(rogue_s):
    """It is alone for the whole hold, so it does set the start: the next trade corrects it."""
    clock = Clock()
    n = 200
    step = None

    async def handler(ws):
        step.ack.clear()
        await ws.send(tmsg(0, T0 + rogue_s * S))
        await asyncio.sleep(0.01)
        clock.exchange += 6 * S  # the hold ends
        await step.ack.wait()
        for i in range(1, n + 1):
            t = T0 + 6 * S + i * S // 10
            await step.send(ws, tmsg(i, t), t)
        await ws.wait_closed()

    async def main():
        nonlocal step
        step = Lockstep(clock)
        builder = None
        late = 0
        ticks, gaps = [], []
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock, start_hold_s=5)
            async for kind, v in acked(sensor, step):
                if kind == "preroll_done":
                    builder = BarBuilder(SYM, 2, start_us=v)
                elif kind == "gap":  # the only reset: a pipeline rebuilds here
                    gaps.append(v)
                    late += builder.late
                    builder = BarBuilder(SYM, 2, start_us=v[1] // S * S)
                    ticks.append([])
                elif kind == "tick":
                    if ticks and ticks[-1]:
                        assert v > ticks[-1][-1]
                    if not ticks:
                        ticks.append([])
                    ticks[-1].append(v)
                    builder.on_tick(v)
                else:
                    builder.on_trade(v)
                    if v.trade_id == n:
                        break
            await sensor.aclose()
        return late + builder.late, ticks, gaps, sensor

    late, ticks, gaps, sensor = run(main())
    assert late == 0 and sensor.gaps == 1
    frm, to = gaps[0]
    assert frm > to and to == pytest.approx(T0 + 6 * S, abs=2 * S)
    assert ticks[-1] and ticks[-1][-1] < T0 + 60 * S  # back on the true time


@pytest.mark.parametrize("rogue_s", [3600, DAY_S])
def test_a_lone_rogue_first_trade_end_to_end_in_a_pipeline(store, rogue_s):
    chief = cheap_chief(CFG, "TESTBTC")
    clock = Clock()
    n_quiet, burst = 1820, 60
    start = T0 + 6 * S
    step = None

    async def handler(ws):
        step.ack.clear()
        await ws.send(tmsg(10**6, T0 + rogue_s * S, "TESTBTC"))
        await asyncio.sleep(0.01)
        clock.exchange += 6 * S
        await step.ack.wait()
        i = 0
        for k in range(n_quiet):
            t = start + k * S + 1
            i += 1
            await step.send(ws, tmsg(i, t, "TESTBTC", q=str(7 + k % 5)), t)
        for k in range(burst):
            t = start + n_quiet * S + k * 100_000
            i += 1
            await step.send(ws, tmsg(i, t, "TESTBTC", q=str(7 + k % 5)), t)
        for k in range(30):
            t = start + (n_quiet + 10 + k) * S
            i += 1
            await step.send(ws, tmsg(i, t, "TESTBTC", q=str(7 + k % 5)), t)
        await ws.wait_closed()

    total = n_quiet + burst + 30

    async def main():
        nonlocal step
        step = Lockstep(clock)
        bus = Bus(maxsize=100_000)
        pipe = StreamPipeline(CFG, store, chief.detectors, chief, bus, "live")
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock, symbols=["TESTBTC"], start_hold_s=5)

            async def source():
                seen = 0
                async for ev in acked(sensor, step):
                    yield ev
                    if ev[0] == "trade":
                        seen += 1
                        if seen == total + 1:
                            return

            await pipe.run(source())
            await sensor.aclose()
        return pipe, sensor

    pipe, sensor = asyncio.run(main())
    assert sensor.gaps == 1 and pipe.late_trades() == 0
    series = pipe.series("TESTBTC")
    assert series and series[0]["t_end_us"] >= start + 1800 * S - 10 * S
    assert series[-1]["t_end_us"] < start + (n_quiet + 60) * S  # true time, published
    alerts = store.list_alerts()
    assert [a for a in alerts if a.window[1] >= start + n_quiet * S]
    assert all(a.window[1] < start + (n_quiet + 60) * S for a in alerts)


def test_a_backward_gap_undoes_what_was_measured_at_the_wrong_time(store):
    """A rogue start followed by a long lull: rows, samples and series a day ahead, then the
    correction. Rows must not arrive out of order, and nothing of the wrong time may stay."""
    chief = cheap_chief(CFG, "TESTBTC")
    pipe = StreamPipeline(CFG, store, chief.detectors, chief, Bus(maxsize=100_000), "live")
    wrong = T0 + DAY_S * S
    sample = live_trades(wrong, 1)[1][1]

    def second(t_us, trade_id):
        tr = sample.model_copy(update={"ts_us": t_us + 1, "recv_us": t_us, "trade_id": trade_id})
        return [("trade", tr), ("tick", t_us + S)]

    async def source():
        yield "preroll_done", wrong
        for k in range(1900):
            for ev in second(wrong + k * S, k):
                yield ev
        assert pipe.baseline_samples("TESTBTC") > 0 and pipe.series("TESTBTC")
        yield "gap", (wrong + 1900 * S, T0)
        assert pipe.baseline_samples("TESTBTC") == 0
        assert pipe.series("TESTBTC") == []
        for k in range(1900):
            for ev in second(T0 + k * S, 10_000 + k):
                yield ev

    asyncio.run(pipe.run(source()))
    series = pipe.series("TESTBTC")
    assert series and series[-1]["t_end_us"] <= T0 + 1900 * S
    assert pipe.baseline_samples("TESTBTC") > 0  # sampling went on from the true time


def test_delivered_ticks_do_not_go_back_when_the_offset_falls():
    """Enough rogues push the offset up; when they age out of the window it falls by under the
    restart limit, so ticks wait for the clock instead of repeating values."""
    clock = Clock()
    n = 1100
    step = None

    async def handler(ws):
        for i in range(n):
            t = T0 + i * S // 100
            await step.send(ws, tmsg(i, t + 30 * S if 10 <= i < 16 else t), t)
        await ws.wait_closed()

    async def main():
        nonlocal step
        step = Lockstep(clock)
        ticks, offsets = [], []
        async with server(handler) as url:
            sensor = sensor_for(url, clock=clock)
            async for kind, v in acked(sensor, step):
                offsets.append(sensor.clock_skew_ms)
                if kind == "tick":
                    ticks.append(v)
                elif kind == "trade" and v.trade_id == n - 1:
                    break
            await sensor.aclose()
        return ticks, offsets, sensor

    ticks, offsets, sensor = run(main())
    assert max(offsets) > 25_000 and abs(offsets[-1]) < 1000  # up, and back down
    assert sensor.gaps == 0
    assert ticks == sorted(set(ticks))


def test_the_skew_window_equals_the_sorted_form():
    rng = np.random.default_rng(3)
    for size, rank, n, spread in [(1000, 5, 5000, 10**6), (1000, 5, 3000, 7), (3, 5, 50, 5)]:
        window = live.SkewWindow(size, rank)
        recent: list[int] = []
        for x in rng.integers(-spread, spread, n).tolist():
            recent = [*recent, x][-size:]
            assert window.add(x) == sorted(recent)[-min(rank, len(recent))]


def test_warmup_filters_the_archive_to_the_sensor_symbols(archive):
    small = archive.filter(pl.col("ts_us") < T0 + 60 * S)
    other = small.with_columns(pl.lit("OTHERBTC").alias("symbol"))
    sensor = FakeSensor(live_trades(LIVE_T, 1), symbols=["TESTBTC"])

    async def main():
        return [ev async for ev in warmup_then_live(pl.concat([small, other]), sensor)]

    trades = [e[1] for e in run(main()) if e[0] == "trade"]
    assert {t.symbol for t in trades} == {"TESTBTC"}


def test_warmup_with_none_of_the_sensor_symbols_is_just_the_sensor(archive):
    sensor = FakeSensor(live_trades(LIVE_T, 1), symbols=["ETHUSDT"])

    async def main():
        return [ev async for ev in warmup_then_live(archive.head(10), sensor)]

    assert run(main()) == sensor.events


def test_baseline_samples_counts_only_the_window_ending_at_the_last_tick(archive, store):
    """An archive ending 20 h before connecting: most of its samples are about to age out."""
    chief = cheap_chief(CFG, "TESTBTC")
    end = T0 + 7 * 3600 * S
    connect = end + 20 * 3600 * S
    pipe = StreamPipeline(CFG, store, chief.detectors, chief, Bus(), "live")
    asyncio.run(pipe.run(warmup_then_live(archive, FakeSensor(live_trades(connect, 5)))))
    b = pipe._windows["TESTBTC"].kernel.baseline
    last_tick = connect + 5 * S + CFG.live.bar_grace_s * S  # the end of the source drains the grace
    times = b._t.view()
    want = int(((times >= last_tick - 86_400 * S) & (times < last_tick)).sum())
    assert 0 < want < times.size
    assert pipe.baseline_samples("TESTBTC") == want
    assert want < CFG.features.min_baseline_samples
    assert Auditor(CFG).warmup(pipe.baseline_samples("TESTBTC")).status == "warn"


def test_late_trades_are_summed_and_kept_across_a_gap(store):
    chief = cheap_chief(CFG, "TESTBTC")
    pipe = StreamPipeline(CFG, store, chief.detectors, chief, Bus(), "live")
    assert pipe.late_trades() == 0
    ev = live_trades(T0, 1)[1]
    old = ev[1].model_copy(update={"trade_id": 5})
    newer = ev[1].model_copy(update={"trade_id": 6, "ts_us": T0 + 7200 * S})

    async def source():
        yield ev
        yield "tick", T0 + 100 * S
        yield "trade", old  # its second closed long ago
        yield "gap", (T0 + 100 * S, T0 + 7200 * S)
        assert pipe.late_trades() == 1
        yield "trade", old  # older than the new origin
        yield "trade", newer

    asyncio.run(pipe.run(source()))
    assert pipe.late_trades() == 2


# ------------------------------------------------------------------ the guards stay
def test_injection_is_still_refused_in_live():
    with pytest.raises(EthicsError):
        assert_injection_allowed("live")
    frame = make_trades(date(2025, 3, 2), 1, seed=4, rate=0.01)
    sc = Scenario(
        family="pump_burst",
        technique_id="MM-T001",
        sub_technique="MM-T001.001",
        t_start_us=T0 + 3600 * S,
        duration_s=120.0,
        impact=0.15,
        vol_mult=30.0,
        rush_size=5,
        dump_tau_s=300.0,
        seed=7,
        region="test",
    )
    with pytest.raises(EthicsError):
        Replayer(frame, speed=0).inject(sc, 1.0, "live")


def test_live_source_is_kept_on_a_parsed_trade():
    assert parse_trade(body(1)).source == "live"
    assert np.isclose(parse_trade(body(1)).quote_qty, 10.0)


# ------------------------------------------------------------------ the one real connection
@pytest.mark.network
def test_btcusdt_yields_at_least_10_trades_in_10_s():
    async def main():
        cfg = load_settings()
        sensor = LiveTradeSensor([SYM], cfg.live.ws_url)
        n = 0
        try:
            async with asyncio.timeout(10):
                async for kind, _ in sensor:
                    n += kind == "trade"
                    if n >= 10:
                        break
        except TimeoutError:
            pass
        finally:
            await sensor.aclose()
        return n

    assert asyncio.run(main()) >= 10
