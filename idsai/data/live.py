"""LIVE sensor: public Binance trades as the same async event stream as `replay.Replayer`.

Read-only by construction: the only hosts are Binance's public market-data host and loopback (for
tests), redirects are not followed, and nothing is ever sent: no application message, no header
with a key, no credentials in the URL. The library answers pings; there is no order code here.

Events: `("trade", Trade)`, `("tick", t_us)`, `("preroll_done", t_us)` and `("gap", (from, to))`.

Clock: ticks are wall-clock, but a local clock ahead of the exchange by more than the bar grace
would make `BarBuilder` drop every trade as late. So ticks read `local now + offset`, with `offset`
the largest `T - recv` of the last 1000 trades (the smallest network delay seen), and never go back.

Start: `preroll_done` comes first and once, as soon as the clock can be trusted: at the first trade,
in its second, so that trade is not late for a pipeline taking its origin from the marker; or
60 s after connecting if the market is silent. Ticks start with it.

Outages: no ticks while disconnected. Back within 60 s: the missed ticks follow, a short empty fill
(the trade_id gap is the Auditor's job). Longer, however the time got lost (a sleeping laptop
counts): `("gap", (last tick, now))` first, and ticks resume from there.
"""

import asyncio
import json
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

import polars as pl
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from idsai.contracts.events import Trade
from idsai.data.replay import Replayer
from idsai.data.store import _check_symbol

S_US = 1_000_000
ROTATE_S = 23 * 3600  # Binance closes a connection at 24 h
BACKOFF_FIRST_S = 1
BACKOFF_MAX_S = 60
GAP_RESTART_S = 60
SILENT_START_S = 60
DEDUP_SIZE = 10_000
SKEW_SAMPLES = 1000
QUEUE_SIZE = 10_000
MAX_POLL_S = 0.25  # longest wait for a message before the clock is looked at again
MIN_POLL_S = 0.01
CLOSE_TIMEOUT_S = 2
MIN_TS_US = 10**15  # 2001 in microseconds: a smaller T is in another unit, not an old trade
PUBLIC_HOST = "data-stream.binance.vision"
LOOPBACK = ("127.0.0.1", "localhost", "::1")
_FIELDS = ("e", "E", "s", "t", "p", "q", "T", "m")
_LOST = (OSError, WebSocketException)


def _now_us() -> int:
    return time.time_ns() // 1000


def _positive(msg: dict[str, Any], key: str) -> float:
    if isinstance(msg[key], bool) or not isinstance(msg[key], str | int | float):
        raise ValueError(f"trade field {key!r} is not a number: {msg[key]!r}")
    x = float(msg[key])  # ValueError for text that is no number
    if not (math.isfinite(x) and x > 0):
        raise ValueError(f"trade field {key!r} must be finite and > 0, got {msg[key]!r}")
    return x


def _integer(msg: dict[str, Any], key: str) -> int:
    if not isinstance(msg[key], int) or isinstance(msg[key], bool):
        raise ValueError(f"trade field {key!r} is not an integer: {msg[key]!r}")
    return msg[key]


def parse_trade(msg: str | bytes | dict[str, Any], recv_us: int | None = None) -> Trade:
    """One `@trade` message, combined-stream envelope or bare, as a `Trade`.

    Raises ValueError for anything that is not a well-formed trade; it never guesses a field.
    """
    if isinstance(msg, str | bytes):
        msg = json.loads(msg)  # JSONDecodeError and UnicodeDecodeError are ValueErrors
    if isinstance(msg, dict) and "data" in msg:
        msg = msg["data"]
    if not isinstance(msg, dict):
        raise ValueError(f"not a trade message: {msg!r}")
    missing = [k for k in _FIELDS if k not in msg]
    if missing:
        raise ValueError(f"trade message without {missing}")
    if msg["e"] != "trade":
        raise ValueError(f"not a trade event: {msg['e']!r}")
    if not isinstance(msg["s"], str):
        raise ValueError(f"trade symbol is not text: {msg['s']!r}")
    if not isinstance(msg["m"], bool):
        raise ValueError(f"trade field 'm' is not a boolean: {msg['m']!r}")
    price, qty = _positive(msg, "p"), _positive(msg, "q")
    quote = price * qty
    if not math.isfinite(quote):
        raise ValueError(f"trade quote value overflows: {price!r} * {qty!r}")
    ts_us, trade_id = _integer(msg, "T"), _integer(msg, "t")
    if ts_us < MIN_TS_US:
        raise ValueError(f"trade time is not in microseconds: {ts_us}")
    if trade_id < 0:  # negative ids are the simulator's
        raise ValueError(f"trade id must be >= 0, got {trade_id}")
    return Trade(
        ts_us=ts_us,
        recv_us=_now_us() if recv_us is None else recv_us,
        symbol=_check_symbol(msg["s"]),
        price=price,
        qty=qty,
        quote_qty=quote,
        taker_side="sell" if msg["m"] else "buy",
        trade_id=trade_id,
        n_fills=1,
        source="live",
    )


def _check_url(url: str) -> str:
    """The URL, if it can only reach the public read-only host or loopback."""
    parts = urlsplit(url)
    host = parts.hostname
    if host != PUBLIC_HOST and host not in LOOPBACK:
        raise ValueError(f"host {host!r} not allowed: only {PUBLIC_HOST} or loopback")
    if parts.scheme != "wss" and not (parts.scheme == "ws" and host in LOOPBACK):
        raise ValueError(f"scheme {parts.scheme!r} not allowed for {host}")
    if parts.username or parts.password:
        raise ValueError("the URL must not carry credentials")
    if parts.query or parts.fragment:
        raise ValueError("the URL must not carry a query: the sensor builds its own")
    return url


def _check_symbols(symbols: list[str]) -> list[str]:
    if isinstance(symbols, str) or not symbols:
        raise ValueError(f"symbols must be a non-empty list, got {symbols!r}")
    for s in symbols:
        if not isinstance(s, str):
            raise ValueError(f"invalid symbol {s!r}")
        _check_symbol(s)
    return list(dict.fromkeys(s.upper() for s in symbols))


class _NoRedirectConnect(connect):
    """A redirect could lead away from the allowed hosts: it is a failed handshake instead."""

    def process_redirect(self, exc: Exception) -> Exception:
        return exc


class LiveTradeSensor:
    """One-consumer async iterator of live events; one use only, `aclose()` when done.

    Connecting starts at the first `__anext__`. Cancelling a waiting `__anext__` closes the sensor,
    so a cancelled consumer leaves no task or socket behind: do not wrap it in a timeout per event.

    Counters: `reconnects` = connections opened after a lost one or a failed attempt, `rotations` =
    proactive ones after `rotate_s` (no backoff), `dupes`, `gaps`, `bad` (malformed or unasked).
    A slow consumer blocks the reader (bounded queue, then TCP): nothing is dropped here, and if the
    exchange gives up on us, that is an ordinary reconnect.
    """

    def __init__(
        self,
        symbols: list[str],
        url: str,
        rotate_s: float = ROTATE_S,
        *,
        clock: Callable[[], int] = _now_us,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        queue_size: int = QUEUE_SIZE,
    ):
        if not rotate_s > 0:
            raise ValueError(f"rotate_s must be > 0, got {rotate_s}")
        self.symbols = _check_symbols(symbols)
        self.url = _check_url(url)
        self.rotate_s = rotate_s
        self.reconnects = 0
        self.rotations = 0
        self.dupes = 0
        self.bad = 0
        self.gaps = 0
        self._clock = clock
        self._sleep = sleep
        self._loopback = urlsplit(url).hostname in LOOPBACK
        self._subscribed = frozenset(self.symbols)
        self._queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None
        self._closed = False
        self._seen: set[tuple[str, int]] = set()
        self._seen_order: deque[tuple[str, int]] = deque()
        self._skews: deque[int] = deque(maxlen=SKEW_SAMPLES)
        self._offset_us = 0
        self._next_tick: int | None = None  # None until the start
        self._delay_s = BACKOFF_FIRST_S

    @property
    def stream_url(self) -> str:
        streams = "/".join(f"{s.lower()}@trade" for s in self.symbols)
        return f"{self.url}?streams={streams}&timeUnit=MICROSECOND"

    @property
    def clock_skew_ms(self) -> float:
        """Added to the local clock for ticks; about -10 000 with a local clock 10 s ahead."""
        return self._offset_us / 1000

    # ---- iterator ----

    def __aiter__(self) -> AsyncIterator[tuple[str, Any]]:
        return self

    async def __anext__(self) -> tuple[str, Any]:
        if self._closed:
            raise StopAsyncIteration
        if self._task is None:
            self._task = asyncio.create_task(self._produce(), name="live-trade-sensor")
        try:
            ev = await self._queue.get()
        except asyncio.CancelledError:
            await self.aclose()
            raise
        if ev is None:
            error = self._error
            await self.aclose()
            if error is not None:
                raise error
            raise StopAsyncIteration
        return ev

    async def aclose(self) -> None:
        """Stop the reader and close its socket; safe to call twice."""
        self._closed = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # ---- reader ----

    async def _produce(self) -> None:
        try:
            await self._connect_forever()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # a bug, not a network state: the consumer must hear about it
            self._error = e
            await self._queue.put(None)

    async def _connect_forever(self) -> None:
        lost = False
        while True:
            rotated = False
            try:
                async with _NoRedirectConnect(
                    self.stream_url,
                    proxy=None if self._loopback else True,
                    close_timeout=CLOSE_TIMEOUT_S,
                ) as ws:
                    if lost:
                        self.reconnects += 1
                    lost = False
                    rotated = await self._session(ws)
            except _LOST:
                pass
            if rotated:
                self.rotations += 1
                continue
            lost = True
            await self._sleep(self._delay_s)
            self._delay_s = min(self._delay_s * 2, BACKOFF_MAX_S)

    async def _session(self, ws: Any) -> bool:
        """Read one connection until it is lost (raises) or due for rotation (True)."""
        connected = self._clock()
        rotate_us = self.rotate_s * S_US
        while True:
            now = self._clock()
            if now - connected >= rotate_us:
                return True
            wait = await self._advance(now, connected)
            try:
                async with asyncio.timeout(wait):
                    raw = await ws.recv()
            except TimeoutError:
                continue
            self._delay_s = BACKOFF_FIRST_S
            await self._on_message(raw)

    async def _on_message(self, raw: str | bytes) -> None:
        recv = self._clock()
        try:
            trade = parse_trade(raw, recv_us=recv)
        except ValueError:
            self.bad += 1
            return
        if trade.symbol not in self._subscribed:
            self.bad += 1
            return
        key = (trade.symbol, trade.trade_id)
        if key in self._seen:
            self.dupes += 1
            return
        self._seen.add(key)
        self._seen_order.append(key)
        if len(self._seen_order) > DEDUP_SIZE:
            self._seen.discard(self._seen_order.popleft())
        self._skews.append(trade.ts_us - recv)
        self._offset_us = max(self._skews)
        if self._next_tick is None:
            await self._start(trade.ts_us)
        await self._queue.put(("trade", trade))

    async def _start(self, t_us: int) -> None:
        second = t_us // S_US * S_US
        self._next_tick = second + S_US
        await self._queue.put(("preroll_done", second))

    async def _advance(self, now: int, connected: int) -> float:
        """Emit what the clock has reached; returns how long to wait for a message, in seconds."""
        t = now + self._offset_us
        if self._next_tick is None:
            if now - connected < SILENT_START_S * S_US:
                return MAX_POLL_S
            await self._start(t)
        last_tick = self._next_tick - S_US
        if t - last_tick > GAP_RESTART_S * S_US:
            self.gaps += 1
            await self._queue.put(("gap", (last_tick, t)))
            self._next_tick = t // S_US * S_US + S_US
        while self._next_tick <= t:
            await self._queue.put(("tick", self._next_tick))
            self._next_tick += S_US
        return min(max((self._next_tick - t) / S_US, MIN_POLL_S), MAX_POLL_S)


async def warmup_then_live(
    archive_trades: pl.DataFrame | None, sensor: Any
) -> AsyncIterator[tuple[str, Any]]:
    """Warm the features up on an archive, unpublished, then go on with `sensor`.

    The archive plays at max speed with no `preroll_done`, so nothing of it is published. Then
    `("gap", (archive end, sensor start))`, since nothing between them was observed, and the
    sensor's events, whose own first `preroll_done` starts publishing. No archive (or an empty one):
    just the sensor. The sensor connects only after the archive has played, and its first event
    supplies "now", on the clock it has corrected, so a skewed local clock cannot misplace the new
    origin. The sensor is closed with this generator.
    """
    events = aiter(sensor)
    try:
        if archive_trades is not None and not archive_trades.is_empty():
            end = None
            replay = aiter(Replayer(archive_trades, speed=0))
            try:
                async for kind, payload in replay:
                    if kind == "preroll_done":
                        continue
                    if kind == "tick":
                        end = payload
                    yield kind, payload
            finally:
                await replay.aclose()
            first = await anext(events, None)
            if first is None:
                return
            kind, payload = first
            now = payload.ts_us if kind == "trade" else payload
            yield "gap", (end, max(end, now))
            yield first
        async for ev in events:
            yield ev
    finally:
        aclose = getattr(events, "aclose", None)
        if aclose is not None:
            await aclose()
