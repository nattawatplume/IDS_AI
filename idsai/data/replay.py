"""Time Machine: replay a frame of trades in event time as an async stream of events.

Events: `("trade", Trade)`, `("tick", t_us)`, `("preroll_done", t_us)` and `("seek", t_us)`.

Order: trades by `(ts_us, trade_id)` (`symbol` breaks ties between symbols); `("tick", t)` comes
after every trade with `ts_us < t` and before any trade at `t`, one tick per second including
empty seconds, from the first second + 1 s to the last trade's second + 1 s. The first second is
the first trade's, or `preroll_from_us` / `start_us` when that is earlier.

Pre-roll: events before `start_us` run at max speed from `preroll_from_us`, so a consumer can warm
up its features. `("preroll_done", start)` then says that everything before `start` was delivered;
`("tick", start)`, which closes the row ending at `start`, comes right after it. Without a pre-roll
the marker comes first, so a consumer never waits for it forever.

`seek(t)` restarts from `max(first second, t - 24 h)`: `("seek", t)` first, so the consumer drops
its state, then the pre-roll, then `("preroll_done", t)`. A replay that ran out cannot be sought
(RuntimeError), nor can one whose iterator was closed: start a new Replayer.

Speed only changes the delays, never the events. All controls are plain synchronous calls: with a
single event loop they always run between two events, so they cannot tear the replay state.
"""

import asyncio
import math
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import polars as pl

from idsai.contracts.events import Trade
from idsai.contracts.frames import TRADES, conform
from idsai.detection.simulator.guard import assert_injection_allowed
from idsai.detection.simulator.inject import inject as simulator_inject
from idsai.detection.simulator.scenarios import Scenario

S_US = 1_000_000
PREROLL_US = 86_400 * S_US
CHUNK_ROWS = 4096  # trades turned into Python objects at a time
YIELD_EVERY = 256  # events between two yields to the event loop at max speed
PAUSE_POLL_S = 0.01
MIN_SLEEP_S = 0.005  # shorter delays are saved up: timers cannot do better on every platform
MAX_SLEEP_S = 0.05  # longest sleep before pause / seek / set_speed are looked at again
ORDER = ["ts_us", "trade_id", "symbol"]


def _check_speed(x: float) -> float:
    try:
        x = float(x)
    except OverflowError:
        raise ValueError("speed is too large") from None
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"speed must be finite and >= 0 (0 = max), got {x}")
    return x


class Replayer:
    """One-consumer async iterator over `trades`; the input frame is never modified."""

    def __init__(
        self,
        trades: pl.DataFrame,
        speed: float = 1.0,
        start_us: int | None = None,
        preroll_from_us: int | None = None,
    ):
        self._speed = _check_speed(speed)
        if start_us is not None and preroll_from_us is not None and preroll_from_us > start_us:
            raise ValueError("preroll_from_us must not be after start_us")
        self._columns = TRADES.names()
        # a requested range may begin before the first trade: its first bars are then empty, and
        # rows match a batch run from the same first bar only if those bars exist here too
        requested = preroll_from_us if preroll_from_us is not None else start_us
        self._requested_first = None if requested is None else requested // S_US * S_US
        self._load(conform(trades, TRADES).sort(ORDER))
        start = self._first if start_us is None else self._clamp(start_us)
        if preroll_from_us is not None and preroll_from_us > start:
            raise ValueError("preroll_from_us must not be after the start of the data range")
        origin = start if preroll_from_us is None else self._clamp(preroll_from_us)
        self._paused = False
        self._announce_seek: int | None = None
        self._ended = False
        self._begin(origin, start)

    # ---- state ----

    def _load(self, df: pl.DataFrame) -> None:
        self._df = df
        self._ts = df["ts_us"].to_numpy().astype(np.int64, copy=False)
        self._chunk_lo = 0
        self._chunk: list[tuple] = []
        data_first = int(self._ts[0]) // S_US * S_US if self._ts.size else None
        firsts = [t for t in (data_first, self._requested_first) if t is not None]
        self._first = min(firsts, default=0)
        self._last_tick = int(self._ts[-1]) // S_US * S_US + S_US if self._ts.size else self._first

    def _clamp(self, t_us: int) -> int:
        return min(max(t_us // S_US * S_US, self._first), self._last_tick)

    def _begin(self, origin: int, start: int) -> None:
        self._i = int(np.searchsorted(self._ts, origin, side="left"))
        self._next_tick = origin + S_US
        self._start = start
        self._pos = origin
        self._preroll_pending = True
        self._pace_us = start
        self._debt_us = 0.0

    @property
    def cursor_us(self) -> int:
        """Play head: no event at or after it has been delivered since the last pre-roll."""
        return max(self._pos, self._start)

    @property
    def end_us(self) -> int:
        """Event time of the last tick: nothing is delivered after it."""
        return self._last_tick

    @property
    def ended(self) -> bool:
        """The iterator ran out or was closed: nothing more will be delivered."""
        return self._ended

    @property
    def symbols(self) -> list[str]:
        """The symbols in the data, sorted."""
        return self._df["symbol"].unique().sort().to_list()

    # ---- controls ----

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def set_speed(self, x: float) -> None:
        """Event-time seconds per wall second; 0 = max speed."""
        self._speed = _check_speed(x)

    def seek(self, t_us: int) -> None:
        """Jump to `t_us` (floored to the second, clamped to the data range).

        Raises RuntimeError once the iterator has ended or been closed. Leaving an `async for`
        with `break` does not close it: until `aclose()` runs, the seek is accepted and never plays.
        """
        if self._ended:
            raise RuntimeError("this replay has ended and cannot seek: start a new Replayer")
        t = self._clamp(t_us)
        self._begin(max(self._first, t - PREROLL_US), t)
        self._announce_seek = t

    def inject(
        self, sc: Scenario, base_quote_per_s: float, mode: str, symbol: str | None = None
    ) -> pl.DataFrame:
        """Inject `sc` into the undelivered trades; returns the simulator's LABELS row.

        `mode` must allow injection (never "live"). The scenario may not start before `cursor_us`,
        so nothing delivered changes. Only this replay's in-memory copy changes, and it stays
        changed after a `seek`. `symbol` is required when the input has several symbols.
        """
        assert_injection_allowed(mode)
        symbols = self.symbols
        if symbol is None:
            if len(symbols) != 1:
                raise ValueError(f"inject needs symbol= on multi-symbol input, got {symbols}")
            symbol = symbols[0]
        if int(sc.t_start_us) < self.cursor_us:
            raise ValueError(
                f"scenario starts at {sc.t_start_us}, before the cursor {self.cursor_us}"
            )
        mine = self._df.filter(pl.col("symbol") == symbol)
        new, label = simulator_inject(mine, sc, base_quote_per_s, mode)
        rest = self._df.filter(pl.col("symbol") != symbol)
        # rows before t_start are unchanged and nothing is added before it, so self._i still counts
        # the same delivered rows
        self._load(pl.concat([rest, new]).sort(ORDER))
        return label

    # ---- events ----

    def _trade(self, i: int) -> Trade:
        if not self._chunk_lo <= i < self._chunk_lo + len(self._chunk):
            self._chunk_lo = i
            self._chunk = self._df.slice(i, CHUNK_ROWS).rows()
        return Trade(**dict(zip(self._columns, self._chunk[i - self._chunk_lo], strict=True)))

    def _next_event(self) -> tuple[str, int] | None:
        """`(kind, event time)` of the next event, without consuming it."""
        ts = int(self._ts[self._i]) if self._i < self._ts.size else None
        if self._preroll_pending and self._next_tick >= self._start:
            if ts is None or ts >= self._start:
                return "preroll_done", self._start
        if ts is not None and ts < self._next_tick:
            return "trade", ts
        if self._next_tick <= self._last_tick:
            return "tick", self._next_tick
        return None

    def _consume(self, kind: str, t: int) -> Any:
        if kind == "trade":
            trade = self._trade(self._i)
            self._i += 1
            self._pos = t + 1
            return trade
        if kind == "tick":
            self._next_tick += S_US
            self._pos = t
        else:
            self._preroll_pending = False
            self._pace_us = t
        return t

    async def _pace(self, t: int) -> bool:
        """Wait out the event-time delay up to `t`; False if a control interrupted the wait."""
        self._debt_us += max(0, t - self._pace_us)
        self._pace_us = max(self._pace_us, t)
        while self._speed > 0 and self._debt_us / self._speed >= MIN_SLEEP_S * 1e6:
            delay = min(self._debt_us / self._speed / 1e6, MAX_SLEEP_S)
            t0 = time.monotonic()
            await asyncio.sleep(delay)
            # charge what really passed, so timer overshoot does not add up over a run
            self._debt_us -= (time.monotonic() - t0) * self._speed * 1e6
            if self._paused or self._announce_seek is not None:
                return False
        if self._speed == 0:
            self._debt_us = 0.0
        return True

    async def _events(self) -> AsyncIterator[tuple[str, Any]]:
        since_yield = 0
        try:
            while True:
                if self._announce_seek is not None:
                    t, self._announce_seek = self._announce_seek, None
                    yield "seek", t
                    continue
                if self._paused:
                    await asyncio.sleep(PAUSE_POLL_S)
                    continue
                ev = self._next_event()
                if ev is None:
                    return
                kind, t = ev
                if not self._preroll_pending and not await self._pace(t):
                    continue
                yield kind, self._consume(kind, t)
                since_yield += 1
                if since_yield >= YIELD_EVERY:
                    since_yield = 0
                    await asyncio.sleep(0)
        finally:
            self._ended = True

    def __aiter__(self) -> AsyncIterator[tuple[str, Any]]:
        return self._events()
