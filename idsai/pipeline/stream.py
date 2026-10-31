"""Stream pipeline: event source -> bars -> window rows -> scores -> Chief -> store + bus.

Rows are rounded through float32 exactly as `load_day_windows` returns them (detectors are fitted
and calibrated on those), then scored by the same `score_windows` and fused by the same `Chief`.

Parity with the batch path: bit-identical from the same first bar. Across days, batch restarts its
kernel per day while stream runs on, so the first hour of later days differs (z-scores and F
included) and episodes crossing midnight may merge; after that hour only float32 rounding noise
remains (about 3e-8). After a seek, rows equal a fresh run from the re-pre-roll point.

A backward seek equals a clean run only when the target lies outside every episode's span plus
suppress gap. A target inside drops or truncates that episode: an open episode whose last
exceedance is before the target is discarded and not recreated, because the pre-roll is unscored.

A source is an async iterable of `("trade", Trade)`, `("tick", t_us)`, `("preroll_done", t_us)` and
`("seek", t_us)` events (`data.replay.Replayer`). The first event fixes the first second of every
symbol's bars, so symbols that show up late still match a batch run over the same range. Nothing
is scored, stored or published until `preroll_done`; the pre-roll only warms up the features. A
source that wants output must therefore send `preroll_done`.

`("gap", (from_us, to_us))` says that nothing was observed in that span (a LIVE outage, or the time
between an archive pre-roll and the first live trade). Empty bars there would enter the baseline as
zero activity and widen its scale, hiding real anomalies. So bars and kernels restart at `to_us`
and only the `Baseline` goes on: no rows until the new kernel has its `max(W)` bars (no measurement,
no alert; CUSUM and the Kamps-Kleinberg history start over too). Open episodes stay open.

A gap with `to_us < from_us` says the source's clock was ahead (a rogue first trade set it). What
was measured since the new origin has wrong times: samples and series points are dropped, episodes
are closed, and publishing starts from the new origin, since rows must go on in time order.

Every default detector is row-independent (no state between `score` calls), so skipping them in
the pre-roll cannot change a later score; CUSUM state lives in the feature kernel.

Bus messages: `bars` = bar dict, `scores` = `{symbol, **series point}`, `alerts` = `Alert`, or on a
seek one `{"event": "alerts_reset", "t_us": target, "deleted": [alert ids]}`, telling a dashboard
to drop those rows. A seek discards open episodes without any alert, and deletes their stored rows
plus every stored alert of this provenance whose window ends at or after the target; it is refused
in live mode, and live rows are never touched.
"""

import asyncio
import math
import time
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

import numpy as np
import polars as pl

from idsai.ai.chief import FUSION_LAYERS, Chief, ChiefStream, p_column
from idsai.bus import Bus
from idsai.config import Settings, quote_unit_for
from idsai.contracts.alert import Alert
from idsai.contracts.detector import Detector
from idsai.contracts.events import Trade
from idsai.data.store import Store
from idsai.detection.features.bars import S_US, BarBuilder
from idsai.detection.features.kernel import Baseline
from idsai.detection.features.streaming import StreamingWindows
from idsai.detection.features.windows import windows_schema
from idsai.pipeline.batch import round_f32, score_windows

MODES = ("live", "replay", "sim")
LATENCY_SAMPLES = 10_000
SERIES_S = 86_400
BACKLOG_YIELD_BARS = 1000  # bars closed by one tick between two yields to the event loop


def _finite_or_none(x: Any) -> Any:
    return None if isinstance(x, float) and not math.isfinite(x) else x


class StreamPipeline:
    """One run of the stream path. `mode` becomes the provenance of every alert."""

    def __init__(
        self,
        cfg: Settings,
        store: Store,
        detectors: list[Detector],
        chief: Chief,
        bus: Bus,
        mode: str,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.cfg = cfg
        self.store = store
        self.detectors = detectors
        self.chief = chief
        self.bus = bus
        self.mode = mode
        self.latency_ms: deque[float] = deque(maxlen=LATENCY_SAMPLES)
        self._schema = windows_schema(cfg.features)
        self._grace_s = cfg.live.bar_grace_s if mode == "live" else 0  # replay input is in order
        self._ring_maxlen = SERIES_S // cfg.features.stride_s
        self._ring: dict[str, deque[dict[str, Any]]] = {}
        self._lanes = [
            (layer, [p_column(d.name) for d in chief.detectors if d.layer == layer])
            for layer in FUSION_LAYERS
        ]
        self._reset()

    def _reset(self) -> None:
        self._builders: dict[str, BarBuilder] = {}
        self._windows: dict[str, StreamingWindows] = {}
        self._stream = ChiefStream(self.chief, self.mode)
        self._origin_us: int | None = None
        self._publish_from_us: int | None = None
        self._last_tick_us: int | None = None
        self._latest_us: int | None = None  # the last tick or gap end, for what is in the baseline
        self._late_before_gaps = 0
        self._ring.clear()

    def series(self, symbol: str) -> list[dict[str, Any]]:
        """Up to the last 24 h of `{t_end_us, close, F, lane_rule, lane_stat, lane_ai}`, oldest
        first; non-finite values are None, so the result is strict JSON."""
        return list(self._ring.get(symbol, ()))

    def baseline_samples(self, symbol: str) -> int:
        """Baseline samples of `symbol` in the window ending at the last tick (0 when unknown), for
        `Auditor.warmup`: samples of an old archive that are about to age out do not count."""
        w = self._windows.get(symbol)
        if w is None:
            return 0
        baseline = w.kernel.baseline
        return (
            baseline.n_samples
            if self._latest_us is None
            else baseline.samples_in_window(self._latest_us)
        )

    def late_trades(self) -> int:
        """Trades dropped as late so far, over all symbols and across gaps."""
        return self._late_before_gaps + sum(b.late for b in self._builders.values())

    async def run(self, source: AsyncIterable[tuple[str, Any]]) -> None:
        """Consume `source` to its end, then close the open episodes and store / publish them.

        The source's iterator is closed (`aclose`, if it has one) whenever this returns or raises,
        so a `Replayer` that fed this run refuses a later `seek`.
        """
        events = aiter(source)
        try:
            await self._consume(events)
        finally:
            aclose = getattr(events, "aclose", None)
            if aclose is not None:
                await aclose()

    async def _consume(self, events: AsyncIterator[tuple[str, Any]]) -> None:
        try:
            async for kind, payload in events:
                if kind == "trade":
                    self._on_trade(payload)
                elif kind == "tick":
                    await self._on_tick(payload)
                elif kind == "preroll_done":
                    self._on_preroll_done(payload)
                elif kind == "seek":
                    self._on_seek(payload)
                elif kind == "gap":
                    await self._on_gap(payload)
                else:
                    raise ValueError(f"unknown stream event {kind!r}")
        except asyncio.CancelledError:
            await self._finish()
            raise
        await self._finish()

    # ---- events ----

    def _on_trade(self, t: Trade) -> None:
        if self._origin_us is None:
            self._origin_us = t.ts_us // S_US * S_US
        self._builder(t.symbol).on_trade(t)

    def _builder(self, symbol: str, baseline: Baseline | None = None) -> BarBuilder:
        b = self._builders.get(symbol)
        if b is None:
            b = BarBuilder(symbol, self._grace_s, start_us=self._origin_us)
            self._builders[symbol] = b
            unit = quote_unit_for(symbol, self.cfg.features)
            self._windows[symbol] = StreamingWindows(
                symbol, self.cfg.features, unit, baseline=baseline
            )
        return b

    async def _on_tick(self, t_us: int) -> None:
        received = time.monotonic()
        if self._origin_us is None:
            self._origin_us = (t_us - 1) // S_US * S_US  # this tick closes the first second
        self._last_tick_us = t_us
        self._latest_us = t_us
        n = 0
        for b in self._builders.values():
            for bar in b.on_tick(t_us):
                self._on_bar(bar, received)
                n += 1
                if n % BACKLOG_YIELD_BARS == 0:  # a late symbol catches up on many empty bars
                    await asyncio.sleep(0)

    async def _on_gap(self, span: tuple[int, int]) -> None:
        backward = span[1] < span[0]
        if self._grace_s and self._last_tick_us is not None and not backward:
            await self._on_tick(self._last_tick_us + self._grace_s * S_US)  # bars still in grace
        baselines = {s: w.kernel.baseline for s, w in self._windows.items()}
        self._late_before_gaps = self.late_trades()
        self._builders.clear()
        self._windows.clear()
        self._origin_us = span[1] // S_US * S_US
        self._last_tick_us = None
        self._latest_us = span[1]
        for symbol, baseline in baselines.items():
            self._builder(symbol, baseline)
        if backward:
            self._undo_from(self._origin_us)

    def _undo_from(self, t_us: int) -> None:
        for w in self._windows.values():
            w.kernel.baseline.drop_from(t_us)
        for ring in self._ring.values():
            while ring and ring[-1]["t_end_us"] >= t_us:
                ring.pop()
        self._emit(self._stream.flush())
        self._stream = ChiefStream(self.chief, self.mode)
        if self._publish_from_us is not None:
            self._publish_from_us = min(self._publish_from_us, t_us)

    def _on_seek(self, target_us: int) -> None:
        if self.mode == "live":
            raise ValueError("a live pipeline cannot seek")
        deleted = self.store.delete_alerts(self.mode, self._stream.discard_open())
        deleted += self.store.delete_alerts_from(self.mode, target_us)
        self.bus.publish("alerts", {"event": "alerts_reset", "t_us": target_us, "deleted": deleted})
        self._reset()

    def _on_preroll_done(self, t_us: int) -> None:
        if self._origin_us is None:
            self._origin_us = t_us // S_US * S_US
        self._publish_from_us = t_us

    def _on_bar(self, bar: dict[str, Any], received: float) -> None:
        row = self._windows[bar["symbol"]].push_bar(bar)
        t_close = bar["t_us"] + S_US
        if self._publish_from_us is None or t_close < self._publish_from_us:
            return
        self.bus.publish("bars", {k: _finite_or_none(v) for k, v in bar.items()})
        if row is None:
            return
        frame = round_f32(pl.DataFrame([row], schema=self._schema), self.cfg.features)
        scored = score_windows(frame, self.detectors)
        alerts = self._stream.step(scored.to_dicts()[0])
        self.latency_ms.append((time.monotonic() - received) * 1000.0)

        point = self._point(scored, row["t_end_us"], bar["close"])
        symbol = bar["symbol"]
        ring = self._ring.get(symbol)
        if ring is None:
            ring = self._ring[symbol] = deque(maxlen=self._ring_maxlen)
        ring.append(point)
        self.bus.publish("scores", {"symbol": symbol, **point})
        self._emit(alerts)

    def _point(self, scored: pl.DataFrame, t_end_us: int, close: float) -> dict[str, Any]:
        fused = self.chief.fuse(scored)
        point: dict[str, Any] = {"t_end_us": t_end_us, "close": close, "F": fused["F"][0]}
        for layer, cols in self._lanes:
            # the same -log10 p the Chief reads from its table; + 0.0 turns -0.0 (p = 1) into 0.0
            nlp = -np.log10(np.array([fused[c][0] for c in cols])) if cols else None
            point[f"lane_{layer}"] = None if nlp is None else float(nlp.max()) + 0.0
        return {k: _finite_or_none(v) for k, v in point.items()}

    def _emit(self, alerts: list[Alert]) -> None:
        for a in alerts:
            self.store.upsert_alert(a)
            self.bus.publish("alerts", a)

    async def _finish(self) -> None:
        if self._grace_s and self._last_tick_us is not None:
            await self._on_tick(self._last_tick_us + self._grace_s * S_US)  # bars still in grace
        self._emit(self._stream.flush())
