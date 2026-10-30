"""Stream pipeline: event source -> bars -> window rows -> scores -> Chief -> store + bus.

Reproduces the batch path of `pipeline.batch` on the same trades: rows are rounded through float32
exactly as `load_day_windows` returns them (detectors are fitted and calibrated on those), then
scored by the same `score_windows` and fused by the same `Chief`.

A source is an async iterable of `("trade", Trade)`, `("tick", t_us)`, `("preroll_done", t_us)` and
`("seek", t_us)` events (`data.replay.Replayer`). The first event fixes the first second of every
symbol's bars, so symbols that show up late still match a batch run over the same range. Nothing
is scored, stored or published until `preroll_done`; the pre-roll only warms up the features. A
source that wants output must therefore send `preroll_done`.

Every default detector is row-independent (no state between `score` calls), so skipping them in
the pre-roll cannot change a later score; CUSUM state lives in the feature kernel.
"""

import asyncio
import math
import time
from collections import deque
from collections.abc import AsyncIterable
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
from idsai.detection.features.streaming import StreamingWindows
from idsai.detection.features.windows import windows_schema
from idsai.pipeline.batch import _round_f32, score_windows

MODES = ("live", "replay", "sim")
LATENCY_SAMPLES = 10_000
SERIES_S = 86_400


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
        self._ring.clear()

    def series(self, symbol: str) -> list[dict[str, Any]]:
        """Up to the last 24 h of `{t_end_us, close, F, lane_rule, lane_stat, lane_ai}`, oldest
        first; non-finite values are None, so the result is strict JSON."""
        return list(self._ring.get(symbol, ()))

    async def run(self, source: AsyncIterable[tuple[str, Any]]) -> None:
        """Consume `source` to its end, then close the open episodes and store / publish them."""
        try:
            async for kind, payload in source:
                if kind == "trade":
                    self._on_trade(payload)
                elif kind == "tick":
                    self._on_tick(payload)
                elif kind == "preroll_done":
                    self._on_preroll_done(payload)
                elif kind == "seek":
                    self._close_episodes()
                    self._reset()
                else:
                    raise ValueError(f"unknown stream event {kind!r}")
        except asyncio.CancelledError:
            self._finish()
            raise
        self._finish()

    # ---- events ----

    def _on_trade(self, t: Trade) -> None:
        if self._origin_us is None:
            self._origin_us = t.ts_us // S_US * S_US
        self._builder(t.symbol).on_trade(t)

    def _builder(self, symbol: str) -> BarBuilder:
        b = self._builders.get(symbol)
        if b is None:
            b = BarBuilder(symbol, self._grace_s, start_us=self._origin_us)
            self._builders[symbol] = b
            unit = quote_unit_for(symbol, self.cfg.features)
            self._windows[symbol] = StreamingWindows(symbol, self.cfg.features, unit)
        return b

    def _on_tick(self, t_us: int) -> None:
        received = time.monotonic()
        if self._origin_us is None:
            self._origin_us = (t_us - 1) // S_US * S_US  # this tick closes the first second
        self._last_tick_us = t_us
        for b in self._builders.values():
            for bar in b.on_tick(t_us):
                self._on_bar(bar, received)

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
        frame = _round_f32(pl.DataFrame([row], schema=self._schema), self.cfg.features)
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

    def _close_episodes(self) -> None:
        self._emit(self._stream.flush())

    def _finish(self) -> None:
        if self._grace_s and self._last_tick_us is not None:
            self._on_tick(self._last_tick_us + self._grace_s * S_US)  # bars still in their grace
        self._close_episodes()
