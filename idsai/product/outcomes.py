"""Phase 2.3: the track record (ผลงานย้อนหลัง): what the price did 1 h after each stored alert.

Product layer only: it reads the store, the pipeline's ring and the trade archive, never the
detectors, and changes no config. Only `live` and `replay` alerts count, never `sim`.

An outcome row (`Store.alert_outcomes`) is `pending` until both prices are known (`done`); when one
of them still cannot be found 24 h after the alert it becomes `unknown`. `price_at` is the close of
the newest ring point at or before the alert, `price_1h` the close of the first ring point at or
after 1 h later, used only once the pipeline's clock (`latest_us`: the replay clock in replay) has
passed that time; the trade archive is the fallback for each. A price more than `MAX_GAP_US` away
from the time it stands for is not used.

The share of done alerts whose price was lower 1 h later is history, not a forecast and not a
score of the system: nothing here is advice to buy or sell.
"""

import asyncio
import logging
import statistics
from typing import Any

from idsai.contracts.alert import Alert
from idsai.data.store import COUNTED_PROVENANCE, Store, _day_of
from idsai.product.check import DISCLAIMER_TH, now_us_of

log = logging.getLogger(__name__)

S_US = 1_000_000
MIN_US = 60 * S_US
HOUR_US = 60 * MIN_US
OUTCOME_AFTER_US = HOUR_US  # price_1h is this long after the alert
UNKNOWN_AFTER_US = 24 * HOUR_US  # still no price this long after the alert: unknown
LOOKBACK_US = 25 * HOUR_US  # the tracker looks at alerts this recent, plus pending rows
MAX_GAP_US = 15 * MIN_US  # a price further than this from the time it stands for is not used
TRACK_PERIOD_S = 60
TRACK_LIMIT = 200  # the track-record page's rows, and the tracker's first pass
YIELD_EVERY = 50  # the tracker gives the event loop a turn after this many alerts

NOTE_TH = [
    "นับเฉพาะการแจ้งเตือนที่เซิร์ฟเวอร์นี้เก็บไว้ ดิสก์ของเซิร์ฟเวอร์ไม่ถาวร "
    "เมื่อเซิร์ฟเวอร์เริ่มใหม่หรือถูกติดตั้งใหม่ ตัวเลขทั้งหมดจะเริ่มนับใหม่",
    "การแจ้งเตือนคือความผิดปกติทางสถิติของการซื้อขาย ไม่ได้กล่าวหาผู้ใด",
    "ราคา 1 ชม. หลังเตือนเป็นข้อมูลย้อนหลัง ไม่ใช่การพยากรณ์ "
    "และสัดส่วนที่ราคาต่ำกว่าตอนเตือนใน 1 ชม. ไม่ได้วัดว่าระบบถูกหรือผิด",
    "«เตือนผิด» คือผู้ใช้ตัดสินว่าการแจ้งเตือนนั้นเป็นการเตือนผิด "
    "«ผิดปกติจริง» คือผู้ใช้ตัดสินว่าผิดปกติจริง",
]


# ---- price lookups: the ring first, then the trade archive ----


def _ring_at(pipeline: Any, symbol: str, t_us: int) -> tuple[int, float] | None:
    fn = getattr(pipeline, "close_at", None) if pipeline is not None else None
    return fn(symbol, t_us) if fn is not None else None


def _ring_after(pipeline: Any, symbol: str, t_us: int) -> tuple[int, float] | None:
    fn = getattr(pipeline, "close_after", None) if pipeline is not None else None
    return fn(symbol, t_us) if fn is not None else None


def _archive_at(store: Store, symbol: str, t_us: int) -> tuple[int, float] | None:
    """The last archived trade at or before `t_us` (within `MAX_GAP_US`); only read when that UTC
    day is stored, and only that window: never a walk back over earlier days."""
    if not store.has_day(symbol, _day_of(t_us)):
        return None
    df = store.read_trades(symbol, t_us - MAX_GAP_US, t_us + 1)
    return None if df.height == 0 else (int(df["ts_us"][-1]), float(df["price"][-1]))


def _archive_after(store: Store, symbol: str, t_us: int) -> tuple[int, float] | None:
    """The first archived trade at or after `t_us` (within `MAX_GAP_US`); only when stored."""
    if not store.has_day(symbol, _day_of(t_us)):
        return None
    df = store.read_trades(symbol, t_us, t_us + MAX_GAP_US + 1)
    return None if df.height == 0 else (int(df["ts_us"][0]), float(df["price"][0]))


def _near(hit: tuple[int, float] | None, t_us: int) -> tuple[int, float] | None:
    if hit is None or abs(hit[0] - t_us) > MAX_GAP_US or not hit[1] > 0:
        return None
    return hit


# ---- the pure step: one alert's outcome ----


def resolve_outcome(
    alert: Alert,
    prev: dict | None,
    now_us: int,
    pipeline: Any,
    store: Store,
    updated_us: int,
) -> dict:
    """The outcome row of `alert` at `now_us` (the pipeline's clock). `prev` is its stored row, if
    any: a known `price_at` is kept, so the ring may roll on. Same inputs, same row."""
    ts, sym = alert.ts_event_us, alert.symbol
    same = prev is not None and prev.get("ts_event_us") == ts
    price_at = prev.get("price_at") if same else None
    if price_at is None:
        hit = _near(_ring_at(pipeline, sym, ts), ts) or _near(_archive_at(store, sym, ts), ts)
        price_at = None if hit is None else hit[1]
    t1h = ts + OUTCOME_AFTER_US
    price_1h = t_1h = None
    settled_1h = False  # the archive holds the day of t1h and its window is over
    if now_us >= t1h:
        hit = _near(_ring_after(pipeline, sym, t1h), t1h)
        if hit is not None and hit[0] > now_us:  # the ring cannot be ahead of its clock
            hit = None
        if hit is None:
            hit = _near(_archive_after(store, sym, t1h), t1h)
            if hit is not None and hit[0] > now_us:  # the archive of a replay is: not yet
                hit = None
            settled_1h = (
                hit is None and now_us >= t1h + MAX_GAP_US and store.has_day(sym, _day_of(t1h))
            )
        if hit is not None:
            t_1h, price_1h = hit
    if price_at is not None and price_1h is not None:
        status = "done"
    elif now_us > ts + UNKNOWN_AFTER_US or settled_1h:  # no price will ever come
        status = "unknown"
    else:
        status = "pending"
    return {
        "alert_id": alert.alert_id,
        "symbol": sym,
        "ts_event_us": ts,
        "price_at": price_at,
        "price_1h": price_1h,
        "t_1h_us": t_1h,
        "status": status,
        "updated_us": updated_us,
    }


def _same_row(a: dict | None, b: dict) -> bool:
    return a is not None and all(a.get(k) == v for k, v in b.items() if k != "updated_us")


def _track_iter(store: Store, pipeline: Any, wall_us: int | None, backfill: bool):
    """The work of `track_step`, one alert at a time: yields after each alert it looks at and
    returns the rows written (None when the pipeline has no clock yet)."""
    if pipeline is not None and getattr(pipeline, "latest_us", None) is None:
        return None
    now_us = now_us_of(pipeline)
    updated = now_us if wall_us is None else wall_us
    if backfill:
        store.delete_orphan_outcomes()
        found = store.counted_alerts(until_us=now_us, limit=TRACK_LIMIT)
    else:
        found = store.counted_alerts(since_us=now_us - LOOKBACK_US, until_us=now_us)
    alerts = {a.alert_id: a for a in found}
    gone = []
    for row in store.list_outcomes("pending"):
        if row["alert_id"] in alerts:
            continue
        a = store.get_alert(row["alert_id"])
        if a is None or a.provenance not in COUNTED_PROVENANCE:
            gone.append(row["alert_id"])
        else:
            alerts[a.alert_id] = a
    if gone:
        store.delete_outcomes(gone)
    known = store.get_outcomes(list(alerts))
    n = 0
    for aid, a in alerts.items():
        prev = known.get(aid)
        if (
            prev is not None
            and prev["status"] in ("done", "unknown")
            and prev["ts_event_us"] == a.ts_event_us
        ):
            continue
        row = resolve_outcome(a, prev, now_us, pipeline, store, updated)
        if not _same_row(prev, row):
            store.upsert_outcome(row)
            n += 1
        yield
    return n


def track_step(
    store: Store, pipeline: Any, wall_us: int | None = None, backfill: bool = False
) -> int | None:
    """Bring the outcomes of recent alerts up to date; returns the rows written, or None when the
    pipeline has no clock yet (nothing is done before its first tick).

    Looks only at `live` / `replay` alerts of the last `LOOKBACK_US` up to the pipeline's clock
    (never one newer than it) and at rows still `pending`; a `done` / `unknown` row is final.
    `backfill` (the tracker's first pass only) looks instead at the newest `TRACK_LIMIT` stored
    alerts, so alerts stored before the tracker ran get a row once, and drops outcome rows whose
    alert is gone. A pending row whose alert is gone (a replay seek deleted it) is dropped too.
    Running it again changes nothing.
    """
    it = _track_iter(store, pipeline, wall_us, backfill)
    while True:
        try:
            next(it)
        except StopIteration as done:
            return done.value


async def track_step_async(
    store: Store, pipeline: Any, wall_us: int | None = None, backfill: bool = False
) -> int | None:
    """`track_step` that gives the event loop a turn every `YIELD_EVERY` alerts. It stays on the
    loop (never a thread): the pipeline's ring is not thread-safe."""
    it = _track_iter(store, pipeline, wall_us, backfill)
    k = 0
    while True:
        try:
            next(it)
        except StopIteration as done:
            return done.value
        k += 1
        if k % YIELD_EVERY == 0:
            await asyncio.sleep(0)


class OutcomeTracker:
    """Runs `track_step` every `period_s` seconds beside the app's other tasks. `state` is read
    each time (its `pipeline` may be replaced), so it is any object with `store` and `pipeline`."""

    def __init__(self, state: Any, period_s: float = TRACK_PERIOD_S):
        self.state = state
        self.period_s = period_s
        self._backfilled = False

    def step(self) -> int | None:
        n = track_step(self.state.store, self.state.pipeline, backfill=not self._backfilled)
        if n is not None:
            self._backfilled = True
        return n

    async def astep(self) -> int | None:
        """`step` on the event loop, yielding every `YIELD_EVERY` alerts."""
        n = await track_step_async(
            self.state.store, self.state.pipeline, backfill=not self._backfilled
        )
        if n is not None:
            self._backfilled = True
        return n

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.period_s)
            try:
                await self.astep()
            except Exception:  # one bad pass must not end the tracker (or the server)
                log.exception("outcome tracker pass failed")


# ---- the track-record JSON ----


def change_pct(price_at: float | None, price_1h: float | None) -> float | None:
    if price_at is None or price_1h is None or not price_at > 0:
        return None
    return (price_1h / price_at - 1.0) * 100.0


def track_record(store: Store, limit: int = TRACK_LIMIT) -> dict[str, Any]:
    """`{items, summary, note_th, disclaimer_th}`: the newest `limit` live / replay alerts with
    their 1 h outcome and feedback counts, and a summary over every such stored alert."""
    alerts = store.counted_alerts(limit=limit)
    ids = [a.alert_id for a in alerts]
    outs = store.get_outcomes(ids)
    fb = store.feedback_counts(ids)
    items = []
    for a in alerts:
        o = outs.get(a.alert_id)
        if o is not None and o["ts_event_us"] != a.ts_event_us:
            o = None  # a row of an older alert under the same id: not this one's
        price_at = None if o is None else o["price_at"]
        price_1h = None if o is None else o["price_1h"]
        done = o is not None and o["status"] == "done"
        items.append(
            {
                "alert_id": a.alert_id,
                "symbol": a.symbol,
                "ts_event_us": a.ts_event_us,
                "technique_id": a.technique_id,
                "phase": a.phase,
                "severity": a.severity,
                "price_at": price_at,
                "price_1h": price_1h,
                "change_1h_pct": change_pct(price_at, price_1h) if done else None,
                "status": "pending" if o is None else o["status"],
                "feedback": fb.get(a.alert_id, {"tp": 0, "fp": 0, "unsure": 0}),
            }
        )
    n_alerts, since_us = store.counted_alert_stats()
    done_prices = store.done_outcome_prices()
    changes = [c for p0, p1 in done_prices if (c := change_pct(p0, p1)) is not None]
    n_done = len(done_prices)
    return {
        "items": items,
        "summary": {
            "n_alerts": n_alerts,
            "n_done": n_done,
            "share_lower_1h": (sum(1 for p0, p1 in done_prices if p1 < p0) / n_done)
            if n_done
            else None,
            "median_change_1h_pct": float(statistics.median(changes)) if changes else None,
            "feedback_totals": store.feedback_totals(),
            "since_us": since_us,
        },
        "note_th": NOTE_TH,
        "disclaimer_th": DISCLAIMER_TH,
    }
