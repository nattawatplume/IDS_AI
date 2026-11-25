"""The web alert history line (1.3) and the track record (2.3): outcomes, API and the page."""

import asyncio
from types import SimpleNamespace

import polars as pl
import pytest
from starlette.testclient import TestClient

from idsai import cli
from idsai.contracts.frames import TRADES, conform
from idsai.data.store import Store
from idsai.product import outcomes as oc
from idsai.product.api import create_app
from idsai.product.explainer import HISTORY_TH, history_line
from idsai.product.notify import telegram as tg
from tests.product.test_api import D_US, SYM, env, make_alert, point  # noqa: F401
from tests.product.test_api_auth import aenv, auth_client  # noqa: F401

S = 1_000_000
MIN = 60 * S
HOUR = 60 * MIN


@pytest.fixture
def client(env):  # noqa: F811
    with TestClient(create_app(env.state)) as c:
        yield c


# ---- 1.3: the history line on the web alert detail ----


@pytest.mark.parametrize(
    ("phase", "want"), [("pump", HISTORY_TH), (None, HISTORY_TH), ("sell", None)]
)
def test_alert_detail_history_line_by_phase(client, env, phase, want):  # noqa: F811
    env.store.upsert_alert(make_alert("h1", phase=phase))
    body = client.get("/api/alerts/h1").json()
    assert body["history_th"] == want
    assert history_line(make_alert("x", phase=phase)) == want


def test_telegram_text_keeps_the_same_history_line():
    assert tg.HISTORY_TH is HISTORY_TH
    assert HISTORY_TH == (
        "ข้อมูลในอดีต (pump จริงที่ระบบจับได้ คู่ BTC 2018–2021): 1 ชม. หลังเตือน "
        "ราคาต่ำกว่าตอนเตือน 89% ของครั้ง แต่ในชั่วโมงนั้นราคายังขึ้นต่อได้ (มัธยฐาน +15%)"
    )
    ts = 1_700_000_000 * S
    pump = tg.message_text(make_alert("p", ts=ts, phase="pump"))
    assert pump.splitlines()[3] == HISTORY_TH
    assert HISTORY_TH not in tg.message_text(make_alert("s", ts=ts, phase="sell"))


def test_web_alert_detail_shows_the_history_line_muted_with_its_source():
    from tests.product.test_web_static import text

    js = text("app.js")
    note = js[js.index("function historyNote(a)") :]
    note = note[: note.index("\n}\n")]
    assert "a.history_th" in note and '"small muted history-note"' in note
    assert 'const HISTORY_SOURCE = "ที่มา: ผลเฟส 0 ใน docs/ROADMAP.md";' in js
    show = js[js.index("async function showAlert(id)") :]
    assert "historyNote(a)" in show[: show.index("\n}\n")]


# ---- 2.3: outcomes ----

T0 = D_US + 3 * HOUR  # the alert time in these tests


class RingPipe:
    """The ring reads the tracker uses: a list of (t_end_us, close), oldest first."""

    def __init__(self, points=(), latest_us=None):
        self.points = list(points)
        self.latest_us = latest_us

    def close_at(self, symbol, t_us):
        hits = [p for p in self.points if p[0] <= t_us and symbol == SYM]
        return hits[-1] if hits else None

    def close_after(self, symbol, t_us):
        hits = [p for p in self.points if p[0] >= t_us and symbol == SYM]
        return hits[0] if hits else None


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "d", tmp_path / "o.duckdb") as st:
        yield st


def trades(rows):
    n = len(rows)
    df = pl.DataFrame(
        {
            "ts_us": [t for t, _ in rows],
            "recv_us": [t for t, _ in rows],
            "symbol": [SYM] * n,
            "price": [p for _, p in rows],
            "qty": [1.0] * n,
            "quote_qty": [p for _, p in rows],
            "taker_side": ["buy"] * n,
            "trade_id": list(range(1, n + 1)),
            "n_fills": [1] * n,
            "source": ["test"] * n,
        }
    )
    return conform(df, TRADES)


def outcome(store, aid):
    return store.get_outcomes([aid]).get(aid)


def test_ring_hit_gives_a_done_outcome(store):
    store.upsert_alert(make_alert("r", ts=T0))
    pipe = RingPipe([(T0 - 5 * S, 100.0), (T0 + HOUR + 5 * S, 90.0)], latest_us=T0 + HOUR + MIN)
    assert oc.track_step(store, pipe) == 1
    o = outcome(store, "r")
    assert o["status"] == "done" and o["price_at"] == 100.0 and o["price_1h"] == 90.0
    assert o["t_1h_us"] == T0 + HOUR + 5 * S and o["symbol"] == SYM and o["ts_event_us"] == T0


def test_archive_fallback_when_the_ring_has_nothing(store):
    store.write_trades(trades([(T0 - 30 * S, 50.0), (T0 + HOUR + 2 * S, 55.0)]))
    store.upsert_alert(make_alert("a", ts=T0))
    oc.track_step(store, RingPipe(latest_us=T0 + 2 * HOUR))
    o = outcome(store, "a")
    assert (o["status"], o["price_at"], o["price_1h"]) == ("done", 50.0, 55.0)
    assert o["t_1h_us"] == T0 + HOUR + 2 * S


def test_pending_before_one_hour_even_with_the_archive_ahead(store):
    # a replay's archive holds the "future": it is not read before the clock passes 1 h
    store.write_trades(trades([(T0 - 30 * S, 50.0), (T0 + HOUR + 2 * S, 55.0)]))
    store.upsert_alert(make_alert("p", ts=T0))
    pipe = RingPipe([(T0, 50.0), (T0 + HOUR + 5 * S, 60.0)], latest_us=T0 + HOUR - S)
    oc.track_step(store, pipe)
    o = outcome(store, "p")
    assert (o["status"], o["price_at"], o["price_1h"], o["t_1h_us"]) == (
        "pending",
        50.0,
        None,
        None,
    )
    # past 1 h: the ring's next point is beyond the clock, the archive trade at +1 h 2 s is not
    pipe.latest_us = T0 + HOUR + 3 * S
    oc.track_step(store, pipe)
    o = outcome(store, "p")
    assert o["status"] == "done" and o["price_1h"] == 55.0


def test_unknown_after_24_hours_without_a_price(store):
    store.upsert_alert(make_alert("u", ts=T0))
    pipe = RingPipe(latest_us=T0 + 2 * HOUR)
    oc.track_step(store, pipe)
    assert outcome(store, "u")["status"] == "pending"
    pipe.latest_us = T0 + 24 * HOUR + S
    oc.track_step(store, pipe)
    o = outcome(store, "u")
    assert o["status"] == "unknown" and o["price_at"] is None and o["price_1h"] is None
    pipe.points = [(T0, 1.0), (T0 + HOUR, 2.0)]  # final: a late price changes nothing
    assert oc.track_step(store, pipe) == 0 and outcome(store, "u")["status"] == "unknown"


def test_a_far_price_is_not_used(store):
    store.upsert_alert(make_alert("f", ts=T0))
    gap = oc.MAX_GAP_US + S
    oc.track_step(store, RingPipe([(T0 - gap, 1.0), (T0 + HOUR + gap, 2.0)], T0 + 3 * HOUR))
    o = outcome(store, "f")
    assert o["status"] == "pending" and o["price_at"] is None and o["price_1h"] is None


def test_sim_alerts_are_never_tracked(store):
    store.upsert_alert(make_alert("s", ts=T0, prov="sim"))
    store.upsert_alert(make_alert("l", ts=T0, prov="live"))
    pipe = RingPipe([(T0, 1.0), (T0 + HOUR, 2.0)], latest_us=T0 + 2 * HOUR)
    oc.track_step(store, pipe)
    assert outcome(store, "s") is None and outcome(store, "l")["status"] == "done"
    assert [i["alert_id"] for i in oc.track_record(store)["items"]] == ["l"]


def test_re_runs_are_idempotent(store):
    store.upsert_alert(make_alert("i", ts=T0))
    store.upsert_alert(make_alert("j", ts=T0 + 10 * MIN))
    pipe = RingPipe([(T0, 10.0), (T0 + HOUR, 11.0)], latest_us=T0 + HOUR + MIN)
    assert oc.track_step(store, pipe, wall_us=1) == 2  # i done, j pending
    first = store.list_outcomes()
    assert {r["alert_id"]: r["status"] for r in first} == {"i": "done", "j": "pending"}
    assert oc.track_step(store, pipe, wall_us=2) == 0
    assert oc.track_step(store, pipe, wall_us=3) == 0
    assert store.list_outcomes() == first  # not even updated_us moved


def test_only_recent_alerts_and_pending_rows_are_looked_at(store, monkeypatch):
    store.upsert_alert(make_alert("old", ts=T0 - 30 * HOUR))
    store.upsert_alert(make_alert("new", ts=T0))
    pipe = RingPipe(latest_us=T0 + MIN)
    seen = []
    real = oc.resolve_outcome

    def spy(a, *rest):
        seen.append(a.alert_id)
        return real(a, *rest)

    monkeypatch.setattr(oc, "resolve_outcome", spy)
    oc.track_step(store, pipe)
    assert seen == ["new"]  # 30 h old and without a row: not scanned each minute
    seen.clear()
    oc.track_step(store, pipe, backfill=True)  # the tracker's first pass only
    assert sorted(seen) == ["new", "old"]
    assert outcome(store, "old")["status"] == "unknown"
    # a pending row is followed past the lookback until it is final
    store.upsert_alert(make_alert("p", ts=T0 - 20 * HOUR))
    pipe.latest_us = T0 - 20 * HOUR + MIN
    oc.track_step(store, pipe)
    assert outcome(store, "p")["status"] == "pending"
    pipe.latest_us = T0 + 6 * HOUR  # 26 h after p
    seen.clear()
    oc.track_step(store, pipe)
    assert "p" in seen and outcome(store, "p")["status"] == "unknown"


def test_pending_row_of_a_deleted_alert_is_dropped(store):
    store.upsert_alert(make_alert("d", ts=T0))
    oc.track_step(store, RingPipe(latest_us=T0 + MIN))
    assert outcome(store, "d")["status"] == "pending"
    store.delete_alerts("replay", ["d"])
    oc.track_step(store, RingPipe(latest_us=T0 + 2 * MIN))
    assert outcome(store, "d") is None


def test_nothing_runs_before_the_pipeline_has_a_clock(store):
    store.upsert_alert(make_alert("c", ts=T0))
    assert oc.track_step(store, RingPipe(latest_us=None)) is None
    assert outcome(store, "c") is None
    tracker = oc.OutcomeTracker(SimpleNamespace(store=store, pipeline=RingPipe()))
    assert tracker.step() is None and not tracker._backfilled
    tracker.state.pipeline.latest_us = T0 + MIN
    assert tracker.step() == 1 and tracker._backfilled


def test_archive_price_at_reads_only_the_window_before_the_alert(store, monkeypatch):
    def no_walk(*a, **k):
        raise AssertionError("last_trade_before walks back over days")

    monkeypatch.setattr(store, "last_trade_before", no_walk)
    store.write_trades(
        trades(
            [
                (T0 - 40 * MIN, 40.0),  # a walk back would find it; the window never does
                (T0 - 10 * MIN, 45.0),
                (T0 - S, 47.0),  # the last trade at or before the alert
                (T0 + S, 99.0),  # after the alert: never the price at it
            ]
        )
    )
    assert oc._archive_at(store, SYM, T0) == (T0 - S, 47.0)
    assert oc._archive_at(store, SYM, T0 - 11 * MIN) is None  # [T0-26m, T0-11m] has no trade
    assert oc._archive_at(store, SYM, T0 - 40 * MIN + oc.MAX_GAP_US) == (T0 - 40 * MIN, 40.0)
    assert oc._archive_at(store, SYM, T0 + 2 * 86_400 * S) is None  # that day is not stored


def test_unknown_at_once_when_the_archive_day_has_no_trade_near_1h(store):
    store.write_trades(trades([(T0 - S, 50.0), (T0 + HOUR + oc.MAX_GAP_US + MIN, 51.0)]))
    store.upsert_alert(make_alert("g", ts=T0))
    pipe = RingPipe(latest_us=T0 + HOUR + oc.MAX_GAP_US - S)
    oc.track_step(store, pipe)
    assert outcome(store, "g")["status"] == "pending"  # the window after 1 h is not over yet
    pipe.latest_us = T0 + HOUR + oc.MAX_GAP_US
    oc.track_step(store, pipe)
    o = outcome(store, "g")
    assert o["status"] == "unknown" and o["price_at"] == 50.0 and o["price_1h"] is None
    # without the archive day the row waits for the ring, up to 24 h
    store.upsert_alert(make_alert("w", ts=T0 + 2 * 86_400 * S))
    pipe.latest_us = T0 + 2 * 86_400 * S + 2 * HOUR
    oc.track_step(store, pipe)
    assert outcome(store, "w")["status"] == "pending"


def test_alerts_newer_than_the_clock_are_not_tracked(store):
    store.upsert_alert(make_alert("now", ts=T0))
    store.upsert_alert(make_alert("ahead", ts=T0 + MIN))
    pipe = RingPipe(latest_us=T0)
    oc.track_step(store, pipe)
    oc.track_step(store, pipe, backfill=True)
    assert outcome(store, "now") is not None and outcome(store, "ahead") is None


def test_backfill_is_bounded_and_drops_orphan_rows(store, monkeypatch):
    monkeypatch.setattr(oc, "TRACK_LIMIT", 2)
    for k in range(4):
        store.upsert_alert(make_alert(f"b{k}", ts=T0 - (40 + k) * HOUR))
    store.upsert_outcome(
        {
            "alert_id": "orphan",
            "symbol": SYM,
            "ts_event_us": T0 - 50 * HOUR,
            "price_at": 1.0,
            "price_1h": 2.0,
            "t_1h_us": T0 - 49 * HOUR,
            "status": "done",
            "updated_us": 1,
        }
    )
    oc.track_step(store, RingPipe(latest_us=T0), backfill=True)
    assert {r["alert_id"] for r in store.list_outcomes()} == {"b0", "b1"}  # the newest two
    store.upsert_alert(make_alert("orphan", ts=T0 - 50 * HOUR))  # a normal pass keeps rows
    store.upsert_outcome({**store.get_outcomes(["b0"])["b0"], "alert_id": "orphan"})
    store.delete_alerts("replay", ["orphan"])
    oc.track_step(store, RingPipe(latest_us=T0))
    assert outcome(store, "orphan") is not None
    oc.track_step(store, RingPipe(latest_us=T0), backfill=True)
    assert outcome(store, "orphan") is None


def test_async_pass_yields_to_the_loop(store, monkeypatch):
    monkeypatch.setattr(oc, "YIELD_EVERY", 2)
    for k in range(5):
        store.upsert_alert(make_alert(f"y{k}", ts=T0 - k * MIN))
    turns = []
    real_sleep = asyncio.sleep

    async def sleep(d, *a):
        if d == 0:
            turns.append(1)
        return await real_sleep(d, *a)

    monkeypatch.setattr(oc.asyncio, "sleep", sleep)
    tracker = oc.OutcomeTracker(SimpleNamespace(store=store, pipeline=RingPipe(latest_us=T0)))
    assert asyncio.run(tracker.astep()) == 5 and tracker._backfilled
    assert len(turns) == 2  # after alerts 2 and 4
    assert asyncio.run(tracker.astep()) == 0  # same rows as the sync pass would write


def test_tracker_runs_periodically_and_survives_a_bad_pass(store):
    calls = []

    class Flaky(oc.OutcomeTracker):
        async def astep(self):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return 0

    async def main():
        task = asyncio.create_task(Flaky(SimpleNamespace(store=store, pipeline=None), 0.01).run())
        await asyncio.sleep(0.1)
        assert not task.done()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())
    assert len(calls) >= 2


def test_serve_runs_the_tracker_beside_the_session_purge(monkeypatch):
    steps = []

    async def astep(self):
        steps.append(1)

    monkeypatch.setattr(oc.OutcomeTracker, "astep", astep)
    monkeypatch.setattr(oc.OutcomeTracker.__init__, "__defaults__", (0.01,))
    ran = []

    async def pipeline():
        ran.append(1)  # ends at once, as a finished replay does

    async def main():
        state = SimpleNamespace(store=None, pipeline=None)
        task = asyncio.create_task(cli.with_outcomes(state, pipeline)())
        await asyncio.sleep(0.1)
        assert not task.done()  # it outlives the pipeline, as the purge does
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())
    assert ran == [1] and len(steps) >= 2
    src = open(cli.__file__, encoding="utf-8").read()
    assert "with_outcomes(state, with_session_purge(store, background))" in src


# ---- 2.3: the track-record API ----


def seed_record(st):
    """Five counted alerts (and a sim one) with known outcomes and votes."""
    rows = [  # alert id, ts, price_at, price_1h, status
        ("a", T0, 100.0, 90.0, "done"),  # -10 %
        ("b", T0 + HOUR, 100.0, 104.0, "done"),  # +4 %
        ("c", T0 + 2 * HOUR, 200.0, 190.0, "done"),  # -5 %
        ("d", T0 + 3 * HOUR, 10.0, None, "pending"),
    ]
    for aid, ts, p0, p1, status in rows:
        st.upsert_alert(make_alert(aid, ts=ts, phase="sell" if aid == "b" else "pump"))
        st.upsert_outcome(
            {
                "alert_id": aid,
                "symbol": SYM,
                "ts_event_us": ts,
                "price_at": p0,
                "price_1h": p1,
                "t_1h_us": None if p1 is None else ts + HOUR,
                "status": status,
                "updated_us": 1,
            }
        )
    st.upsert_alert(make_alert("e", ts=T0 + 4 * HOUR, prov="live"))  # no outcome row yet
    st.upsert_alert(make_alert("sim", ts=T0 + 5 * HOUR, prov="sim"))
    st.insert_feedback("a", "fp", "ann")
    st.insert_feedback("a", "tp", "bob")
    st.insert_feedback("a", "fp", "bob")  # bob changed his mind: his latest vote counts once
    st.insert_feedback("b", "unsure", "ann")
    st.insert_feedback("sim", "tp", "ann")  # never counted


ITEM_KEYS = {
    "alert_id",
    "symbol",
    "ts_event_us",
    "technique_id",
    "phase",
    "severity",
    "price_at",
    "price_1h",
    "change_1h_pct",
    "status",
    "feedback",
}
TRACK_FORBIDDEN = ("ความน่าจะเป็น", "probability", "accuracy", "แม่นยำ", "ควรซื้อ", "ควรขาย")


def test_track_record_shape_maths_and_feedback(client, env):  # noqa: F811
    seed_record(env.store)
    r = client.get("/api/track-record")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"items", "summary", "note_th", "disclaimer_th"}
    items = body["items"]
    assert [i["alert_id"] for i in items] == ["e", "d", "c", "b", "a"]  # newest first, no sim
    assert all(set(i) == ITEM_KEYS for i in items)
    by = {i["alert_id"]: i for i in items}
    assert by["a"]["change_1h_pct"] == pytest.approx(-10.0)
    assert by["b"]["change_1h_pct"] == pytest.approx(4.0) and by["b"]["phase"] == "sell"
    assert by["d"]["status"] == "pending" and by["d"]["change_1h_pct"] is None
    assert by["e"]["status"] == "pending" and by["e"]["price_at"] is None
    assert by["a"]["feedback"] == {"tp": 0, "fp": 2, "unsure": 0}
    assert by["b"]["feedback"] == {"tp": 0, "fp": 0, "unsure": 1}
    assert by["c"]["feedback"] == {"tp": 0, "fp": 0, "unsure": 0}
    s = body["summary"]
    assert s["n_alerts"] == 5 and s["n_done"] == 3
    assert s["share_lower_1h"] == pytest.approx(2 / 3)
    assert s["median_change_1h_pct"] == pytest.approx(-5.0)
    assert s["feedback_totals"] == {"tp": 0, "fp": 2, "unsure": 1}
    assert s["since_us"] == T0
    assert body["disclaimer_th"] == "ข้อมูลนี้ไม่ใช่คำแนะนำการลงทุน"
    note = " ".join(body["note_th"])
    for want in ("เซิร์ฟเวอร์", "ไม่ถาวร", "เริ่มนับใหม่", "ไม่ใช่การพยากรณ์", "เตือนผิด"):
        assert want in note, want
    assert "ความผิดปกติทางสถิติ" in note
    text = r.content.decode("utf-8")
    for word in TRACK_FORBIDDEN:
        assert word not in text, word
    small = client.get("/api/track-record?limit=2").json()
    assert [i["alert_id"] for i in small["items"]] == ["e", "d"]
    assert small["summary"] == s  # the summary is over every stored alert, not the page
    for bad in (0, 201):
        assert client.get(f"/api/track-record?limit={bad}").status_code == 422


def test_track_record_when_empty(client):
    body = client.get("/api/track-record").json()
    assert body["items"] == []
    assert body["summary"] == {
        "n_alerts": 0,
        "n_done": 0,
        "share_lower_1h": None,
        "median_change_1h_pct": None,
        "feedback_totals": {"tp": 0, "fp": 0, "unsure": 0},
        "since_us": None,
    }


def test_track_record_needs_sign_in(auth_client):  # noqa: F811
    assert auth_client().get("/api/track-record").status_code == 401
    assert auth_client("viewer").get("/api/track-record").status_code == 200


def test_track_fields_read_by_app_js_exist(client, env):  # noqa: F811
    from tests.product.test_web_static import reads

    seed_record(env.store)
    body = client.get("/api/track-record").json()
    assert reads("rec") and reads("rec") <= set(body)
    assert reads("sum") and reads("sum") <= set(body["summary"])
    assert reads("it") and reads("it") <= ITEM_KEYS
    assert reads("tot") <= set(body["summary"]["feedback_totals"])


def test_track_page_markup_and_wording():
    from tests.product.test_web_static import text

    html, js, css = text("index.html"), text("app.js"), text("styles.css")
    assert '<a href="#/track" data-route="track">' in html and "<span>ผลงานย้อนหลัง</span>" in html
    page = html[html.index('id="page-track"') : html.index("<!-- Simulate")]
    for want in (
        "ความผิดปกติทางสถิติ",
        "ข้อมูลนี้ไม่ใช่คำแนะนำการลงทุน",
        "สัดส่วนที่ราคาต่ำกว่าตอนเตือนใน 1 ชม.",
        "ไม่ใช่การพยากรณ์",
        "เตือนผิด",  # false alarms are listed, never hidden
        'id="track-body"',
        'class="table-wrap"',  # a phone scrolls the table, not the page
    ):
        assert want in page, want
    block = js[js.index("const TRACK_POLL_MS") : js.index("function renderTrack(")]
    block += js[js.index("function renderTrack(") :].split("\n}\n")[0]
    for want in (
        '"รอครบ 1 ชม."',
        '"ไม่มีข้อมูลราคา"',
        'api("/api/track-record")',
        "alertHref(it.alert_id)",
    ):
        assert want in block, want
    assert "const TRACK_POLL_MS = 60000;" in js
    assert 'state.route === "track") loadTrack();' in js  # polls only while the page is shown
    assert 'track: "page-track"' in js and '"#/track"' in js
    assert "fmtThaiDateTime(it.ts_event_us)" in block and "TH_OFFSET_US" in block
    for word in TRACK_FORBIDDEN:
        assert word not in page and word not in block, word
    assert ".track-tiles { grid-template-columns: minmax(0, 1fr); }" in css


def test_track_prices_never_use_exponent_notation(tmp_path):
    from tests.product.test_web_static import _run_dom, text

    js = text("app.js")
    fn = js[js.index("function fmtPrice(x)") :].split("\n}\n")[0]
    assert "toPrecision" not in fn and "toExponential" not in fn
    assert 'maximumSignificantDigits: 4' in fn and "toFixed(8)" in fn
    _run_dom(tmp_path, "true", FMT_PRICE_BODY)


FMT_PRICE_BODY = r"""
for (const [x, want] of [
  [5.98e-6, "0.00000598"], [0.0000001, "0.00000010"], [0.000512, "0.000512"],
  [0.0123456, "0.01235"], [12.3456, "12.35"], [43210.5, "43,210"], [0, "0"], [NaN, "–"],
]) {
  const got = m.fmtPrice(x);
  assert.strictEqual(got, want, `${x}: ${got}`);
  assert.ok(!/e/i.test(got), got);
}
console.log("ok");
"""


def test_restart_note_names_restarts_too():
    assert "เมื่อเซิร์ฟเวอร์เริ่มใหม่หรือถูกติดตั้งใหม่ ตัวเลขทั้งหมดจะเริ่มนับใหม่" in oc.NOTE_TH[0]


def test_track_error_is_drawn_only_on_the_track_page():
    from tests.product.test_web_static import text

    js = text("app.js")
    load = js[js.index("async function loadTrack(") :].split("\n}\n")[0]
    catch = load[load.index("} catch (e) {") :]
    assert catch.index('state.route !== "track"') < catch.index('$("track-error")')


def test_track_page_renders_rows_as_text(tmp_path):
    """The page draws an API answer into the DOM with text nodes only (node + the dom shim)."""
    from tests.product.test_web_static import _run_dom

    _run_dom(tmp_path, "true", TRACK_DOM_BODY)


TRACK_DOM_BODY = r"""
(async () => {
  const at = Date.UTC(2021, 0, 18, 16, 30) * 1000;  // 23:30 in Bangkok
  const rec = {
    items: [
      { alert_id: "x<b>", symbol: "STEEMBTC", ts_event_us: at, technique_id: "MM-T001",
        phase: "pump", severity: 2, price_at: 0.0005, price_1h: 0.00045, change_1h_pct: -10,
        status: "done",
        feedback: { tp: 0, fp: 2, unsure: 0 } },
      { alert_id: "y", symbol: "STEEMBTC", ts_event_us: at - 3600e6, technique_id: "MM-T001",
        phase: null, severity: 1, price_at: null, price_1h: null, change_1h_pct: null,
        status: "unknown", feedback: { tp: 1, fp: 0, unsure: 0 } },
      { alert_id: "z", symbol: "STEEMBTC", ts_event_us: at - 7200e6, technique_id: "MM-T001",
        phase: "sell", severity: 1, price_at: 0.0005, price_1h: null, change_1h_pct: null,
        status: "pending", feedback: { tp: 0, fp: 0, unsure: 0 } },
    ],
    summary: { n_alerts: 3, n_done: 1, share_lower_1h: 1, median_change_1h_pct: -10,
      feedback_totals: { tp: 1, fp: 2, unsure: 0 }, since_us: at - 7200e6 },
    note_th: ["<i>หมายเหตุ</i>"],
    disclaimer_th: "ข้อมูลนี้ไม่ใช่คำแนะนำการลงทุน",
  };
  globalThis.fetch = async (path) => ({ ok: true, status: 200, json: async () =>
    (path.startsWith("/api/track-record") ? rec : {}) });
  m.setRoute("track");
  await m.loadTrack();
  const rows = walk(document.getElementById("track-body"), (n) => n.tagName === "TR");
  assert.strictEqual(rows.length, 3);
  const cells = (r) => r.children.map((c) => c.textContent);
  assert.ok(cells(rows[0])[0].includes("18/01/2021 23:30 น."), cells(rows[0])[0]);
  assert.ok(cells(rows[0])[5].includes("-10.00%"), cells(rows[0])[5]);
  assert.strictEqual(cells(rows[0])[3], "กำลังพุ่ง");
  assert.ok(cells(rows[0])[6].includes("เตือนผิด 2"), cells(rows[0])[6]);
  assert.strictEqual(cells(rows[1])[5], "ไม่มีข้อมูลราคา");
  assert.strictEqual(cells(rows[2])[5], "รอครบ 1 ชม.");
  const link = walk(rows[0], (n) => n.tagName === "A")[0];
  assert.strictEqual(link.getAttribute("href"), "#/alerts/x%3Cb%3E");
  assert.strictEqual(document.getElementById("track-lower").textContent, "100%");
  assert.ok(document.getElementById("track-lower-sub").textContent.includes("1 ครั้ง"));
  const notes = walk(document.getElementById("track-notes"), (n) => n.tagName === "LI");
  assert.strictEqual(notes[0].textContent, "<i>หมายเหตุ</i>");  // text, never markup
  assert.strictEqual(document.getElementById("track-empty").hidden, true);
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_series_output_is_unchanged_by_the_track_record(client, env):  # noqa: F811
    env.pipe.points[SYM] = [point(i) for i in range(3)]
    before = client.get(f"/api/series?symbol={SYM}").content
    seed_record(env.store)
    oc.track_step(env.store, env.pipe)
    client.get("/api/track-record")
    assert client.get(f"/api/series?symbol={SYM}").content == before
