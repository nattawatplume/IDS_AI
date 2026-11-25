"""History persistence: live alerts, feedback and outcomes copied to a backend and restored."""

import pytest

from idsai.data import store as store_mod
from idsai.data.history import (
    OUTCOME_COLS,
    HistoryStoreError,
    MemoryHistory,
    PostgresHistory,
    build_backend,
)
from idsai.data.store import _OUTCOME_COLS, Store
from tests.product.test_api import D_US, SYM, make_alert

S = 1_000_000
HOUR = 3600 * S
T0 = D_US + 3 * HOUR


def new_store(tmp_path, name):
    return Store(tmp_path / name, tmp_path / name / "app.duckdb")


@pytest.fixture
def store(tmp_path):
    with new_store(tmp_path, "a") as s:
        yield s


def out_row(aid, ts=T0, status="pending", p1=None):
    return {
        "alert_id": aid,
        "symbol": SYM,
        "ts_event_us": ts,
        "price_at": 100.0,
        "price_1h": p1,
        "t_1h_us": None if p1 is None else ts + HOUR,
        "status": status,
        "updated_us": 1,
    }


def test_outcome_columns_match_the_store():
    assert OUTCOME_COLS == _OUTCOME_COLS


def test_build_backend_none():
    assert build_backend(None) is None
    assert build_backend("") is None
    assert isinstance(build_backend("postgresql://x"), PostgresHistory)


def test_no_backend_queues_nothing(store):
    store.upsert_alert(make_alert("l", prov="live"))
    store.insert_feedback("l", "tp", "ann")
    store.upsert_outcome(out_row("l"))
    assert not store.history_attached and store.history_since_us is None
    assert store.history_pending() == (0, 0, 0)
    assert store.history_sync() is True


def test_round_trip_live_only(tmp_path, store):
    mem = MemoryHistory()
    store.attach_history(mem, since_us=0)
    assert store.history_attached and store.history_since_us == 0
    store.upsert_alert(make_alert("l1", ts=T0, prov="live"))
    store.insert_alert(make_alert("l2", ts=T0 + HOUR, prov="live"))
    store.upsert_alert(make_alert("r", ts=T0, prov="replay"))
    store.upsert_alert(make_alert("s", ts=T0, prov="sim"))
    store.insert_feedback("l1", "fp", "ann")
    store.insert_feedback("r", "tp", "ann")
    store.insert_feedback("s", "tp", "ann")
    store.upsert_outcome(out_row("l1", status="done", p1=90.0))
    store.upsert_outcome(out_row("r"))
    assert store.history_pending() == (2, 1, 1)
    assert store.history_sync() is True
    assert store.history_pending() == (0, 0, 0)
    assert set(mem.alerts) == {"l1", "l2"}
    assert [k[0] for k in mem.feedback] == ["l1"]
    assert set(mem.outcomes) == {"l1"} and mem.outcomes["l1"]["price_1h"] == 90.0

    with new_store(tmp_path, "b") as b:
        b.attach_history(mem, since_us=0)
        assert {a.alert_id for a in b.list_alerts()} == {"l1", "l2"}
        assert b.get_alert("l1") == store.get_alert("l1")
        fb = b.list_feedback("l1")
        assert [(f["verdict"], f["analyst"]) for f in fb] == [("fp", "ann")]
        assert fb[0]["ts_us"] == store.list_feedback("l1")[0]["ts_us"]
        assert b.get_outcomes(["l1"])["l1"] == store.get_outcomes(["l1"])["l1"]
        assert b.history_pending() == (0, 0, 0)  # restored rows are not queued again


def test_updated_episode_replaces_the_backend_row(store):
    mem = MemoryHistory()
    store.attach_history(mem, since_us=0)
    store.upsert_alert(make_alert("e", ts=T0, prov="live", severity=1))
    store.history_sync()
    store.upsert_alert(make_alert("e", ts=T0 + 60 * S, prov="live", severity=3))
    store.history_sync()
    assert len(mem.alerts) == 1
    assert mem.alerts["e"]["ts_event_us"] == T0 + 60 * S
    assert '"severity":3' in mem.alerts["e"]["json"]


def test_failure_keeps_the_queue_and_detection_never_raises(store):
    mem = MemoryHistory()
    store.attach_history(mem, since_us=0)
    mem.fail = True
    store.upsert_alert(make_alert("x", prov="live"))  # the write path never touches the backend
    store.insert_feedback("x", "unsure", "ann")
    store.upsert_outcome(out_row("x"))
    with pytest.raises(HistoryStoreError):
        store.history_sync()
    assert store.history_pending() == (1, 1, 1)
    store.upsert_alert(make_alert("y", prov="live"))  # queued while the batch was out
    with pytest.raises(HistoryStoreError):
        store.history_sync()
    assert store.history_pending() == (2, 1, 1)
    mem.fail = False
    assert store.history_sync() is True
    assert set(mem.alerts) == {"x", "y"} and len(mem.feedback) == 1 and set(mem.outcomes) == {"x"}
    assert store.history_pending() == (0, 0, 0)


def test_requeue_keeps_the_newer_writes_newest(store, monkeypatch):
    mem = MemoryHistory()
    store.attach_history(mem, since_us=0)
    store.upsert_alert(make_alert("old", prov="live"))

    def fail_and_write(rows):
        store.upsert_alert(make_alert("new", prov="live"))  # arrives while the batch is out
        raise HistoryStoreError("down")

    monkeypatch.setattr(mem, "upsert_alerts", fail_and_write)
    with pytest.raises(HistoryStoreError):
        store.history_sync()
    assert list(store._pending_alerts) == ["old", "new"]


def test_restore_skips_duplicates_and_cuts_at_since(tmp_path, store):
    mem = MemoryHistory()
    old = make_alert("old", ts=T0 - 40 * 24 * HOUR, prov="live")
    keep = make_alert("keep", ts=T0, prov="live")
    for a in (old, keep):
        mem.upsert_alerts(
            [
                dict(
                    zip(
                        ("alert_id", "ts_event_us", "symbol", "json"),
                        Store._alert_row(a),
                        strict=True,
                    )
                )
            ]
        )
    mem.insert_feedback(
        [
            {"alert_id": "keep", "analyst": "ann", "ts_us": 5, "verdict": "tp"},
            {"alert_id": "old", "analyst": "ann", "ts_us": 6, "verdict": "fp"},
        ]
    )
    mem.upsert_outcomes(
        [out_row("keep", status="done", p1=99.0), out_row("old", ts=old.ts_event_us)]
    )
    # a store that already has `keep` (locally changed) and its feedback row
    local = make_alert("keep", ts=T0, prov="live", severity=3)
    store.upsert_alert(local)
    store._db().execute("INSERT INTO feedback VALUES ('keep', 'tp', 'ann', 5)")
    store.attach_history(mem, since_us=T0 - 30 * 24 * HOUR)
    assert {a.alert_id for a in store.list_alerts()} == {"keep"}
    assert store.get_alert("keep").severity == 3  # the stored alert is kept
    assert len(store.list_feedback("keep")) == 1  # the exact duplicate is skipped
    assert store.list_feedback("old") == []
    assert store.get_outcomes(["keep"])["keep"]["price_1h"] == 99.0  # outcomes are replaced
    assert store.get_outcomes(["old"]) == {}
    assert store.history_pending() == (0, 0, 0)
    # attaching twice changes nothing
    store.attach_history(mem, since_us=T0 - 30 * 24 * HOUR)
    assert len(store.list_feedback("keep")) == 1


def test_attach_failure_leaves_the_store_detached(store):
    with pytest.raises(HistoryStoreError):
        store.attach_history(MemoryHistory(fail=True), since_us=0)
    assert not store.history_attached
    store.upsert_alert(make_alert("l", prov="live"))
    assert store.history_pending() == (0, 0, 0)


def test_pending_queues_are_bounded(store, monkeypatch, caplog):
    monkeypatch.setattr(store_mod, "HISTORY_MAX_PENDING", 5)
    mem = MemoryHistory(fail=True)
    store._history = mem  # attached, the backend down
    for i in range(8):
        store.upsert_alert(make_alert(f"a{i}", ts=T0 + i * S, prov="live"))
        store.insert_feedback(f"a{i}", "tp", "ann")
    assert list(store._pending_alerts) == [f"a{i}" for i in range(3, 8)]
    assert len(store._pending_feedback) == 5
    warned = [r for r in caplog.records if "history queue" in r.getMessage()]
    assert len(warned) == 1  # said once
    with pytest.raises(HistoryStoreError):
        store.history_sync()
    assert store.history_pending()[0] == 5


# ---- PostgresHistory with a mocked connection ----


class _Cur:
    def __init__(self, con):
        self.con = con

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, args=()):
        self.con.log.append(("execute", sql, args))

    def executemany(self, sql, rows):
        self.con.log.append(("many", sql, list(rows)))

    def fetchall(self):
        return self.con.results.pop(0) if self.con.results else []


class _Con:
    closed = False

    def __init__(self):
        self.log = []
        self.results = []

    def execute(self, sql, *a):
        self.log.append(("con", sql))

    def cursor(self):
        return _Cur(self)


def test_postgres_history_batches_and_maps_rows(monkeypatch):
    import psycopg

    con = _Con()
    monkeypatch.setattr(psycopg, "connect", lambda url, **kw: con)
    be = PostgresHistory("postgresql://u:p@h/db")
    be.ensure_schema()
    assert all("ids_" in sql for kind, sql, *_ in con.log if kind == "execute")
    con.log.clear()
    be.upsert_alerts([{"alert_id": "a", "ts_event_us": 1, "symbol": SYM, "json": "{}"}] * 3)
    be.insert_feedback([{"alert_id": "a", "analyst": "x", "ts_us": 2, "verdict": "tp"}])
    be.upsert_outcomes([out_row("a")])
    be.upsert_alerts([])  # nothing to send: no statement
    kinds = [k for k, *_ in con.log]
    assert kinds == ["many", "many", "many"]
    assert "ON CONFLICT (alert_id) DO UPDATE" in con.log[0][1] and len(con.log[0][2]) == 3
    assert (
        "ON CONFLICT (alert_id, analyst, ts_us) DO UPDATE SET verdict = EXCLUDED.verdict"
        in (con.log[1][1])
    )
    assert con.log[2][1].endswith("WHERE ids_alert_outcomes.status = 'pending'")
    con.log.clear()
    con.results = [
        [("a", 1, SYM, "{}")],
        [("a", "x", 2, "tp")],
        [tuple(out_row("a")[c] for c in OUTCOME_COLS)],
    ]
    alerts, fb, outs = be.load(0)
    assert len(con.log) == 3  # one query per table, never one per row
    assert alerts == [{"alert_id": "a", "ts_event_us": 1, "symbol": SYM, "json": "{}"}]
    assert fb == [{"alert_id": "a", "analyst": "x", "ts_us": 2, "verdict": "tp"}]
    assert outs == [out_row("a")]


def test_postgres_history_errors_do_not_leak_url(monkeypatch):
    import psycopg

    def boom(url, **kw):
        raise psycopg.OperationalError("server secret-host.neon.tech, user bob failed")

    monkeypatch.setattr(psycopg, "connect", boom)
    be = PostgresHistory("postgresql://bob:pw@secret-host.neon.tech/db")
    with pytest.raises(HistoryStoreError) as ei:
        be.load(0)
    e = ei.value
    assert e.__cause__ is None and e.__suppress_context__
    assert "secret-host" not in str(e) and "bob" not in str(e)


def test_postgres_history_retries_once_after_a_dropped_connection(monkeypatch):
    import psycopg

    cons = []

    class Flaky(_Con):
        def cursor(self):
            if len(cons) == 1:
                raise psycopg.OperationalError("connection lost secret")
            return _Cur(self)

        def close(self):
            self.closed = True

    def connect(url, **kw):
        cons.append(Flaky())
        return cons[-1]

    monkeypatch.setattr(psycopg, "connect", connect)
    PostgresHistory("postgresql://x").upsert_outcomes([out_row("a")])
    assert len(cons) == 2 and cons[0].closed


# ---- reviewer fixes: final outcomes, feedback updates, in-flight writes, the bound ----


def test_backend_never_downgrades_a_final_outcome():
    mem = MemoryHistory()
    mem.upsert_outcomes([out_row("a")])
    mem.upsert_outcomes([out_row("a", status="done", p1=90.0)])
    mem.upsert_outcomes([out_row("a")])  # a stale pending copy arrives late
    assert mem.outcomes["a"]["status"] == "done" and mem.outcomes["a"]["price_1h"] == 90.0
    mem.upsert_outcomes([out_row("b", status="unknown"), out_row("b", status="done", p1=1.0)])
    assert mem.outcomes["b"]["status"] == "unknown"


def test_backend_feedback_takes_the_new_verdict():
    mem = MemoryHistory()
    row = {"alert_id": "a", "analyst": "ann", "ts_us": 5, "verdict": "tp"}
    mem.insert_feedback([row])
    mem.insert_feedback([{**row, "verdict": "fp"}])
    assert [f["verdict"] for f in mem.feedback.values()] == ["fp"]


def test_restore_never_replaces_a_final_local_outcome(store):
    mem = MemoryHistory()
    for aid in ("done_here", "pending_here", "absent_here"):
        a = make_alert(aid, ts=T0, prov="live")
        mem.upsert_alerts(
            [
                dict(
                    zip(
                        ("alert_id", "ts_event_us", "symbol", "json"),
                        Store._alert_row(a),
                        strict=True,
                    )
                )
            ]
        )
    mem.upsert_outcomes(
        [
            out_row("done_here"),
            out_row("pending_here", status="done", p1=80.0),
            out_row("absent_here", status="done", p1=70.0),
        ]
    )
    store.upsert_outcome(out_row("done_here", status="done", p1=90.0))
    store.upsert_outcome(out_row("pending_here"))
    store.attach_history(mem, since_us=0)
    got = store.get_outcomes(["done_here", "pending_here", "absent_here"])
    assert (got["done_here"]["status"], got["done_here"]["price_1h"]) == ("done", 90.0)
    assert (got["pending_here"]["status"], got["pending_here"]["price_1h"]) == ("done", 80.0)
    assert got["absent_here"]["price_1h"] == 70.0


def test_a_write_during_a_sync_goes_out_with_the_next_one(store, monkeypatch):
    mem = MemoryHistory()
    store.attach_history(mem, since_us=0)
    store.upsert_alert(make_alert("first", prov="live"))
    real = mem.upsert_alerts
    sent = []

    def slow(rows):
        sent.append([r["alert_id"] for r in rows])
        if len(sent) == 1:  # while the first batch is on the wire
            store.upsert_alert(make_alert("during", prov="live"))
            store.upsert_alert(make_alert("first", ts=T0 + S, prov="live"))  # updated again
        real(rows)

    monkeypatch.setattr(mem, "upsert_alerts", slow)
    assert store.history_sync() is True
    assert store.history_pending()[0] == 2
    assert store.history_sync() is True
    assert [sorted(b) for b in sent] == [["first"], ["during", "first"]]
    assert mem.alerts["first"]["ts_event_us"] == T0 + S
    assert store.history_pending() == (0, 0, 0)


def test_sync_without_wait_skips_when_one_is_running(store):
    store.attach_history(MemoryHistory(), since_us=0)
    store.upsert_alert(make_alert("a", prov="live"))
    with store._sync_lock:
        assert store.history_sync(wait=False) is False
    assert store.history_pending()[0] == 1
    assert store.history_sync(wait=False) is True


def test_overflow_is_said_again_after_a_recovery(store, monkeypatch, caplog):
    monkeypatch.setattr(store_mod, "HISTORY_MAX_PENDING", 2)
    mem = MemoryHistory(fail=True)
    store.attach_history(MemoryHistory(), since_us=0)
    store._history = mem

    def overflow(tag):
        for i in range(3):
            store.upsert_alert(make_alert(f"{tag}{i}", prov="live"))

    overflow("a")
    overflow("b")  # the same outage: not said twice
    mem.fail = False
    store.history_sync()
    mem.fail = True
    overflow("c")
    warned = [r for r in caplog.records if "history queue" in r.getMessage()]
    assert len(warned) == 2
