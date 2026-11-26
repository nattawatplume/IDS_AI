"""DATABASE_URL wiring of the alert history: restore at serve start, the sync task, the note."""

import asyncio
import logging

import pytest

from idsai import cli
from idsai.data.history import HISTORY_DAYS, HistoryStoreError, MemoryHistory
from idsai.data.store import Store
from idsai.product import outcomes as oc
from tests.product.test_api import D_US, SYM, make_alert
from tests.product.test_track_record import RingPipe, trades
from tests.test_cli import write_cfg

S = 1_000_000
HOUR = 3600 * S
DAY = 24 * HOUR
URL = "postgresql://u:s3cretpw@secret-host.example/x"


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "data", tmp_path / "data" / "app.duckdb") as s:
        yield s


# ---- attach_history_from_env ----


def test_no_database_url_attaches_nothing_and_runs_no_task(store, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert cli.attach_history_from_env(store) is False
    assert not store.history_attached

    async def bg():
        return None

    assert cli.with_history_sync(store, bg) is bg  # no task, no queueing
    store.upsert_alert(make_alert("l", prov="live"))
    assert store.history_pending() == (0, 0, 0)


def test_attach_restores_the_last_30_days(store, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", URL)
    mem = MemoryHistory()
    now = D_US + 40 * DAY
    seen = []

    def factory(url):
        seen.append(url)
        return mem

    assert cli.attach_history_from_env(store, backend_factory=factory, now_us=now) is True
    assert store.history_attached and store.history_since_us == now - HISTORY_DAYS * DAY
    assert seen == [URL]


def test_attach_failure_warns_in_thai_and_returns(store, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setattr(cli, "ATTACH_WAIT_S", 0)
    calls = []

    def factory(url):
        calls.append(1)
        return MemoryHistory(fail=True)

    assert cli.attach_history_from_env(store, backend_factory=factory) is False
    assert len(calls) == cli.ATTACH_TRIES
    err = capsys.readouterr().err
    assert cli.HISTORY_WARN_TH in err
    assert "s3cret" not in err and "secret-host" not in err
    assert not store.history_attached


def _serve_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(cli, "attach_accounts_from_env", lambda s: None)
    monkeypatch.setattr(cli, "ATTACH_WAIT_S", 0)
    return write_cfg(tmp_path)


def test_serve_live_prepares_after_a_failed_history_attach(tmp_path, monkeypatch, capsys):
    cfg_path = _serve_env(tmp_path, monkeypatch)

    def broken(url):
        raise HistoryStoreError("down " + url)

    monkeypatch.setattr("idsai.data.history.build_backend", broken)
    order = []

    def fake_prepare(args, cfg, st):
        order.append(("prepare", st.history_attached))
        raise cli.CliError("stop here")

    monkeypatch.setattr(cli, "prepare_live", fake_prepare)
    assert cli.main(["--config", str(cfg_path), "serve", "--mode", "live"]) == 1
    assert order == [("prepare", False)]
    err = capsys.readouterr().err
    assert cli.HISTORY_WARN_TH in err and "s3cret" not in err


def test_serve_replay_never_attaches_history(tmp_path, monkeypatch):
    cfg_path = _serve_env(tmp_path, monkeypatch)
    built = []
    monkeypatch.setattr(
        "idsai.data.history.build_backend", lambda url: built.append(url) or MemoryHistory()
    )
    seen = []

    def fake_prepare(args, cfg, st):
        seen.append(st.history_attached)
        raise cli.CliError("stop here")

    monkeypatch.setattr(cli, "prepare_replay", fake_prepare)
    assert cli.main(["--config", str(cfg_path), "serve"]) == 1
    assert seen == [False] and built == []


def test_serve_live_restores_before_it_prepares_and_syncs_what_it_stores(tmp_path, monkeypatch):
    """The pipeline sees the restored rows; a live alert stored while serving reaches the
    database (here through the shutdown sync, as run_server cancels the background)."""
    from idsai.product.api.state import AppState

    cfg_path = _serve_env(tmp_path, monkeypatch)
    mem = MemoryHistory()
    old = make_alert("old", ts=_now_us() - DAY, prov="live")
    mem.upsert_alerts(
        [
            dict(
                zip(
                    ("alert_id", "ts_event_us", "symbol", "json"),
                    Store._alert_row(old),
                    strict=True,
                )
            )
        ]
    )
    monkeypatch.setattr("idsai.data.history.build_backend", lambda url: mem)
    seen = {}

    def fake_prepare(args, cfg, st):
        seen["restored"] = st.get_alert("old") is not None
        seen["store"] = st

        async def background():
            st.upsert_alert(make_alert("new", prov="live"))
            await asyncio.Event().wait()

        return AppState(cfg, st, "live", None, None), background

    def fake_run_server(app, host, port, background):
        async def main():
            task = asyncio.create_task(background())
            await asyncio.sleep(0.05)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        asyncio.run(main())

    monkeypatch.setattr(cli, "prepare_live", fake_prepare)
    monkeypatch.setattr(cli, "run_server", fake_run_server)
    assert cli.main(["--config", str(cfg_path), "serve", "--mode", "live"]) == 0
    assert seen["restored"] is True
    assert set(mem.alerts) == {"old", "new"}


def _now_us():
    return cli._now_us()


# ---- with_history_sync ----


def test_sync_task_copies_periodically_and_once_more_at_shutdown(store, monkeypatch):
    mem = MemoryHistory()
    store.attach_history(mem, since_us=0)
    monkeypatch.setattr(cli, "HISTORY_SYNC_S", 0.01)

    async def main():
        stop = asyncio.Event()

        async def bg():
            store.upsert_alert(make_alert("a", prov="live"))
            await stop.wait()

        task = asyncio.create_task(cli.with_history_sync(store, bg)())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if "a" in mem.alerts:
                break
        assert "a" in mem.alerts
        monkeypatch.setattr(cli, "HISTORY_SYNC_S", 3600)
        await asyncio.sleep(0.05)  # the loop now sleeps an hour
        store.upsert_alert(make_alert("b", prov="live"))
        task.cancel()  # as run_server does at shutdown
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())
    assert set(mem.alerts) == {"a", "b"}  # b arrived through the shutdown sync


def test_sync_failures_are_logged_rarely_without_the_url(store, monkeypatch, caplog):
    mem = MemoryHistory()
    store.attach_history(mem, since_us=0)
    mem.fail = True
    store.upsert_alert(make_alert("a", prov="live"))
    monkeypatch.setattr(cli, "HISTORY_SYNC_S", 0.005)
    monkeypatch.setenv("DATABASE_URL", URL)
    caplog.set_level(logging.WARNING)

    async def main():
        task = asyncio.create_task(cli.with_history_sync(store, asyncio.Event().wait)())
        await asyncio.sleep(0.15)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())
    warned = [r.getMessage() for r in caplog.records if r.name == "idsai.cli"]
    assert len(warned) == 1 and "HistoryStoreError" in warned[0]
    assert all("s3cret" not in m and "secret-host" not in m for m in warned)
    assert store.history_pending()[0] == 1  # still queued for the next try
    mem.fail = False
    store.history_sync()
    assert "a" in mem.alerts


# ---- outcomes of restored alerts ----

T0 = D_US + 3 * HOUR


def _restored(tmp_path, out_status="pending"):
    """A store B restored from a backend holding a live alert at T0 and its pending outcome."""
    mem = MemoryHistory()
    with Store(tmp_path / "a", tmp_path / "a" / "app.duckdb") as a:
        a.attach_history(mem, since_us=0)
        a.upsert_alert(make_alert("old", ts=T0, prov="live"))
        a.upsert_outcome(
            {
                "alert_id": "old",
                "symbol": SYM,
                "ts_event_us": T0,
                "price_at": 50.0 if out_status == "done" else None,
                "price_1h": 55.0 if out_status == "done" else None,
                "t_1h_us": T0 + HOUR if out_status == "done" else None,
                "status": out_status,
                "updated_us": 1,
            }
        )
        a.history_sync()
    b = Store(tmp_path / "b", tmp_path / "b" / "app.duckdb")
    b.attach_history(mem, since_us=0)
    return b, mem


def test_restored_old_pending_outcome_resolves_from_the_archive(tmp_path):
    b, mem = _restored(tmp_path)
    with b:
        b.write_trades(trades([(T0 - 30 * S, 50.0), (T0 + HOUR + 2 * S, 55.0)]))
        # the ring holds nothing that old: the archive fallback settles it
        oc.track_step(b, RingPipe(latest_us=T0 + 3 * HOUR), backfill=True)
        o = b.get_outcomes(["old"])["old"]
        assert (o["status"], o["price_at"], o["price_1h"]) == ("done", 50.0, 55.0)
        b.history_sync()
    assert mem.outcomes["old"]["status"] == "done"


def test_restored_old_pending_outcome_becomes_unknown_after_24_h(tmp_path):
    b, mem = _restored(tmp_path)
    with b:
        oc.track_step(b, RingPipe(latest_us=T0 + 25 * HOUR), backfill=True)
        assert b.get_outcomes(["old"])["old"]["status"] == "unknown"
        assert b.history_pending() == (0, 0, 1)
        b.history_sync()
    assert mem.outcomes["old"]["status"] == "unknown"


def test_restored_done_outcome_is_not_queued_again(tmp_path):
    b, _ = _restored(tmp_path, out_status="done")
    with b:
        oc.track_step(b, RingPipe(latest_us=T0 + 3 * HOUR), backfill=True)
        assert b.get_outcomes(["old"])["old"]["status"] == "done"
        assert b.history_pending() == (0, 0, 0)


# ---- the track-record note ----


def test_note_without_history_keeps_the_restart_text(store):
    note = oc.track_record(store)["note_th"]
    assert note == oc.NOTE_TH
    assert "ตัวเลขทั้งหมดจะเริ่มนับใหม่" in note[0]


def test_note_with_history_names_the_database_and_the_exact_cut(store):
    cut = D_US + 7 * HOUR + 5 * 60 * S + 1  # 07:05 and a microsecond: shown as 07:06, never 07:05
    store.attach_history(MemoryHistory(), since_us=cut)
    note = oc.track_record(store)["note_th"]
    assert note[1:] == oc.NOTE_TH[1:]
    assert "ฐานข้อมูล" in note[0] and "ตัวเลขทั้งหมดจะเริ่มนับใหม่" not in note[0]
    assert "ราวทุก 30 วินาที" in note[0] and "อาจหายได้" in note[0]
    assert f"{HISTORY_DAYS} วันล่าสุด" in note[0]
    assert f"ตั้งแต่ {oc._day_of(D_US).isoformat()} 07:06 UTC" in note[0]
    assert oc._utc_minute(D_US + 7 * HOUR) == f"{oc._day_of(D_US).isoformat()} 07:00"
