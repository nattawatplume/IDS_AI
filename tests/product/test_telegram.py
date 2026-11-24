"""Telegram alerts: linking, /stop, unlink, server-only pref, watchlist, notifier, token safety.

Every Bot API call goes to an `httpx.MockTransport`; nothing reaches the network.
"""

import asyncio
import json
import logging
import shutil
import subprocess
import traceback
from types import SimpleNamespace

import httpx
import pytest
from starlette.testclient import TestClient

from idsai.ai.auditor import Auditor
from idsai.bus import Bus
from idsai.config import load_settings
from idsai.data.store import Store
from idsai.product.api import AppState, create_app
from idsai.product.api.app import WEB_DIR
from idsai.product.explainer import BANNED_TH
from idsai.product.notify import LinkCodes, TelegramClient, TelegramError, TelegramService
from idsai.product.notify import telegram as tg
from tests.product.test_api import SYM, FakePipe, make_alert, make_replayer
from tests.product.test_api_auth import HASH, login, send

TOKEN = "123456:SECRET-test-token-abcdef"
OTHER = "OTHERBTC"


class FakeTG:
    """A scripted Bot API. `script[method]` is a list of (status, body) answers used in turn (the
    last one repeats) or an Exception to raise; unscripted methods answer ok."""

    def __init__(self):
        self.calls: list[tuple[str, dict, str]] = []
        self.script: dict[str, list] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        body = json.loads(request.content or b"{}")
        self.calls.append((method, body, str(request.url)))
        answers = self.script.get(method)
        if answers:
            a = answers.pop(0) if len(answers) > 1 else answers[0]
            if isinstance(a, Exception):
                raise a
            status, payload = a
            return httpx.Response(status, json=payload)
        if method == "getMe":
            return httpx.Response(200, json={"ok": True, "result": {"username": "ids_bot"}})
        if method == "getUpdates":
            return httpx.Response(200, json={"ok": True, "result": []})
        return httpx.Response(200, json={"ok": True, "result": {}})

    def sent(self) -> list[tuple[int, str]]:
        return [(b["chat_id"], b["text"]) for m, b, _ in self.calls if m == "sendMessage"]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def make_service(store, fake=None, clock=None, sleep=None, public_url=None):
    fake = fake or FakeTG()
    client = TelegramClient(TOKEN, "https://tg.test", transport=httpx.MockTransport(fake))
    kw = {"token": TOKEN, "clock": clock or Clock()}
    if sleep is not None:
        kw["sleep"] = sleep
    svc = TelegramService(client, store, public_url, **kw)
    svc.bot_username = "ids_bot"
    return svc, fake


def start_update(chat_id, text, chat_type="private", update_id=1):
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat_id, "type": chat_type}, "text": text},
    }


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "d", tmp_path / "a.duckdb")
    s.add_user("ann", "Ann A", "viewer", HASH, 1)
    s.add_user("bob", "", "analyst", HASH, 1)
    yield s
    s.close()


def uid(store, name):
    return store.get_user_by_name(name)["user_id"]


@pytest.fixture
def svc(store):
    service, fake = make_service(store)
    yield SimpleNamespace(svc=service, fake=fake, clock=service._clock)
    tg.remove_redaction(service._filter)


def run(coro):
    return asyncio.run(coro)


# ---- link codes ----


def test_link_codes_one_time_ttl_and_replace():
    clock = Clock()
    codes = LinkCodes(ttl_s=600, clock=clock)
    c1 = codes.issue("u1")
    assert 0 < len(c1) <= 64 and all(ch.isalnum() or ch in "_-" for ch in c1)
    c2 = codes.issue("u1")  # a new request replaces the old code
    assert codes.take(c1) is None
    assert codes.take(c2) == "u1"
    assert codes.take(c2) is None  # used up
    c3 = codes.issue("u2")
    clock.t += 601
    assert codes.take(c3) is None  # expired
    assert codes.take("nope") is None


# ---- linking over the poller ----


def test_start_with_a_valid_code_links_and_greets(svc, store):
    ann = uid(store, "ann")
    out = svc.svc.link_url(ann)
    assert out["expires_s"] == 600
    assert out["url"].startswith("https://t.me/ids_bot?start=")
    code = out["url"].split("start=", 1)[1]
    run(svc.svc.handle_update(start_update(555, f"/start {code}")))
    assert store.get_prefs(ann)[tg.PREF_KEY] == 555
    # the username, never the free-text display name ("Ann A")
    assert svc.fake.sent() == [(555, "เชื่อมบัญชี ann แล้ว พิมพ์ /stop เพื่อยกเลิก")]
    rows = [r for r in store.list_audit(50) if r["action"] == "telegram_link"]
    assert len(rows) == 1 and rows[0]["username"] == "ann"
    # the code is used up
    run(svc.svc.handle_update(start_update(556, f"/start {code}")))
    assert svc.fake.sent()[-1] == (556, tg.EXPIRED_TH)
    assert store.get_prefs(ann)[tg.PREF_KEY] == 555


def test_start_with_an_expired_or_wrong_code_does_not_link(svc, store):
    bob = uid(store, "bob")
    code = svc.svc.link_url(bob)["url"].split("start=", 1)[1]
    svc.clock.t += 601
    run(svc.svc.handle_update(start_update(7, f"/start {code}")))
    run(svc.svc.handle_update(start_update(7, "/start wrong-code")))
    run(svc.svc.handle_update(start_update(7, "/start")))
    assert tg.PREF_KEY not in store.get_prefs(bob)
    assert svc.fake.sent() == [(7, tg.EXPIRED_TH)] * 3
    assert "หมดอายุ" in tg.EXPIRED_TH and "เว็บ" in tg.EXPIRED_TH


def test_messages_outside_a_private_chat_are_ignored(svc, store):
    ann = uid(store, "ann")
    code = svc.svc.link_url(ann)["url"].split("start=", 1)[1]
    for kind in ("group", "supergroup", "channel"):
        run(svc.svc.handle_update(start_update(-100, f"/start {code}", chat_type=kind)))
    run(svc.svc.handle_update({"update_id": 3}))
    run(svc.svc.handle_update({"update_id": 4, "message": {"chat": {"id": 1, "type": "private"}}}))
    assert svc.fake.sent() == []
    assert tg.PREF_KEY not in store.get_prefs(ann)
    assert svc.svc.codes.take(code) == ann  # still valid: a group message did not spend it


def test_local_user_links_with_auth_off(svc, store):
    code = svc.svc.link_url("local")["url"].split("start=", 1)[1]
    run(svc.svc.handle_update(start_update(9, f"/start {code}")))
    assert store.get_prefs("local")[tg.PREF_KEY] == 9
    assert svc.fake.sent() == [(9, "เชื่อมบัญชี local แล้ว พิมพ์ /stop เพื่อยกเลิก")]


def test_stop_unlinks_every_user_of_that_chat(svc, store):
    ann, bob = uid(store, "ann"), uid(store, "bob")
    store.set_prefs(ann, {tg.PREF_KEY: 42, "theme": "dark"})
    store.set_prefs(bob, {tg.PREF_KEY: 42})
    store.set_prefs("local", {tg.PREF_KEY: 43})
    run(svc.svc.handle_update(start_update(42, "/stop")))
    assert store.get_prefs(ann) == {"theme": "dark"}
    assert store.get_prefs(bob) == {}
    assert store.get_prefs("local") == {tg.PREF_KEY: 43}
    run(svc.svc.handle_update(start_update(42, "/stop")))
    assert svc.fake.sent() == [(42, tg.STOPPED_TH), (42, tg.NOT_LINKED_TH)]


def test_poll_once_learns_the_bot_and_advances_the_offset(store):
    service, fake = make_service(store)
    service.bot_username = None
    code = service.codes.issue("local")
    fake.script["getUpdates"] = [
        (200, {"ok": True, "result": [start_update(5, f"/start {code}", update_id=10)]}),
        (200, {"ok": True, "result": []}),
    ]
    run(service.poll_once())
    run(service.poll_once())
    assert service.bot_username == "ids_bot"
    updates = [b for m, b, _ in fake.calls if m == "getUpdates"]
    assert "offset" not in updates[0] and updates[1]["offset"] == 11
    assert updates[0]["timeout"] == 25
    assert [m for m, _, _ in fake.calls].count("getMe") == 1
    assert store.get_prefs("local")[tg.PREF_KEY] == 5
    tg.remove_redaction(service._filter)


class _Stop(Exception):
    pass


def test_poller_honours_retry_after_and_backs_off(store):
    waits: list[float] = []

    async def sleep(s):
        waits.append(s)
        if len(waits) >= 4:
            raise _Stop

    service, fake = make_service(store, sleep=sleep)
    fake.script["getUpdates"] = [
        (429, {"ok": False, "error_code": 429, "parameters": {"retry_after": 17}}),
        (500, {"ok": False}),
        (502, {"ok": False}),
        httpx.ConnectError("boom"),
    ]
    with pytest.raises(_Stop):
        run(service.poll_forever())
    assert waits == [17, 1.0, 2.0, 4.0]
    tg.remove_redaction(service._filter)


# ---- the API ----


@pytest.fixture
def api(store, tmp_path):
    cfg = load_settings(
        overrides={
            "paths": {"data_dir": str(tmp_path / "d"), "app_db": str(tmp_path / "a.duckdb")},
            "auth": {"enabled": True},
        }
    )
    service, fake = make_service(store)
    state = AppState(
        cfg,
        store,
        "replay",
        FakePipe(Bus(maxsize=50)),
        make_replayer(),
        Auditor(cfg),
        symbols=[SYM, OTHER],
        auth_enabled=True,
        telegram=service,
    )
    with TestClient(create_app(state)) as c:
        yield SimpleNamespace(c=c, state=state, svc=service, fake=fake)
    tg.remove_redaction(service._filter)


def test_api_link_status_and_unlink(api, store):
    c = api.c
    r = login(c, "ann")
    assert r.status_code == 200
    assert c.get("/api/me").json()["telegram"] == {"enabled": True, "linked": False}
    r = send(c, "POST", "/api/me/telegram/link")
    assert r.status_code == 200
    out = r.json()
    assert out["expires_s"] == 600 and out["url"].startswith("https://t.me/ids_bot?start=")
    assert TOKEN not in r.text
    code = out["url"].split("start=", 1)[1]
    run(api.svc.handle_update(start_update(77, f"/start {code}")))
    assert c.get("/api/me").json()["telegram"] == {"enabled": True, "linked": True}
    r = send(c, "DELETE", "/api/me/telegram")
    assert r.status_code == 200 and r.json() == {"linked": False}
    assert tg.PREF_KEY not in store.get_prefs(uid(store, "ann"))
    assert c.get("/api/me").json()["telegram"] == {"enabled": True, "linked": False}
    # cross-site requests are refused
    evil = {"origin": "http://evil.example"}
    assert send(c, "POST", "/api/me/telegram/link", headers=evil).status_code == 403
    assert send(c, "DELETE", "/api/me/telegram", headers=evil).status_code == 403


def test_api_link_needs_sign_in_and_a_known_bot(api):
    c = api.c
    assert send(c, "POST", "/api/me/telegram/link").status_code == 401
    assert send(c, "DELETE", "/api/me/telegram").status_code == 401
    login(c, "ann")
    api.svc.bot_username = None
    r = send(c, "POST", "/api/me/telegram/link")
    assert r.status_code == 503 and TOKEN not in r.text


def test_server_only_pref_never_leaves_and_is_never_accepted(api, store):
    c = api.c
    ann = uid(store, "ann")
    store.set_prefs(ann, {tg.PREF_KEY: 99, "theme": "dark"})
    r = login(c, "ann")
    assert r.json()["prefs"] == {"theme": "dark"}
    assert c.get("/api/me").json()["prefs"] == {"theme": "dark"}
    r = send(c, "PUT", "/api/me/prefs", {"theme": "light"})
    assert r.status_code == 200 and r.json() == {"theme": "light"}
    for responses in (r.text, c.get("/api/me").text):
        assert tg.PREF_KEY not in responses and "99" not in responses
    assert store.get_prefs(ann)[tg.PREF_KEY] == 99  # a prefs save keeps the link
    for bad in ({tg.PREF_KEY: 1}, {tg.PREF_KEY: None}):
        assert send(c, "PUT", "/api/me/prefs", bad).status_code == 422
    assert store.get_prefs(ann)[tg.PREF_KEY] == 99


def test_watch_validation(api, store):
    c = api.c
    login(c, "ann")
    r = send(c, "PUT", "/api/me/prefs", {"watch": [SYM, SYM]})
    assert r.status_code == 200 and r.json() == {"watch": [SYM]}
    assert send(c, "PUT", "/api/me/prefs", {"watch": []}).json() == {"watch": []}
    for bad in (["NOPE"], [SYM, "NOPE"], "BTC", [1], [SYM] * 51):
        assert send(c, "PUT", "/api/me/prefs", {"watch": bad}).status_code == 422, bad
    assert c.get("/api/me").json()["prefs"] == {"watch": []}
    assert send(c, "PUT", "/api/me/prefs", {"watch": None}).json() == {}  # null removes it


def test_feature_off_without_the_token(store, tmp_path, monkeypatch):
    monkeypatch.delenv(tg.TOKEN_ENV, raising=False)
    assert TelegramService.from_env(store) is None
    assert TelegramService.from_env(store, env={tg.TOKEN_ENV: "  "}) is None
    cfg = load_settings(
        overrides={"paths": {"data_dir": str(tmp_path / "d"), "app_db": str(tmp_path / "a.db")}}
    )
    state = AppState(cfg, store, "replay", None, None, symbols=[SYM])
    with TestClient(create_app(state)) as c:
        assert c.get("/api/me").json()["telegram"] == {"enabled": False, "linked": False}
        assert send(c, "POST", "/api/me/telegram/link").status_code == 404
        assert send(c, "DELETE", "/api/me/telegram").status_code == 404


def test_from_env_reads_base_and_public_url(store):
    fake = FakeTG()
    env = {
        tg.TOKEN_ENV: TOKEN,
        tg.API_BASE_ENV: "http://fake.local:9",
        tg.PUBLIC_URL_ENV: "https://ids.example",
    }
    service = TelegramService.from_env(store, env=env, transport=httpx.MockTransport(fake))
    assert service is not None and service.public_url == "https://ids.example"
    run(service.client.get_me())
    assert fake.calls[0][2] == f"http://fake.local:9/bot{TOKEN}/getMe"
    run(service.stop())


# ---- notifier ----


def notify_and_wait(service, alert):
    async def go():
        tasks = service.notify(alert)
        await asyncio.gather(*tasks)
        return len(tasks)

    return run(go())


def test_notifier_sends_only_open_alerts_on_watched_symbols(svc, store):
    ann, bob = uid(store, "ann"), uid(store, "bob")
    store.set_prefs(ann, {tg.PREF_KEY: 1})  # no watch: every symbol
    store.set_prefs(bob, {tg.PREF_KEY: 2, "watch": [OTHER]})
    store.set_prefs("local", {tg.PREF_KEY: 3, "watch": []})  # empty: nothing
    store.set_prefs("x", {"watch": [SYM]})  # not linked
    s = svc.svc
    assert notify_and_wait(s, {"event": "alerts_reset", "t_us": 0}) == 0
    assert notify_and_wait(s, {"event": "replay_restarted", "t_us": 0}) == 0
    assert notify_and_wait(s, make_alert("u", status="updated")) == 0
    assert notify_and_wait(s, make_alert("c", status="closed")) == 0
    assert svc.fake.sent() == []
    assert notify_and_wait(s, make_alert("a1")) == 1
    assert [c for c, _ in svc.fake.sent()] == [1]
    assert notify_and_wait(s, make_alert("a2", symbol=OTHER)) == 2
    assert sorted(c for c, _ in svc.fake.sent()[1:]) == [1, 2]


def test_rate_limit_per_chat_and_symbol(svc, store):
    store.set_prefs(uid(store, "ann"), {tg.PREF_KEY: 1})
    s = svc.svc
    assert notify_and_wait(s, make_alert("a1")) == 1
    svc.clock.t += 599
    assert notify_and_wait(s, make_alert("a2")) == 0
    assert notify_and_wait(s, make_alert("b1", symbol=OTHER)) == 1  # another symbol
    svc.clock.t += 1
    assert notify_and_wait(s, make_alert("a3")) == 1
    assert len(svc.fake.sent()) == 3


def test_403_unlinks_and_does_not_affect_other_chats(svc, store):
    ann, bob = uid(store, "ann"), uid(store, "bob")
    store.set_prefs(ann, {tg.PREF_KEY: 1, "watch": [SYM]})
    store.set_prefs(bob, {tg.PREF_KEY: 2})

    def answer(request):
        body = json.loads(request.content)
        if body.get("chat_id") == 1:
            return httpx.Response(403, json={"ok": False, "error_code": 403})
        return httpx.Response(200, json={"ok": True, "result": {}})

    svc.svc.client = TelegramClient(TOKEN, "https://tg.test", httpx.MockTransport(answer))
    seen = []
    orig = svc.svc.client.send_message

    async def spy(chat_id, text):
        seen.append(chat_id)
        await orig(chat_id, text)

    svc.svc.client.send_message = spy
    assert notify_and_wait(svc.svc, make_alert("a1")) == 2
    assert sorted(seen) == [1, 2]
    assert store.get_prefs(ann) == {"watch": [SYM]}
    assert store.get_prefs(bob) == {tg.PREF_KEY: 2}


def test_send_honours_retry_after_once(store):
    waits = []

    async def sleep(s):
        waits.append(s)

    service, fake = make_service(store, sleep=sleep)
    store.set_prefs("local", {tg.PREF_KEY: 5})
    fake.script["sendMessage"] = [
        (429, {"ok": False, "parameters": {"retry_after": 3}}),
        (200, {"ok": True, "result": {}}),
    ]
    assert notify_and_wait(service, make_alert("a1")) == 1
    assert waits == [3] and len(fake.sent()) == 2
    tg.remove_redaction(service._filter)


def test_message_text_is_thai_plain_and_carries_the_disclaimer(svc, store):
    a = make_alert("a1", ts=1_700_000_000 * 1_000_000)
    text = tg.message_text(a, "https://ids.example")
    lines = text.split("\n")
    assert lines[0] == f"⚠️ ตรวจพบความผิดปกติทางสถิติ · {SYM}"
    # 2023-11-14 22:13:20 UTC is 05:13 in Bangkok; labelled, since the explainer uses UTC
    assert lines[1] == "05:13:20 15/11/2023 (เวลาไทย)"
    assert "แต่ในชั่วโมงนั้นราคายังขึ้นต่อได้ (มัธยฐาน +15%)" in tg.HISTORY_TH
    assert lines[-1] == "https://ids.example"
    assert tg.DISCLAIMER_TH in lines and tg.HISTORY_TH in lines
    assert "ไม่ใช่คำแนะนำการลงทุน" in text and "ไม่ได้กล่าวหาผู้ใด" in text
    for word in ("ควรซื้อ", "ควรขาย", "ควรถือ", "ความน่าจะเป็น", "probability", *BANNED_TH):
        assert word not in text, word
    assert tg.message_text(a).split("\n")[-1] == tg.DISCLAIMER_TH  # no URL when unset
    store.set_prefs("local", {tg.PREF_KEY: 8})
    notify_and_wait(svc.svc, a)
    body = next(b for m, b, _ in svc.fake.calls if m == "sendMessage")
    assert "parse_mode" not in body and body["text"] == tg.message_text(a)


def test_message_text_names_the_phase_and_the_bear_raid_neutrally():
    ts = 1_700_000_000 * 1_000_000
    raid = make_alert("r1", ts=ts, technique_id="MM-T006", signature_id=100601, phase="sell")
    text = tg.message_text(raid)
    assert "MM-T006 ทุบราคา" in text and "ช่วง: กำลังเทขาย" in text
    assert "p ≈" in text and tg.DISCLAIMER_TH in text.splitlines()
    assert tg.HISTORY_TH not in text  # measured on the rise of real pumps only
    for word in ("ควรซื้อ", "ควรขาย", "ความน่าจะเป็น", "probability", *BANNED_TH):
        assert word not in text, word
    pump = tg.message_text(make_alert("p1", ts=ts, technique_id="MM-T001.001", phase="pump"))
    assert "ช่วง: กำลังพุ่ง" in pump and tg.HISTORY_TH in pump.splitlines()
    assert "ช่วง:" not in tg.message_text(make_alert("n1", ts=ts))  # no phase, no line


def test_service_runs_on_the_bus_and_stops_cleanly(store):
    service, fake = make_service(store)
    fake.script["getUpdates"] = [httpx.ConnectError("down")]  # the poller backs off for real
    store.set_prefs("local", {tg.PREF_KEY: 5})
    bus = Bus(maxsize=10)

    async def go():
        service.start(bus)
        assert len(bus._subs["alerts"]) == 1
        bus.publish("alerts", {"event": "alerts_reset"})
        bus.publish("alerts", make_alert("a1"))
        for _ in range(200):
            if fake.sent():
                break
            await asyncio.sleep(0.01)
        await service.stop()
        assert bus._subs["alerts"] == []

    run(go())
    assert [c for c, _ in fake.sent()] == [5]


# ---- the token stays secret ----


def test_token_never_in_logs_or_errors(store, caplog):
    caplog.set_level(logging.DEBUG)
    errors = []

    async def sleep(s):
        if len(errors) >= 3:
            raise _Stop

    service, fake = make_service(store, sleep=sleep)
    fake.script["getUpdates"] = [
        httpx.ConnectError(f"cannot reach https://tg.test/bot{TOKEN}/getUpdates"),
        (500, {"ok": False, "description": "oops"}),
        (401, {"ok": False}),
    ]

    async def go():
        for _ in range(3):
            try:
                await service.client.get_updates(None)
            except TelegramError as e:
                errors.append(e)
        await service.client.get_me()  # httpx logs this request (URL included) at INFO

    run(go())
    fake.script["getUpdates"] = [httpx.ConnectError(f"x {TOKEN}")]
    with pytest.raises(_Stop):
        run(service.poll_forever())
    store.set_prefs("local", {tg.PREF_KEY: 5})
    fake.script["sendMessage"] = [httpx.ReadTimeout(f"t {TOKEN}")]
    notify_and_wait(service, make_alert("a1"))
    assert [e.status for e in errors] == [None, 500, 401]
    for e in errors:
        shown = "".join(traceback.format_exception(e))
        assert TOKEN not in shown and "getUpdates" in str(e)
    assert "HTTP Request" in caplog.text  # httpx did log the requests ...
    assert TOKEN not in caplog.text  # ... without the token
    assert "<redacted>" in caplog.text
    assert "telegram getUpdates failed" in caplog.text
    tg.remove_redaction(service._filter)


# ---- the web page ----

WEB_BODY = r"""
const assert = require("assert");
globalThis.matchMedia = () => ({ matches: false });
const m = require(process.argv[2]);
const store = {};
globalThis.window = { localStorage: {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); } } };
const calls = [];
globalThis.fetch = async (path, opts) => {
  calls.push([path, opts]); return { ok: true, status: 200, json: async () => ({}) }; };
(async () => {
  assert.strictEqual(m.tgWatched({}, "BTCUSDT"), true);
  assert.strictEqual(m.tgWatched({ watch: [] }, "BTCUSDT"), false);
  assert.strictEqual(m.tgWatched({ watch: ["BTCUSDT"] }, "BTCUSDT"), true);
  assert.strictEqual(m.isTelegramLink("https://t.me/ids_bot?start=abc"), true);
  const bads = ["http://t.me/x", "https://t.me.evil.example/x", "javascript:alert(1)", "", null];
  for (const bad of bads) {
    assert.strictEqual(m.isTelegramLink(bad), false, String(bad));
  }
  // auth off: the watchlist still reaches the server (the notifier reads it there)
  m.setMe({ username: "local", role: "admin", auth_enabled: false, prefs: {} });
  m.savePrefs({ symbol: "ETHUSDT", watch: ["BTCUSDT"] });
  await new Promise((r) => setTimeout(r, 700));
  assert.strictEqual(calls.length, 1);
  assert.deepStrictEqual(JSON.parse(calls[0][1].body), { watch: ["BTCUSDT"] });
  assert.strictEqual(store["idsai.symbol"], "ETHUSDT");
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_web_watchlist_reaches_the_server_and_section_is_thai(tmp_path):
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    for needle in ('id="tg-section"', "แจ้งเตือน Telegram", 'id="tg-link"', 'id="tg-unlink"'):
        assert needle in html, needle
    assert 'id="tg-section" class="tg-section" aria-labelledby="tg-title" hidden' in html
    assert "isTelegramLink(url)" in js  # only a Telegram link is ever shown
    for word in ("ควรซื้อ", "ควรขาย", "ความน่าจะเป็น"):
        assert word not in html and word not in js
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = tmp_path / "tg.js"
    script.write_text(WEB_BODY, encoding="utf-8")
    r = subprocess.run(
        [node, str(script), str(WEB_DIR / "app.js")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert r.returncode == 0, r.stderr


# ---- review fixes: who may receive, disabled accounts, store failures, audit ----


def test_recipients_skip_disabled_unknown_and_local_with_auth_on(store):
    ann, bob = uid(store, "ann"), uid(store, "bob")
    store.set_prefs(ann, {tg.PREF_KEY: 1})
    store.set_prefs(bob, {tg.PREF_KEY: 2})
    store.set_prefs("ghost", {tg.PREF_KEY: 3})  # no such account
    store.set_prefs("local", {tg.PREF_KEY: 4})
    off, _ = make_service(store)
    assert off.recipients(SYM) == [1, 2, 4]  # auth off: the local user counts
    on, _ = make_service(store)
    on.auth_enabled = True
    assert on.recipients(SYM) == [1, 2]
    with store._lock:  # disabled without going through set_disabled (it would unlink)
        store._db().execute("UPDATE users SET disabled = TRUE WHERE username = 'bob'")
    assert on.recipients(SYM) == [1]
    for s in (off, on):
        tg.remove_redaction(s._filter)


def test_disabling_an_account_drops_its_telegram_link(store):
    ann = uid(store, "ann")
    store.set_prefs(ann, {tg.PREF_KEY: 1, "watch": [SYM]})
    store.set_disabled("ann", True)
    assert store.get_prefs(ann) == {"watch": [SYM]}
    store.set_disabled("ann", False)
    assert tg.PREF_KEY not in store.get_prefs(ann)  # re-enabling needs a new link


def test_403_with_a_failing_store_is_logged_and_contained(svc, store, caplog):
    store.set_prefs(uid(store, "ann"), {tg.PREF_KEY: 1})
    svc.fake.script["sendMessage"] = [(403, {"ok": False, "error_code": 403})]

    def boom(*a, **k):
        raise RuntimeError(f"db down {TOKEN}")

    svc.svc.unlink_chat = boom
    tasks_ok = notify_and_wait(svc.svc, make_alert("a1"))  # gather would raise on a task error
    assert tasks_ok == 1
    assert "unlink failed (RuntimeError)" in caplog.text and TOKEN not in caplog.text


def test_link_store_failure_replies_thai_and_keeps_code_used(svc, store):
    ann = uid(store, "ann")
    code = svc.svc.link_url(ann)["url"].split("start=", 1)[1]

    def boom(*a, **k):
        raise RuntimeError("store down")

    svc.svc._link = boom
    run(svc.svc.handle_update(start_update(5, f"/start {code}")))
    assert svc.fake.sent() == [(5, tg.LINK_FAILED_TH)]
    assert tg.LINK_FAILED_TH == "เชื่อมไม่สำเร็จ ลองกดเชื่อมใหม่บนเว็บ"
    assert svc.svc.codes.take(code) is None


def test_link_stop_and_block_are_audited(svc, store):
    ann, bob = uid(store, "ann"), uid(store, "bob")
    store.set_prefs(ann, {tg.PREF_KEY: 1})
    store.set_prefs(bob, {tg.PREF_KEY: 2})
    run(svc.svc.handle_update(start_update(1, "/stop")))
    svc.fake.script["sendMessage"] = [(403, {"ok": False, "error_code": 403})]
    notify_and_wait(svc.svc, make_alert("a1"))
    got = {(r["username"], r["action"]) for r in store.list_audit(50)}
    assert {("ann", "telegram_stop"), ("bob", "telegram_blocked")} <= got


def test_api_link_and_unlink_are_audited(api, store):
    c = api.c
    login(c, "ann")
    send(c, "POST", "/api/me/telegram/link")
    send(c, "DELETE", "/api/me/telegram")  # not linked: nothing to audit
    store.set_prefs(uid(store, "ann"), {tg.PREF_KEY: 9})
    send(c, "DELETE", "/api/me/telegram")
    acts = [r["action"] for r in store.list_audit(50) if r["action"].startswith("telegram")]
    assert sorted(acts) == ["telegram_link_code", "telegram_unlink"]
