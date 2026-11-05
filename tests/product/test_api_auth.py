"""API with sign-in on: cookies, the generic failure, the lock, 401 / 403, prefs, audit, logout."""

import json
from datetime import date
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from idsai.ai.auditor import Auditor
from idsai.bus import Bus
from idsai.config import load_settings
from idsai.data.store import Store
from idsai.product.api import AppState, create_app
from idsai.product.api import app as app_mod
from idsai.product.auth import ROLES, LoginLimiter, hash_password, token_hash
from tests.product.test_api import SYM, FakePipe, inject_body, make_alert, make_replayer
from tests.synth import make_trades

PW = "correct-horse-1"
HASH = hash_password(PW)
FAIL_TH = "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง"
LOGIN_TH = "กรุณาเข้าสู่ระบบ"
S = 1_000_000
T0 = 1_800_000_000 * S


def send(c, method, url, body=None, headers=None):
    content = None if body is None else json.dumps(body)
    hdrs = {"content-type": "application/json", **(headers or {})}
    return c.request(method, url, content=content, headers=hdrs)


def login(c, username, password=PW, remember=False):
    body = {"username": username, "password": password, "remember": remember}
    return send(c, "POST", "/api/auth/login", body)


def actions(store) -> list[str]:
    return [r["action"] for r in store.list_audit(500)]


@pytest.fixture
def aenv(tmp_path):
    cfg = load_settings(
        overrides={
            "paths": {"data_dir": str(tmp_path / "d"), "app_db": str(tmp_path / "a.duckdb")},
            "auth": {"enabled": True},
        }
    )
    store = Store(cfg.paths.data_dir, cfg.paths.app_db)
    store.write_trades(make_trades(date(2025, 3, 1), 2, symbol=SYM, seed=3, rate=0.05))
    replayer = make_replayer()
    state = AppState(
        cfg,
        store,
        "replay",
        FakePipe(Bus(maxsize=50)),
        replayer,
        Auditor(cfg),
        symbols=[SYM, "OTHERBTC"],
        auth_enabled=True,
    )
    for role in ROLES:
        store.add_user(role, role.title(), role, HASH, 1)
    store.upsert_alert(make_alert("a1"))
    yield SimpleNamespace(cfg=cfg, store=store, state=state, replayer=replayer)
    store.close()


@pytest.fixture
def auth_client(aenv):
    opened: list[TestClient] = []

    def make(role: str | None = None, remember: bool = False) -> TestClient:
        c = TestClient(create_app(aenv.state))
        c.__enter__()
        opened.append(c)
        if role is not None:
            r = login(c, role, remember=remember)
            assert r.status_code == 200, r.text
        return c

    yield make
    for c in opened:
        c.__exit__(None, None, None)


def cookie_parts(r) -> list[str]:
    cookies = r.headers.get_list("set-cookie")
    assert len(cookies) == 1
    return [p.strip() for p in cookies[0].split(";")]


# ---- login ----


@pytest.mark.parametrize("remember", [False, True])
@pytest.mark.parametrize("secure", [False, True])
def test_login_sets_cookie_flags(aenv, auth_client, monkeypatch, remember, secure):
    monkeypatch.setattr(app_mod, "_now_us", lambda: T0)
    auth = aenv.cfg.auth.model_copy(update={"cookie_secure": secure})
    aenv.state.cfg = aenv.cfg.model_copy(update={"auth": auth})
    c = auth_client()
    r = login(c, "analyst", remember=remember)
    assert r.status_code == 200
    assert r.json() == {
        "username": "analyst",
        "display_name": "Analyst",
        "role": "analyst",
        "prefs": {},
    }
    parts = cookie_parts(r)
    name, token = parts[0].split("=", 1)
    assert name == "idsai_session" and len(token) >= 40
    flags = parts[1:]
    assert "HttpOnly" in flags and "SameSite=Strict" in flags and "Path=/" in flags
    assert ("Secure" in flags) is secure
    max_age = [f for f in flags if f.lower().startswith("max-age")]
    assert max_age == (["Max-Age=2592000"] if remember else [])
    assert not any(f.lower().startswith("expires") for f in flags) or remember
    # the store keeps only the hash, and the server expiry matches the lifetime
    life = 30 * 86_400 * S if remember else 12 * 3600 * S
    assert aenv.store.get_session(token, T0) is None
    assert aenv.store.get_session(token_hash(token), T0 + life - 1) is not None
    assert aenv.store.get_session(token_hash(token), T0 + life) is None
    assert token not in r.text and "pw_hash" not in r.text and "scrypt" not in r.text


def test_login_failure_is_generic_and_audited(aenv, auth_client):
    c = auth_client()
    aenv.store.set_disabled("viewer", True)
    unknown = login(c, "nobody")
    wrong = login(c, "analyst", "wrong-password")
    disabled = login(c, "viewer")
    bad_name = login(c, "Not A Name!")
    for r in (unknown, wrong, disabled, bad_name):
        assert r.status_code == 401
        assert r.json() == {"detail": FAIL_TH}
        assert r.headers.get_list("set-cookie") == []
    assert unknown.text == wrong.text == disabled.text
    assert actions(aenv.store).count("login_failed") == 4
    assert PW not in json.dumps(aenv.store.list_audit(500))
    assert "wrong-password" not in json.dumps(aenv.store.list_audit(500))


def test_lone_surrogate_password_is_a_failed_login_not_a_500(aenv, auth_client, monkeypatch):
    c = auth_client()
    for raw in (  # the JSON parser refuses these before the route (422, no echo)
        '{"username": "analyst", "password": "\\ud800abcdefghij"}',
        '{"username": "\\ud800bad", "password": "abcdefghijkl"}',
    ):
        r = c.post("/api/auth/login", content=raw, headers={"content-type": "application/json"})
        assert r.status_code in (401, 422), r.text
        assert "Traceback" not in r.text and "ud800" not in r.text
        if r.status_code == 401:
            assert r.json() == {"detail": FAIL_TH}

    def boom(pw, stored):  # what verify does when the password cannot be encoded
        raise UnicodeEncodeError("utf-8", pw, 0, 1, "surrogates not allowed")

    monkeypatch.setattr(app_mod, "verify_or_dummy", boom)
    r = login(c, "analyst")
    assert r.status_code == 401 and r.json() == {"detail": FAIL_TH}
    assert actions(aenv.store)[0] == "login_failed"


def test_login_lock_after_five(aenv, auth_client):
    now = [0.0]
    aenv.state.limiter = LoginLimiter(clock=lambda: now[0])
    c = auth_client()
    for _ in range(5):
        assert login(c, "analyst", "wrong-password").status_code == 401
    # locked: the right password still fails, with the same message, and is audited
    for t in (1.0, 30.0, 59.0):
        now[0] = t
        r = login(c, "analyst")
        assert r.status_code == 401 and r.json() == {"detail": FAIL_TH}
    assert actions(aenv.store).count("login_failed") == 6  # the locked attempts are one row
    assert "login" not in actions(aenv.store)
    # attempts during the lock did not extend it
    now[0] = 61.0
    assert login(c, "analyst").status_code == 200
    # another username was never locked
    now[0] = 2.0
    assert login(auth_client(), "admin").status_code == 200


def run_async(app, *requests):
    """Send `(method, url, body)` requests to `app` concurrently on one event loop."""
    import asyncio

    import httpx

    async def main():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await asyncio.gather(*(c.request(m, u, json=b) for m, u, b in requests))

    return asyncio.run(main())


def wrong(username="analyst"):
    return ("POST", "/api/auth/login", {"username": username, "password": "wrong-password"})


def test_password_check_runs_off_the_event_loop(aenv, monkeypatch):
    import asyncio
    import time

    seen = []
    real = app_mod.verify_or_dummy

    def slow(pw, stored):
        try:
            asyncio.get_running_loop()
            seen.append("on the loop")
        except RuntimeError:
            seen.append("thread")
        time.sleep(0.5)
        return real(pw, stored)

    monkeypatch.setattr(app_mod, "verify_or_dummy", slow)
    app = create_app(aenv.state)
    done = []

    async def main():
        import httpx

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:

            async def go(m, u, b=None):
                r = await c.request(m, u, json=b)
                done.append(u)
                return r

            body = {"username": "analyst", "password": PW}
            return await asyncio.gather(
                go("POST", "/api/auth/login", body), go("GET", "/api/health")
            )

    login_r, health_r = asyncio.run(main())
    assert login_r.status_code == 200 and health_r.status_code == 200
    assert seen == ["thread"]
    assert done == ["/api/health", "/api/auth/login"]  # health was not stuck behind scrypt


def test_concurrent_wrong_attempts_still_lock(aenv):
    app = create_app(aenv.state)
    rs = run_async(app, *[wrong() for _ in range(7)])
    assert all(r.status_code == 401 and r.json() == {"detail": FAIL_TH} for r in rs)
    reasons = [r["detail"]["reason"] for r in aenv.store.list_audit(500)]
    assert reasons.count("password") == 5 and reasons.count("locked") == 1
    (ok,) = run_async(app, ("POST", "/api/auth/login", {"username": "analyst", "password": PW}))
    assert ok.status_code == 401  # still locked
    (other,) = run_async(app, ("POST", "/api/auth/login", {"username": "admin", "password": PW}))
    assert other.status_code == 200


def test_success_after_four_failures_clears_the_count(aenv, auth_client):
    c = auth_client()
    for _ in range(2):
        for _ in range(4):
            assert login(c, "analyst", "wrong-password").status_code == 401
        assert login(c, "analyst").status_code == 200
    assert not aenv.state.limiter.locked("analyst")


def test_login_refuses_cross_origin(auth_client):
    c = auth_client()
    body = {"username": "admin", "password": PW}
    r = send(c, "POST", "/api/auth/login", body, headers={"origin": "http://evil.example"})
    assert r.status_code == 403


# ---- 401 / 403 ----

READS = [
    "/api/alerts",
    "/api/alerts/a1",
    f"/api/series?symbol={SYM}",
    "/api/symbols",
    "/api/experiments",
    "/api/taxonomy",
    "/api/me",
    "/api/audit",
]
WRITES = [
    ("POST", "/api/alerts/a1/feedback", {"verdict": "tp"}),
    ("POST", "/api/replay/control", {"action": "pause"}),
    ("POST", "/api/redteam/inject", inject_body()),
    ("PUT", "/api/me/prefs", {"theme": "dark"}),
    ("POST", "/api/auth/logout", None),
]


def test_unauthenticated_gets_401_on_reads_and_ws_4401(aenv, auth_client):
    c = auth_client()
    for url in READS:
        r = c.get(url)
        assert r.status_code == 401, url
        assert r.json() == {"detail": LOGIN_TH}
    for method, url, body in WRITES:
        assert send(c, method, url, body).status_code == 401, url
    c.cookies.set("idsai_session", "forged-token")
    assert c.get("/api/alerts").status_code == 401
    with pytest.raises(WebSocketDisconnect) as e:
        with c.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
    assert e.value.code == 4401
    assert aenv.store.list_feedback("a1") == [] and not aenv.replayer._paused
    signed = auth_client("viewer")
    with signed.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["type"] == "health"


MUTATING = {
    "feedback": ("POST", "/api/alerts/a1/feedback", {"verdict": "tp"}, 201),
    "replay_control": ("POST", "/api/replay/control", {"action": "pause"}, 200),
    "redteam_inject": ("POST", "/api/redteam/inject", inject_body(), 202),
}


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("action", sorted(MUTATING))
def test_role_matrix(aenv, auth_client, role, action):
    method, url, body, ok = MUTATING[action]
    c = auth_client(role)
    for read in READS[:-1]:
        assert c.get(read).status_code == 200, read
    r = send(c, method, url, body)
    if role == "viewer":
        assert r.status_code == 403
        assert r.json() == {"detail": "สิทธิ์ไม่พอ (ต้องเป็น analyst)"}
        assert action not in actions(aenv.store)
    else:
        assert r.status_code == ok, r.text
        row = aenv.store.list_audit(1)[0]
        assert row["action"] == action and row["username"] == role
    audit = c.get("/api/audit")
    if role == "admin":
        assert audit.status_code == 200
        assert audit.json()[0]["action"] == action
    else:
        assert audit.status_code == 403
        assert audit.json() == {"detail": "สิทธิ์ไม่พอ (ต้องเป็น admin)"}


def test_audit_details_of_each_action(aenv, auth_client):
    c = auth_client("admin")
    send(c, "POST", "/api/alerts/a1/feedback", {"verdict": "fp"})
    send(c, "POST", "/api/replay/control", {"action": "speed", "value": 2})
    assert send(c, "POST", "/api/redteam/inject", inject_body()).status_code == 202
    rows = c.get("/api/audit?limit=3").json()
    assert [(r["action"], r["detail"]) for r in rows] == [
        ("redteam_inject", {"family": "pump_burst", "region": "train", "symbol": SYM}),
        ("replay_control", {"action": "speed", "value": 2}),
        ("feedback", {"verdict": "fp"}),
    ]
    assert rows[2]["target"] == "a1"
    older = c.get(f"/api/audit?before_us={rows[1]['ts_us']}").json()
    assert all(r["ts_us"] < rows[1]["ts_us"] for r in older)
    assert [r["action"] for r in older][-1] == "login"


def test_role_change_applies_next_request(aenv, auth_client):
    c = auth_client("viewer")
    assert send(c, "POST", "/api/replay/control", {"action": "pause"}).status_code == 403
    aenv.store.set_role("viewer", "analyst")
    assert send(c, "POST", "/api/replay/control", {"action": "pause"}).status_code == 200
    assert c.get("/api/me").json()["role"] == "analyst"
    aenv.store.set_disabled("viewer", True)
    assert c.get("/api/me").status_code == 401


def test_feedback_analyst_comes_from_session(aenv, auth_client):
    c = auth_client("analyst")
    body = {"verdict": "tp", "analyst": "someone-else"}
    r = send(c, "POST", "/api/alerts/a1/feedback", body)
    assert r.status_code == 201 and r.json()["analyst"] == "analyst"
    assert send(c, "POST", "/api/alerts/a1/feedback", {"verdict": "fp"}).status_code == 201
    fb = aenv.store.list_feedback("a1")
    assert [f["analyst"] for f in fb] == ["analyst", "analyst"]


def test_session_touch_at_most_once_a_minute(aenv, auth_client, monkeypatch):
    clock = [T0]
    monkeypatch.setattr(app_mod, "_now_us", lambda: clock[0])
    c = auth_client("viewer")
    token = token_hash(c.cookies["idsai_session"])
    clock[0] = T0 + 30 * S
    c.get("/api/me")
    assert aenv.store.get_session(token, clock[0])["last_seen_us"] == T0
    clock[0] = T0 + 61 * S
    c.get("/api/me")
    assert aenv.store.get_session(token, clock[0])["last_seen_us"] == T0 + 61 * S


# ---- me / prefs ----


def test_prefs_whitelist_and_round_trip(aenv, auth_client):
    c = auth_client("analyst")
    prefs = {"theme": "light", "symbol": SYM, "eval_budget": 2}
    r = send(c, "PUT", "/api/me/prefs", prefs)
    assert r.status_code == 200 and r.json() == prefs
    me = c.get("/api/me").json()
    assert me == {
        "username": "analyst",
        "display_name": "Analyst",
        "role": "analyst",
        "auth_enabled": True,
        "prefs": prefs,
    }
    # a partial update keeps the other keys
    assert send(c, "PUT", "/api/me/prefs", {"theme": "auto"}).json() == {**prefs, "theme": "auto"}
    for bad in (
        {"colour": "red"},
        {"theme": "pink"},
        {"symbol": "NOPEBTC"},
        {"eval_budget": 3},
        {"eval_budget": "2"},
        {"eval_budget": True},
    ):
        assert send(c, "PUT", "/api/me/prefs", bad).status_code == 422, bad
    assert c.get("/api/me").json()["prefs"] == {**prefs, "theme": "auto"}
    # prefs follow the user to a fresh client, and stay theirs
    again = login(auth_client(), "analyst")
    assert again.json()["prefs"] == {**prefs, "theme": "auto"}
    assert auth_client("admin").get("/api/me").json()["prefs"] == {}
    r = send(
        c, "PUT", "/api/me/prefs", {"theme": "dark"}, headers={"origin": "http://evil.example"}
    )
    assert r.status_code == 403


def test_logout_kills_session(aenv, auth_client):
    c = auth_client("analyst")
    token = c.cookies["idsai_session"]
    r = send(c, "POST", "/api/auth/logout")
    assert r.status_code == 204
    parts = cookie_parts(r)
    assert parts[0] in ('idsai_session=""', "idsai_session=")
    assert "Max-Age=0" in parts
    assert aenv.store.get_session(token_hash(token), app_mod._now_us()) is None
    assert actions(aenv.store)[0] == "logout"
    fresh = auth_client()
    fresh.cookies.set("idsai_session", token)
    assert fresh.get("/api/me").status_code == 401


def test_health_is_public(auth_client):
    c = auth_client()
    assert c.get("/api/health").status_code == 200
    if app_mod.WEB_DIR.is_dir():
        assert c.get("/").status_code == 200
        assert c.get("/app.js").status_code == 200


def test_auth_off_me_is_local(aenv):
    aenv.state.auth_enabled = False
    with TestClient(create_app(aenv.state)) as c:
        assert c.get("/api/me").json() == {
            "username": "local",
            "display_name": "local",
            "role": "admin",
            "auth_enabled": False,
            "prefs": {},
        }
        assert c.get("/api/audit").status_code == 200
        r = send(c, "POST", "/api/alerts/a1/feedback", {"verdict": "tp", "analyst": "x"})
        assert r.status_code == 201 and r.json()["analyst"] == "x"


def test_put_prefs_with_no_symbols_is_a_422_not_a_500(aenv, auth_client):
    c = auth_client("analyst")
    aenv.state.symbols = None
    assert send(c, "PUT", "/api/me/prefs", {"symbol": SYM}).status_code == 422
    assert send(c, "PUT", "/api/me/prefs", {"theme": "dark"}).status_code == 200


@pytest.mark.parametrize("name", ["nobody1", "Bad Name!", "x", "a" * 33])
def test_login_always_verifies_against_a_dummy_for_unknown_or_malformed_names(
    aenv, auth_client, monkeypatch, name
):
    seen = []

    def spy(pw, stored):
        seen.append(stored)
        return False

    monkeypatch.setattr(app_mod, "verify_or_dummy", spy)
    r = login(auth_client(), name)
    assert r.status_code == 401 and r.json() == {"detail": FAIL_TH}
    assert seen == [None]  # exactly one verify, with no stored hash: same cost as a real user


def test_locked_audit_coalesced(aenv, auth_client):
    c = auth_client()
    for _ in range(5):
        assert login(c, "analyst", "wrong-password").status_code == 401
    for _ in range(5):
        r = login(c, "analyst")
        assert r.status_code == 401 and r.json() == {"detail": FAIL_TH}
    reasons = [
        r["detail"]["reason"] for r in aenv.store.list_audit(500) if r["action"] == "login_failed"
    ]
    assert reasons.count("locked") == 1 and reasons.count("password") == 5


def test_login_waiter_cap(aenv, monkeypatch):
    import asyncio
    import secrets
    import threading

    import httpx

    gate = threading.Event()
    called: list[str] = []

    def blocked(pw, stored):
        called.append(pw)
        gate.wait(10)
        return False

    monkeypatch.setattr(app_mod, "verify_or_dummy", blocked)
    app = create_app(aenv.state)
    n = app_mod.MAX_LOGIN_WAITERS + 1
    creds = [(f"user{i:02d}", secrets.token_urlsafe(12)) for i in range(n)]

    async def main():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            tasks = [
                asyncio.create_task(
                    c.post("/api/auth/login", json={"username": u, "password": p})
                )
                for u, p in creds
            ]
            try:
                done, _ = await asyncio.wait(tasks, timeout=10, return_when=asyncio.FIRST_COMPLETED)
                assert done  # the over-cap one answers while the others still wait
            finally:
                gate.set()
            return await asyncio.gather(*tasks)

    rs = asyncio.run(main())
    busy = [i for i, r in enumerate(rs) if r.status_code == 429]
    assert busy
    for i in busy:
        assert rs[i].json() == {"detail": "ระบบไม่ว่าง ลองใหม่อีกครั้ง"}
        u, p = creds[i]
        assert u not in aenv.state.limiter._fails
        assert p not in called
    assert all(r.status_code == 401 for i, r in enumerate(rs) if i not in busy)


def ws_close_code_after(aenv, c, monkeypatch, sign_out) -> int:
    monkeypatch.setattr(app_mod, "WS_RECHECK_S", 0.05)
    with pytest.raises(WebSocketDisconnect) as e:
        with c.websocket_connect("/ws/stream") as ws:
            assert ws.receive_json()["type"] == "health"
            sign_out()
            for _ in range(50):  # health arrives once a second; the close comes first
                ws.receive_json()
    return e.value.code


def test_ws_closes_when_session_deleted(aenv, auth_client, monkeypatch):
    c = auth_client("viewer")
    th = token_hash(c.cookies["idsai_session"])
    assert ws_close_code_after(aenv, c, monkeypatch, lambda: aenv.store.delete_session(th)) == 4401


def test_ws_closes_when_user_disabled(aenv, auth_client, monkeypatch):
    c = auth_client("viewer")
    off = lambda: aenv.store.set_disabled("viewer", True)  # noqa: E731
    assert ws_close_code_after(aenv, c, monkeypatch, off) == 4401


def test_ws_stays_open_while_signed_in(aenv, auth_client, monkeypatch):
    monkeypatch.setattr(app_mod, "WS_RECHECK_S", 0.05)
    monkeypatch.setattr(app_mod, "HEALTH_S", 0.2)
    c = auth_client("viewer")
    with c.websocket_connect("/ws/stream") as ws:
        for _ in range(3):  # several re-checks pass in between
            assert ws.receive_json()["type"] == "health"
