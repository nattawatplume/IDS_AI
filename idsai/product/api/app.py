"""FastAPI app: REST routes and the `/ws/stream` WebSocket.

Every route is `async def`, so it runs on the one event loop that owns the store, the bus, the
replayer and the pipeline. There is no CORS: a POST, PUT or WebSocket whose `Origin` host differs
from `Host` is refused, which stops an ordinary page on another site. It does not stop DNS
rebinding, where `Origin` and `Host` agree on the attacker's name: that takes a host allowlist where
the bind address is chosen (`create_app(allowed_hosts=...)`, which `idsai serve` sets). A request
without `Origin` (curl, tests) is allowed. JSON is strict: non-finite floats go out as null.

WebSocket messages are `{type: bar|score|alert|health, data}`. Per client and symbol, bar and score
go out at most 4 Hz and a held value is replaced by a newer one, so the client always ends on the
latest. Alerts bypass the throttle and have a queue per client; if the bus dropped alerts during
the connection, `{type: "alert", data: {"event": "resync"}}` tells the client to re-read
`GET /api/alerts`. A client that does not take a message within `SEND_TIMEOUT_S` counts as gone.

Sign-in (`state.auth_enabled`): the `idsai_session` cookie holds a random token whose SHA-256 is the
session key in the store. Every `/api/*` data route and the WebSocket need a session
(`current_user`; 401, WS close 4401); feedback, replay control and inject need an analyst, the
audit log an admin (`require`; 403). `/api/health`, login and the dashboard files stay public. With
auth off every request is `LOCAL_USER`, role admin. Logins, failed logins, logouts and every
analyst action go to the audit log.
"""

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    field_validator,
)
from starlette._utils import parse_host_header
from starlette.requests import HTTPConnection
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect

from idsai.contracts.alert import Alert
from idsai.data.store import Verdict
from idsai.detection.simulator import base_quote_rate
from idsai.detection.simulator.guard import EthicsError, assert_injection_allowed
from idsai.detection.simulator.scenarios import (
    Family,
    Region,
    load_scenario_ranges,
    sample_scenario,
)
from idsai.detection.taxonomy import load_taxonomy
from idsai.product.api.state import AppState, clean, health
from idsai.product.auth import (
    ROLES,
    check_username,
    new_token,
    role_at_least,
    token_hash,
    verify_or_dummy,
)
from idsai.product.explainer import explain

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
S_US = 1_000_000
DAY_S = 86_400
THROTTLE_S = 0.25  # bar / score: 4 Hz per client and symbol
HEALTH_S = 1.0
SEND_TIMEOUT_S = 10.0  # a client that stopped reading blocks its sends: it is then disconnected
LANES = ("lane_rule", "lane_stat", "lane_ai")
REPLAY_ENDED_TH = "การเล่นย้อนหลังนี้จบแล้ว ต้องเริ่มใหม่"
REFUSED_TH = "Red-Team ปิดอยู่ในโหมด LIVE เพราะระบบไม่แก้ไขข้อมูลตลาดจริง (ใช้ได้เฉพาะโหมด replay / sim)"
COOKIE = "idsai_session"
TOUCH_US = 60 * S_US  # last_seen_us is written at most once a minute
LOGIN_FAILED_TH = "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง"
LOGIN_NEEDED_TH = "กรุณาเข้าสู่ระบบ"
WS_UNAUTHORIZED = 4401
VERIFY_SLOTS = 2  # password checks run in worker threads, at most this many at once
MAX_LOGIN_WAITERS = 16  # logins queued on or holding a verify slot; beyond it: 429
BUSY_TH = "ระบบไม่ว่าง ลองใหม่อีกครั้ง"
WS_RECHECK_S = 60  # an open WebSocket re-reads its session this often; gone or disabled: 4401
LOCAL_USER = {"user_id": "local", "username": "local", "display_name": "local", "role": "admin"}
_USER_KEYS = ("user_id", "username", "display_name", "role")


def _now_us() -> int:
    return time.time_ns() // 1000


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeedbackIn(_Body):
    """`analyst` is required with auth off; with auth on the session's username is used."""

    verdict: Verdict
    analyst: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)] | None
    ) = None

    @field_validator("analyst")
    @classmethod
    def _printable(cls, v: str | None) -> str | None:
        if v is not None and not v.isprintable():
            raise ValueError("analyst must not contain control characters")
        return v


class ControlIn(_Body):
    action: Literal["play", "pause", "speed", "seek"]
    value: StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)] | None = None


class InjectIn(_Body):
    family: Family
    region: Region
    offset_s: (
        Annotated[StrictInt, Field(gt=0, le=3600)]
        | Annotated[StrictFloat, Field(gt=0, le=3600, allow_inf_nan=False)]
    )
    symbol: str | None = Field(default=None, max_length=32)


class LoginIn(_Body):
    username: str = Field(max_length=64)
    password: str = Field(max_length=1024)
    remember: bool = False


class PrefsIn(_Body):
    """The whitelist; `symbol` and `eval_budget` are checked against the app in the route. A key
    sent as null is removed."""

    theme: Literal["dark", "light", "auto"] | None = None
    symbol: str | None = Field(default=None, max_length=32)
    eval_budget: StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)] | None = None
    tour_done: StrictBool | None = None


class _ClientGone(Exception):
    """A send failed or timed out: the client is closed, vanished or no longer reading."""


def _dumps(obj: Any) -> str:
    return json.dumps(clean(obj), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _json(obj: Any, status: int = 200) -> Response:
    return Response(_dumps(obj), status_code=status, media_type="application/json")


def alert_item(a: Alert) -> dict[str, Any]:
    """One alert as `GET /api/alerts` and the WebSocket serve it."""
    return clean({**a.model_dump(mode="json"), "explanation_th": explain(a).text_th})


def _cross_origin(conn: HTTPConnection) -> bool:
    origin = conn.headers.get("origin")
    if origin is None:
        return False
    try:
        netloc = urlsplit(origin).netloc
    except ValueError:  # unparsable: it cannot be shown to be this host
        return True
    return netloc.lower() != (conn.headers.get("host") or "").lower()


async def same_origin(conn: HTTPConnection) -> None:
    if _cross_origin(conn):
        raise HTTPException(403, "คำขอจากเว็บไซต์อื่นถูกปฏิเสธ (cross-origin)")


async def current_user(conn: HTTPConnection) -> dict:
    """The signed-in user `{user_id, username, display_name, role}` (never `pw_hash`), or 401.

    With auth off it is `LOCAL_USER`. The role is read from `users` on every request, so a role
    change or a disable applies on the next one."""
    state: AppState = conn.app.state.idsai
    if not state.auth_enabled:
        return dict(LOCAL_USER)
    token = conn.cookies.get(COOKIE)
    if not token:
        raise HTTPException(401, LOGIN_NEEDED_TH)
    th = token_hash(token)
    now = _now_us()
    s = state.store.get_session(th, now)
    if s is None:
        raise HTTPException(401, LOGIN_NEEDED_TH)
    if now - s["last_seen_us"] >= TOUCH_US:
        state.store.touch_session(th, now)
    return {k: s[k] for k in _USER_KEYS}


def require(role: str):
    """A dependency giving the user when their role is at least `role`, else 403."""
    if role not in ROLES:
        raise ValueError(f"unknown role: {role!r}")

    async def dep(user: Annotated[dict, Depends(current_user)]) -> dict:
        if user["role"] not in ROLES or not role_at_least(user["role"], role):
            raise HTTPException(403, f"สิทธิ์ไม่พอ (ต้องเป็น {role})")
        return user

    return dep


Signed = Annotated[dict, Depends(current_user)]
Analyst = Annotated[dict, Depends(require("analyst"))]
Admin = Annotated[dict, Depends(require("admin"))]


BAD_HOST_TH = (
    "ชื่อเครื่อง (Host) ที่ใช้เรียกไม่อยู่ในรายการที่อนุญาต (กันการโจมตีแบบ DNS rebinding) "
    "ให้เรียกด้วย 127.0.0.1 หรือ localhost หรือเปิด serve ด้วย --host <IP ของเครื่องนี้> "
    "แล้วเรียกผ่าน IP นั้น"
)


def _health_probe(scope: Scope) -> bool:
    return (
        scope["type"] == "http"
        and scope.get("method") in ("GET", "HEAD")
        and scope.get("path") == "/api/health"
    )


class HostAllowlist:
    """ASGI middleware: HTTP and WebSocket requests whose `Host` (port ignored, IPv6 in brackets)
    is not in `allowed` get 400 with a Thai hint; a WebSocket is refused before it is accepted.
    `GET`/`HEAD /api/health` skips the check: a host's own probes (e.g. Hugging Face) call it by
    an internal IP, and it returns nothing private."""

    def __init__(self, app: ASGIApp, allowed: list[str]):
        self.app = app
        self.allowed = {h.lower() for h in allowed}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket") and not _health_probe(scope):
            raw = dict(scope.get("headers") or []).get(b"host", b"").decode("latin-1")
            parsed = parse_host_header(raw)
            if parsed is None or parsed.host.lower() not in self.allowed:
                response = PlainTextResponse(BAD_HOST_TH, status_code=400)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class _NoCacheStatic(StaticFiles):
    """Dashboard files revalidate on every load (ETag / Last-Modified still give a 304)."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


def create_app(state: AppState, allowed_hosts: list[str] | None = None) -> FastAPI:
    """The app over `state`. `allowed_hosts` (e.g. loopback names plus the bound host) turns on a
    `Host` header allowlist against DNS rebinding: any other host gets 400, HTTP and WebSocket
    alike. None (the default) checks no host."""
    app = FastAPI(title="Market IDS", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.idsai = state  # read by `current_user`
    signed = [Depends(current_user)]
    verify_slots = asyncio.Semaphore(VERIFY_SLOTS)  # concurrent logins queue here
    login_waiters = 0  # logins waiting on or holding `verify_slots`
    if allowed_hosts is not None:
        app.add_middleware(HostAllowlist, allowed=list(allowed_hosts))

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> Response:
        # only where and why: the default body echoes the input, which may hold NaN
        detail = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return _json({"detail": detail}, 422)

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException) -> Response:
        return _json({"detail": exc.detail}, exc.status_code)

    def audit(user: dict, action: str, target: str = "", detail: dict | None = None) -> None:
        state.store.add_audit(_now_us(), user["user_id"], user["username"], action, target, detail)

    def cookie_flags() -> dict[str, Any]:
        return {
            "path": "/",
            "secure": state.cfg.auth.cookie_secure,
            "httponly": True,
            "samesite": "Strict",
        }

    # ---- sign-in ----

    @app.post("/api/auth/login", dependencies=[Depends(same_origin)])
    async def login(body: LoginIn):
        nonlocal login_waiters
        if not state.auth_enabled:
            raise HTTPException(409, "ระบบล็อกอินปิดอยู่")
        store, limiter, now = state.store, state.limiter, _now_us()
        name = body.username
        shown = name.encode("utf-8", "replace").decode("utf-8")  # lone surrogates cannot be stored

        def failed(user: dict | None, reason: str) -> HTTPException:
            uid = None if user is None else user["user_id"]
            store.add_audit(now, uid, shown, "login_failed", shown, {"reason": reason})
            return HTTPException(401, LOGIN_FAILED_TH)

        if limiter.locked(name):  # no fail() here: it would extend the lock
            if limiter.first_locked_hit(name):  # one audit row per lock, not one per attempt
                raise failed(None, "locked")
            raise HTTPException(401, LOGIN_FAILED_TH)
        try:
            check_username(name)
        except ValueError:
            user = None  # such a name cannot exist: no lookup
        else:
            user = store.get_user_by_name(name)
        if login_waiters >= MAX_LOGIN_WAITERS:  # a flood is refused before it costs a count
            raise HTTPException(429, BUSY_TH)
        # the attempt is counted before the await, so concurrent guesses cannot pass the lock
        # check together; a success clears it again
        limiter.fail(name)
        stored = None if user is None else user["pw_hash"]
        login_waiters += 1
        try:
            async with verify_slots:  # scrypt off the loop, a bounded number at a time
                ok = await asyncio.to_thread(verify_or_dummy, body.password, stored)
        except Exception:  # e.g. a password with lone surrogates cannot be encoded
            ok = False
        finally:
            login_waiters -= 1
        if ok:  # the user may have changed during the await
            user = store.get_user_by_name(name)
            ok = user is not None and user["pw_hash"] == stored
        if user is None or not ok or user["disabled"]:
            raise failed(user, "unknown" if user is None else "disabled" if ok else "password")
        limiter.ok(name)
        auth = state.cfg.auth
        life_s = auth.remember_days * DAY_S if body.remember else auth.session_hours * 3600
        token = new_token()
        store.create_session(
            user["user_id"], token_hash(token), now, now + life_s * S_US, body.remember
        )
        me = {k: user[k] for k in _USER_KEYS}  # never pw_hash
        store.add_audit(
            now, me["user_id"], me["username"], "login", "", {"remember": body.remember}
        )
        out = {k: me[k] for k in ("username", "display_name", "role")}
        resp = _json({**out, "prefs": store.get_prefs(me["user_id"])})
        resp.set_cookie(COOKIE, token, max_age=life_s if body.remember else None, **cookie_flags())
        return resp

    @app.post("/api/auth/logout", status_code=204, dependencies=[Depends(same_origin)])
    async def logout(request: Request, user: Signed):
        token = request.cookies.get(COOKIE)
        if state.auth_enabled and token:
            state.store.delete_session(token_hash(token))
            audit(user, "logout")
        resp = Response(status_code=204)
        resp.delete_cookie(COOKIE, **cookie_flags())
        return resp

    @app.get("/api/me")
    async def me(user: Signed):
        return _json(
            {
                "username": user["username"],
                "display_name": user["display_name"],
                "role": user["role"],
                "auth_enabled": state.auth_enabled,
                "prefs": state.store.get_prefs(user["user_id"]),
            }
        )

    @app.put("/api/me/prefs", dependencies=[Depends(same_origin)])
    async def put_prefs(body: PrefsIn, user: Signed):
        if body.symbol is not None and body.symbol not in (state.symbols or []):
            raise HTTPException(422, "ไม่รู้จัก symbol นี้")
        budgets = state.cfg.eval.budgets_per_symbol_day
        if body.eval_budget is not None and body.eval_budget not in budgets:
            raise HTTPException(422, f"eval_budget ต้องเป็นหนึ่งใน {budgets}")
        prefs = state.store.get_prefs(user["user_id"])
        for k, v in body.model_dump(exclude_unset=True).items():
            if v is None:
                prefs.pop(k, None)
            else:
                prefs[k] = v
        state.store.set_prefs(user["user_id"], prefs)
        return _json(prefs)

    @app.get("/api/audit")
    async def audit_log(
        user: Admin,
        limit: Annotated[int, Query(ge=1, le=500)] = 200,
        before_us: int | None = None,
    ):
        return _json(state.store.list_audit(limit, before_us))

    # ---- alerts ----

    @app.get("/api/alerts", dependencies=signed)
    async def alerts(
        symbol: Annotated[str | None, Query(max_length=32)] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ):
        return _json([alert_item(a) for a in state.store.list_alerts(limit, symbol)])

    @app.get("/api/alerts/{alert_id}", dependencies=signed)
    async def alert_detail(alert_id: str):
        a = state.store.get_alert(alert_id)
        if a is None:
            raise HTTPException(404, "ไม่พบการแจ้งเตือนนี้")
        e = explain(a)
        return _json(
            {
                **alert_item(a),
                "explanation": {
                    "text_th": e.text_th,
                    "facts": [{"label": k, "value": v} for k, v in e.facts],
                },
                "feedback": state.store.list_feedback(alert_id),
            }
        )

    @app.get("/api/feedback", dependencies=signed)
    async def feedback_list():
        """Every verdict, oldest first: the alert list's filters and verdict chips read it."""
        return _json(state.store.list_feedback())

    @app.post(
        "/api/alerts/{alert_id}/feedback", status_code=201, dependencies=[Depends(same_origin)]
    )
    async def feedback(alert_id: str, body: FeedbackIn, user: Analyst):
        analyst = user["username"] if state.auth_enabled else body.analyst
        if analyst is None:
            raise HTTPException(422, "ต้องระบุ analyst")
        if state.store.get_alert(alert_id) is None:
            raise HTTPException(404, "ไม่พบการแจ้งเตือนนี้")
        state.store.insert_feedback(alert_id, body.verdict, analyst)
        audit(user, "feedback", alert_id, {"verdict": body.verdict})
        return _json({"alert_id": alert_id, "verdict": body.verdict, "analyst": analyst}, 201)

    # ---- series / reads ----

    @app.get("/api/series", dependencies=signed)
    async def series(
        symbol: Annotated[str, Query(max_length=32)],
        start_us: int | None = None,
        end_us: int | None = None,
    ):
        if symbol not in state.symbols:
            raise HTTPException(404, "ไม่รู้จัก symbol นี้")
        if start_us is not None and end_us is not None and start_us > end_us:
            raise HTTPException(422, "start_us ต้องไม่มากกว่า end_us")
        pts = [] if state.pipeline is None else state.pipeline.series(symbol)
        pts = [
            p
            for p in pts
            if (start_us is None or p["t_end_us"] >= start_us)
            and (end_us is None or p["t_end_us"] <= end_us)
        ]
        out: dict[str, Any] = {
            k: [p.get(k) for p in pts] for k in ("t_end_us", "close", "F", *LANES)
        }
        out["tau"] = None if state.pipeline is None else state.pipeline.chief.tau
        return _json(out)

    @app.get("/api/symbols", dependencies=signed)
    async def symbols():
        return _json(state.symbols)

    @app.get("/api/health")
    async def health_route():
        return _json(health(state))

    @app.get("/api/experiments", dependencies=signed)
    async def experiments():
        try:
            report = json.loads(state.cfg.paths.report_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = None
        return _json(report if isinstance(report, dict) else {"runs": 0})

    @app.get("/api/taxonomy", dependencies=signed)
    async def taxonomy():
        return _json(load_taxonomy())

    # ---- replay / red team ----

    @app.post("/api/replay/control", dependencies=[Depends(same_origin)])
    async def replay_control(body: ControlIn, user: Analyst):
        rp = state.replayer
        if state.mode == "live" or rp is None:
            raise HTTPException(409, "ควบคุมการเล่นย้อนหลังไม่ได้ในโหมดนี้")
        if rp.ended:
            raise HTTPException(409, REPLAY_ENDED_TH)
        if body.action == "pause":
            rp.pause()
        elif body.action == "play":
            rp.resume()
        elif body.action == "speed":
            if body.value is None:
                raise HTTPException(422, "speed ต้องมี value")
            try:
                rp.set_speed(body.value)
            except ValueError as e:
                raise HTTPException(422, str(e)) from None
        else:
            if not isinstance(body.value, int):
                raise HTTPException(422, "seek ต้องมี value เป็นจำนวนเต็ม (µs)")
            rp.seek(body.value)
        audit(user, "replay_control", "", {"action": body.action, "value": body.value})
        return _json({"action": body.action, "cursor_us": rp.cursor_us})

    @app.post("/api/redteam/inject", status_code=202, dependencies=[Depends(same_origin)])
    async def inject(body: InjectIn, user: Analyst):
        try:
            assert_injection_allowed(state.mode)
        except EthicsError:
            raise HTTPException(403, REFUSED_TH) from None
        rp = state.replayer
        if rp is None:
            raise HTTPException(409, "ไม่มีตัวเล่นย้อนหลังสำหรับฉีดข้อมูลจำลอง")
        if rp.ended:  # the scenario would never play
            raise HTTPException(409, REPLAY_ENDED_TH)
        symbol = body.symbol
        if symbol is None and len(rp.symbols) == 1:
            symbol = rp.symbols[0]
        if symbol is None or symbol not in state.symbols:
            raise HTTPException(422, "ต้องระบุ symbol ที่รู้จัก")
        cursor = rp.cursor_us
        t_start = cursor + int(round(body.offset_s * S_US))
        rng = np.random.default_rng([state.cfg.eval.split_seed, len(state.injected)])
        sc = sample_scenario(body.family, body.region, t_start, rng, load_scenario_ranges())
        if t_start + int(round(sc.duration_s * S_US)) > rp.end_us:
            raise HTTPException(422, "สถานการณ์จำลองจะจบหลังข้อมูลสุดท้าย ลดค่า offset_s")
        try:
            prev = state.store.read_trades(symbol, cursor - DAY_S * S_US, cursor)
            label = rp.inject(sc, base_quote_rate(prev), mode=state.mode, symbol=symbol)
        except EthicsError:
            raise HTTPException(403, REFUSED_TH) from None
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        result = clean({"scenario": asdict(sc), "label": label.to_dicts()[0]})
        state.injected.append(result)
        detail = {"family": body.family, "region": body.region, "symbol": symbol}
        audit(user, "redteam_inject", symbol, detail)
        return _json(result, 202)

    # ---- websocket ----

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket):
        if _cross_origin(ws):
            await ws.close(code=1008)
            return
        try:
            await current_user(ws)
        except HTTPException:
            await ws.accept()  # accepted first, so a browser sees the code; nothing is sent
            await ws.close(code=WS_UNAUTHORIZED)
            return
        await ws.accept()
        bus = state.bus
        queues = {t: bus.subscribe(t) for t in ("bars", "scores", "alerts")} if bus else {}
        tasks: list[asyncio.Task] = []
        try:
            lock = asyncio.Lock()
            latest: dict[tuple[str, str], Any] = {}
            wake = asyncio.Event()
            dropped0 = bus.dropped["alerts"] if bus else 0

            async def send(kind: str, data: Any) -> None:
                text = _dumps({"type": kind, "data": data})
                async with lock:
                    try:
                        async with asyncio.timeout(SEND_TIMEOUT_S):
                            await ws.send_text(text)
                    except (WebSocketDisconnect, RuntimeError, OSError) as e:  # with TimeoutError
                        raise _ClientGone from e

            async def collect(kind: str, q: asyncio.Queue) -> None:
                while True:
                    m = await q.get()
                    latest[(kind, str(m.get("symbol", "")))] = m  # a newer value replaces it
                    wake.set()

            async def flush() -> None:
                next_ok: dict[tuple[str, str], float] = {}
                while True:
                    await wake.wait()
                    now = time.monotonic()
                    for key in [k for k in latest if now >= next_ok.get(k, 0.0)]:
                        if key in latest:  # a reset may have dropped it during the last send
                            next_ok[key] = now + THROTTLE_S
                            await send(key[0], latest.pop(key))
                    if not latest:
                        wake.clear()
                    else:
                        due = min(next_ok.get(k, 0.0) for k in latest)
                        await asyncio.sleep(max(due - time.monotonic(), 0.0))

            async def alert_pump(q: asyncio.Queue) -> None:
                seen = dropped0
                while True:
                    m = await q.get()
                    if isinstance(m, dict) and m.get("event") == "alerts_reset":
                        latest.clear()  # a held value from before the seek must not follow it
                    if bus.dropped["alerts"] != seen:  # some were lost: the client must re-read
                        seen = bus.dropped["alerts"]
                        await send("alert", {"event": "resync"})
                    await send("alert", m if isinstance(m, dict) else alert_item(m))

            async def pulse() -> None:
                while True:
                    await send("health", health(state))
                    await asyncio.sleep(HEALTH_S)

            async def listen() -> None:
                while (await ws.receive())["type"] != "websocket.disconnect":
                    pass

            async def recheck() -> None:  # a logout, a disable or an expiry ends the stream
                while True:
                    await asyncio.sleep(WS_RECHECK_S)
                    try:
                        await current_user(ws)
                    except HTTPException:
                        async with lock:  # not in the middle of a send
                            try:
                                await ws.close(code=WS_UNAUTHORIZED)
                            except (WebSocketDisconnect, RuntimeError, OSError):
                                pass  # already gone
                        return

            tasks = [
                asyncio.create_task(listen()),
                asyncio.create_task(pulse()),
                asyncio.create_task(recheck()),
            ]
            if bus:
                tasks += [
                    asyncio.create_task(collect("bar", queues["bars"])),
                    asyncio.create_task(collect("score", queues["scores"])),
                    asyncio.create_task(flush()),
                    asyncio.create_task(alert_pump(queues["alerts"])),
                ]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:  # only a failed send means the client left; anything else is a bug
                if not t.cancelled() and not isinstance(t.exception(), _ClientGone | None):
                    raise t.exception()
        finally:
            for t in tasks:  # no await here: a cancelled handler must still clean up in full
                t.cancel()
            for topic, q in queues.items():
                bus.unsubscribe(topic, q)

    if WEB_DIR.is_dir():
        app.mount("/", _NoCacheStatic(directory=WEB_DIR, html=True), name="web")
    return app
