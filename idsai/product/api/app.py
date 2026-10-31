"""FastAPI app: REST routes and the `/ws/stream` WebSocket.

Every route is `async def`, so it runs on the one event loop that owns the store, the bus, the
replayer and the pipeline. There is no login and no CORS: a POST or WebSocket whose `Origin` host
differs from `Host` is refused, which stops an ordinary page on another site. It does not stop DNS
rebinding, where `Origin` and `Host` agree on the attacker's name: that takes a host allowlist where
the bind address is chosen. A request without `Origin` (curl, tests) is allowed. JSON is strict:
non-finite floats go out as null.

WebSocket messages are `{type: bar|score|alert|health, data}`. Per client and symbol, bar and score
go out at most 4 Hz and a held value is replaced by a newer one, so the client always ends on the
latest. Alerts bypass the throttle and have a queue per client; if the bus dropped alerts during
the connection, `{type: "alert", data: {"event": "resync"}}` tells the client to re-read
`GET /api/alerts`. A client that does not take a message within `SEND_TIMEOUT_S` counts as gone.
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
    StrictFloat,
    StrictInt,
    StringConstraints,
    field_validator,
)
from starlette.requests import HTTPConnection
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


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeedbackIn(_Body):
    verdict: Verdict
    analyst: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]

    @field_validator("analyst")
    @classmethod
    def _printable(cls, v: str) -> str:
        if not v.isprintable():
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


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="Market IDS", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> Response:
        # only where and why: the default body echoes the input, which may hold NaN
        detail = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return _json({"detail": detail}, 422)

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException) -> Response:
        return _json({"detail": exc.detail}, exc.status_code)

    # ---- alerts ----

    @app.get("/api/alerts")
    async def alerts(
        symbol: Annotated[str | None, Query(max_length=32)] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ):
        return _json([alert_item(a) for a in state.store.list_alerts(limit, symbol)])

    @app.get("/api/alerts/{alert_id}")
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

    @app.post(
        "/api/alerts/{alert_id}/feedback", status_code=201, dependencies=[Depends(same_origin)]
    )
    async def feedback(alert_id: str, body: FeedbackIn):
        if state.store.get_alert(alert_id) is None:
            raise HTTPException(404, "ไม่พบการแจ้งเตือนนี้")
        state.store.insert_feedback(alert_id, body.verdict, body.analyst)
        return _json({"alert_id": alert_id, "verdict": body.verdict, "analyst": body.analyst}, 201)

    # ---- series / reads ----

    @app.get("/api/series")
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

    @app.get("/api/symbols")
    async def symbols():
        return _json(state.symbols)

    @app.get("/api/health")
    async def health_route():
        return _json(health(state))

    @app.get("/api/experiments")
    async def experiments():
        try:
            report = json.loads(state.cfg.paths.report_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = None
        return _json(report if isinstance(report, dict) else {"runs": 0})

    @app.get("/api/taxonomy")
    async def taxonomy():
        return _json(load_taxonomy())

    # ---- replay / red team ----

    @app.post("/api/replay/control", dependencies=[Depends(same_origin)])
    async def replay_control(body: ControlIn):
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
        return _json({"action": body.action, "cursor_us": rp.cursor_us})

    @app.post("/api/redteam/inject", status_code=202, dependencies=[Depends(same_origin)])
    async def inject(body: InjectIn):
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
        return _json(result, 202)

    # ---- websocket ----

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket):
        if _cross_origin(ws):
            await ws.close(code=1008)
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

            tasks = [asyncio.create_task(listen()), asyncio.create_task(pulse())]
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
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
