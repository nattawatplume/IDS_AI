"""REST + WebSocket API: shapes, errors, strict JSON, browser safety, stream and cleanup."""

import asyncio
import inspect
import json
import time
from collections import deque
from datetime import date
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest
from fastapi.routing import APIRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from idsai.ai.auditor import Auditor
from idsai.bus import Bus
from idsai.config import load_settings
from idsai.contracts.alert import Alert, DetectorRef
from idsai.data.replay import Replayer
from idsai.data.store import Store
from idsai.detection.simulator.guard import EthicsError
from idsai.detection.simulator.scenarios import load_scenario_ranges, sample_scenario
from idsai.pipeline.stream import StreamPipeline
from idsai.product.api import AppState, create_app
from idsai.product.api import app as app_mod
from idsai.product.explainer import BANNED_TH
from tests.pipeline.test_stream_gap import cheap_chief
from tests.synth import day_us, make_trades

S = 1_000_000
SYM = "TESTBTC"
D = date(2025, 3, 2)
D_US = day_us(D)


def strict_loads(text: str) -> Any:
    def bad(c: str):
        raise ValueError(f"non-finite constant {c} in JSON")

    return json.loads(text, parse_constant=bad)


def make_alert(aid="a1", ts=D_US + 3600 * S, symbol=SYM, prov="replay", **kw) -> Alert:
    base = dict(
        alert_id=aid,
        ts_event_us=ts,
        symbol=symbol,
        technique_id="MM-T001",
        severity=2,
        status="open",
        fused_p=0.0004,
        detectors=[DetectorRef(name="rule.x", layer="rule", p=0.01, score=1.0)],
        agreement=2,
        evidence=[("z_quote_vol_60", 7.1, 7.1), ("ret_60", 0.042, 0.5)],
        window=(ts - 60 * S, ts),
        dedup_key="k",
        suppress_for_s=60,
        config_hash="h",
        model_versions={},
        provenance=prov,
    )
    base.update(kw)
    return Alert(**base)


class FakePipe:
    """What the API reads from a StreamPipeline."""

    def __init__(self, bus: Bus, mode: str = "replay"):
        self.bus = bus
        self.mode = mode
        self.chief = SimpleNamespace(tau=3.5)
        self.latency_ms: deque[float] = deque([5.0, 8.0, 12.0])
        self.points: dict[str, list[dict]] = {}
        self.samples = 400
        self.samples_by: dict[str, int] = {}
        self.late = 2

    def series(self, symbol):
        return list(self.points.get(symbol, ()))

    def baseline_samples(self, symbol):
        return self.samples_by.get(symbol, self.samples)

    def late_trades(self):
        return self.late


def point(i: int, **kw) -> dict:
    p = {
        "t_end_us": D_US + i * 10 * S,
        "close": 100.0 + i,
        "F": 0.5 * i,
        "lane_rule": 1.0,
        "lane_stat": 2.0,
        "lane_ai": 3.0,
    }
    p.update(kw)
    return p


def make_replayer() -> Replayer:
    trades = make_trades(date(2025, 3, 1), 2, symbol=SYM, seed=3, rate=0.05)
    return Replayer(trades, speed=1.0, start_us=D_US + 3600 * S)


def finish(rp: Replayer) -> None:
    """End `rp` the way a cancelled or crashed pipeline does: its iterator is closed."""

    async def run():
        it = aiter(rp)
        await anext(it)
        await it.aclose()

    asyncio.run(run())


@pytest.fixture
def env(tmp_path):
    cfg = load_settings(
        overrides={"paths": {"data_dir": str(tmp_path / "d"), "app_db": str(tmp_path / "a.duckdb")}}
    )
    store = Store(cfg.paths.data_dir, cfg.paths.app_db)
    store.write_trades(make_trades(date(2025, 3, 1), 2, symbol=SYM, seed=3, rate=0.05))
    bus = Bus(maxsize=50)
    pipe = FakePipe(bus)
    replayer = make_replayer()
    state = AppState(cfg, store, "replay", pipe, replayer, Auditor(cfg), symbols=[SYM, "OTHERBTC"])
    yield SimpleNamespace(cfg=cfg, store=store, bus=bus, pipe=pipe, replayer=replayer, state=state)
    store.close()


@pytest.fixture
def client(env):
    with TestClient(create_app(env.state)) as c:
        yield c


def post(c, url, body):
    return c.post(url, content=json.dumps(body), headers={"content-type": "application/json"})


def subs(bus: Bus) -> int:
    return sum(len(q) for q in bus._subs.values())


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ---- structure ----


def test_every_route_and_dependency_runs_on_the_loop(env):
    app = create_app(env.state)
    seen = 0
    for r in app.routes:
        if isinstance(r, APIRoute):
            seen += 1
            assert inspect.iscoroutinefunction(r.endpoint), r.path
            for d in r.dependant.dependencies:
                assert inspect.iscoroutinefunction(d.call), (r.path, d.call)
        elif hasattr(r, "endpoint"):
            assert inspect.iscoroutinefunction(r.endpoint), r.path
    assert seen >= 9


def test_root_is_404_without_web_dir_and_served_with_one(env, tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "WEB_DIR", tmp_path / "nope")
    with TestClient(create_app(env.state)) as c:
        assert c.get("/").status_code == 404
        assert c.get("/api/health").status_code == 200
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    monkeypatch.setattr(app_mod, "WEB_DIR", web)
    with TestClient(create_app(env.state)) as c:
        assert "hi" in c.get("/").text
        assert c.get("/api/health").status_code == 200


def test_bad_mode_is_refused(env):
    with pytest.raises(ValueError):
        AppState(env.cfg, env.store, "paper", None, None, Auditor(env.cfg))


@pytest.mark.parametrize("mode", ["live", "sim"])
def test_mode_must_be_the_pipelines(env, mode):
    with pytest.raises(ValueError, match="differs"):
        AppState(env.cfg, env.store, mode, env.pipe, env.replayer, Auditor(env.cfg))


def test_symbols_default_to_the_replayers(env):
    st = AppState(env.cfg, env.store, "replay", env.pipe, env.replayer, Auditor(env.cfg))
    with TestClient(create_app(st)) as c:
        assert c.get("/api/symbols").json() == [SYM]
        assert c.get(f"/api/series?symbol={SYM}").status_code == 200
        names = {x["name"]: x["status"] for x in c.get("/api/health").json()["checks"]}
        assert names["warmup"] == "ok"
    no_replayer = AppState(env.cfg, env.store, "replay", env.pipe, None, Auditor(env.cfg))
    assert no_replayer.symbols == list(env.cfg.live.symbols)


# ---- alerts / feedback ----


def test_alerts_list_filter_limit_and_explanation(client, env):
    env.store.upsert_alert(make_alert("a1", D_US + 100 * S))
    env.store.upsert_alert(make_alert("a2", D_US + 200 * S, symbol="OTHERBTC"))
    env.store.upsert_alert(make_alert("a3", D_US + 300 * S))
    r = client.get("/api/alerts")
    assert r.status_code == 200
    assert [a["alert_id"] for a in r.json()] == ["a3", "a2", "a1"]
    assert r.json()[0]["explanation_th"].startswith("พบความผิดปกติทางสถิติ (MM-T001")
    assert [a["alert_id"] for a in client.get("/api/alerts?symbol=OTHERBTC").json()] == ["a2"]
    assert len(client.get("/api/alerts?limit=1").json()) == 1
    for bad in ("0", "1001", "x", "-3"):
        assert client.get(f"/api/alerts?limit={bad}").status_code == 422


def test_alert_detail_and_feedback(client, env):
    env.store.upsert_alert(make_alert("a1"))
    assert client.get("/api/alerts/nope").status_code == 404
    body = client.get("/api/alerts/a1").json()
    assert body["alert_id"] == "a1" and body["feedback"] == []
    assert body["explanation"]["text_th"] == body["explanation_th"]
    assert all(set(f) == {"label", "value"} for f in body["explanation"]["facts"])
    assert (
        post(client, "/api/alerts/a1/feedback", {"verdict": "tp", "analyst": "สมชาย"}).status_code
        == 201
    )
    fb = client.get("/api/alerts/a1").json()["feedback"]
    assert [(x["verdict"], x["analyst"]) for x in fb] == [("tp", "สมชาย")]
    r = post(client, "/api/alerts/nope/feedback", {"verdict": "fp", "analyst": "a"})
    assert r.status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"verdict": "maybe", "analyst": "a"},
        {"verdict": "tp", "analyst": ""},
        {"verdict": "tp", "analyst": "x" * 65},
        {"verdict": "tp", "analyst": "   "},
        {"verdict": "tp", "analyst": "a\u0000b"},
        {"verdict": "tp", "analyst": "a\nb"},
        {"verdict": "tp"},
        {"verdict": "tp", "analyst": "a", "extra": 1},
    ],
)
def test_feedback_validation(client, env, body):
    env.store.upsert_alert(make_alert("a1"))
    r = post(client, "/api/alerts/a1/feedback", body)
    assert r.status_code == 422
    assert "Traceback" not in r.text and ".py" not in r.text
    assert env.store.list_feedback("a1") == []


# ---- series ----


def test_series_shape_filter_and_nulls(client, env):
    pts = [point(i) for i in range(6)]
    del pts[2]["lane_ai"]
    pts[3]["lane_stat"] = None
    env.pipe.points[SYM] = pts
    b = client.get(f"/api/series?symbol={SYM}").json()
    assert set(b) == {"t_end_us", "close", "F", "lane_rule", "lane_stat", "lane_ai", "tau"}
    assert b["tau"] == 3.5
    keys = ("close", "F", "lane_rule", "lane_stat", "lane_ai")
    assert len(b["t_end_us"]) == 6 and all(len(b[k]) == 6 for k in keys)
    assert b["t_end_us"] == sorted(b["t_end_us"])
    assert b["lane_ai"][2] is None and b["lane_stat"][3] is None
    lo, hi = pts[1]["t_end_us"], pts[4]["t_end_us"]
    b = client.get(f"/api/series?symbol={SYM}&start_us={lo}&end_us={hi}").json()
    assert b["t_end_us"] == [p["t_end_us"] for p in pts[1:5]]
    assert all(len(b[k]) == 4 for k in keys)


def test_series_errors_and_empty(client):
    b = client.get(f"/api/series?symbol={SYM}")
    assert b.status_code == 200 and b.json()["t_end_us"] == [] and b.json()["F"] == []
    assert client.get("/api/series?symbol=NOPE").status_code == 404
    assert client.get("/api/series").status_code == 422
    assert client.get(f"/api/series?symbol={SYM}&start_us=5&end_us=4").status_code == 422
    assert client.get(f"/api/series?symbol={SYM}&start_us=x").status_code == 422


def test_series_without_pipeline_is_empty(env):
    st = AppState(env.cfg, env.store, "replay", None, None, Auditor(env.cfg), symbols=[SYM])
    with TestClient(create_app(st)) as c:
        b = c.get(f"/api/series?symbol={SYM}").json()
        assert b["t_end_us"] == [] and b["tau"] is None


# ---- strict JSON ----


def test_nan_in_series_and_alert_is_strict_json(client, env):
    env.pipe.points[SYM] = [point(0, F=float("nan"), close=float("inf"))]
    env.store.upsert_alert(
        make_alert("a1", evidence=[("z_quote_vol_60", float("nan"), 7.1), ("ret_60", 0.04, 0.5)])
    )
    for url in (f"/api/series?symbol={SYM}", "/api/alerts", "/api/alerts/a1"):
        r = client.get(url)
        assert r.status_code == 200, url
        strict_loads(r.text)
    b = strict_loads(client.get(f"/api/series?symbol={SYM}").text)
    assert b["F"] == [None] and b["close"] == [None]
    assert client.get("/api/alerts/a1").json()["evidence"] == [["ret_60", 0.04, 0.5]]


def test_an_old_invalid_row_costs_only_itself(client, env, caplog):
    env.store.upsert_alert(make_alert("good", D_US + 100 * S))
    bad = make_alert("bad", D_US + 200 * S).model_dump(mode="json")
    bad["evidence"] = [["ret_60", None, 0.1]]  # how a NaN was stored before the store dropped it
    env.store._db().execute(
        "INSERT INTO alerts(alert_id, ts_event_us, symbol, json) VALUES (?, ?, ?, ?)",
        ["bad", D_US + 200 * S, SYM, json.dumps(bad)],
    )
    r = client.get("/api/alerts")
    assert r.status_code == 200 and [a["alert_id"] for a in r.json()] == ["good"]
    assert "unreadable alert row" in caplog.text
    assert client.get("/api/alerts/bad").status_code == 404
    assert (
        post(client, "/api/alerts/bad/feedback", {"verdict": "tp", "analyst": "a"}).status_code
        == 404
    )


def test_nan_in_ws_message_is_strict_json(client, env):
    with client.websocket_connect("/ws/stream") as ws:
        strict_loads(ws.receive_text())  # health
        alert = make_alert("a9", evidence=[("ret_60", float("nan"), 0.1)])
        client.portal.call(env.bus.publish, "alerts", alert)
        msg = strict_loads(ws.receive_text())
        assert msg["type"] == "alert" and msg["data"]["alert_id"] == "a9"


# ---- misc reads ----


def test_symbols_taxonomy_experiments(client, env, tmp_path):
    assert client.get("/api/symbols").json() == [SYM, "OTHERBTC"]
    assert "MM-T001" in client.get("/api/taxonomy").json()
    assert client.get("/api/experiments").json() == {"runs": 0}
    report = tmp_path / "report.json"
    report.write_text('{"runs": 3, "x": NaN}', encoding="utf-8")
    env.cfg.paths.report_json = report
    r = client.get("/api/experiments")
    assert strict_loads(r.text) == {"runs": 3, "x": None}
    assert str(tmp_path) not in r.text
    report.write_text("{broken", encoding="utf-8")
    assert client.get("/api/experiments").json() == {"runs": 0}


# ---- health ----


def test_health_reports_what_the_app_has(client, env):
    h = client.get("/api/health").json()
    names = {c["name"]: c["status"] for c in h["checks"]}
    assert names["latency"] == "ok" and names["warmup"] == "ok"
    assert h["late_trades"] == 2 and h["mode"] == "replay"
    assert set(h["dropped"]) == {"bars", "scores", "alerts", "health"}
    assert h["overall"] == "ok"
    env.pipe.samples = 3
    env.pipe.latency_ms.clear()
    h = client.get("/api/health").json()
    names = {c["name"]: c["status"] for c in h["checks"]}
    assert names["warmup"] == "warn" and names["latency"] == "warn"  # no data is not ok
    assert h["overall"] == "warn"


def test_health_without_pipeline_is_not_ok(env):
    st = AppState(env.cfg, env.store, "replay", None, None, Auditor(env.cfg), symbols=[SYM])
    with TestClient(create_app(st)) as c:
        h = c.get("/api/health").json()
        assert h["overall"] != "ok" and h["dropped"] == {} and h["late_trades"] is None


def test_warmup_is_decided_by_the_slowest_symbol(client, env):
    for slow in (SYM, "OTHERBTC"):
        env.pipe.samples_by = {slow: 3}
        h = client.get("/api/health").json()
        assert {c["name"]: c["status"] for c in h["checks"]}["warmup"] == "warn", slow


def test_ws_health_matches_rest_health(client):
    with client.websocket_connect("/ws/stream") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "health"
    rest = client.get("/api/health").json()
    assert {**msg["data"], "ts_us": 0} == {**rest, "ts_us": 0}


def test_ws_health_comes_about_once_a_second(client):
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["type"] == "health"
        t0 = time.monotonic()
        for _ in range(2):
            assert ws.receive_json()["type"] == "health"
        elapsed = time.monotonic() - t0
    assert 1.5 < elapsed < 4.0


# ---- replay control ----


def test_replay_control(client, env):
    r = env.replayer
    assert post(client, "/api/replay/control", {"action": "pause"}).status_code == 200
    assert r._paused
    assert post(client, "/api/replay/control", {"action": "play"}).status_code == 200
    assert not r._paused
    assert post(client, "/api/replay/control", {"action": "speed", "value": 5}).status_code == 200
    assert r._speed == 5.0
    t = D_US + 7200 * S
    assert post(client, "/api/replay/control", {"action": "seek", "value": t}).status_code == 200
    assert r.cursor_us == t


@pytest.mark.parametrize(
    "body",
    [
        {"action": "speed"},
        {"action": "speed", "value": -1},
        {"action": "speed", "value": "fast"},
        {"action": "speed", "value": 10**400},
        {"action": "seek"},
        {"action": "seek", "value": 1.5},
        {"action": "seek", "value": "x"},
        {"action": "seek", "value": True},
        {"action": "jump"},
        {"action": "pause", "extra": 1},
    ],
)
def test_replay_control_bad_input_is_422(client, body):
    assert post(client, "/api/replay/control", body).status_code == 422


def test_replay_control_non_finite_is_422(client):
    raw = '{"action": "speed", "value": NaN}'
    h = {"content-type": "application/json"}
    r = client.post("/api/replay/control", content=raw, headers=h)
    assert r.status_code == 422
    strict_loads(r.text)


def test_replay_control_conflicts(env):
    pipe = FakePipe(env.bus, "live")
    live = AppState(env.cfg, env.store, "live", pipe, env.replayer, Auditor(env.cfg))
    none = AppState(env.cfg, env.store, "replay", env.pipe, None, Auditor(env.cfg))
    for st in (live, none):
        with TestClient(create_app(st)) as c:
            assert post(c, "/api/replay/control", {"action": "pause"}).status_code == 409
    assert not env.replayer._paused


@pytest.mark.parametrize(
    "body",
    [
        {"action": "pause"},
        {"action": "play"},
        {"action": "speed", "value": 5},
        {"action": "seek", "value": D_US + 7200 * S},
    ],
)
def test_every_control_on_an_ended_replay_is_409(client, env, body):
    finish(env.replayer)
    cursor, speed = env.replayer.cursor_us, env.replayer._speed
    assert post(client, "/api/replay/control", body).status_code == 409
    assert env.replayer.cursor_us == cursor and env.replayer._speed == speed
    assert not env.replayer._paused


def test_inject_on_an_ended_replay_is_409(client, env):
    finish(env.replayer)
    h0 = env.replayer._df.height
    assert post(client, "/api/redteam/inject", inject_body()).status_code == 409
    assert env.replayer._df.height == h0 and env.state.injected == []


# ---- red team ----


def inject_body(**kw) -> dict:
    return {"family": "pump_burst", "region": "train", "offset_s": 60, "symbol": SYM, **kw}


def test_inject_accepted(client, env):
    h0, c0 = env.replayer._df.height, env.replayer.cursor_us
    r = post(client, "/api/redteam/inject", inject_body())
    assert r.status_code == 202, r.text
    b = r.json()
    assert b["scenario"]["family"] == "pump_burst"
    assert b["scenario"]["t_start_us"] == c0 + 60 * S
    assert b["label"]["symbol"] == SYM
    assert env.replayer._df.height > h0
    assert len(env.state.injected) == 1 and env.state.injected[0]["label"] == b["label"]
    assert env.store.read_labels().height == 0  # never written to the store


def test_inject_is_repeatable_per_app_counter(env):
    out = []
    for _ in range(2):
        st = AppState(
            env.cfg, env.store, "replay", env.pipe, make_replayer(), Auditor(env.cfg), symbols=[SYM]
        )
        with TestClient(create_app(st)) as c:
            out.append(post(c, "/api/redteam/inject", inject_body()).json()["scenario"])
    assert out[0] == out[1]


def test_each_inject_of_an_app_draws_a_new_scenario(client):
    a = post(client, "/api/redteam/inject", inject_body()).json()["scenario"]
    b = post(client, "/api/redteam/inject", inject_body()).json()["scenario"]
    assert a != b


def test_inject_symbol_may_be_omitted_with_a_one_symbol_replayer(client, env):
    assert len(env.state.symbols) == 2  # the replayer decides, not the known set
    body = inject_body()
    del body["symbol"]
    r = post(client, "/api/redteam/inject", body)
    assert r.status_code == 202, r.text
    assert r.json()["label"]["symbol"] == SYM


def test_inject_symbol_is_required_with_a_many_symbol_replayer(env):
    two = pl.concat([make_trades(D, 1, symbol=s, seed=3, rate=0.05) for s in (SYM, "OTHERBTC")])
    rp = Replayer(two, speed=1.0, start_us=D_US + 3600 * S)
    st = AppState(env.cfg, env.store, "replay", env.pipe, rp, Auditor(env.cfg))
    assert st.symbols == ["OTHERBTC", SYM]
    body = inject_body()
    del body["symbol"]
    with TestClient(create_app(st)) as c:
        assert post(c, "/api/redteam/inject", body).status_code == 422


@pytest.mark.parametrize("mode", ["replay", "sim"])
def test_inject_passes_the_real_mode_to_the_replayer(env, monkeypatch, mode):
    real, seen = env.replayer.inject, []

    def spy(*a, **k):
        seen.append(k["mode"])
        return real(*a, **k)

    monkeypatch.setattr(env.replayer, "inject", spy)
    st = AppState(env.cfg, env.store, mode, FakePipe(env.bus, mode), env.replayer, Auditor(env.cfg))
    with TestClient(create_app(st)) as c:
        assert post(c, "/api/redteam/inject", inject_body()).status_code == 202
    assert seen == [mode]


def test_replayer_itself_refuses_live_if_the_pre_check_is_gone(env, monkeypatch):
    # the refusal rests on inject(..., mode=state.mode): a hard-coded mode would get through here
    monkeypatch.setattr(app_mod, "assert_injection_allowed", lambda mode: None)
    h0, c0 = env.replayer._df.height, env.replayer.cursor_us
    pipe = FakePipe(env.bus, "live")
    st = AppState(env.cfg, env.store, "live", pipe, env.replayer, Auditor(env.cfg), symbols=[SYM])
    with TestClient(create_app(st)) as c:
        assert post(c, "/api/redteam/inject", inject_body()).status_code == 403
    assert env.replayer._df.height == h0 and env.replayer.cursor_us == c0
    assert st.injected == []


def test_inject_refused_in_live_leaves_everything_unchanged(env):
    h0, c0 = env.replayer._df.height, env.replayer.cursor_us
    pipe = FakePipe(env.bus, "live")
    for replayer in (env.replayer, None):
        st = AppState(env.cfg, env.store, "live", pipe, replayer, Auditor(env.cfg))
        with TestClient(create_app(st)) as c:
            r = post(c, "/api/redteam/inject", inject_body())
            assert r.status_code == 403
            d = r.json()["detail"]
            assert "LIVE" in d and "ข้อมูลตลาดจริง" in d
            assert not any(w in d for w in BANNED_TH)
        assert st.injected == []
    assert env.replayer._df.height == h0 and env.replayer.cursor_us == c0


def test_inject_ethics_error_from_replayer_maps_to_403(env, monkeypatch):
    def boom(*a, **k):
        raise EthicsError("x")

    monkeypatch.setattr(env.replayer, "inject", boom)
    with TestClient(create_app(env.state)) as c:
        assert post(c, "/api/redteam/inject", inject_body()).status_code == 403


@pytest.mark.parametrize(
    "body",
    [
        inject_body(family="rug_pull"),
        inject_body(region="prod"),
        inject_body(offset_s=0),
        inject_body(offset_s=-5),
        inject_body(offset_s=3601),
        inject_body(offset_s="soon"),
        inject_body(offset_s="60"),
        inject_body(offset_s=True),
        inject_body(offset_s=10**400),
        inject_body(extra=1),
        inject_body(symbol="NOPE"),
    ],
)
def test_inject_bad_input_is_422(client, env, body):
    h0 = env.replayer._df.height
    r = post(client, "/api/redteam/inject", body)
    assert r.status_code == 422, r.text
    assert env.replayer._df.height == h0 and env.state.injected == []


def test_inject_without_replayer_is_409(env):
    pipe = FakePipe(env.bus, "sim")
    st = AppState(env.cfg, env.store, "sim", pipe, None, Auditor(env.cfg), symbols=[SYM])
    with TestClient(create_app(st)) as c:
        assert post(c, "/api/redteam/inject", inject_body()).status_code == 409


# ---- browser safety ----


def test_cross_origin_post_is_403_and_same_or_no_origin_is_allowed(client, env):
    env.store.upsert_alert(make_alert("a1"))
    body = json.dumps({"verdict": "tp", "analyst": "a"})
    h = {"content-type": "application/json"}
    url = "/api/alerts/a1/feedback"
    for origin in ("http://evil.example", "null", "http://testserver:9", "http://[bad"):
        r = client.post(url, content=body, headers={**h, "origin": origin})
        assert r.status_code == 403, origin
    assert env.store.list_feedback("a1") == []
    r = client.post(url, content=body, headers={**h, "origin": "http://testserver"})
    assert r.status_code == 201
    assert client.post(url, content=body, headers=h).status_code == 201
    got = client.get("/api/health", headers={"origin": "http://evil.example"})
    assert "access-control-allow-origin" not in got.headers


def test_cross_origin_control_and_inject_blocked(client, env):
    h = {"content-type": "application/json", "origin": "http://evil.example"}
    c0 = env.replayer.cursor_us
    for url, body in (
        ("/api/replay/control", {"action": "pause"}),
        ("/api/redteam/inject", inject_body()),
    ):
        assert client.post(url, content=json.dumps(body), headers=h).status_code == 403
    assert not env.replayer._paused and env.replayer.cursor_us == c0
    assert env.state.injected == []


def test_cross_origin_websocket_is_refused_before_accept(client, env):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/stream", headers={"origin": "http://evil.example"}):
            pass
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/stream", headers={"origin": "http://[bad"}):
            pass
    assert subs(env.bus) == 0
    with client.websocket_connect("/ws/stream", headers={"origin": "http://testserver"}) as ws:
        assert ws.receive_json()["type"] == "health"


# ---- websocket ----


def test_ws_alert_arrives_and_matches_rest_item(client, env):
    alert = make_alert("a5")
    env.store.upsert_alert(alert)
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["type"] == "health"
        client.portal.call(env.bus.publish, "alerts", alert)
        msg = ws.receive_json()
    assert msg["type"] == "alert" and msg["data"]["alert_id"] == "a5"
    assert msg["data"]["explanation_th"]
    assert msg["data"] == client.get("/api/alerts").json()[0]


def test_ws_alerts_reset_passes_through(client, env):
    reset = {"event": "alerts_reset", "t_us": 5, "deleted": ["a1"]}
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        client.portal.call(env.bus.publish, "alerts", reset)
        assert ws.receive_json() == {"type": "alert", "data": reset}


def test_ws_overflow_sends_resync(client, env):
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()

        def burst():  # the loop cannot run between these, so the 50-slot queue drops the oldest
            for i in range(80):
                env.bus.publish("alerts", make_alert(f"b{i}", D_US + i * S))

        client.portal.call(burst)
        got = []
        for _ in range(60):
            got.append(ws.receive_json())
            if got[-1]["data"].get("alert_id") == "b79":
                break
    assert {"type": "alert", "data": {"event": "resync"}} in got
    assert got[-1]["data"]["alert_id"] == "b79"


def test_ws_throttles_bars_and_scores_but_ends_on_the_latest(client, env):
    n_pub = 60

    async def spaced() -> float:  # with room for the loop between publishes, or collect coalesces
        t0 = time.monotonic()
        for i in range(n_pub):
            env.bus.publish("scores", {"symbol": SYM, **point(i)})
            env.bus.publish("bars", {"symbol": SYM, "t_us": i, "close": float(i)})
            await asyncio.sleep(0.02)
        return time.monotonic() - t0

    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        elapsed = client.portal.call(spaced)
        n = {"bar": 0, "score": 0}
        last: dict[str, dict] = {}
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            m = ws.receive_json()
            if m["type"] == "health":
                continue
            n[m["type"]] += 1
            last[m["type"]] = m["data"]
            if (
                last.get("score", {}).get("close") == 100.0 + n_pub - 1
                and last.get("bar", {}).get("t_us") == n_pub - 1
            ):
                break
    assert last["score"]["close"] == 100.0 + n_pub - 1 and last["bar"]["t_us"] == n_pub - 1
    for kind, count in n.items():  # 4 Hz over the run, plus the first and the last value
        assert 3 <= count <= elapsed * 4 + 2, (kind, count, elapsed)


def test_ws_throttle_is_per_symbol(client, env):
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()

        def two():
            env.bus.publish("scores", {"symbol": SYM, **point(1)})
            env.bus.publish("scores", {"symbol": "OTHERBTC", **point(2)})

        client.portal.call(two)
        got = {ws.receive_json()["data"]["symbol"] for _ in range(2)}
    assert got == {SYM, "OTHERBTC"}


@pytest.mark.parametrize("abrupt", [False, True])
def test_ws_disconnect_leaves_no_subscriber(client, env, abrupt):
    base = subs(env.bus)
    try:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            assert subs(env.bus) > base
            if abrupt:
                raise RuntimeError("client died")
    except RuntimeError:
        pass
    assert wait_for(lambda: subs(env.bus) == base)


def test_ws_reset_drops_the_held_values(client, env):
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()

        def next_data(kind):
            while (m := ws.receive_json())["type"] != kind:
                pass
            return m["data"]

        client.portal.call(env.bus.publish, "scores", {"symbol": SYM, **point(1)})
        assert next_data("score")["close"] == 101.0

        def held_then_reset():  # the first score used the slot, so this one waits for the next
            env.bus.publish("scores", {"symbol": SYM, **point(2)})
            env.bus.publish("alerts", {"event": "alerts_reset", "t_us": 5, "deleted": []})

        client.portal.call(held_then_reset)
        assert next_data("alert")["event"] == "alerts_reset"
        time.sleep(0.4)  # past the slot the held value would have used
        client.portal.call(env.bus.publish, "scores", {"symbol": SYM, **point(3)})
        assert next_data("score")["close"] == 103.0


async def _run_ws(app, on_send) -> tuple[int, int, int]:
    """Run the handler on this loop until it returns; `on_send(inbox)` gets every sent message.

    Returns the tasks before, at the first send, and after. No client close is sent by itself.
    """
    inbox: asyncio.Queue = asyncio.Queue()
    inbox.put_nowait({"type": "websocket.connect"})
    during: list[int] = []

    async def send(msg):
        if msg["type"] != "websocket.accept":
            during.append(len(asyncio.all_tasks()))
            await on_send(inbox)

    scope = {
        "type": "websocket",
        "path": "/ws/stream",
        "raw_path": b"/ws/stream",
        "query_string": b"",
        "headers": [],
        "subprotocols": [],
        "scheme": "ws",
        "server": ("testserver", 80),
        "client": ("testclient", 1),
        "app": app,
    }
    before = len(asyncio.all_tasks())
    await asyncio.wait_for(app(scope, inbox.get, send), 3.0)
    for _ in range(5):  # cancelled tasks finish on their own: give them their turn
        await asyncio.sleep(0)
    return before, during[0], len(asyncio.all_tasks())


async def _client_closes(inbox) -> None:
    inbox.put_nowait({"type": "websocket.disconnect", "code": 1000})


async def _send_fails(inbox) -> None:
    raise OSError("connection reset")


async def _send_never_returns(inbox) -> None:
    await asyncio.Event().wait()


@pytest.mark.parametrize("on_send", [_client_closes, _send_fails, _send_never_returns])
def test_ws_any_end_leaves_no_task_and_no_subscriber(env, monkeypatch, on_send):
    monkeypatch.setattr(app_mod, "SEND_TIMEOUT_S", 0.05)
    app = create_app(env.state)
    before, during, after = asyncio.run(_run_ws(app, on_send))
    assert during > before  # the count does see the handler's tasks
    assert after == before
    assert subs(env.bus) == 0


def test_ws_real_bug_is_not_taken_for_a_disconnect(env, monkeypatch):
    def boom(state):
        raise RuntimeError("store is closed")

    monkeypatch.setattr(app_mod, "health", boom)
    with pytest.raises(RuntimeError, match="store is closed"):
        asyncio.run(_run_ws(create_app(env.state), _client_closes))
    assert subs(env.bus) == 0


def test_ws_clients_are_independent(client, env):
    with client.websocket_connect("/ws/stream") as a, client.websocket_connect("/ws/stream") as b:
        a.receive_json()
        b.receive_json()
        client.portal.call(env.bus.publish, "alerts", make_alert("z1"))
        assert a.receive_json()["data"]["alert_id"] == "z1"
        assert b.receive_json()["data"]["alert_id"] == "z1"


# ---- text rules ----


def test_api_texts_avoid_banned_words(client, env):
    env.store.upsert_alert(make_alert("a1"))
    texts = [
        client.get("/api/alerts/a1").text,
        client.get("/api/health").text,
        client.get("/api/taxonomy").text,
        post(client, "/api/alerts/zz/feedback", {"verdict": "tp", "analyst": "a"}).text,
    ]
    for t in texts:
        assert not any(w in t for w in BANNED_TH)


def test_inject_past_the_end_of_the_data_is_422(env):
    trades = make_trades(date(2025, 3, 1), 2, symbol=SYM, seed=3, rate=0.05)
    rp = Replayer(trades, speed=1.0, start_us=day_us(date(2025, 3, 3)) - 600 * S)
    st = AppState(env.cfg, env.store, "replay", env.pipe, rp, Auditor(env.cfg), symbols=[SYM])
    h0 = rp._df.height
    with TestClient(create_app(st)) as c:
        assert post(c, "/api/redteam/inject", inject_body(offset_s=3600)).status_code == 422
    assert rp._df.height == h0 and st.injected == []


def test_inject_that_starts_in_the_data_but_ends_past_it_is_422(env):
    trades = make_trades(date(2025, 3, 1), 2, symbol=SYM, seed=3, rate=0.05)
    end = Replayer(trades).end_us
    rng = np.random.default_rng([env.cfg.eval.split_seed, 0])  # what the app's first inject draws
    sc = sample_scenario("pump_burst", "train", 0, rng, load_scenario_ranges())
    offset_s = 60
    rp = Replayer(trades, speed=1.0, start_us=end - int((offset_s + sc.duration_s / 2) * S))
    assert rp.cursor_us + offset_s * S < rp.end_us  # it starts before the end
    st = AppState(env.cfg, env.store, "replay", env.pipe, rp, Auditor(env.cfg), symbols=[SYM])
    h0 = rp._df.height
    with TestClient(create_app(st)) as c:
        assert post(c, "/api/redteam/inject", inject_body(offset_s=offset_s)).status_code == 422
    assert rp._df.height == h0 and st.injected == []


# ---- real pipeline ----


def test_smoke_real_pipeline_store_and_replayer_through_the_routes(env):
    trades = make_trades(D, 1, symbol=SYM, seed=4, rate=0.5).filter(
        pl.col("ts_us") < D_US + 3600 * S
    )
    rp = Replayer(trades, speed=0, start_us=D_US)
    rp.pause()
    chief = cheap_chief(env.cfg)
    pipe = StreamPipeline(env.cfg, env.store, chief.detectors, chief, Bus(maxsize=1000), "replay")
    st = AppState(env.cfg, env.store, "replay", pipe, rp, Auditor(env.cfg))
    assert st.symbols == [SYM]

    with TestClient(create_app(st)) as c:
        with c.websocket_connect("/ws/stream") as ws:

            def until(cond, seconds=30.0):
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    m = strict_loads(ws.receive_text())
                    if cond(m):
                        return m
                raise AssertionError("no such message")

            run = c.portal.start_task_soon(pipe.run, rp)
            try:
                assert until(lambda m: m["type"] == "health")["data"]["mode"] == "replay"
                # paused: a seek is still delivered, and it resets what the client holds
                seek = {"action": "seek", "value": D_US + 600 * S}
                assert post(c, "/api/replay/control", seek).status_code == 200
                reset = until(lambda m: m["type"] == "alert")["data"]
                assert reset["event"] == "alerts_reset" and reset["t_us"] == D_US + 600 * S

                body = {"family": "pump_burst", "region": "train", "offset_s": 1800}
                r = post(c, "/api/redteam/inject", body)  # no symbol: the replayer has one
                assert r.status_code == 202, r.text
                assert env.store.read_labels().height == 0

                assert post(c, "/api/replay/control", {"action": "play"}).status_code == 200
                alert = until(lambda m: m["type"] == "alert" and "alert_id" in m["data"])["data"]
                assert alert["explanation_th"]
                run.result(timeout=60)
            finally:
                run.cancel()

        assert rp.ended
        items = c.get("/api/alerts").json()
        assert alert["alert_id"] in [a["alert_id"] for a in items]
        aid = alert["alert_id"]
        fb = {"verdict": "unsure", "analyst": "a"}
        assert post(c, f"/api/alerts/{aid}/feedback", fb).status_code == 201
        assert len(c.get(f"/api/alerts/{aid}").json()["feedback"]) == 1

        b = strict_loads(c.get(f"/api/series?symbol={SYM}").text)
        assert b["t_end_us"] and b["t_end_us"] == sorted(b["t_end_us"])
        keys = ("close", "F", "lane_rule", "lane_stat", "lane_ai")
        assert all(len(b[k]) == len(b["t_end_us"]) for k in keys)
        assert b["tau"] == chief.tau
        assert any(x is not None for x in b["lane_stat"])

        h = c.get("/api/health").json()
        assert h["late_trades"] == 0 and h["dropped"]["alerts"] == 0
        assert {x["name"] for x in h["checks"]} >= {"latency", "warmup"}

        assert post(c, "/api/replay/control", {"action": "pause"}).status_code == 409
        assert post(c, "/api/redteam/inject", body).status_code == 409
    assert subs(pipe.bus) == 0
