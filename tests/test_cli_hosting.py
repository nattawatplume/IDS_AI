"""Hosting flags of `serve`: --public-host, --secure-cookies, --loop."""

import argparse
import asyncio
import json
import secrets
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from starlette.testclient import TestClient

from idsai import cli
from idsai.ai.auditor import Auditor
from idsai.bus import Bus
from idsai.config import load_settings
from idsai.data.replay import Replayer
from idsai.data.store import Store
from idsai.detection.simulator import Scenario
from idsai.product.api import AppState, create_app
from idsai.product.auth import hash_password
from tests.product.test_api import SYM, FakePipe, make_replayer
from tests.synth import day_us, make_trades
from tests.test_cli import write_cfg

PW = secrets.token_urlsafe(16)
PUBLIC = "x.hf.space"


@pytest.fixture
def cfg_path(tmp_path):
    return write_cfg(tmp_path)


def build_state(cfg, store):
    store.write_trades(make_trades(date(2025, 3, 1), 2, symbol=SYM, seed=3, rate=0.05))
    return AppState(
        cfg,
        store,
        "replay",
        FakePipe(Bus(maxsize=50)),
        make_replayer(),
        Auditor(cfg),
        symbols=[SYM],
        auth_enabled=True,
    )


@pytest.fixture
def env(tmp_path):
    cfg = load_settings(
        overrides={
            "paths": {"data_dir": str(tmp_path / "d"), "app_db": str(tmp_path / "a.duckdb")},
            "auth": {"enabled": True},
        }
    )
    store = Store(cfg.paths.data_dir, cfg.paths.app_db)
    store.add_user("root", "Root", "admin", hash_password(PW), 1)
    state = build_state(cfg, store)
    yield cfg, store, state
    store.close()


def login(c, password, headers):
    body = {"username": "root", "password": password, "remember": False}
    return c.post(
        "/api/auth/login",
        content=json.dumps(body),
        headers={"content-type": "application/json", **headers},
    )


@pytest.mark.parametrize("bad", ["https://x.hf.space", "x.hf.space:443", "x/y", ""])
def test_public_host_parse_rejects(bad, capsys):
    with pytest.raises(SystemExit) as e:
        cli.build_parser().parse_args(["serve", "--public-host", bad])
    assert e.value.code == 1
    assert "ชื่อโฮสต์" in capsys.readouterr().err


def test_public_host_parse_ok():
    a = cli.build_parser().parse_args(["serve", "--public-host", "X.HF.space"])
    assert a.public_host == ["x.hf.space"]
    assert cli.build_parser().parse_args(["serve"]).public_host == []
    assert cli.build_parser().parse_args(["serve"]).secure_cookies is False
    assert cli.build_parser().parse_args(["serve"]).loop is False


def test_allowed_hosts_adds_public():
    hosts = cli.allowed_hosts("0.0.0.0", ["X.hf.space"])
    assert "x.hf.space" in hosts and "0.0.0.0" in hosts and "127.0.0.1" in hosts
    assert cli.allowed_hosts("127.0.0.1") == cli.allowed_hosts("127.0.0.1", [])


def test_public_host_allowlist(env):
    _, _, state = env
    app = create_app(state, allowed_hosts=cli.allowed_hosts("0.0.0.0", [PUBLIC]))
    with TestClient(app) as c:
        assert c.get("/api/health", headers={"host": PUBLIC}).status_code == 200
        assert c.get("/api/symbols", headers={"host": PUBLIC}).status_code == 401  # past the check
        assert c.get("/api/symbols", headers={"host": "evil.com"}).status_code == 400


def test_health_probe_skips_host_check(env):
    _, _, state = env
    app = create_app(state, allowed_hosts=cli.allowed_hosts("0.0.0.0", [PUBLIC]))
    probe = {"host": "10.0.0.5:7860"}  # the Space's own probe, by its internal IP
    with TestClient(app) as c:
        assert c.get("/api/health", headers=probe).status_code == 200
        assert c.head("/api/health", headers=probe).status_code != 400
        assert c.get("/api/me", headers=probe).status_code == 400
        assert c.get("/api/symbols", headers=probe).status_code == 400
        assert c.post("/api/health", headers=probe).status_code == 400
        assert c.get("/api/health/", headers=probe).status_code == 400
        with pytest.raises(Exception):  # noqa: B017 - the handshake is refused before accept
            with c.websocket_connect("/ws/stream", headers=probe):
                pass


def test_public_host_case_and_port(env):
    _, _, state = env
    app = create_app(state, allowed_hosts=cli.allowed_hosts("0.0.0.0", [PUBLIC]))
    with TestClient(app) as c:
        assert c.get("/api/health", headers={"host": "X.HF.Space:443"}).status_code == 200


def test_public_host_same_origin(env):
    _, _, state = env
    app = create_app(state, allowed_hosts=cli.allowed_hosts("0.0.0.0", [PUBLIC]))
    with TestClient(app) as c:
        h = {"host": PUBLIC, "origin": f"https://{PUBLIC}"}
        assert login(c, "wrong-password-1", h).status_code == 401
        assert login(c, PW, h).status_code == 200
        h2 = {"host": PUBLIC, "origin": "https://evil.com"}
        assert login(c, PW, h2).status_code == 403


def test_secure_cookies_flag(cfg_path, monkeypatch):
    captured = {}
    scfg = load_settings(cfg_path)
    with Store(scfg.paths.data_dir, scfg.paths.app_db) as st:
        st.add_user("root", "Root", "admin", hash_password(PW), 1)

    def fake_prepare(args, cfg, store):
        captured["cfg"] = cfg
        return build_state(cfg, store), lambda: None

    def fake_run(app, host, port, background):
        with TestClient(app) as c:  # the store is still open here
            captured["login"] = login(c, PW, {"host": PUBLIC, "origin": f"https://{PUBLIC}"})

    monkeypatch.setattr(cli, "prepare_replay", fake_prepare)
    monkeypatch.setattr(cli, "run_server", fake_run)
    argv = ["--config", str(cfg_path), "serve", "--auth", "--host", "0.0.0.0",
            "--public-host", PUBLIC]
    assert cli.main([*argv, "--secure-cookies"]) == 0
    assert captured["cfg"].auth.cookie_secure is True
    r = captured["login"]
    assert r.status_code == 200
    assert "secure" in r.headers["set-cookie"].lower()

    captured.clear()
    assert cli.main(argv) == 0
    assert captured["cfg"].auth.cookie_secure is False
    assert "secure" not in captured["login"].headers["set-cookie"].lower()


def test_loop_live_is_usage_error(cfg_path, capsys):
    assert cli.main(["--config", str(cfg_path), "serve", "--mode", "live", "--loop"]) == 1
    assert "--loop" in capsys.readouterr().err


# ---- --loop: the replay day starts again by itself ----

S = 1_000_000
DAY = date(2025, 3, 1)
D0 = day_us(DAY)
START = D0 + 60 * S


class RoundPipe:
    """Stands in for StreamPipeline: each `run` plays the source to its end, cancels at `stop`."""

    def __init__(self, stop: int = 3, before=None):
        self.bus = Bus(maxsize=50)
        self.mode = "replay"
        self.stop = stop
        self.before = before  # called with (round number, replayer) before a round plays
        self.end_speed: float | None = None  # what the analyst left the speed at, if set
        self.rounds: list[tuple[Replayer, tuple, int]] = []  # (source, first event, trades)
        self.speeds: list[float] = []  # each round's speed as it started

    async def run(self, rp):
        if self.before is not None:
            self.before(len(self.rounds) + 1, rp)
        speed = rp.speed
        self.speeds.append(speed)
        rp.set_speed(0.0)  # play at max speed so the test is fast
        first, trades = None, 0
        async for ev in rp:
            first = ev if first is None else first
            trades += ev[0] == "trade"
        rp.set_speed(speed if self.end_speed is None else self.end_speed)
        self.rounds.append((rp, first, trades))
        if len(self.rounds) >= self.stop:
            asyncio.current_task().cancel()
            await asyncio.sleep(0)


@pytest.fixture
def replay_env(tmp_path, monkeypatch):
    cfg = load_settings(
        overrides={"paths": {"data_dir": str(tmp_path / "d"), "app_db": str(tmp_path / "a.duckdb")}}
    )
    store = Store(cfg.paths.data_dir, cfg.paths.app_db)
    trades = make_trades(DAY, 1, symbol=SYM, seed=3, rate=1.0)
    trades = trades.filter(pl.col("ts_us") < D0 + 180 * S)  # about 3 min
    store.write_trades(trades)
    pipes: list[RoundPipe] = []

    def fake_pipeline(*a, **k):
        pipes.append(RoundPipe())
        return pipes[-1]

    target = (SYM, DAY, START, Path("."))
    monkeypatch.setattr(cli, "_replay_target", lambda args, cfg, store: target)
    monkeypatch.setattr(cli, "_load", lambda models, cfg: ([], None, {}))
    monkeypatch.setattr("idsai.pipeline.stream.StreamPipeline", fake_pipeline)
    yield cfg, store, trades, pipes
    store.close()


def serve_args(loop: bool) -> argparse.Namespace:
    return argparse.Namespace(speed=0.0, loop=loop, auth=False)


def run_background(background) -> None:
    async def go():
        with pytest.raises(asyncio.CancelledError):
            await asyncio.create_task(background())

    asyncio.run(go())


def test_loop_restarts(replay_env):
    cfg, store, trades, pipes = replay_env
    state, background = cli.prepare_replay(serve_args(True), cfg, store)
    pipe = pipes[0]
    q = pipe.bus.subscribe("alerts")
    seen = []
    pipe.before = lambda n, rp: seen.append(rp is state.replayer)
    run_background(background)
    sources = [rp for rp, _, _ in pipe.rounds]
    assert len(sources) == 3 and len({id(rp) for rp in sources}) == 3
    assert seen == [True, True, True]  # each round plays the replayer the routes see
    assert state.replayer is sources[-1]
    # round 1 ran at max speed (0): the next rounds fall back to the serve default
    assert pipe.speeds == [0.0, cli.DEFAULT_SPEED, cli.DEFAULT_SPEED]
    for _, first, n in pipe.rounds[1:]:
        assert first == ("seek", START)
        assert n == trades.height
    msgs = []
    while not q.empty():
        msgs.append(q.get_nowait())
    restarted = [m for m in msgs if isinstance(m, dict) and m.get("event") == "replay_restarted"]
    assert restarted and restarted[0] == {"event": "replay_restarted", "t_us": START}


def test_loop_carries_a_nonzero_speed(replay_env):
    cfg, store, _, pipes = replay_env
    state, background = cli.prepare_replay(serve_args(True), cfg, store)
    pipe = pipes[0]
    pipe.end_speed = 5.0  # an analyst set 5x during each round
    run_background(background)
    assert pipe.speeds == [0.0, 5.0, 5.0]
    assert state.replayer.speed == 5.0


def test_loop_restarts_after_inject(replay_env):
    cfg, store, trades, pipes = replay_env
    state, background = cli.prepare_replay(serve_args(True), cfg, store)
    pipe = pipes[0]
    pipe.stop = 2

    def inject_first(n, rp):
        if n == 1:
            sc = Scenario(
                family="pump_burst", technique_id="MM-T001", sub_technique="MM-T001.001",
                t_start_us=START + 10 * S, duration_s=30.0, impact=0.1, vol_mult=10.0,
                rush_size=3, dump_tau_s=60.0, seed=3, region="test",
            )
            rp.inject(sc, 1.0, "replay")
            state.injected.append({"scenario": "x"})
        else:
            assert state.injected == []

    pipe.before = inject_first
    run_background(background)
    (_, _, n1), (_, _, n2) = pipe.rounds
    assert n1 > trades.height  # round 1 played the injected trades
    assert n2 == trades.height  # round 2 plays the original frame
    assert state.injected == []


def test_no_loop_unchanged(replay_env):
    cfg, store, _, pipes = replay_env
    state, background = cli.prepare_replay(serve_args(False), cfg, store)
    pipe = pipes[0]
    pipe.stop = 99
    rp = state.replayer
    asyncio.run(background())  # returns after one run, nothing restarts
    assert len(pipe.rounds) == 1 and pipe.rounds[0][0] is rp
    assert state.replayer is rp and rp.ended
