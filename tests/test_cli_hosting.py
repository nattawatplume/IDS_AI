"""Hosting flags of `serve`: --public-host, --secure-cookies, --loop."""

import json
import secrets
from datetime import date

import pytest
from starlette.testclient import TestClient

from idsai import cli
from idsai.ai.auditor import Auditor
from idsai.bus import Bus
from idsai.config import load_settings
from idsai.data.store import Store
from idsai.product.api import AppState, create_app
from idsai.product.auth import hash_password
from tests.product.test_api import SYM, FakePipe, make_replayer
from tests.synth import make_trades
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
        assert c.get("/api/health", headers={"host": "evil.com"}).status_code == 400


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
