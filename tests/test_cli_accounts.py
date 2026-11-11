"""DATABASE_URL wiring: accounts load from the backend at boot of serve and the user commands."""

import secrets

import pytest

from idsai import cli
from idsai.data.accounts import AccountStoreError, MemoryAccounts
from idsai.data.store import Store
from tests.test_cli import write_cfg

THAI = "เชื่อมต่อฐานข้อมูลบัญชีไม่ได้ (DATABASE_URL)"


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "data", tmp_path / "data" / "app.duckdb") as s:
        yield s


def test_no_database_url_attaches_nothing(store, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    cli.attach_accounts_from_env(store)
    assert store._accounts is None


def test_attach_retries_then_cli_error(store, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:secret@127.0.0.1:1/x")
    monkeypatch.setattr(cli, "ATTACH_WAIT_S", 0)
    calls = []

    class Bad:
        def ensure_schema(self):
            raise AccountStoreError("boom secret")

    def factory(url):
        calls.append(url)
        return Bad()

    with pytest.raises(cli.CliError) as ei:
        cli.attach_accounts_from_env(store, backend_factory=factory)
    assert str(ei.value) == THAI
    assert len(calls) == 3
    assert "secret" not in str(ei.value)
    assert store._accounts is None


def test_attach_succeeds_after_a_failure(store, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setattr(cli, "ATTACH_WAIT_S", 0)
    backend = MemoryAccounts()
    seq = iter([RuntimeError("down"), backend])

    def factory(url):
        item = next(seq)
        if isinstance(item, Exception):
            raise item
        return item

    cli.attach_accounts_from_env(store, backend_factory=factory)
    assert store._accounts is backend


def test_seed_env_writes_through_and_leaves_signups(tmp_path, monkeypatch):
    cfg_path = write_cfg(tmp_path)
    backend = MemoryAccounts()
    backend.users["walkin"] = {
        "user_id": "u-walkin",
        "username": "walkin",
        "display_name": "Walk In",
        "role": "viewer",
        "pw_hash": "h-walkin",
        "created_us": 7,
        "disabled": False,
    }
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setattr(cli, "build_backend", lambda url: backend)
    monkeypatch.setenv("IDSAI_USERS", f"boss:admin:{secrets.token_urlsafe(12)}")
    assert cli.main(["--config", str(cfg_path), "user", "seed-env"]) == 0
    assert backend.users["boss"]["role"] == "admin"
    assert backend.users["walkin"]["role"] == "viewer"
    assert backend.users["walkin"]["pw_hash"] == "h-walkin"


def test_serve_and_user_commands_attach(tmp_path, monkeypatch):
    cfg_path = write_cfg(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "attach_accounts_from_env", lambda s: calls.append(s))
    assert cli.main(["--config", str(cfg_path), "user", "list"]) == 0
    assert len(calls) == 1

    # serve: stop at the live/loop usage error, which comes right after the attach
    calls.clear()
    assert cli.main(["--config", str(cfg_path), "serve", "--mode", "live", "--loop"]) == 1
    assert len(calls) == 1
