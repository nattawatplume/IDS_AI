"""`idsai user seed-env`: accounts created or updated from the IDSAI_USERS variable."""

import secrets

import pytest

from idsai import cli
from idsai.config import load_settings
from idsai.data.store import Store
from idsai.product.auth import hash_password, verify_password
from tests.test_cli import write_cfg


@pytest.fixture
def cfg_path(tmp_path):
    return write_cfg(tmp_path)


def seed(cfg_path, monkeypatch, text):
    if text is None:
        monkeypatch.delenv("IDSAI_USERS", raising=False)
    else:
        monkeypatch.setenv("IDSAI_USERS", text)
    return cli.main(["--config", str(cfg_path), "user", "seed-env"])


def open_store(cfg_path):
    s = load_settings(cfg_path)
    return Store(s.paths.data_dir, s.paths.app_db)


def pw():
    return secrets.token_urlsafe(12)


def test_seed_creates_and_updates(cfg_path, monkeypatch, capsys):
    p1, p2, p3 = pw(), pw(), pw()
    with open_store(cfg_path) as st:
        st.add_user("vwr1", "", "viewer", hash_password(pw()), 1)
        st.set_disabled("vwr1", True)
    assert seed(cfg_path, monkeypatch, f"adm1:admin:{p1}\nvwr2:viewer:{p2}") == 0
    assert "created 2" in capsys.readouterr().out
    with open_store(cfg_path) as st:
        assert {u["username"] for u in st.list_users()} == {"adm1", "vwr1", "vwr2"}
    capsys.readouterr()
    assert seed(cfg_path, monkeypatch, f"adm1:admin:{p1}\nvwr1:analyst:{p3}") == 0
    assert "updated 2" in capsys.readouterr().out
    with open_store(cfg_path) as st:
        vwr1 = st.get_user_by_name("vwr1")
        assert vwr1["role"] == "analyst"
        assert verify_password(p3, vwr1["pw_hash"])
        assert not vwr1["disabled"]


def test_seed_counts_on_fresh_store(cfg_path, monkeypatch, capsys):
    assert seed(cfg_path, monkeypatch, f"adm1:admin:{pw()}\nvwr1:viewer:{pw()}") == 0
    assert "created 2" in capsys.readouterr().out


def test_seed_password_with_colon_and_thai(cfg_path, monkeypatch):
    p = "ab:cd ผ่าน12345"
    assert seed(cfg_path, monkeypatch, f"adm1:admin:{p}") == 0
    with open_store(cfg_path) as st:
        assert verify_password(p, st.get_user_by_name("adm1")["pw_hash"])


def test_seed_crlf(cfg_path, monkeypatch):
    p = pw()
    assert seed(cfg_path, monkeypatch, f"adm1:admin:{p}\r\n") == 0
    with open_store(cfg_path) as st:
        h = st.get_user_by_name("adm1")["pw_hash"]
        assert verify_password(p, h)
        assert not verify_password(p + "\r", h)


def test_seed_skips_blank_and_comment(cfg_path, monkeypatch):
    assert seed(cfg_path, monkeypatch, f"\n# note\n\nadm1:admin:{pw()}\n  \n") == 0
    with open_store(cfg_path) as st:
        assert len(st.list_users()) == 1


def test_seed_bad_line_writes_nothing(cfg_path, monkeypatch, capsys):
    p1, p2 = pw(), pw()
    assert seed(cfg_path, monkeypatch, f"adm1:admin:{p1}\nbob1:boss:{p2}") == 1
    cap = capsys.readouterr()
    assert "บรรทัด 2" in cap.err
    for p in (p1, p2):
        assert p not in cap.err and p not in cap.out
    with open_store(cfg_path) as st:
        assert st.list_users() == []


def test_seed_needs_admin(cfg_path, monkeypatch):
    assert seed(cfg_path, monkeypatch, f"vwr1:viewer:{pw()}") == 1


@pytest.mark.parametrize("val", [None, "", "  \n# only comment\n"])
def test_seed_unset_or_empty(cfg_path, monkeypatch, val):
    assert seed(cfg_path, monkeypatch, val) == 1


@pytest.mark.parametrize("name", ["cli", "local"])
def test_seed_reserved_name(cfg_path, monkeypatch, name):
    assert seed(cfg_path, monkeypatch, f"{name}:admin:{pw()}") == 1


def test_seed_audits(cfg_path, monkeypatch):
    assert seed(cfg_path, monkeypatch, f"adm1:admin:{pw()}\nvwr1:viewer:{pw()}") == 0
    with open_store(cfg_path) as st:
        rows = [r for r in st.list_audit() if r["action"] == "user_seed"]
    assert len(rows) == 2
    assert {r["username"] for r in rows} == {"cli"}
    assert {r["target"] for r in rows} == {"adm1", "vwr1"}
