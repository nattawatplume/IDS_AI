"""Admin user API: list, role change, disable / enable, the last-admin guard, audit, 503."""

import pytest

from idsai.data.accounts import MemoryAccounts
from tests.product.test_api_auth import HASH, actions, aenv, auth_client, send  # noqa: F401

LAST_ADMIN_TH = "ต้องมีผู้ดูแลอย่างน้อย 1 คน"
FAILED_TH = "บันทึกบัญชีไม่สำเร็จ ลองใหม่อีกครั้ง"
KEYS = {"username", "display_name", "role", "disabled", "created_us", "last_seen_us", "from_env"}


def role(c, name, value):
    return send(c, "POST", f"/api/users/{name}/role", {"role": value})


def disabled(c, name, value):
    return send(c, "POST", f"/api/users/{name}/disabled", {"disabled": value})


def test_users_admin_only(auth_client):  # noqa: F811
    for r in ("viewer", "analyst"):
        assert auth_client(r).get("/api/users").status_code == 403
    assert auth_client().get("/api/users").status_code == 401
    resp = auth_client("admin").get("/api/users")
    assert resp.status_code == 200
    rows = resp.json()
    assert {u["username"] for u in rows} == {"viewer", "analyst", "admin"}
    assert all(set(u) == KEYS for u in rows)
    assert "pw_hash" not in resp.text and "user_id" not in resp.text


def test_from_env_flag(aenv, auth_client):  # noqa: F811
    aenv.state.env_usernames = frozenset({"admin"})
    flags = {u["username"]: u["from_env"] for u in auth_client("admin").get("/api/users").json()}
    assert flags == {"admin": True, "viewer": False, "analyst": False}


def test_change_role_and_audit(aenv, auth_client):  # noqa: F811
    c = auth_client("admin")
    r = role(c, "viewer", "analyst")
    assert r.status_code == 200 and r.json()["role"] == "analyst"
    assert set(r.json()) == KEYS
    rows = {u["username"]: u for u in c.get("/api/users").json()}
    assert rows["viewer"]["role"] == "analyst"
    log = [a for a in aenv.store.list_audit(50) if a["action"] == "role_change"]
    assert len(log) == 1 and log[0]["target"] == "viewer" and log[0]["username"] == "admin"
    assert "viewer" in str(log[0]["detail"]) and "analyst" in str(log[0]["detail"])
    assert role(c, "viewer", "root").status_code == 422
    assert role(c, "nobody", "viewer").status_code == 404
    assert disabled(c, "nobody", True).status_code == 404
    assert aenv.store.get_user_by_name("viewer")["role"] == "analyst"


def test_same_value_is_noop(aenv, auth_client):  # noqa: F811
    c = auth_client("admin")
    assert role(c, "viewer", "viewer").status_code == 200
    assert disabled(c, "viewer", False).status_code == 200
    assert not {"role_change", "user_enable", "user_disable"} & set(actions(aenv.store))


def test_last_admin_cannot_be_demoted_or_disabled(aenv, auth_client):  # noqa: F811
    c = auth_client("admin")
    r = role(c, "admin", "viewer")
    assert r.status_code == 409 and r.json()["detail"] == LAST_ADMIN_TH
    # a second admin that is disabled does not count
    aenv.store.add_user("admin2", "Admin2", "admin", "x", 2)
    aenv.store.set_disabled("admin2", True)
    r = role(c, "admin", "analyst")
    assert r.status_code == 409 and r.json()["detail"] == LAST_ADMIN_TH
    assert aenv.store.get_user_by_name("admin")["role"] == "admin"
    r = disabled(c, "admin", True)  # the last enabled admin is guarded before the self check
    assert r.status_code == 409 and r.json()["detail"] == LAST_ADMIN_TH
    # with another enabled admin the demotion goes through
    aenv.store.set_disabled("admin2", False)
    assert role(c, "admin", "analyst").status_code == 200


def test_admin_cannot_disable_self(aenv, auth_client):  # noqa: F811
    aenv.store.add_user("admin2", "Admin2", "admin", "x", 2)
    c = auth_client("admin")
    r = disabled(c, "admin", True)
    assert r.status_code == 409 and r.json()["detail"] == "ปิดบัญชีตัวเองไม่ได้"
    assert not aenv.store.get_user_by_name("admin")["disabled"]


def test_disable_kills_open_session(aenv, auth_client):  # noqa: F811
    admin, viewer = auth_client("admin"), auth_client("viewer")
    assert viewer.get("/api/me").status_code == 200
    r = disabled(admin, "viewer", True)
    assert r.status_code == 200 and r.json()["disabled"] is True
    assert viewer.get("/api/me").status_code == 401
    assert "user_disable" in actions(aenv.store)
    r = disabled(admin, "viewer", False)
    assert r.status_code == 200 and r.json()["disabled"] is False
    assert "user_enable" in actions(aenv.store)


def test_admin_write_backend_failure_is_503(aenv, auth_client):  # noqa: F811
    backend = MemoryAccounts()
    aenv.store.attach_accounts(backend)  # replaces the fixture users with the backend's rows
    aenv.store.add_user("admin", "Admin", "admin", HASH, 1)
    aenv.store.add_user("viewer", "Viewer", "viewer", HASH, 1)
    c = auth_client("admin")
    backend.fail = True
    r = role(c, "viewer", "analyst")
    assert r.status_code == 503 and r.json()["detail"] == FAILED_TH
    r = disabled(c, "viewer", True)
    assert r.status_code == 503 and r.json()["detail"] == FAILED_TH
    backend.fail = False
    assert aenv.store.get_user_by_name("viewer")["role"] == "viewer"
    assert not aenv.store.get_user_by_name("viewer")["disabled"]
    assert not {"role_change", "user_disable"} & set(actions(aenv.store))


def test_user_writes_refuse_cross_origin(auth_client):  # noqa: F811
    c = auth_client("admin")
    h = {"origin": "http://evil.example"}
    assert send(c, "POST", "/api/users/viewer/role", {"role": "admin"}, h).status_code == 403
    assert send(c, "POST", "/api/users/viewer/disabled", {"disabled": True}, h).status_code == 403


@pytest.mark.parametrize("bad", [{"role": 1}, {"role": "viewer", "x": 1}, {}])
def test_role_body_is_strict(auth_client, bad):  # noqa: F811
    assert send(auth_client("admin"), "POST", "/api/users/viewer/role", bad).status_code == 422
