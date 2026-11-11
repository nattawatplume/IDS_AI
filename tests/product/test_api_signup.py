"""Self sign-up: viewer role, input checks, per-IP / site-wide limits, backend failure, origin."""

import pytest

from idsai.data.accounts import MemoryAccounts
from idsai.product.auth import SignupLimiter
from tests.product.test_api_auth import actions, aenv, auth_client, send  # noqa: F401

PW = "correct-horse-1"
TAKEN_TH = "ชื่อนี้มีคนใช้แล้ว"
LIMITED_TH = "สมัครถี่เกินไป ลองใหม่ภายหลัง"
FAILED_TH = "บันทึกบัญชีไม่สำเร็จ ลองใหม่อีกครั้ง"


def signup(c, username="newbie", password=PW, display_name=None, headers=None):
    body = {"username": username, "password": password}
    if display_name is not None:
        body["display_name"] = display_name
    return send(c, "POST", "/api/auth/signup", body, headers)


@pytest.fixture
def open_env(aenv):  # noqa: F811
    aenv.state.allow_signup = True
    return aenv


def test_signup_limiter_windows():
    lim = SignupLimiter()
    assert (lim.per_ip, lim.total) == (10, 60)
    assert [lim.allow("1.1.1.1", float(t)) for t in range(11)] == [True] * 10 + [False]
    assert lim.allow("1.1.1.1", 3601.0) is True
    lim = SignupLimiter()
    assert all(lim.allow(f"10.0.0.{i}", float(i)) for i in range(60))
    assert lim.allow("10.0.1.1", 61.0) is False


def test_signup_limiter_refund():
    lim = SignupLimiter(per_ip=1, total=1)
    assert lim.allow("1.1.1.1", 5.0)
    assert not lim.allow("1.1.1.1", 6.0)
    lim.refund("1.1.1.1", 5.0)
    assert lim.allow("1.1.1.1", 7.0)
    lim.refund("9.9.9.9", 1.0)  # unknown slot: harmless


def test_signup_name_taken_does_not_use_quota(open_env, auth_client):  # noqa: F811
    open_env.state.signup_limiter = SignupLimiter(per_ip=2, total=5)
    c = auth_client()
    assert signup(c, "user-one").status_code == 200
    for _ in range(4):
        assert signup(c, "user-one").status_code == 409
    assert signup(c, "user-two").status_code == 200
    assert signup(c, "user-three").status_code == 429


def test_config_reports_signup(aenv, auth_client):  # noqa: F811
    c = auth_client()
    assert c.get("/api/auth/config").json() == {"signup": False}
    aenv.state.allow_signup = True
    assert c.get("/api/auth/config").json() == {"signup": True}
    aenv.state.auth_enabled = False
    assert c.get("/api/auth/config").json() == {"signup": False}


def test_signup_off_is_404(aenv, auth_client):  # noqa: F811
    r = signup(auth_client())
    assert r.status_code == 404
    assert aenv.store.get_user_by_name("newbie") is None


def test_signup_success_signs_in_as_viewer(open_env, auth_client):  # noqa: F811
    c = auth_client()
    r = signup(c)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "username": "newbie",
        "display_name": "newbie",
        "role": "viewer",
        "prefs": {},
    }
    assert "idsai_session" in r.headers["set-cookie"]
    me = c.get("/api/me").json()
    assert me["role"] == "viewer" and me["username"] == "newbie" and me["signup"] is True
    row = [a for a in open_env.store.list_audit(50) if a["action"] == "signup"]
    assert len(row) == 1 and row[0]["username"] == "newbie" and row[0]["user_id"]
    assert row[0]["target"] == ""
    assert PW not in str(open_env.store.list_audit(50))


def test_signup_trims_display_name(open_env, auth_client):  # noqa: F811
    r = signup(auth_client(), display_name="  สมชาย  ")
    assert r.status_code == 200 and r.json()["display_name"] == "สมชาย"


def test_signup_same_name_twice_is_409(open_env, auth_client):  # noqa: F811
    c = auth_client()
    assert signup(c).status_code == 200
    r = signup(auth_client())
    assert r.status_code == 409 and r.json()["detail"] == TAKEN_TH


@pytest.mark.parametrize(
    "kw",
    [
        {"username": "Somchai"},
        {"username": "ab"},
        {"username": "local"},
        {"username": "cli"},
        {"password": "123456789"},
        {"username": "samename01", "password": "samename01"},
        {"display_name": "x" * 41},
        {"display_name": "bad\x07name"},
    ],
)
def test_signup_rejects_bad_input(open_env, auth_client, kw):  # noqa: F811
    r = signup(auth_client(), **kw)
    assert r.status_code == 422
    assert kw.get("password", PW) not in r.text or kw.get("password") is None
    assert open_env.store.get_user_by_name(kw.get("username", "newbie")) is None


def test_signup_bad_input_does_not_use_quota(open_env, auth_client):  # noqa: F811
    open_env.state.signup_limiter = SignupLimiter(per_ip=1, total=5)
    c = auth_client()
    assert signup(c, username="ab").status_code == 422
    assert signup(c).status_code == 200


def test_signup_rate_limited_per_ip_and_site(open_env, auth_client):  # noqa: F811
    open_env.state.signup_limiter = SignupLimiter(per_ip=1, total=2)
    c = auth_client()
    assert signup(c, "user-one").status_code == 200
    r = signup(c, "user-two")
    assert r.status_code == 429 and r.json()["detail"] == LIMITED_TH
    # without a secure cookie the header is ignored: still the same client address
    r = signup(c, "user-two", headers={"x-forwarded-for": "9.9.9.9"})
    assert r.status_code == 429
    auth = open_env.cfg.auth.model_copy(update={"cookie_secure": True})
    open_env.state.cfg = open_env.cfg.model_copy(update={"auth": auth})
    r = signup(c, "user-two", headers={"x-forwarded-for": " 9.9.9.9 , 10.0.0.1"})
    assert r.status_code == 200
    r = signup(c, "user-three", headers={"x-forwarded-for": "8.8.8.8"})  # site-wide cap of 2
    assert r.status_code == 429 and r.json()["detail"] == LIMITED_TH
    assert open_env.store.get_user_by_name("user-three") is None


def test_signup_backend_failure_is_503(open_env, auth_client):  # noqa: F811
    backend = MemoryAccounts()
    open_env.store.attach_accounts(backend)
    backend.fail = True
    c = auth_client()
    r = signup(c)
    assert r.status_code == 503 and r.json()["detail"] == FAILED_TH
    assert open_env.store.get_user_by_name("newbie") is None
    assert "signup" not in actions(open_env.store)
    assert c.get("/api/me").status_code == 401


def test_signup_refuses_cross_origin(open_env, auth_client):  # noqa: F811
    r = signup(auth_client(), headers={"origin": "http://evil.example"})
    assert r.status_code == 403
    assert open_env.store.get_user_by_name("newbie") is None


def test_put_prefs_backend_failure_is_503(open_env, auth_client):  # noqa: F811
    backend = MemoryAccounts()
    open_env.store.attach_accounts(backend)
    c = auth_client()
    assert signup(c).status_code == 200
    backend.fail = True
    r = send(c, "PUT", "/api/me/prefs", {"eval_budget": None})
    assert r.status_code == 503 and r.json()["detail"] == FAILED_TH
