import pytest

from idsai.data.store import Store


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "data", tmp_path / "data" / "app.duckdb") as s:
        yield s


def test_auth_tables_created(store):
    names = {r[0] for r in store._db().execute("SHOW TABLES").fetchall()}
    assert {"users", "sessions", "user_prefs", "audit_log"} <= names


def test_add_user_unique(store):
    uid = store.add_user("alice", "Alice", "analyst", "h1", 1)
    assert len(uid) == 32
    assert store.get_user_by_name("alice")["user_id"] == uid
    assert store.get_user(uid)["username"] == "alice"
    assert store.get_user("nope") is None
    assert store.get_user_by_name("bob") is None
    with pytest.raises(ValueError):
        store.add_user("alice", "Other", "viewer", "h2", 2)


def test_list_users_hides_hash(store):
    store.add_user("alice", "Alice", "analyst", "secret-hash", 1)
    rows = store.list_users()
    assert len(rows) == 1
    assert "pw_hash" not in rows[0]
    assert rows[0]["username"] == "alice"
    assert "secret-hash" not in str(rows)


def test_roles_admin_and_flags(store):
    assert store.has_enabled_admin() is False
    store.add_user("root", "Root", "admin", "h", 1)
    assert store.has_enabled_admin() is True
    store.set_disabled("root", True)
    assert store.has_enabled_admin() is False
    store.set_disabled("root", False)
    store.set_role("root", "viewer")
    assert store.get_user_by_name("root")["role"] == "viewer"
    assert store.has_enabled_admin() is False


def test_session_expiry_boundary_is_expired(store):
    uid = store.add_user("alice", "Alice", "analyst", "h", 1)
    store.create_session(uid, "t1", 100, 200, True)
    s = store.get_session("t1", 199)
    assert s["username"] == "alice" and s["role"] == "analyst"
    assert s["remember"] is True and s["user_id"] == uid
    assert set(s) == {"user_id", "username", "display_name", "role", "remember", "last_seen_us"}
    assert store.get_session("t1", 200) is None
    assert store.get_session("t1", 201) is None
    assert store.get_session("missing", 150) is None


def test_touch_delete_purge(store):
    uid = store.add_user("alice", "Alice", "analyst", "h", 1)
    store.create_session(uid, "t1", 100, 200, False)
    store.create_session(uid, "t2", 100, 500, False)
    store.touch_session("t1", 150)
    assert store.get_session("t1", 150)["last_seen_us"] == 150
    assert store.purge_sessions(300) == 1
    assert store.get_session("t2", 300) is not None
    store.delete_session("t2")
    assert store.get_session("t2", 300) is None


def test_disabled_user_session_is_rejected_and_deleted(store):
    uid = store.add_user("alice", "Alice", "analyst", "h", 1)
    store.create_session(uid, "t1", 100, 1000, False)
    store.set_disabled("alice", True)
    assert store.get_session("t1", 150) is None
    store.set_disabled("alice", False)
    assert store.get_session("t1", 150) is None  # sessions were deleted


def test_set_password_drops_sessions(store):
    uid = store.add_user("alice", "Alice", "analyst", "old", 1)
    other = store.add_user("bob", "Bob", "viewer", "h", 1)
    store.create_session(uid, "t1", 100, 1000, False)
    store.create_session(other, "t2", 100, 1000, False)
    store.set_password("alice", "new")
    assert store.get_session("t1", 150) is None
    assert store.get_session("t2", 150) is not None
    assert store.get_user_by_name("alice")["pw_hash"] == "new"


def test_prefs_round_trip(store):
    uid = store.add_user("alice", "Alice", "analyst", "h", 1)
    assert store.get_prefs(uid) == {}
    store.set_prefs(uid, {"theme": "dark", "eval_budget": 0.5})
    assert store.get_prefs(uid) == {"theme": "dark", "eval_budget": 0.5}
    store.set_prefs(uid, {"theme": "light"})
    assert store.get_prefs(uid) == {"theme": "light"}


def test_audit_newest_first_and_paging(store):
    for i in range(1, 6):
        store.add_audit(i * 10, "u", "alice", "login", target=f"t{i}", detail={"i": i})
    rows = store.list_audit()
    assert [r["ts_us"] for r in rows] == [50, 40, 30, 20, 10]
    assert rows[0]["detail"] == {"i": 5}
    assert rows[0]["target"] == "t5" and rows[0]["action"] == "login"
    page = store.list_audit(limit=2, before_us=40)
    assert [r["ts_us"] for r in page] == [30, 20]
    store.add_audit(60, None, "cli", "user_add")
    assert store.list_audit(limit=1)[0]["detail"] == {}


def test_audit_limit_capped_at_500(store):
    for i in range(520):
        store.add_audit(i, "u", "alice", "login")
    assert len(store.list_audit(limit=10_000)) == 500
    assert len(store.list_audit(limit=3)) == 3


def test_list_users_shows_last_seen_from_sessions(store):
    a = store.add_user("alice", "Alice", "analyst", "h", 5)
    store.add_user("bob", "Bob", "viewer", "h", 6)
    assert [(r["username"], r["created_us"], r["last_seen_us"]) for r in store.list_users()] == [
        ("alice", 5, None),
        ("bob", 6, None),
    ]
    store.create_session(a, "t1", 100, 10**15, False)
    store.create_session(a, "t2", 200, 10**15, True)
    rows = {r["username"]: r for r in store.list_users()}
    assert rows["alice"]["last_seen_us"] == 200 and rows["bob"]["last_seen_us"] is None
    assert "pw_hash" not in rows["alice"]
