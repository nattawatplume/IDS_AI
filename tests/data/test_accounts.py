import pytest

from idsai.data.accounts import AccountStoreError, MemoryAccounts, build_backend
from idsai.data.store import Store


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "data", tmp_path / "data" / "app.duckdb") as s:
        yield s


def _neo(uid="u-neo"):
    return {
        "user_id": uid,
        "username": "neo",
        "display_name": "Neo",
        "role": "admin",
        "pw_hash": "h-neo",
        "created_us": 5,
        "disabled": False,
    }


def test_no_backend_is_unchanged(store):
    assert store._accounts is None


def test_build_backend_none():
    assert build_backend(None) is None
    assert build_backend("") is None


def test_attach_loads_rows_and_replaces_local(store):
    store.add_user("old", "Old", "viewer", "h", 1)
    backend = MemoryAccounts()
    backend.users["neo"] = _neo()
    backend.prefs["u-neo"] = '{"theme": "dark"}'
    store.attach_accounts(backend)
    assert store.get_user_by_name("old") is None
    assert store.get_user_by_name("neo")["role"] == "admin"
    assert store.get_prefs("u-neo") == {"theme": "dark"}
    assert store.has_enabled_admin() is True


def test_writes_go_to_backend_and_duckdb(store):
    backend = MemoryAccounts()
    store.attach_accounts(backend)
    uid = store.add_user("amy", "Amy", "viewer", "h1", 7)
    assert backend.users["amy"] == {
        "user_id": uid,
        "username": "amy",
        "display_name": "Amy",
        "role": "viewer",
        "pw_hash": "h1",
        "created_us": 7,
        "disabled": False,
    }
    assert store.get_user_by_name("amy")["user_id"] == uid

    store.set_role("amy", "analyst")
    assert backend.users["amy"]["role"] == "analyst"
    assert store.get_user_by_name("amy")["role"] == "analyst"

    store.set_disabled("amy", True)
    assert backend.users["amy"]["disabled"] is True
    assert store.get_user_by_name("amy")["disabled"] is True

    store.create_session(uid, "tok", 1, 10**12, False)
    store.set_password("amy", "h2")
    assert backend.users["amy"]["pw_hash"] == "h2"
    assert store.get_user_by_name("amy")["pw_hash"] == "h2"
    assert store._rows("SELECT * FROM sessions", []) == []

    store.set_prefs(uid, {"a": 1})
    assert backend.prefs[uid] == '{"a": 1}'
    assert store.get_prefs(uid) == {"a": 1}


def test_backend_error_leaves_duckdb_unchanged(store):
    backend = MemoryAccounts()
    store.attach_accounts(backend)
    uid = store.add_user("amy", "Amy", "viewer", "h1", 1)
    backend.fail = True
    with pytest.raises(AccountStoreError):
        store.add_user("bob", "Bob", "viewer", "h", 2)
    assert store.get_user_by_name("bob") is None
    with pytest.raises(AccountStoreError):
        store.set_role("amy", "admin")
    assert store.get_user_by_name("amy")["role"] == "viewer"
    with pytest.raises(AccountStoreError):
        store.set_disabled("amy", True)
    assert store.get_user_by_name("amy")["disabled"] is False
    with pytest.raises(AccountStoreError):
        store.set_password("amy", "zzz")
    assert store.get_user_by_name("amy")["pw_hash"] == "h1"
    with pytest.raises(AccountStoreError):
        store.set_prefs(uid, {"x": 1})
    assert store.get_prefs(uid) == {}


def test_taken_name_raises_value_error_from_backend(store):
    backend = MemoryAccounts()
    store.attach_accounts(backend)
    backend.users["neo"] = _neo()  # only in the backend, DuckDB does not know it
    with pytest.raises(ValueError):
        store.add_user("neo", "Other", "viewer", "h", 1)
    assert store.get_user_by_name("neo") is None


def test_attach_failure_keeps_local_state(store):
    store.add_user("old", "Old", "viewer", "h", 1)
    with pytest.raises(AccountStoreError):
        store.attach_accounts(MemoryAccounts(fail=True))
    assert store._accounts is None
    assert store.get_user_by_name("old") is not None
