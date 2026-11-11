"""PostgresAccounts against a real server. Needs IDSAI_TEST_PG_URL (a scratch database)."""

import json
import os
import uuid

import pytest

from idsai.data.accounts import PostgresAccounts

URL = os.environ.get("IDSAI_TEST_PG_URL")

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not URL, reason="IDSAI_TEST_PG_URL not set"),
]


def _row(name="neo", uid="u-neo"):
    return {
        "user_id": uid,
        "username": name,
        "display_name": "Neo",
        "role": "viewer",
        "pw_hash": "h",
        "created_us": 5,
        "disabled": False,
    }


def test_round_trip_in_private_schema():
    import psycopg

    schema = f"idsai_test_{uuid.uuid4().hex}"
    with psycopg.connect(URL, autocommit=True, connect_timeout=10) as admin:
        admin.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in URL else "?"
    scoped = f"{URL}{sep}options=-csearch_path%3D{schema}"
    try:
        be = PostgresAccounts(scoped)
        be.ensure_schema()
        be.insert_user(_row())
        with pytest.raises(ValueError) as ei:
            be.insert_user(_row(uid="u-other"))
        assert URL not in str(ei.value)
        be.update_user("neo", role="admin", disabled=True)
        be.set_prefs("u-neo", json.dumps({"theme": "dark"}))
        be.set_prefs("u-neo", json.dumps({"theme": "light"}))
        users, prefs = be.load_all()
        assert len(users) == 1
        assert users[0]["role"] == "admin" and users[0]["disabled"] is True
        assert prefs == [{"user_id": "u-neo", "json": json.dumps({"theme": "light"})}]
    finally:
        with psycopg.connect(URL, autocommit=True, connect_timeout=10) as admin:
            admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def test_errors_do_not_leak_url():
    from idsai.data.accounts import AccountStoreError

    bad = PostgresAccounts("postgresql://u:s3cretpw@127.0.0.1:1/x")
    with pytest.raises(AccountStoreError) as ei:
        bad.ensure_schema()
    assert "s3cretpw" not in str(ei.value)
