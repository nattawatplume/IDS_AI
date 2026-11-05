import hashlib

import pytest

from idsai.product.auth import (
    LoginLimiter,
    check_password,
    check_username,
    hash_password,
    new_token,
    role_at_least,
    token_hash,
    verify_or_dummy,
    verify_password,
)


def test_hash_round_trip_and_format():
    h = hash_password("correct horse")
    assert h.startswith("scrypt$n=16384,r=8,p=1$")
    assert h != hash_password("correct horse")
    assert verify_password("correct horse", h)
    assert not verify_password("wrong horse", h)


def test_verify_or_dummy_unknown_user_is_false():
    assert verify_or_dummy("whatever", None) is False
    h = hash_password("abcdefghij")
    assert verify_or_dummy("abcdefghij", h) is True


def test_role_order():
    assert role_at_least("admin", "viewer")
    assert role_at_least("analyst", "analyst")
    assert not role_at_least("viewer", "analyst")
    assert not role_at_least("analyst", "admin")
    with pytest.raises(ValueError):
        role_at_least("root", "viewer")
    with pytest.raises(ValueError):
        role_at_least("viewer", "root")


def test_username_rules():
    assert check_username("alice") == "alice"
    assert check_username("a_b.c-1") == "a_b.c-1"
    for bad in ("Al", " alice", "ALICE", "a" * 33):
        with pytest.raises(ValueError):
            check_username(bad)


def test_password_min_length():
    with pytest.raises(ValueError):
        check_password("123456789")
    assert check_password("1234567890") is None


def test_token_hash_is_sha256_hex():
    t = new_token()
    assert len(t) >= 43
    assert new_token() != t
    assert token_hash("abc") == hashlib.sha256(b"abc").hexdigest()


def test_limiter_locks_after_five_then_releases():
    now = [1000.0]
    lim = LoginLimiter(clock=lambda: now[0])
    for _ in range(4):
        lim.fail("bob")
    assert not lim.locked("bob")
    lim.fail("bob")
    assert lim.locked("bob")
    assert not lim.locked("carol")
    now[0] += 61
    assert not lim.locked("bob")
    for _ in range(4):
        lim.fail("bob")
    lim.ok("bob")
    lim.fail("bob")
    assert not lim.locked("bob")
