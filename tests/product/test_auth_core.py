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


def test_limiter_prunes():
    now = [0.0]
    lim = LoginLimiter(clock=lambda: now[0])
    for i in range(1000):
        lim.fail(f"user{i}")
    for _ in range(5):
        lim.fail("locked1")
    assert lim.locked("locked1")
    assert lim.size() == 1001
    now[0] += max(lim.window_s, lim.lock_s) + 1
    assert not lim.locked("x")
    assert lim.size() == 0


def test_limiter_prunes_at_most_once_a_second_and_keeps_live_entries():
    now = [0.0]
    lim = LoginLimiter(clock=lambda: now[0])
    lim.fail("old")
    now[0] = lim.window_s - 0.2
    lim.locked("x")  # prunes now: "old" is still inside the window
    now[0] = lim.window_s + 0.5
    lim.fail("fresh")
    lim.locked("x")  # within the second after the last prune: nothing dropped yet
    assert lim.size() == 2
    now[0] += 1
    lim.locked("x")
    assert lim.size() == 1 and "fresh" in lim._fails


def test_limiter_first_locked_hit_once_per_lock():
    now = [0.0]
    lim = LoginLimiter(clock=lambda: now[0])
    for _ in range(5):
        lim.fail("bob")
    assert lim.locked("bob")
    assert lim.first_locked_hit("bob")
    assert not lim.first_locked_hit("bob")
    assert not lim.first_locked_hit("bob")
    now[0] += lim.lock_s + 1
    assert not lim.locked("bob")
    for _ in range(5):
        lim.fail("bob")
    assert lim.locked("bob")
    assert lim.first_locked_hit("bob")  # a new lock: reported again


def test_prune_expiring_a_lock_also_clears_its_failures():
    now = [0.0]
    lim = LoginLimiter(clock=lambda: now[0])
    for _ in range(5):
        lim.fail("bob")
    assert lim.locked("bob")
    now[0] += lim.lock_s + 2  # past the lock, still inside the window, > 1 s since last prune
    lim.locked("x")  # the prune expires bob's lock
    for _ in range(4):
        lim.fail("bob")
    assert not lim.locked("bob")
    lim.fail("bob")
    assert lim.locked("bob")
