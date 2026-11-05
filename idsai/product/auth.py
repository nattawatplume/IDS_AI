"""Auth core: password hashing, tokens, roles, login limiter (stdlib only, no I/O)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import time
from collections.abc import Callable

ROLES = ("viewer", "analyst", "admin")

_N, _R, _P = 16384, 8, 1
_PREFIX = f"scrypt$n={_N},r={_R},p={_P}$"
_MAXMEM = 64 * 1024 * 1024
_USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,32}$")
_MIN_PASSWORD = 10


def role_at_least(role: str, need: str) -> bool:
    if role not in ROLES or need not in ROLES:
        raise ValueError(f"unknown role: {role!r} / {need!r}")
    return ROLES.index(role) >= ROLES.index(need)


def _scrypt(pw: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        pw.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, maxmem=_MAXMEM, dklen=32
    )


def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)

    def b64(b: bytes) -> str:
        return base64.b64encode(b).decode("ascii")

    return f"{_PREFIX}{b64(salt)}${b64(_scrypt(pw, salt))}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        if not stored.startswith(_PREFIX):
            return False
        salt_b64, hash_b64 = stored[len(_PREFIX):].split("$")
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(_scrypt(pw, salt), expected)


_DUMMY_HASH = hash_password("dummy-password-for-timing")


def verify_or_dummy(pw: str, stored: str | None) -> bool:
    """Spend the same scrypt time for unknown users, then return False."""
    if stored is None:
        verify_password(pw, _DUMMY_HASH)
        return False
    return verify_password(pw, stored)


def check_username(u: str) -> str:
    if not _USERNAME_RE.fullmatch(u):
        raise ValueError(
            "ชื่อผู้ใช้ต้องมี 3-32 ตัว ใช้ได้เฉพาะ a-z 0-9 _ . -"
        )
    return u


def check_password(pw: str) -> None:
    if len(pw) < _MIN_PASSWORD:
        raise ValueError(f"รหัสผ่านต้องมีอย่างน้อย {_MIN_PASSWORD} ตัวอักษร")


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


class LoginLimiter:
    """In-memory: max_fail failures per username within window_s -> lock for lock_s.

    Names whose failures are all past the window and whose lock is over are pruned (at most once
    a second, from `locked`), so many distinct names cannot grow it without bound."""

    def __init__(
        self,
        max_fail: int = 5,
        window_s: float = 600,
        lock_s: float = 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_fail = max_fail
        self.window_s = window_s
        self.lock_s = lock_s
        self._clock = clock
        self._fails: dict[str, list[float]] = {}
        self._locked_until: dict[str, float] = {}
        self._hit: set[str] = set()  # names whose current lock already had a rejected attempt
        self._pruned_at: float | None = None

    def prune(self) -> None:
        now = self._clock()
        self._pruned_at = now
        for name, until in list(self._locked_until.items()):
            if now >= until:
                del self._locked_until[name]
                self._hit.discard(name)
        for name, ts in list(self._fails.items()):
            if name not in self._locked_until and all(now - t >= self.window_s for t in ts):
                del self._fails[name]

    def size(self) -> int:
        return len(self._fails.keys() | self._locked_until.keys())

    def locked(self, username: str) -> bool:
        if self._pruned_at is None or self._clock() - self._pruned_at >= 1.0:
            self.prune()
        until = self._locked_until.get(username)
        if until is None:
            return False
        if self._clock() < until:
            return True
        del self._locked_until[username]
        self._fails.pop(username, None)
        self._hit.discard(username)
        return False

    def first_locked_hit(self, username: str) -> bool:
        """True only for the first rejected attempt within the current lock."""
        if username in self._hit:
            return False
        self._hit.add(username)
        return True

    def fail(self, username: str) -> None:
        now = self._clock()
        recent = [t for t in self._fails.get(username, []) if now - t < self.window_s]
        recent.append(now)
        self._fails[username] = recent
        if len(recent) >= self.max_fail:
            self._locked_until[username] = now + self.lock_s
            self._hit.discard(username)

    def ok(self, username: str) -> None:
        self._fails.pop(username, None)
        self._locked_until.pop(username, None)
        self._hit.discard(username)
