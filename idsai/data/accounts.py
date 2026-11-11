"""Account backends: source of truth for users and prefs (Postgres, or in memory for tests).

The DuckDB Store mirrors these rows (write-through) so read paths never touch the backend.
"""

from __future__ import annotations

from typing import Any, Protocol


class AccountStoreError(RuntimeError):
    """The account backend could not complete the operation."""


class AccountBackend(Protocol):
    def ensure_schema(self) -> None: ...

    def load_all(self) -> tuple[list[dict], list[dict]]:
        """Return (users, prefs). User keys: user_id, username, display_name, role, pw_hash,
        created_us, disabled. Pref keys: user_id, json."""
        ...

    def insert_user(self, row: dict) -> None:
        """Raise ValueError if the username is taken, AccountStoreError on other failures."""
        ...

    def update_user(self, username: str, **fields: Any) -> None:
        """fields is a subset of role, pw_hash, disabled."""
        ...

    def set_prefs(self, user_id: str, json_text: str) -> None: ...


_UPDATABLE = ("role", "pw_hash", "disabled")


class MemoryAccounts:
    """In-memory backend. `users` is a dict keyed by username, `prefs` a dict keyed by user_id
    (value: the JSON text). When `fail` is True every call raises AccountStoreError."""

    def __init__(self, fail: bool = False) -> None:
        self.users: dict[str, dict] = {}
        self.prefs: dict[str, str] = {}
        self.fail = fail

    def _check(self) -> None:
        if self.fail:
            raise AccountStoreError("account store unavailable")

    def ensure_schema(self) -> None:
        self._check()

    def load_all(self) -> tuple[list[dict], list[dict]]:
        self._check()
        users = [dict(u) for u in self.users.values()]
        prefs = [{"user_id": k, "json": v} for k, v in self.prefs.items()]
        return users, prefs

    def insert_user(self, row: dict) -> None:
        self._check()
        if row["username"] in self.users:
            raise ValueError(f"username already taken: {row['username']}")
        self.users[row["username"]] = dict(row)

    def update_user(self, username: str, **fields: Any) -> None:
        self._check()
        bad = set(fields) - set(_UPDATABLE)
        if bad:
            raise ValueError(f"cannot update fields: {sorted(bad)}")
        if username in self.users:
            self.users[username].update(fields)

    def set_prefs(self, user_id: str, json_text: str) -> None:
        self._check()
        self.prefs[user_id] = json_text


_DDL = (
    "CREATE TABLE IF NOT EXISTS users("
    "user_id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, display_name TEXT, "
    "role TEXT, pw_hash TEXT, created_us BIGINT, disabled BOOLEAN)",
    "CREATE TABLE IF NOT EXISTS user_prefs(user_id TEXT PRIMARY KEY, json TEXT)",
)
_USER_COLS = ("user_id", "username", "display_name", "role", "pw_hash", "created_us", "disabled")


class PostgresAccounts:
    """Postgres backend. psycopg is imported lazily; error messages never include the URL."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._con: Any = None

    def _connect(self) -> Any:
        import psycopg

        try:
            con = psycopg.connect(
                self._url,
                connect_timeout=5,
                autocommit=True,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=3,
            )
            # a SET, not the `options` parameter, so `options` already in the URL survive
            con.execute("SET statement_timeout = 5000")
            return con
        except psycopg.Error:
            raise AccountStoreError("cannot connect to account database") from None

    def _run(self, sql: str, args: tuple = (), fetch: bool = False) -> list[tuple]:
        import psycopg

        for attempt in (1, 2):
            try:
                if self._con is None or self._con.closed:
                    self._con = self._connect()
                with self._con.cursor() as cur:
                    cur.execute(sql, args)
                    return cur.fetchall() if fetch else []
            except psycopg.errors.UniqueViolation:
                raise ValueError("username already taken") from None
            except psycopg.OperationalError:
                try:
                    if self._con is not None:
                        self._con.close()
                except psycopg.Error:
                    pass
                self._con = None
                if attempt == 2:
                    raise AccountStoreError("account database unavailable") from None
            except psycopg.Error:
                raise AccountStoreError("account database error") from None
        return []  # pragma: no cover

    def ensure_schema(self) -> None:
        for ddl in _DDL:
            self._run(ddl)

    def load_all(self) -> tuple[list[dict], list[dict]]:
        urows = self._run(f"SELECT {', '.join(_USER_COLS)} FROM users", fetch=True)
        prows = self._run("SELECT user_id, json FROM user_prefs", fetch=True)
        users = [dict(zip(_USER_COLS, r, strict=True)) for r in urows]
        prefs = [{"user_id": r[0], "json": r[1]} for r in prows]
        return users, prefs

    def insert_user(self, row: dict) -> None:
        self._run(
            f"INSERT INTO users({', '.join(_USER_COLS)}) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            tuple(row[c] for c in _USER_COLS),
        )

    def update_user(self, username: str, **fields: Any) -> None:
        bad = set(fields) - set(_UPDATABLE)
        if bad or not fields:
            raise ValueError(f"cannot update fields: {sorted(bad)}")
        sets = ", ".join(f"{k} = %s" for k in fields)
        self._run(f"UPDATE users SET {sets} WHERE username = %s", (*fields.values(), username))

    def set_prefs(self, user_id: str, json_text: str) -> None:
        self._run(
            "INSERT INTO user_prefs(user_id, json) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO UPDATE SET json = EXCLUDED.json",
            (user_id, json_text),
        )


def build_backend(url: str | None) -> AccountBackend | None:
    if not url:
        return None
    return PostgresAccounts(url)
