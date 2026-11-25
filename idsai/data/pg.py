"""One Postgres connection with bounded connects and one reconnect retry (accounts, history).

psycopg is imported lazily. Errors are raised as the caller's error class with a fixed message:
never the URL, never the psycopg text, never a chained cause.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class PgClient:
    """`what` names the database in the error messages ("account", "history"). With
    `unique_error` set, a unique violation raises ValueError(unique_error) instead."""

    def __init__(
        self,
        url: str,
        error: type[Exception],
        what: str,
        unique_error: str | None = None,
    ) -> None:
        self._url = url
        self._error = error
        self._what = what
        self._unique_error = unique_error
        self._con: Any = None

    def connect(self) -> Any:
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
            raise self._error(f"cannot connect to {self._what} database") from None

    def run(
        self,
        sql: str,
        args: tuple | Sequence[tuple] = (),
        fetch: bool = False,
        many: bool = False,
    ) -> list[tuple]:
        """Execute `sql` (with `many`, once per tuple of `args`, batched by psycopg); retried
        once after an OperationalError on a fresh connection, so statements must be idempotent."""
        import psycopg

        for attempt in (1, 2):
            try:
                if self._con is None or self._con.closed:
                    self._con = self.connect()
                with self._con.cursor() as cur:
                    if many:
                        cur.executemany(sql, list(args))
                        return []
                    cur.execute(sql, args)
                    return cur.fetchall() if fetch else []
            except psycopg.errors.UniqueViolation:
                if self._unique_error is not None:
                    raise ValueError(self._unique_error) from None
                raise self._error(f"{self._what} database error") from None
            except psycopg.OperationalError:
                try:
                    if self._con is not None:
                        self._con.close()
                except psycopg.Error:
                    pass
                self._con = None
                if attempt == 2:
                    raise self._error(f"{self._what} database unavailable") from None
            except psycopg.Error:
                raise self._error(f"{self._what} database error") from None
        return []  # pragma: no cover
