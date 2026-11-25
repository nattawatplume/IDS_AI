"""History backends: a durable copy of the `live` alerts, their feedback and their 1 h outcomes
(Postgres, or in memory for tests).

The DuckDB Store stays the only thing read paths and detection touch. With a backend attached it
restores the recent rows at start-up (`Store.attach_history`) and copies new ones out in batches
(`Store.history_sync`, a background task), so Postgres can never slow or break detection.

Neon's free tier holds 0.5 GB; an alert row is a few KB, so years of live alerts fit. Old rows are
never deleted here; start-up only loads the last `HISTORY_DAYS`.
"""

from __future__ import annotations

from typing import Any, Protocol

from idsai.data.pg import PgClient

HISTORY_DAYS = 30  # start-up restores this many days of history (Neon keeps everything)

ALERT_COLS = ("alert_id", "ts_event_us", "symbol", "json")
FEEDBACK_COLS = ("alert_id", "analyst", "ts_us", "verdict")
# = idsai.data.store._OUTCOME_COLS (a test checks it)
OUTCOME_COLS = (
    "alert_id",
    "symbol",
    "ts_event_us",
    "price_at",
    "price_1h",
    "t_1h_us",
    "status",
    "updated_us",
)


class HistoryStoreError(RuntimeError):
    """The history backend could not complete the operation."""


class HistoryBackend(Protocol):
    def ensure_schema(self) -> None: ...

    def upsert_alerts(self, rows: list[dict]) -> None:
        """Rows with the keys of `ALERT_COLS`; an existing alert_id is replaced."""
        ...

    def insert_feedback(self, rows: list[dict]) -> None:
        """Rows with the keys of `FEEDBACK_COLS`; an existing (alert_id, analyst, ts_us) is kept."""
        ...

    def upsert_outcomes(self, rows: list[dict]) -> None:
        """Rows with the keys of `OUTCOME_COLS`; an existing alert_id is replaced."""
        ...

    def load(self, since_us: int) -> tuple[list[dict], list[dict], list[dict]]:
        """(alerts, feedback, outcomes) of the alerts with `ts_event_us >= since_us`."""
        ...


class MemoryHistory:
    """In-memory backend: `alerts` and `outcomes` keyed by alert_id, `feedback` keyed by
    (alert_id, analyst, ts_us). When `fail` is True every call raises HistoryStoreError."""

    def __init__(self, fail: bool = False) -> None:
        self.alerts: dict[str, dict] = {}
        self.feedback: dict[tuple[str, str, int], dict] = {}
        self.outcomes: dict[str, dict] = {}
        self.fail = fail
        self.calls = 0  # write calls that reached the backend

    def _check(self) -> None:
        if self.fail:
            raise HistoryStoreError("history store unavailable")

    def ensure_schema(self) -> None:
        self._check()

    def upsert_alerts(self, rows: list[dict]) -> None:
        self._check()
        self.calls += 1
        for r in rows:
            self.alerts[r["alert_id"]] = {c: r[c] for c in ALERT_COLS}

    def insert_feedback(self, rows: list[dict]) -> None:
        self._check()
        self.calls += 1
        for r in rows:
            self.feedback.setdefault(
                (r["alert_id"], r["analyst"], r["ts_us"]), {c: r[c] for c in FEEDBACK_COLS}
            )

    def upsert_outcomes(self, rows: list[dict]) -> None:
        self._check()
        self.calls += 1
        for r in rows:
            self.outcomes[r["alert_id"]] = {c: r[c] for c in OUTCOME_COLS}

    def load(self, since_us: int) -> tuple[list[dict], list[dict], list[dict]]:
        self._check()
        alerts = [dict(a) for a in self.alerts.values() if a["ts_event_us"] >= since_us]
        ids = {a["alert_id"] for a in alerts}
        feedback = [dict(f) for f in self.feedback.values() if f["alert_id"] in ids]
        outcomes = [dict(o) for o in self.outcomes.values() if o["alert_id"] in ids]
        return alerts, feedback, outcomes


# ids_ prefix: never clashes with the accounts' `users` / `user_prefs` in the same database
_DDL = (
    "CREATE TABLE IF NOT EXISTS ids_alerts("
    "alert_id TEXT PRIMARY KEY, ts_event_us BIGINT, symbol TEXT, json TEXT)",
    "CREATE INDEX IF NOT EXISTS ids_alerts_ts ON ids_alerts(ts_event_us)",
    "CREATE TABLE IF NOT EXISTS ids_feedback("
    "alert_id TEXT, analyst TEXT, ts_us BIGINT, verdict TEXT, "
    "PRIMARY KEY(alert_id, analyst, ts_us))",
    "CREATE TABLE IF NOT EXISTS ids_alert_outcomes("
    "alert_id TEXT PRIMARY KEY, symbol TEXT, ts_event_us BIGINT, price_at DOUBLE PRECISION, "
    "price_1h DOUBLE PRECISION, t_1h_us BIGINT, status TEXT, updated_us BIGINT)",
)


def _upsert_sql(table: str, cols: tuple[str, ...], key: str) -> str:
    sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != key)
    return (
        f"INSERT INTO {table}({', '.join(cols)}) VALUES ({', '.join('%s' for _ in cols)}) "
        f"ON CONFLICT ({key}) DO UPDATE SET {sets}"
    )


class PostgresHistory:
    """Postgres backend. psycopg is imported lazily; error messages never include the URL."""

    def __init__(self, url: str) -> None:
        self._pg = PgClient(url, HistoryStoreError, "history")

    def _run(self, sql: str, args: Any = (), fetch: bool = False, many: bool = False):
        return self._pg.run(sql, args, fetch=fetch, many=many)

    def ensure_schema(self) -> None:
        for ddl in _DDL:
            self._run(ddl)

    def upsert_alerts(self, rows: list[dict]) -> None:
        if rows:
            self._run(
                _upsert_sql("ids_alerts", ALERT_COLS, "alert_id"),
                [tuple(r[c] for c in ALERT_COLS) for r in rows],
                many=True,
            )

    def insert_feedback(self, rows: list[dict]) -> None:
        if rows:
            self._run(
                f"INSERT INTO ids_feedback({', '.join(FEEDBACK_COLS)}) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                [tuple(r[c] for c in FEEDBACK_COLS) for r in rows],
                many=True,
            )

    def upsert_outcomes(self, rows: list[dict]) -> None:
        if rows:
            self._run(
                _upsert_sql("ids_alert_outcomes", OUTCOME_COLS, "alert_id"),
                [tuple(r[c] for c in OUTCOME_COLS) for r in rows],
                many=True,
            )

    def load(self, since_us: int) -> tuple[list[dict], list[dict], list[dict]]:
        arows = self._run(
            f"SELECT {', '.join(ALERT_COLS)} FROM ids_alerts WHERE ts_event_us >= %s",
            (since_us,),
            fetch=True,
        )
        frows = self._run(
            f"SELECT {', '.join('f.' + c for c in FEEDBACK_COLS)} FROM ids_feedback f "
            "JOIN ids_alerts a USING (alert_id) WHERE a.ts_event_us >= %s",
            (since_us,),
            fetch=True,
        )
        orows = self._run(
            f"SELECT {', '.join('o.' + c for c in OUTCOME_COLS)} FROM ids_alert_outcomes o "
            "JOIN ids_alerts a USING (alert_id) WHERE a.ts_event_us >= %s",
            (since_us,),
            fetch=True,
        )
        return (
            [dict(zip(ALERT_COLS, r, strict=True)) for r in arows],
            [dict(zip(FEEDBACK_COLS, r, strict=True)) for r in frows],
            [dict(zip(OUTCOME_COLS, r, strict=True)) for r in orows],
        )


def build_backend(url: str | None) -> HistoryBackend | None:
    if not url:
        return None
    return PostgresHistory(url)
