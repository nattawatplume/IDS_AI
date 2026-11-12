"""Local store: Parquet trade partitions + DuckDB tables (alerts, feedback, labels, runs)."""

import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import duckdb
import polars as pl

from idsai.config import file_sha
from idsai.contracts.alert import Alert
from idsai.contracts.frames import LABELS, TRADES, conform, empty
from idsai.data.accounts import AccountBackend

log = logging.getLogger(__name__)
DAY_US = 86_400_000_000
_EPOCH = date(1970, 1, 1)
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_INDEX_SCHEMA = pl.Schema(
    {"symbol": pl.String, "date": pl.Date, "sha": pl.String, "n_rows": pl.Int64}
)
_DUCK_TYPES = {pl.Int64: "BIGINT", pl.String: "VARCHAR"}
Verdict = Literal["tp", "fp", "unsure"]
_VERDICTS = ("tp", "fp", "unsure")
_DELETABLE_PROVENANCE = ("replay", "sim")


def _now_us() -> int:
    return time.time_ns() // 1000


def _day_of(ts_us: int) -> date:
    return _EPOCH + timedelta(days=ts_us // DAY_US)


def _check_symbol(symbol: str) -> str:
    if not _SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"invalid symbol {symbol!r}")
    return symbol


class Store:
    """Trades live in `data_dir/trades/symbol=S/date=D/part.parquet`; everything else in DuckDB.

    One DuckDB connection per instance (guarded by a lock so API threads can share it). Use as
    a context manager, or call `close()`, to release the database file.
    """

    def __init__(self, data_dir: Path, db_path: Path):
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path)
        self._trades_dir = self.data_dir / "trades"
        self._index_path = self._trades_dir / "_index.parquet"
        self._lock = threading.RLock()
        self._accounts: AccountBackend | None = None
        self._trades_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con: duckdb.DuckDBPyConnection | None = duckdb.connect(str(self.db_path))
        self._create_tables()

    # ---- lifecycle ----

    def close(self) -> None:
        with self._lock:
            if self._con is not None:
                self._con.close()
                self._con = None

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _db(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise RuntimeError("Store is closed")
        return self._con

    def _create_tables(self) -> None:
        label_cols = ", ".join(f"{n} {_DUCK_TYPES[t]}" for n, t in LABELS.items())
        con = self._db()
        con.execute(
            "CREATE TABLE IF NOT EXISTS alerts("
            "alert_id VARCHAR PRIMARY KEY, ts_event_us BIGINT, symbol VARCHAR, json VARCHAR)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS feedback("
            "alert_id VARCHAR, verdict VARCHAR, analyst VARCHAR, ts_us BIGINT)"
        )
        con.execute(f"CREATE TABLE IF NOT EXISTS labels({label_cols})")
        con.execute(
            "CREATE TABLE IF NOT EXISTS runs("
            "run_id VARCHAR PRIMARY KEY, experiment VARCHAR, variant VARCHAR, seed BIGINT, "
            "created_us BIGINT, metrics_json VARCHAR, record_json VARCHAR)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS users("
            "user_id VARCHAR PRIMARY KEY, username VARCHAR UNIQUE, display_name VARCHAR, "
            "role VARCHAR, pw_hash VARCHAR, created_us BIGINT, disabled BOOLEAN)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS sessions("
            "token_hash VARCHAR PRIMARY KEY, user_id VARCHAR, created_us BIGINT, "
            "expires_us BIGINT, remember BOOLEAN, last_seen_us BIGINT)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS user_prefs(user_id VARCHAR PRIMARY KEY, json VARCHAR)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS audit_log("
            "ts_us BIGINT, user_id VARCHAR, username VARCHAR, action VARCHAR, "
            "target VARCHAR, detail VARCHAR)"
        )

    # ---- trades ----

    def _part_path(self, symbol: str, day: date) -> Path:
        return (
            self._trades_dir
            / f"symbol={_check_symbol(symbol)}"
            / f"date={day.isoformat()}"
            / "part.parquet"
        )

    def _read_index(self) -> pl.DataFrame:
        if not self._index_path.exists():
            return empty(_INDEX_SCHEMA)
        return pl.read_parquet(self._index_path)

    def write_trades(self, df: pl.DataFrame) -> None:
        """Write trades into per-(symbol, UTC day) partitions, replacing each touched partition.

        Within a partition rows are de-duplicated on trade_id (first occurrence wins) and sorted
        by (ts_us, trade_id). Every symbol is validated before anything is written.
        """
        df = conform(df, TRADES)
        if df.height == 0:
            return
        for symbol in df["symbol"].unique().to_list():
            _check_symbol(symbol)
        day_col = (pl.col("ts_us") // DAY_US).cast(pl.Int32).cast(pl.Date).alias("_day")
        groups = df.with_columns(day_col).partition_by(["symbol", "_day"], as_dict=True)
        entries = []
        with self._lock:
            for (symbol, day), part in groups.items():
                part = (
                    part.drop("_day")
                    .unique(subset=["trade_id"], keep="first", maintain_order=True)
                    .sort(["ts_us", "trade_id"])
                )
                path = self._part_path(str(symbol), day)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".tmp")
                part.write_parquet(tmp, compression="zstd")
                os.replace(tmp, path)
                entries.append((str(symbol), day, file_sha(path), part.height))
            self._update_index(entries)

    def write_day_streamed(
        self, symbol: str, day: date, build: Callable[[Path], None]
    ) -> int | None:
        """Write one symbol-day without holding it in memory: `build(tmp)` creates the partition's
        Parquet (TRADES schema, rows in file order) at `tmp`.

        The result equals `write_trades` for the same rows only when they are one UTC day with
        ts_us non-decreasing and trade_id strictly increasing (so sorting and de-duplicating are
        no-ops); that is checked on the built file. Returns the row count, or None - nothing is
        stored - when `build` fails, the file is empty or fails the check; the caller then falls
        back to `write_trades`.
        """
        path = self._part_path(symbol, day)
        lo = (day - _EPOCH).days * DAY_US
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            try:
                try:
                    build(tmp)
                    stats = self._order_stats(tmp)
                except duckdb.Error:
                    return None
                n, t_min, t_max, ordered = stats
                if not (n > 0 and t_min >= lo and t_max < lo + DAY_US and ordered):
                    return None
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
                try:
                    path.parent.rmdir()  # only succeeds when the partition was never created
                except OSError:
                    pass
            self._update_index([(symbol, day, file_sha(path), n)])
            return n

    @staticmethod
    def _order_stats(parquet: Path) -> tuple[int, int, int, bool]:
        """(rows, min ts_us, max ts_us, ts_us non-decreasing and trade_id strictly increasing)."""
        con = duckdb.connect()
        try:
            con.execute("SET memory_limit='96MB'")
            con.execute("SET threads=1")
            con.execute(f"SET temp_directory='{parquet.parent.as_posix()}'")
            n, lo, hi, bad = con.execute(
                """SELECT count(*), min(ts_us), max(ts_us), coalesce(bool_or(back), false) FROM (
                       SELECT ts_us, ts_us < lag(ts_us) OVER w OR trade_id <= lag(trade_id) OVER w
                       AS back FROM read_parquet(?) WINDOW w AS ())""",
                [parquet.as_posix()],
            ).fetchone()
        finally:
            con.close()
        return int(n), lo, hi, not bad

    def _update_index(self, entries: list[tuple[str, date, str, int]]) -> None:
        new = pl.DataFrame(entries, schema=_INDEX_SCHEMA, orient="row")
        old = self._read_index()
        keep = old.join(new.select("symbol", "date"), on=["symbol", "date"], how="anti")
        merged = pl.concat([keep, new]).sort(["symbol", "date"])
        tmp = self._index_path.with_name(self._index_path.name + ".tmp")
        merged.write_parquet(tmp)
        os.replace(tmp, self._index_path)

    def has_day(self, symbol: str, day: date) -> bool:
        return self._part_path(symbol, day).exists()

    def partition_sha(self, symbol: str, day: date) -> str | None:
        """12-hex sha256 of the partition file's actual bytes, or None if there is no partition.

        The index is only a cache: the sha is recomputed from the file, and if the index entry is
        missing or stale it is refreshed. Callers use this as a cache key, so it must never lag
        behind the file.
        """
        path = self._part_path(symbol, day)
        with self._lock:
            if not path.exists():
                return None
            actual = file_sha(path)
            hit = self._read_index().filter((pl.col("symbol") == symbol) & (pl.col("date") == day))
            if not hit.height or hit["sha"][0] != actual:
                n_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
                self._update_index([(symbol, day, actual, n_rows)])
            return actual

    def read_trades(self, symbol: str, start_us: int, end_us: int) -> pl.DataFrame:
        """Trades with start_us <= ts_us < end_us, sorted by (ts_us, trade_id)."""
        _check_symbol(symbol)
        if end_us <= start_us:
            return empty(TRADES)
        first, last = _day_of(start_us), _day_of(end_us - 1)
        paths = [
            p
            for n in range((last - first).days + 1)
            if (p := self._part_path(symbol, first + timedelta(days=n))).exists()
        ]
        if not paths:
            return empty(TRADES)
        out = (
            pl.scan_parquet(paths)
            .filter((pl.col("ts_us") >= start_us) & (pl.col("ts_us") < end_us))
            .sort(["ts_us", "trade_id"])
            .collect()
        )
        return conform(out, TRADES)

    def days(self, symbol: str) -> list[date]:
        """Sorted UTC days that have a trades partition for `symbol`."""
        return self._days_for(symbol)

    def _days_for(self, symbol: str) -> list[date]:
        base = self._trades_dir / f"symbol={_check_symbol(symbol)}"
        if not base.exists():
            return []
        days = []
        for d in base.glob("date=*"):
            if (d / "part.parquet").exists():
                days.append(date.fromisoformat(d.name.removeprefix("date=")))
        return sorted(days)

    def last_trade_before(self, symbol: str, ts_us: int) -> dict | None:
        """The latest trade with ts_us strictly before `ts_us` (any earlier day), as a dict."""
        cutoff_day = _day_of(ts_us - 1)
        for day in reversed([d for d in self._days_for(symbol) if d <= cutoff_day]):
            df = (
                pl.scan_parquet(self._part_path(symbol, day))
                .filter(pl.col("ts_us") < ts_us)
                .sort(["ts_us", "trade_id"])
                .tail(1)
                .collect()
            )
            if df.height:
                return df.to_dicts()[0]
        return None

    # ---- labels ----

    def write_labels(self, df: pl.DataFrame) -> None:
        """Upsert label rows by label_id."""
        df = conform(df, LABELS)
        if df.height == 0:
            return
        cols = ", ".join(LABELS.names())
        marks = ", ".join("?" for _ in LABELS)
        with self._lock:
            con = self._db()
            con.execute("BEGIN")
            try:
                con.executemany(
                    "DELETE FROM labels WHERE label_id = ?", [(i,) for i in df["label_id"]]
                )
                con.executemany(f"INSERT INTO labels({cols}) VALUES ({marks})", df.rows())
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise

    def replace_labels(self, df: pl.DataFrame, provenance: str = "REAL") -> int:
        """Replace every label of `provenance` by `df` (whose rows must all carry it), in one
        transaction; returns the number of rows removed.

        Label ids depend on the label time, so a changed offset gives new ids: an upsert would
        keep the old rows next to the new ones.
        """
        df = conform(df, LABELS)
        other = df.filter(pl.col("provenance") != provenance)
        if other.height:
            raise ValueError(f"replace_labels({provenance!r}) got rows of another provenance")
        cols = ", ".join(LABELS.names())
        marks = ", ".join("?" for _ in LABELS)
        with self._lock:
            con = self._db()
            con.execute("BEGIN")
            try:
                n_old = con.execute(
                    "SELECT count(*) FROM labels WHERE provenance = ?", [provenance]
                ).fetchone()[0]
                con.execute("DELETE FROM labels WHERE provenance = ?", [provenance])
                if df.height:
                    con.executemany(f"INSERT INTO labels({cols}) VALUES ({marks})", df.rows())
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
        return int(n_old)

    def read_labels(self, symbol: str | None = None) -> pl.DataFrame:
        cols = ", ".join(LABELS.names())
        sql = f"SELECT {cols} FROM labels"
        args: list[Any] = []
        if symbol is not None:
            sql += " WHERE symbol = ?"
            args.append(symbol)
        sql += " ORDER BY t_start_us, label_id"
        with self._lock:
            rows = self._db().execute(sql, args).fetchall()
        return pl.DataFrame(rows, schema=LABELS, orient="row")

    # ---- alerts / feedback ----

    @staticmethod
    def _alert_row(a: Alert) -> tuple:
        # evidence is explanatory only; JSON cannot carry inf/NaN, so drop such items (as the Chief)
        evidence = [(f, v, c) for f, v, c in a.evidence if math.isfinite(v) and math.isfinite(c)]
        a = a.model_copy(update={"evidence": evidence})
        return (a.alert_id, a.ts_event_us, a.symbol, a.model_dump_json())

    @staticmethod
    def _read_alert(js: str) -> Alert | None:
        """The stored alert, or None with a log line: one bad row must not cost the others."""
        try:
            return Alert.model_validate_json(js)
        except ValueError as e:
            log.warning("skipping an unreadable alert row: %s", e)
            return None

    def insert_alert(self, a: Alert) -> None:
        with self._lock:
            self._db().execute(
                "INSERT INTO alerts(alert_id, ts_event_us, symbol, json) VALUES (?, ?, ?, ?)",
                self._alert_row(a),
            )

    def upsert_alert(self, a: Alert) -> None:
        with self._lock:
            self._db().execute(
                "INSERT OR REPLACE INTO alerts(alert_id, ts_event_us, symbol, json) "
                "VALUES (?, ?, ?, ?)",
                self._alert_row(a),
            )

    @staticmethod
    def _check_deletable(provenance: str) -> None:
        if provenance not in _DELETABLE_PROVENANCE:
            raise ValueError(
                f"alerts of provenance {provenance!r} cannot be deleted; "
                f"only {_DELETABLE_PROVENANCE} can, never live"
            )

    def delete_alerts(self, provenance: str, alert_ids: list[str]) -> list[str]:
        """Delete the given alerts that have this replay / sim provenance; returns the ids deleted.

        Raises ValueError for any other provenance, "live" included.
        """
        self._check_deletable(provenance)
        wanted = set(alert_ids)
        return self._delete_alerts_where(
            lambda aid, a: aid in wanted and a["provenance"] == provenance
        )

    def delete_alerts_from(self, provenance: str, t_us: int) -> list[str]:
        """Delete the alerts of this replay / sim provenance whose window ends at or after `t_us`.

        Returns the ids deleted. Raises ValueError for any other provenance, "live" included.
        """
        self._check_deletable(provenance)
        return self._delete_alerts_where(
            lambda aid, a: a["provenance"] == provenance and a["window"][1] >= t_us
        )

    def _delete_alerts_where(self, match) -> list[str]:
        with self._lock:
            con = self._db()
            rows = con.execute("SELECT alert_id, json FROM alerts").fetchall()
            ids = [aid for aid, js in rows if match(aid, json.loads(js))]
            if ids:
                con.execute("BEGIN")
                try:
                    con.executemany("DELETE FROM alerts WHERE alert_id = ?", [(i,) for i in ids])
                    con.execute("COMMIT")
                except BaseException:
                    con.execute("ROLLBACK")
                    raise
        return ids

    def list_alerts(self, limit: int = 200, symbol: str | None = None) -> list[Alert]:
        sql = "SELECT json FROM alerts"
        args: list[Any] = []
        if symbol is not None:
            sql += " WHERE symbol = ?"
            args.append(symbol)
        sql += " ORDER BY ts_event_us DESC, alert_id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._db().execute(sql, args).fetchall()
        return [a for r in rows if (a := self._read_alert(r[0])) is not None]

    def get_alert(self, alert_id: str) -> Alert | None:
        with self._lock:
            row = (
                self._db()
                .execute("SELECT json FROM alerts WHERE alert_id = ?", [alert_id])
                .fetchone()
            )
        return self._read_alert(row[0]) if row else None

    def insert_feedback(self, alert_id: str, verdict: Verdict, analyst: str) -> None:
        if verdict not in _VERDICTS:
            raise ValueError(f"verdict must be one of {_VERDICTS}, got {verdict!r}")
        with self._lock:
            self._db().execute(
                "INSERT INTO feedback(alert_id, verdict, analyst, ts_us) VALUES (?, ?, ?, ?)",
                [alert_id, verdict, analyst, _now_us()],
            )

    def list_feedback(self, alert_id: str | None = None) -> list[dict]:
        """Verdicts on `alert_id`, oldest first; every alert's when `alert_id` is None."""
        where, args = ("WHERE alert_id = ? ", [alert_id]) if alert_id is not None else ("", [])
        with self._lock:
            cur = self._db().execute(
                "SELECT alert_id, verdict, analyst, ts_us FROM feedback "
                f"{where}ORDER BY ts_us, rowid",
                args,
            )
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]

    # ---- runs ----

    def insert_run(self, record: dict) -> None:
        """Store one experiment run; `record` must carry run_id and experiment.

        `metrics_json` (str) or `metrics` (dict) fills the metrics_json column; the full record is
        kept as JSON so `list_runs` returns it unchanged. Re-inserting a run_id replaces it.
        """
        metrics = record.get("metrics_json", record.get("metrics", {}))
        if not isinstance(metrics, str):
            metrics = json.dumps(metrics, sort_keys=True)
        created = record.get("created_us")
        with self._lock:
            self._db().execute(
                "INSERT OR REPLACE INTO runs(run_id, experiment, variant, seed, created_us, "
                "metrics_json, record_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    record["run_id"],
                    record["experiment"],
                    record.get("variant"),
                    record.get("seed"),
                    _now_us() if created is None else created,
                    metrics,
                    json.dumps(record, default=str),
                ],
            )

    def list_runs(self, experiment: str | None = None) -> list[dict]:
        sql = "SELECT record_json FROM runs"
        args: list[Any] = []
        if experiment is not None:
            sql += " WHERE experiment = ?"
            args.append(experiment)
        sql += " ORDER BY created_us, run_id"
        with self._lock:
            rows = self._db().execute(sql, args).fetchall()
        return [json.loads(r[0]) for r in rows]

    # ---- users, sessions, prefs, audit ----

    def _rows(self, sql: str, args: list[Any]) -> list[dict]:
        with self._lock:
            cur = self._db().execute(sql, args)
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]

    def attach_accounts(self, backend: AccountBackend) -> None:
        """Make `backend` the source of truth: replace local users/prefs with its rows and
        write through to it from now on."""
        backend.ensure_schema()
        users, prefs = backend.load_all()
        with self._lock:
            con = self._db()
            con.execute("BEGIN")
            try:
                con.execute("DELETE FROM users")
                con.execute("DELETE FROM user_prefs")
                for u in users:
                    con.execute(
                        "INSERT INTO users(user_id, username, display_name, role, pw_hash, "
                        "created_us, disabled) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [
                            u["user_id"],
                            u["username"],
                            u["display_name"],
                            u["role"],
                            u["pw_hash"],
                            u["created_us"],
                            bool(u["disabled"]),
                        ],
                    )
                for pref in prefs:
                    con.execute(
                        "INSERT INTO user_prefs(user_id, json) VALUES (?, ?)",
                        [pref["user_id"], pref["json"]],
                    )
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            self._accounts = backend

    def add_user(
        self, username: str, display_name: str, role: str, pw_hash: str, now_us: int
    ) -> str:
        user_id = uuid.uuid4().hex
        with self._lock:
            con = self._db()
            if con.execute("SELECT 1 FROM users WHERE username = ?", [username]).fetchone():
                raise ValueError(f"username already taken: {username}")
            if self._accounts is not None:
                self._accounts.insert_user(
                    {
                        "user_id": user_id,
                        "username": username,
                        "display_name": display_name,
                        "role": role,
                        "pw_hash": pw_hash,
                        "created_us": now_us,
                        "disabled": False,
                    }
                )
            try:
                con.execute(
                    "INSERT INTO users(user_id, username, display_name, role, pw_hash, "
                    "created_us, disabled) VALUES (?, ?, ?, ?, ?, ?, FALSE)",
                    [user_id, username, display_name, role, pw_hash, now_us],
                )
            except duckdb.ConstraintException as e:
                raise ValueError(f"username already taken: {username}") from e
        return user_id

    def get_user_by_name(self, username: str) -> dict | None:
        rows = self._rows("SELECT * FROM users WHERE username = ?", [username])
        return rows[0] if rows else None

    def get_user(self, user_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM users WHERE user_id = ?", [user_id])
        return rows[0] if rows else None

    def list_users(self) -> list[dict]:
        return self._rows(
            "SELECT u.user_id, u.username, u.display_name, u.role, u.created_us, u.disabled, "
            "(SELECT MAX(s.last_seen_us) FROM sessions s WHERE s.user_id = u.user_id) "
            "AS last_seen_us FROM users u ORDER BY u.created_us, u.username",
            [],
        )

    def set_role(self, username: str, role: str) -> None:
        with self._lock:
            if self._accounts is not None:
                self._accounts.update_user(username, role=role)
            self._db().execute("UPDATE users SET role = ? WHERE username = ?", [role, username])

    def _drop_user_sessions(self, username: str) -> None:
        self._db().execute(
            "DELETE FROM sessions WHERE user_id IN (SELECT user_id FROM users WHERE username = ?)",
            [username],
        )

    def set_disabled(self, username: str, flag: bool) -> None:
        with self._lock:
            if self._accounts is not None:
                self._accounts.update_user(username, disabled=flag)
            self._db().execute("UPDATE users SET disabled = ? WHERE username = ?", [flag, username])
            if flag:
                self._drop_user_sessions(username)

    def set_password(self, username: str, pw_hash: str) -> None:
        with self._lock:
            if self._accounts is not None:
                self._accounts.update_user(username, pw_hash=pw_hash)
            self._db().execute(
                "UPDATE users SET pw_hash = ? WHERE username = ?", [pw_hash, username]
            )
            self._drop_user_sessions(username)

    def has_enabled_admin(self) -> bool:
        with self._lock:
            row = (
                self._db()
                .execute("SELECT 1 FROM users WHERE role = 'admin' AND NOT disabled LIMIT 1")
                .fetchone()
            )
        return row is not None

    def create_session(
        self, user_id: str, token_h: str, now_us: int, expires_us: int, remember: bool
    ) -> None:
        with self._lock:
            self._db().execute(
                "INSERT INTO sessions(token_hash, user_id, created_us, expires_us, remember, "
                "last_seen_us) VALUES (?, ?, ?, ?, ?, ?)",
                [token_h, user_id, now_us, expires_us, remember, now_us],
            )

    def get_session(self, token_h: str, now_us: int) -> dict | None:
        rows = self._rows(
            "SELECT u.user_id, u.username, u.display_name, u.role, s.remember, s.last_seen_us "
            "FROM sessions s JOIN users u ON u.user_id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_us > ? AND NOT u.disabled",
            [token_h, now_us],
        )
        return rows[0] if rows else None

    def touch_session(self, token_h: str, now_us: int) -> None:
        with self._lock:
            self._db().execute(
                "UPDATE sessions SET last_seen_us = ? WHERE token_hash = ?", [now_us, token_h]
            )

    def delete_session(self, token_h: str) -> None:
        with self._lock:
            self._db().execute("DELETE FROM sessions WHERE token_hash = ?", [token_h])

    def purge_sessions(self, now_us: int) -> int:
        with self._lock:
            con = self._db()
            n = con.execute(
                "SELECT count(*) FROM sessions WHERE expires_us <= ?", [now_us]
            ).fetchone()[0]
            con.execute("DELETE FROM sessions WHERE expires_us <= ?", [now_us])
        return int(n)

    def get_prefs(self, user_id: str) -> dict:
        with self._lock:
            row = (
                self._db()
                .execute("SELECT json FROM user_prefs WHERE user_id = ?", [user_id])
                .fetchone()
            )
        return json.loads(row[0]) if row else {}

    def set_prefs(self, user_id: str, prefs: dict) -> None:
        text = json.dumps(prefs, sort_keys=True)
        with self._lock:
            if self._accounts is not None:
                self._accounts.set_prefs(user_id, text)
            self._db().execute(
                "INSERT OR REPLACE INTO user_prefs(user_id, json) VALUES (?, ?)",
                [user_id, text],
            )

    def add_audit(
        self,
        now_us: int,
        user_id: str | None,
        username: str,
        action: str,
        target: str = "",
        detail: dict | None = None,
    ) -> None:
        with self._lock:
            self._db().execute(
                "INSERT INTO audit_log(ts_us, user_id, username, action, target, detail) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [now_us, user_id, username, action, target, json.dumps(detail or {}, default=str)],
            )

    def list_audit(self, limit: int = 200, before_us: int | None = None) -> list[dict]:
        limit = max(0, min(int(limit), 500))
        sql = "SELECT ts_us, user_id, username, action, target, detail FROM audit_log"
        args: list[Any] = []
        if before_us is not None:
            sql += " WHERE ts_us < ?"
            args.append(before_us)
        sql += " ORDER BY ts_us DESC, rowid DESC LIMIT ?"
        args.append(limit)
        rows = self._rows(sql, args)
        for r in rows:
            r["detail"] = json.loads(r["detail"]) if r["detail"] else {}
        return rows
