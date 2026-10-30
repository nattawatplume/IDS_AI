"""Local store: Parquet trade partitions + DuckDB tables (alerts, feedback, labels, runs)."""

import json
import os
import re
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import duckdb
import polars as pl

from idsai.config import file_sha
from idsai.contracts.alert import Alert
from idsai.contracts.frames import LABELS, TRADES, conform, empty

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
        return (a.alert_id, a.ts_event_us, a.symbol, a.model_dump_json())

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
        return [Alert.model_validate_json(r[0]) for r in rows]

    def get_alert(self, alert_id: str) -> Alert | None:
        with self._lock:
            row = self._db().execute(
                "SELECT json FROM alerts WHERE alert_id = ?", [alert_id]
            ).fetchone()
        return Alert.model_validate_json(row[0]) if row else None

    def insert_feedback(self, alert_id: str, verdict: Verdict, analyst: str) -> None:
        if verdict not in _VERDICTS:
            raise ValueError(f"verdict must be one of {_VERDICTS}, got {verdict!r}")
        with self._lock:
            self._db().execute(
                "INSERT INTO feedback(alert_id, verdict, analyst, ts_us) VALUES (?, ?, ?, ?)",
                [alert_id, verdict, analyst, _now_us()],
            )

    def list_feedback(self, alert_id: str) -> list[dict]:
        with self._lock:
            cur = self._db().execute(
                "SELECT alert_id, verdict, analyst, ts_us FROM feedback "
                "WHERE alert_id = ? ORDER BY ts_us, rowid",
                [alert_id],
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
