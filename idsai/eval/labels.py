"""Real Telegram pump-and-dump labels (Sapienza, Hu et al.), dedup, and a timestamp alignment check.

These are the only real-world positives in the evaluation, so timestamps are parsed as UTC to the
microsecond and checked against the actual volume peak (`label_alignment`) before any REAL metric
is trusted.
"""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from idsai.contracts.frames import LABELS, conform, empty
from idsai.data.store import Store

TECHNIQUE_ID = "MM-T001"
FAMILY = "telegram_pump"
PROVENANCE = "REAL"
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_US = timedelta(microseconds=1)
_MINUTE_US = 60_000_000


def sapienza_csv_path() -> Path:
    """Path of the vendored Sapienza `pump_telegram.csv` (shipped inside the package)."""
    return Path(__file__).resolve().parent / "data" / "pump_telegram.csv"


def _to_us(dt: datetime) -> int:
    """Microseconds since the Unix epoch; a naive datetime is taken as UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (dt - _EPOCH) // _US


def label_id_for(symbol: str, t_start_us: int) -> str:
    return hashlib.sha1(f"{symbol}|{t_start_us}".encode()).hexdigest()[:12]


def _build(source: str, rows: list[tuple[str, int]]) -> pl.DataFrame:
    """rows: (symbol, t_us). Sorted by (t_start_us, symbol) for a deterministic order."""
    if not rows:
        return empty(LABELS)
    rows = sorted(rows, key=lambda r: (r[1], r[0]))
    return pl.DataFrame(
        [
            (label_id_for(sym, t), source, sym, t, t, TECHNIQUE_ID, FAMILY, None, PROVENANCE)
            for sym, t in rows
        ],
        schema=LABELS,
        orient="row",
    )


def load_sapienza(path: str | Path, offset_s: int = 0) -> pl.DataFrame:
    """Binance rows of the Sapienza `pump_telegram.csv` (`symbol,group,date,hour,exchange`).

    `date` is `YYYY-MM-DD` and `hour` is `HH:MM`, both UTC. Every label becomes a `<coin>BTC`
    symbol with `t_start_us == t_end_us == date + hour + offset_s`.
    """
    raw = pl.read_csv(path, infer_schema=False)
    rows = []
    for sym, date_s, hour_s in (
        raw.filter(pl.col("exchange") == "binance").select("symbol", "date", "hour").iter_rows()
    ):
        dt = datetime.strptime(f"{date_s} {hour_s}", "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        rows.append((f"{sym}BTC", _to_us(dt) + offset_s * 1_000_000))
    return _build("sapienza", rows)


def load_hu(path: str | Path, offset_s: int = 0) -> pl.DataFrame:
    """Binance/BTC rows of the Hu et al. `PD_logs_cleaned` TSV.

    Columns: `channel_id, session_id, coin, exchange, pair, timestamp`; timestamps look like
    `2019-06-04 12:01:33+00:00` and are converted to UTC.
    """
    raw = pl.read_csv(path, separator="\t", infer_schema=False)
    rows = []
    for coin, ts in (
        raw.filter((pl.col("exchange") == "binance") & (pl.col("pair") == "BTC"))
        .select("coin", "timestamp")
        .iter_rows()
    ):
        rows.append((f"{coin}BTC", _to_us(datetime.fromisoformat(ts)) + offset_s * 1_000_000))
    return _build("hu", rows)


def merge_dedup(frames: list[pl.DataFrame], tol_s: int = 300) -> pl.DataFrame:
    """Merge label frames; per symbol, labels chained by gaps <= `tol_s` become one label.

    Clustering is single-linkage on the time-sorted labels (consecutive gap <= tol_s), so a chain
    16:00 / 16:02 / 16:06 is one cluster. Each cluster yields one label at its earliest time, with
    `source` the sorted unique sources joined by `+` and `label_id = sha1(symbol|t_start_us)[:12]`.
    Other fields come from the earliest label (ties broken by source name).
    """
    frames = [conform(f, LABELS) for f in frames]
    if not frames:
        return empty(LABELS)
    allf = pl.concat(frames)
    if allf.height == 0:
        return empty(LABELS)
    tol_us = tol_s * 1_000_000
    allf = allf.sort(["symbol", "t_start_us", "source", "label_id"])
    out_rows = []
    for _, grp in allf.partition_by("symbol", as_dict=True, maintain_order=True).items():
        cluster: list[dict] = []
        last_t = None
        clusters: list[list[dict]] = []
        for row in grp.iter_rows(named=True):
            if last_t is not None and row["t_start_us"] - last_t > tol_us:
                clusters.append(cluster)
                cluster = []
            cluster.append(row)
            last_t = row["t_start_us"]
        clusters.append(cluster)
        for c in clusters:
            first = c[0]
            out_rows.append(
                (
                    label_id_for(first["symbol"], first["t_start_us"]),
                    "+".join(sorted({r["source"] for r in c})),
                    first["symbol"],
                    first["t_start_us"],
                    first["t_start_us"],
                    first["technique_id"],
                    first["family"],
                    first["scenario_seed"],
                    first["provenance"],
                )
            )
    return (
        pl.DataFrame(out_rows, schema=LABELS, orient="row")
        .sort(["t_start_us", "symbol"])
        .select(LABELS.names())
    )


_ALIGN_SCHEMA = pl.Schema(
    {
        "label_id": pl.String,
        "source": pl.String,
        "symbol": pl.String,
        "t_start_us": pl.Int64,
        "has_data": pl.Boolean,
        "lag_s": pl.Float64,
        "peak_quote_volume": pl.Float64,
    }
)


def label_alignment(
    store: Store, labels: pl.DataFrame, half_window_s: int = 10_800
) -> pl.DataFrame:
    """Where is the real volume peak relative to each label time?

    For each label with time t0, trades in `[t0 - half_window_s, t0 + half_window_s)` are bucketed
    into 1-minute buckets aligned to whole UTC minutes. `lag_s` is the start of the bucket with
    the largest quote volume (earliest on ties) minus t0, in seconds. Labels without trades get
    `has_data=False` and `lag_s=None`. A median lag near zero means the label clock is right;
    a consistent shift means `splits.label_offset_s` needs correcting.
    """
    labels = conform(labels, LABELS)
    half_us = half_window_s * 1_000_000
    rows = []
    for lab in labels.iter_rows(named=True):
        t0 = lab["t_start_us"]
        trades = store.read_trades(lab["symbol"], t0 - half_us, t0 + half_us)
        lag = peak = None
        if trades.height:
            buckets = (
                trades.with_columns(((pl.col("ts_us") // _MINUTE_US) * _MINUTE_US).alias("bucket"))
                .group_by("bucket")
                .agg(pl.col("quote_qty").sum().alias("vol"))
                .sort(["vol", "bucket"], descending=[True, False])
            )
            peak_bucket, peak = buckets["bucket"][0], float(buckets["vol"][0])
            lag = (peak_bucket - t0) / 1_000_000
        rows.append((lab["label_id"], lab["source"], lab["symbol"], t0, lag is not None, lag, peak))
    return pl.DataFrame(rows, schema=_ALIGN_SCHEMA, orient="row")
