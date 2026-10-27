"""Binance Vision daily archive: checksum-verified download and zip -> TRADES normalization."""

import hashlib
import io
import time
import zipfile
from datetime import date
from pathlib import Path
from typing import Literal

import httpx
import polars as pl

from idsai.contracts.frames import TRADES, empty

BASE_URL = "https://data.binance.vision/data/spot/daily"
RATE_LIMIT_STATUS = (429, 418)
MAX_RATE_RETRIES = 3
DEFAULT_RETRY_AFTER_S = 5.0
_US_THRESHOLD = 1e14  # raw timestamps above this are already microseconds (spot: from 2025-01-01)

_sleep = time.sleep  # indirection so tests can monkeypatch the backoff

Kind = Literal["trades", "aggTrades"]

_CSV_COLUMNS = ["id", "price", "qty", "quoteQty", "time", "isBuyerMaker", "isBestMatch"]


class ChecksumError(RuntimeError):
    """A downloaded archive did not match its published sha256 (after one retry)."""


def archive_url(symbol: str, day: date, kind: Kind = "trades") -> str:
    return f"{BASE_URL}/{kind}/{symbol}/{symbol}-{kind}-{day.isoformat()}.zip"


def _retry_after(resp: httpx.Response) -> float:
    try:
        return float(resp.headers["Retry-After"])
    except (KeyError, ValueError):
        return DEFAULT_RETRY_AFTER_S


def _get(client: httpx.Client, url: str) -> httpx.Response:
    """GET honouring 429/418 `Retry-After` (at most MAX_RATE_RETRIES retries)."""
    for attempt in range(MAX_RATE_RETRIES + 1):
        resp = client.get(url)
        if resp.status_code not in RATE_LIMIT_STATUS:
            return resp
        if attempt == MAX_RATE_RETRIES:
            break
        _sleep(_retry_after(resp))
    resp.raise_for_status()
    return resp  # pragma: no cover - raise_for_status always raises for 4xx


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_checksum(text: str) -> str:
    """`.CHECKSUM` content is `<sha256hex>  <filename>` (filename may carry a `*` prefix)."""
    tokens = text.split()
    if not tokens:
        raise ChecksumError("empty CHECKSUM file")
    return tokens[0].lower()


def download_day(
    symbol: str,
    day: date,
    dest_dir: Path,
    client: httpx.Client,
    kind: Kind = "trades",
) -> Path | None:
    """Download one verified daily zip into `dest_dir`; None if the day does not exist (404).

    An existing file whose sha256 matches is reused. A mismatching download is discarded and
    retried once; a second mismatch raises ChecksumError. Unverified data never stays on disk.
    """
    url = archive_url(symbol, day, kind)
    dest_dir = Path(dest_dir)
    dest = dest_dir / url.rsplit("/", 1)[1]

    resp = _get(client, url + ".CHECKSUM")
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    expected = _parse_checksum(resp.text)

    if dest.exists():
        if _sha256(dest.read_bytes()) == expected:
            return dest
        dest.unlink()

    dest_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for _attempt in range(2):
        resp = _get(client, url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        if _sha256(resp.content) == expected:
            part.write_bytes(resp.content)
            part.replace(dest)
            return dest
    raise ChecksumError(f"sha256 mismatch for {url} after retry")


def _read_csv_bytes(zip_path: Path) -> bytes:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            return b""
        return zf.read(names[0])


def read_trades_zip(path: Path, symbol: str) -> pl.DataFrame:
    """Parse a spot `trades` daily zip into a TRADES frame sorted by (ts_us, trade_id)."""
    raw = _read_csv_bytes(Path(path))
    if not raw.strip():
        return empty(TRADES)

    first_field = raw.split(b"\n", 1)[0].split(b",", 1)[0].strip().strip(b'"')
    has_header = not first_field.lstrip(b"-").isdigit()
    schema = {
        "id": pl.Int64,
        "price": pl.Float64,
        "qty": pl.Float64,
        "quoteQty": pl.Float64,
        "time": pl.Int64,
        "isBuyerMaker": pl.String,
        "isBestMatch": pl.String,
    }
    df = pl.read_csv(
        io.BytesIO(raw),
        has_header=False,
        skip_rows=1 if has_header else 0,
        new_columns=_CSV_COLUMNS,
        schema_overrides=schema,
    )
    if df.height == 0:
        return empty(TRADES)

    ts_us = pl.when(pl.col("time") > _US_THRESHOLD).then(pl.col("time")).otherwise(
        pl.col("time") * 1000
    )
    out = df.select(
        ts_us.alias("ts_us"),
        ts_us.alias("recv_us"),
        pl.lit(symbol).alias("symbol"),
        pl.col("price"),
        pl.col("qty"),
        pl.col("quoteQty").alias("quote_qty"),
        pl.when(pl.col("isBuyerMaker").str.to_lowercase() == "true")
        .then(pl.lit("sell"))
        .otherwise(pl.lit("buy"))
        .alias("taker_side"),
        pl.col("id").alias("trade_id"),
        pl.lit(1, dtype=pl.Int32).alias("n_fills"),
        pl.lit("archive").alias("source"),
    )
    return out.cast(dict(TRADES)).sort(["ts_us", "trade_id"])
