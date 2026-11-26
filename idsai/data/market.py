"""The M6 market reference: BTCUSDT 1-minute klines from data.binance.vision, and `mkt_absret_z`.

Klines come only from data.binance.vision (`spot/daily/klines/BTCUSDT/1m`), each zip checked
against its published `.CHECKSUM` (`idsai.data.archive.download_verified`, the trades
downloader's code). A verified zip is parsed into `<market_dir>/<YYYY-MM-DD>.parquet` and
deleted; a day whose parquet exists is skipped (it was verified when written), as the trades
ingest does.

`mkt_absret_z` is the size of the latest BTC move as a robust z: `|log(close_i / close_{i-k})|`
over `window_s` (`k = window_s / 60` minutes; M6 uses 60 s, the window of `ret_60`, the return of
the pump-burst and dump rules), centred on the median and scaled by `1.4826 * MAD` of the same
quantity over the minutes whose close lies in the trailing `baseline_s` (24 h) strictly before
minute i. A value is known at its kline's close (`open_time + 60 s`) and a row at time t only
reads values known at or before t (`asof`), so nothing looks ahead.
"""

import io
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from idsai.data.archive import BASE_URL, _has_header, _read_csv_bytes, download_verified

MARKET_SYMBOL = "BTCUSDT"
INTERVAL = "1m"
MINUTE_US = 60_000_000
MKT_RET_WINDOW_S = 60  # the window of ret_60 (rules 100101, 100104, 100601)
MKT_BASELINE_S = 86_400
MKT_MIN_SAMPLES = 360  # as features.min_baseline_samples
MKT_MAX_AGE_S = 300  # a value older than this (a kline gap) is not used
MAD_SCALE = 1.4826
MAD_FLOOR = 1e-5  # log-return units (0.1 bp): a flat day must not divide by zero
_US_THRESHOLD = 1e14  # raw timestamps above this are already microseconds (spot: from 2025)
_CHUNK = 8192  # minutes per vectorised median block (bounds memory to ~100 MB)

KLINES = {
    "open_time_us": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "quote_volume": pl.Float64,
    "n_trades": pl.Int64,
}
_CSV_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def kline_url(day: date, symbol: str = MARKET_SYMBOL, interval: str = INTERVAL) -> str:
    return f"{BASE_URL}/klines/{symbol}/{interval}/{symbol}-{interval}-{day.isoformat()}.zip"


def market_dir(data_dir: Path, symbol: str = MARKET_SYMBOL, interval: str = INTERVAL) -> Path:
    return Path(data_dir) / "market" / f"{symbol}-{interval}"


def kline_path(mdir: Path, day: date) -> Path:
    return Path(mdir) / f"{day.isoformat()}.parquet"


def read_klines_zip(path: Path) -> pl.DataFrame:
    """Parse a daily klines zip into a KLINES frame sorted by `open_time_us`."""
    raw = _read_csv_bytes(Path(path))
    if not raw.strip():
        return pl.DataFrame(schema=KLINES)
    skip = 1 if _has_header(raw.split(b"\n", 1)[0]) else 0
    df = pl.read_csv(
        io.BytesIO(raw),
        has_header=False,
        skip_rows=skip,
        new_columns=_CSV_COLUMNS,
        schema_overrides={"open_time": pl.Int64, "close_time": pl.Int64, "count": pl.Int64},
    )
    t = pl.col("open_time")
    return (
        df.select(
            pl.when(t > _US_THRESHOLD).then(t).otherwise(t * 1000).alias("open_time_us"),
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            pl.col("count").alias("n_trades"),
        )
        .cast(KLINES)
        .sort("open_time_us")
    )


@dataclass
class KlineCounts:
    downloaded: int = 0
    skipped: int = 0
    missing: int = 0
    missing_days: list[date] = field(default_factory=list)

    def text(self) -> str:
        return (
            f"แท่งเทียน {MARKET_SYMBOL} {INTERVAL}: ดาวน์โหลด {self.downloaded} วัน, "
            f"มีอยู่แล้ว (ข้าม) {self.skipped} วัน, ไม่มีในคลังข้อมูล {self.missing} วัน"
        )


def download_klines(
    days: list[date], mdir: Path, client: httpx.Client, symbol: str = MARKET_SYMBOL
) -> KlineCounts:
    """Download, verify and store every day of `days` that `mdir` lacks (see the module doc)."""
    mdir = Path(mdir)
    c = KlineCounts()
    for day in sorted(set(days)):
        out = kline_path(mdir, day)
        if out.exists():
            c.skipped += 1
            continue
        zpath = download_verified(kline_url(day, symbol), mdir / "_zip", client)
        if zpath is None:
            c.missing += 1
            c.missing_days.append(day)
            continue
        try:
            df = read_klines_zip(zpath)
            part = out.with_name(out.name + ".part")
            df.write_parquet(part)
            part.replace(out)
        finally:
            zpath.unlink(missing_ok=True)
        c.downloaded += 1
    return c


def load_klines(mdir: Path, first: date, last: date) -> pl.DataFrame:
    """Every stored kline of `[first, last]` (missing days are simply absent), sorted, unique."""
    frames = []
    d = first
    while d <= last:
        p = kline_path(mdir, d)
        if p.exists():
            frames.append(pl.read_parquet(p).cast(KLINES))
        d += timedelta(days=1)
    if not frames:
        return pl.DataFrame(schema=KLINES)
    return pl.concat(frames).unique("open_time_us", keep="first").sort("open_time_us")


def mkt_absret_z(
    open_time_us: np.ndarray,
    close: np.ndarray,
    *,
    window_s: int = MKT_RET_WINDOW_S,
    baseline_s: int = MKT_BASELINE_S,
    min_samples: int = MKT_MIN_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(t_known_us, absret, z)` per kline, sorted by time (pure; see the module doc).

    `t_known_us = open_time + 60 s`. `absret` is NaN when the minute `window_s` before is not a
    kline (a gap) and `z` is NaN with fewer than `min_samples` finite baseline values.
    """
    if window_s % 60 or window_s <= 0:
        raise ValueError(f"window_s must be a positive multiple of 60, got {window_s}")
    if baseline_s % 60 or baseline_s <= 0:
        raise ValueError(f"baseline_s must be a positive multiple of 60, got {baseline_s}")
    t = np.asarray(open_time_us, dtype=np.int64)
    c = np.asarray(close, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    t, c = t[order], c[order]
    keep = np.concatenate([[True], np.diff(t) != 0]) if t.size else np.zeros(0, bool)
    t, c = t[keep], c[keep]
    if t.size == 0:
        e = np.zeros(0)
        return np.zeros(0, dtype=np.int64), e, e
    if np.any(t % MINUTE_US):
        raise ValueError("kline open times must lie on whole minutes")
    # a regular minute grid: gaps hold NaN closes, so a return across a gap is NaN
    m = (t - t[0]) // MINUTE_US
    grid = np.full(int(m[-1]) + 1, np.nan)
    grid[m] = c
    k = window_s // 60
    ret = np.full(grid.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret[k:] = np.abs(np.log(grid[k:] / grid[:-k]))
    ret[~np.isfinite(ret)] = np.nan
    n = baseline_s // 60  # the n minutes strictly before minute i
    padded = np.concatenate([np.full(n, np.nan), ret])
    z = np.full(grid.size, np.nan)
    for lo in range(0, grid.size, _CHUNK):
        hi = min(lo + _CHUNK, grid.size)
        # rows i in [lo, hi): baseline padded[i : i + n] = ret[i - n : i]
        win = np.lib.stride_tricks.sliding_window_view(padded[lo : hi + n - 1], n)
        cnt = np.isfinite(win).sum(axis=1)
        ok = cnt >= min_samples
        if not ok.any():
            continue
        w = win[ok]
        med = np.nanmedian(w, axis=1)
        mad = np.nanmedian(np.abs(w - med[:, None]), axis=1)
        scale = np.maximum(MAD_SCALE * mad, MAD_FLOOR)
        idx = np.arange(lo, hi)[ok]
        z[idx] = (ret[idx] - med) / scale
    return t + MINUTE_US, ret[m], z[m]


def asof(
    row_t_us: np.ndarray,
    t_known_us: np.ndarray,
    values: np.ndarray,
    max_age_s: int = MKT_MAX_AGE_S,
) -> np.ndarray:
    """The latest value known at or before each row time (NaN if none, or older than
    `max_age_s`). `t_known_us` must be sorted ascending."""
    rt = np.asarray(row_t_us, dtype=np.int64)
    tk = np.asarray(t_known_us, dtype=np.int64)
    v = np.asarray(values, dtype=np.float64)
    out = np.full(rt.size, np.nan)
    if tk.size == 0:
        return out
    i = np.searchsorted(tk, rt, side="right") - 1
    ok = (i >= 0) & (rt - tk[np.maximum(i, 0)] <= max_age_s * 1_000_000)
    out[ok] = v[i[ok]]
    return out


def research_market_days(symdays: list[tuple[str, date]]) -> list[date]:
    """Every date of the research symbol-days (`cli.research_symdays` already holds each day's
    D-1, so a scored day always has its trailing 24 h of klines)."""
    return sorted({d for _, d in symdays})
