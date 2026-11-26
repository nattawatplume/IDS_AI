"""M6 market reference: BTCUSDT 1m kline download (mocked HTTP, CHECKSUM), parsing and
`mkt_absret_z` (no look-ahead, trailing 24 h robust z). No test touches the network or the real
`data/app.duckdb`."""

import hashlib
import io
import zipfile
from datetime import date, timedelta

import httpx
import numpy as np
import polars as pl
import pytest

from idsai import cli
from idsai.data.archive import ChecksumError, download_verified
from idsai.data.market import (
    KLINES,
    MINUTE_US,
    asof,
    download_klines,
    kline_path,
    kline_url,
    load_klines,
    market_dir,
    mkt_absret_z,
    read_klines_zip,
    research_market_days,
)
from idsai.data.store import Store
from tests.test_cli import label_rows, write_cfg

DAY = date(2019, 3, 1)
D0_MS = 1551398400000  # 2019-03-01T00:00Z


def kline_csv(day_ms: int, closes: list[float], header: bool = False, us: bool = False) -> str:
    rows = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,tbv,tbq,ignore"]
    rows = rows if header else []
    for i, c in enumerate(closes):
        t = day_ms + i * 60_000
        t0, t1 = (t * 1000, (t + 59_999) * 1000 + 999) if us else (t, t + 59_999)
        rows.append(f"{t0},{c},{c},{c},{c},1.5,{t1},{1.5 * c},7,0.5,{0.5 * c},0")
    return "\n".join(rows) + "\n"


def zip_of(csv: str, name: str = "BTCUSDT-1m-2019-03-01.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, csv)
    return buf.getvalue()


def checksum(b: bytes) -> bytes:
    return f"{hashlib.sha256(b).hexdigest()}  BTCUSDT-1m-2019-03-01.zip\n".encode()


def client_for(files: dict[str, bytes], calls: list[str] | None = None) -> httpx.Client:
    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if calls is not None:
            calls.append(url)
        if url in files:
            return httpx.Response(200, content=files[url])
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------ download


def test_kline_url_is_the_binance_vision_daily_1m_path():
    assert kline_url(DAY) == (
        "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2019-03-01.zip"
    )


def test_download_verifies_checksum_stores_parquet_and_skips_stored_days(tmp_path):
    z = zip_of(kline_csv(D0_MS, [100.0, 101.0, 99.0]))
    url = kline_url(DAY)
    calls: list[str] = []
    with client_for({url: z, url + ".CHECKSUM": checksum(z)}, calls) as c:
        got = download_klines([DAY, DAY - timedelta(days=1)], tmp_path, c)
        assert (got.downloaded, got.skipped, got.missing) == (1, 0, 1)
        assert got.missing_days == [DAY - timedelta(days=1)]
        df = pl.read_parquet(kline_path(tmp_path, DAY))
        assert dict(df.schema) == KLINES
        assert df["close"].to_list() == [100.0, 101.0, 99.0]
        assert df["open_time_us"][0] == D0_MS * 1000
        assert not list((tmp_path / "_zip").glob("*.zip"))  # the verified zip is deleted
        n = len(calls)
        again = download_klines([DAY], tmp_path, c)
        assert (again.downloaded, again.skipped) == (0, 1)
        assert len(calls) == n  # a stored day costs no request


def test_download_bad_checksum_twice_raises_and_keeps_nothing(tmp_path):
    z = zip_of(kline_csv(D0_MS, [100.0]))
    url = kline_url(DAY)
    with client_for({url: z, url + ".CHECKSUM": checksum(b"other")}) as c:
        with pytest.raises(ChecksumError):
            download_klines([DAY], tmp_path, c)
    assert not kline_path(tmp_path, DAY).exists()
    assert not list(tmp_path.rglob("*.zip")) and not list(tmp_path.rglob("*.part"))


def test_download_verified_refuses_other_hosts(tmp_path):
    with client_for({}) as c, pytest.raises(ValueError):
        download_verified("https://example.com/data/x.zip", tmp_path, c)


def test_read_klines_zip_handles_header_and_microsecond_times(tmp_path):
    p = tmp_path / "k.zip"
    p.write_bytes(zip_of(kline_csv(1735689600000, [5.0, 6.0], header=True, us=True)))
    df = read_klines_zip(p)
    assert df["open_time_us"].to_list() == [1735689600000000, 1735689660000000]
    assert df["n_trades"].to_list() == [7, 7]


def test_load_klines_concatenates_days_and_skips_missing(tmp_path):
    for i, d in enumerate([DAY, DAY + timedelta(days=2)]):
        ms = D0_MS + i * 2 * 86_400_000
        p = tmp_path / "z.zip"
        p.write_bytes(zip_of(kline_csv(ms, [1.0, 2.0])))
        read_klines_zip(p).write_parquet(kline_path(tmp_path, d))
    df = load_klines(tmp_path, DAY, DAY + timedelta(days=2))
    assert df.height == 4 and df["open_time_us"].is_sorted()
    assert load_klines(tmp_path, DAY + timedelta(days=5), DAY + timedelta(days=6)).height == 0


def test_research_market_days_are_the_unique_dates():
    sd = [("A", DAY), ("B", DAY), ("A", DAY - timedelta(days=1))]
    assert research_market_days(sd) == [DAY - timedelta(days=1), DAY]


def test_cli_download_market_fetches_every_research_date(tmp_path, monkeypatch):
    cfg = write_cfg(tmp_path)
    from idsai.config import load_settings

    s = load_settings(cfg)
    t0 = int(D0_MS * 1000) + 12 * 3600 * 1_000_000
    with Store(s.paths.data_dir, s.paths.app_db) as store:
        store.write_labels(label_rows(("sapienza", "AAABTC", t0)))
    want = research_market_days(cli.research_symdays(s, label_rows(("x", "AAABTC", t0))))
    calls: list[str] = []
    monkeypatch.setattr(cli, "make_market_client", lambda: client_for({}, calls))
    assert cli.main(["--config", str(cfg), "download", "--market"]) == 0
    assert {u for u in calls if u.endswith(".CHECKSUM")} == {
        kline_url(d) + ".CHECKSUM" for d in want
    }
    assert DAY in want and DAY - timedelta(days=1) in want
    assert market_dir(s.paths.data_dir).parts[-2:] == ("market", "BTCUSDT-1m")
    assert cli.main(["--config", str(cfg), "download", "--market", "--days", "2"]) == 1


def test_cli_download_market_without_labels_exits_1(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert cli.main(["--config", str(cfg), "download", "--market"]) == 1
    assert "labels" in capsys.readouterr().err


# ------------------------------------------------------------------ mkt_absret_z


def _series(n: int, seed: int = 0, t0_min: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = (t0_min + np.arange(n, dtype=np.int64)) * MINUTE_US
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    return t, close


def _naive(t, close, i, n=1440, min_samples=360):
    """Reference for minute i on a gap-free series: robust z of |r_i| vs r_{i-n..i-1}."""
    r = np.full(t.size, np.nan)
    r[1:] = np.abs(np.diff(np.log(close)))
    base = r[max(0, i - n) : i]
    base = base[np.isfinite(base)]
    if base.size < min_samples:
        return np.nan
    med = np.median(base)
    mad = np.median(np.abs(base - med))
    return (r[i] - med) / max(1.4826 * mad, 1e-5)


def test_z_matches_a_naive_trailing_24h_reference():
    t, c = _series(3000)
    tk, r, z = mkt_absret_z(t, c)
    assert np.array_equal(tk, t + MINUTE_US)
    assert np.isnan(r[0]) and np.isclose(r[1], abs(np.log(c[1] / c[0])))
    for i in (0, 100, 360, 361, 1440, 1441, 2999):
        assert np.allclose(z[i], _naive(t, c, i), equal_nan=True), i
    assert np.isnan(z[:361]).all() and np.isfinite(z[361:]).all()


def test_no_look_ahead_future_klines_do_not_change_past_values():
    t, c = _series(2500, seed=3)
    _, _, z_full = mkt_absret_z(t, c)
    c2 = c.copy()
    c2[2000:] *= 1.5  # a huge future move
    _, _, z_cut = mkt_absret_z(t, c2)
    assert np.array_equal(z_full[:2000], z_cut[:2000], equal_nan=True)
    _, _, z_trunc = mkt_absret_z(t[:2000], c[:2000])
    assert np.array_equal(z_full[:2000], z_trunc, equal_nan=True)


def test_a_btc_jump_gives_a_large_z_and_gaps_give_nan():
    t, c = _series(2000, seed=1)
    c[1800:] *= 1.02  # +2 % in one minute
    _, r, z = mkt_absret_z(t, c)
    assert z[1800] > 10 and abs(z[1801]) < 5
    keep = np.ones(t.size, bool)
    keep[1500] = False  # a missing kline
    tk, r2, z2 = mkt_absret_z(t[keep], c[keep])
    assert tk.size == t.size - 1
    assert np.isnan(r2[1500]) and np.isnan(z2[1500])  # minute 1501: its previous kline is missing


def test_window_of_k_minutes_uses_the_close_k_minutes_back():
    t, c = _series(10)
    _, r, _ = mkt_absret_z(t, c, window_s=300)
    assert np.isnan(r[:5]).all() and np.isclose(r[7], abs(np.log(c[7] / c[2])))
    with pytest.raises(ValueError):
        mkt_absret_z(t, c, window_s=90)


def test_unsorted_and_duplicate_input_is_sorted_and_deduplicated():
    t, c = _series(2000, seed=2)
    _, _, z = mkt_absret_z(t, c)
    perm = np.random.default_rng(0).permutation(t.size)
    tt = np.concatenate([t[perm], t[:5]])
    cc = np.concatenate([c[perm], c[:5]])
    tk, _, z2 = mkt_absret_z(tt, cc)
    assert np.array_equal(tk, t + MINUTE_US) and np.array_equal(z, z2, equal_nan=True)
    assert mkt_absret_z(t[:0], c[:0])[2].size == 0


def test_asof_reads_only_values_known_at_or_before_the_row():
    tk = np.array([60, 120, 180], dtype=np.int64) * 1_000_000
    v = np.array([1.0, 2.0, 3.0])
    rows = np.array([59, 60, 119, 120, 179, 600], dtype=np.int64) * 1_000_000
    got = asof(rows, tk, v, max_age_s=300)
    assert np.array_equal(got, [np.nan, 1.0, 1.0, 2.0, 2.0, np.nan], equal_nan=True)
    assert np.isnan(asof(rows, tk[:0], v[:0])).all()
