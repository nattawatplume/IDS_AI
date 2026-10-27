import hashlib
import zipfile
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest

from idsai.contracts.frames import TRADES
from idsai.data import archive
from idsai.data.archive import ChecksumError, archive_url, download_day, read_trades_zip

DAY = date(2020, 6, 1)
URL = "https://data.binance.vision/data/spot/daily/trades/BRDBTC/BRDBTC-trades-2020-06-01.zip"


def make_zip(path: Path, csv: str, name: str = "BRDBTC-trades-2020-06-01.csv") -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, csv)
    return path


def sha_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def checksum_body(zip_bytes: bytes, name: str = "BRDBTC-trades-2020-06-01.zip") -> bytes:
    return f"{sha_of_bytes(zip_bytes)}  {name}\n".encode()


@pytest.fixture
def zip_bytes(tmp_path_factory) -> bytes:
    p = make_zip(
        tmp_path_factory.mktemp("src") / "a.zip",
        "1,0.0001,10,0.001,1590969600000,True,True\n2,0.0002,5,0.001,1590969601000,False,True\n",
    )
    return p.read_bytes()


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_archive_url_pattern():
    assert archive_url("BRDBTC", DAY) == URL
    assert archive_url("BTCUSDT", date(2025, 1, 2), "aggTrades") == (
        "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/"
        "BTCUSDT-aggTrades-2025-01-02.zip"
    )


def test_download_ok_and_skips_verified_existing(tmp_path, zip_bytes):
    zip_requests = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, content=checksum_body(zip_bytes))
        zip_requests.append(req)
        return httpx.Response(200, content=zip_bytes)

    with make_client(handler) as c:
        p = download_day("BRDBTC", DAY, tmp_path, c)
        assert p is not None and p.read_bytes() == zip_bytes
        assert p.name == "BRDBTC-trades-2020-06-01.zip"
        assert len(zip_requests) == 1
        p2 = download_day("BRDBTC", DAY, tmp_path, c)
        assert p2 == p
        assert len(zip_requests) == 1  # verified file on disk: no re-download


def test_checksum_with_star_prefix_is_parsed(tmp_path, zip_bytes):
    body = f"{sha_of_bytes(zip_bytes)} *BRDBTC-trades-2020-06-01.zip\n".encode()

    def handler(req):
        if req.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, content=body)
        return httpx.Response(200, content=zip_bytes)

    with make_client(handler) as c:
        assert download_day("BRDBTC", DAY, tmp_path, c) is not None


def test_404_returns_none(tmp_path):
    with make_client(lambda req: httpx.Response(404)) as c:
        assert download_day("BRDBTC", DAY, tmp_path, c) is None
    assert list(tmp_path.iterdir()) == []


def test_checksum_mismatch_retries_once_then_raises(tmp_path, zip_bytes):
    zip_requests = []

    def handler(req):
        if req.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, content=f"{'0' * 64}  x.zip\n".encode())
        zip_requests.append(req)
        return httpx.Response(200, content=zip_bytes)

    with make_client(handler) as c, pytest.raises(ChecksumError):
        download_day("BRDBTC", DAY, tmp_path, c)
    assert len(zip_requests) == 2
    assert list(tmp_path.iterdir()) == []  # no unverified zip (or temp file) left behind


def test_corrupt_then_good_succeeds(tmp_path, zip_bytes):
    calls = {"n": 0}

    def handler(req):
        if req.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, content=checksum_body(zip_bytes))
        calls["n"] += 1
        return httpx.Response(200, content=zip_bytes[:-5] if calls["n"] == 1 else zip_bytes)

    with make_client(handler) as c:
        p = download_day("BRDBTC", DAY, tmp_path, c)
    assert p is not None and p.read_bytes() == zip_bytes
    assert calls["n"] == 2


def test_existing_corrupt_file_is_replaced(tmp_path, zip_bytes):
    (tmp_path / "BRDBTC-trades-2020-06-01.zip").write_bytes(b"garbage")

    def handler(req):
        if req.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, content=checksum_body(zip_bytes))
        return httpx.Response(200, content=zip_bytes)

    with make_client(handler) as c:
        p = download_day("BRDBTC", DAY, tmp_path, c)
    assert p is not None and p.read_bytes() == zip_bytes


def test_429_honours_retry_after(tmp_path, zip_bytes, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(archive, "_sleep", sleeps.append)
    state = {"n": 0}

    def handler(req):
        if req.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, content=checksum_body(zip_bytes))
        state["n"] += 1
        if state["n"] <= 2:
            return httpx.Response(429 if state["n"] == 1 else 418, headers={"Retry-After": "7"})
        return httpx.Response(200, content=zip_bytes)

    with make_client(handler) as c:
        assert download_day("BRDBTC", DAY, tmp_path, c) is not None
    assert sleeps == [7.0, 7.0]


def test_rate_limit_gives_up_after_three_retries(tmp_path, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(archive, "_sleep", sleeps.append)

    with make_client(lambda req: httpx.Response(429, headers={"Retry-After": "1"})) as c:
        with pytest.raises(httpx.HTTPStatusError):
            download_day("BRDBTC", DAY, tmp_path, c)
    assert len(sleeps) == 3


def test_read_trades_zip_schema_and_sides(tmp_path):
    z = make_zip(
        tmp_path / "t.zip",
        "2,0.0002,5,0.001,1590969601000,False,True\n"
        "1,0.0001,10,0.001,1590969600000,True,True\n"
        "3,0.0003,1,0.0003,1590969601000,true,False\n",
    )
    df = read_trades_zip(z, "BRDBTC")
    assert df.schema == TRADES
    assert df["trade_id"].to_list() == [1, 2, 3]  # sorted (ts_us, trade_id)
    assert df["taker_side"].to_list() == ["sell", "buy", "sell"]
    assert df["ts_us"].to_list() == [1590969600000000, 1590969601000000, 1590969601000000]
    assert df["recv_us"].to_list() == df["ts_us"].to_list()
    assert set(df["source"]) == {"archive"} and set(df["n_fills"]) == {1}
    assert set(df["symbol"]) == {"BRDBTC"}
    assert df["quote_qty"].to_list() == [0.001, 0.001, 0.0003]
    assert df["price"][0] == pytest.approx(0.0001)


def test_read_trades_zip_tolerates_header(tmp_path):
    z = make_zip(
        tmp_path / "t.zip",
        "id,price,qty,quoteQty,time,isBuyerMaker,isBestMatch\n"
        "1,0.0001,10,0.001,1590969600000,false,true\n",
    )
    df = read_trades_zip(z, "BRDBTC")
    assert df.height == 1 and df["taker_side"][0] == "buy"


def test_read_trades_zip_empty_csv(tmp_path):
    df = read_trades_zip(make_zip(tmp_path / "e.zip", ""), "BRDBTC")
    assert df.schema == TRADES and df.height == 0


def test_ms_us_boundary_concat_is_sorted_and_normalized(tmp_path):
    ms_day = make_zip(
        tmp_path / "a.zip",
        "10,100.0,1,100.0,1735689599999,False,True\n",  # 2024-12-31T23:59:59.999Z in ms
    )
    us_day = make_zip(
        tmp_path / "b.zip",
        "11,100.0,1,100.0,1735689600000000,True,True\n",  # 2025-01-01T00:00:00Z in us
    )
    df = pl.concat([read_trades_zip(ms_day, "BTCUSDT"), read_trades_zip(us_day, "BTCUSDT")])
    assert df["ts_us"].to_list() == [1735689599999000, 1735689600000000]
    assert df["ts_us"].is_sorted()
    assert (df["ts_us"] > 1.7e15).all()
    assert df["taker_side"].to_list() == ["buy", "sell"]


@pytest.mark.network
def test_real_archive_roundtrip(tmp_path):
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        p = download_day("BRDBTC", DAY, tmp_path, c)
    assert p is not None
    df = read_trades_zip(p, "BRDBTC")
    assert df.height > 0
    assert df["ts_us"].is_sorted()
