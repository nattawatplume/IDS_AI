"""Streaming day ingest (zip -> temp CSV -> parquet) equals the in-memory path."""

import zipfile
from datetime import date

import httpx
import polars as pl
import pytest

from idsai import cli
from idsai.contracts.frames import TRADES
from idsai.data import archive
from idsai.data.archive import read_trades_zip
from idsai.data.store import Store

DAY = date(2025, 3, 1)
D0 = 1_740_787_200_000  # 2025-03-01T00:00:00Z in ms
HEADER = "id,price,qty,quoteQty,time,isBuyerMaker,isBestMatch\n"


def make_zip(path, csv: str, name="BTCUSDT-trades-2025-03-01.csv"):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, csv)
    return path


def csv_rows(n=50, scale=1, flags=("True", "False", "true", "FALSE")):
    return "".join(
        f"{100 + i},{100.5 + i / 8},{0.25 * (i % 7 + 1)},{(100.5 + i / 8) * 0.25},"
        f"{(D0 + i * 700) * scale},{flags[i % len(flags)]},True\n"
        for i in range(n)
    )


def old_path(tmp_path, zip_path):
    with Store(tmp_path / "old", tmp_path / "old" / "a.duckdb") as s:
        s.write_trades(read_trades_zip(zip_path, "BTCUSDT"))
        return s.read_trades("BTCUSDT", 0, 4_000_000_000_000_000), s._read_index()


def new_path(tmp_path, zip_path):
    csv = archive.extract_csv(zip_path, tmp_path)
    with Store(tmp_path / "new", tmp_path / "new" / "a.duckdb") as s:
        n = s.write_day_streamed(
            "BTCUSDT", DAY, lambda out: archive.csv_to_parquet(csv, "BTCUSDT", out)
        )
        return n, s.read_trades("BTCUSDT", 0, 4_000_000_000_000_000), s._read_index()


@pytest.mark.parametrize("header", [False, True])
@pytest.mark.parametrize("scale", [1, 1000])  # ms and us timestamps
def test_streamed_day_equals_in_memory(tmp_path, header, scale):
    body = csv_rows(scale=scale)
    z = make_zip(tmp_path / "a.zip", (HEADER if header else "") + body)
    want, want_idx = old_path(tmp_path, z)
    n, got, got_idx = new_path(tmp_path, z)
    assert n == want.height == 50
    assert got.schema == TRADES
    assert got.equals(want)
    assert got_idx.drop("sha").equals(want_idx.drop("sha"))
    part = tmp_path / "new" / "trades" / "symbol=BTCUSDT" / "date=2025-03-01"
    assert [p.name for p in part.iterdir()] == ["part.parquet"]
    assert pl.read_parquet(part / "part.parquet").equals(want)
    assert got_idx["sha"][0] == __import__("idsai.config", fromlist=["file_sha"]).file_sha(
        part / "part.parquet"
    )


def test_unsorted_or_duplicate_or_cross_day_is_refused(tmp_path):
    bad = {
        "unsorted": "".join(reversed(csv_rows(5).splitlines(keepends=True))),
        "dup": csv_rows(3) + csv_rows(3),
        "other_day": csv_rows(3, flags=("True",)).replace(str(D0), str(D0 - 86_400_000)),
    }
    for name, csv in bad.items():
        z = make_zip(tmp_path / f"{name}.zip", csv)
        c = archive.extract_csv(z, tmp_path)
        with Store(tmp_path / name, tmp_path / name / "a.duckdb") as s:
            build = lambda out, c=c: archive.csv_to_parquet(c, "BTCUSDT", out)  # noqa: E731
            assert s.write_day_streamed("BTCUSDT", DAY, build) is None
            assert not s.has_day("BTCUSDT", DAY)
            assert s.days("BTCUSDT") == []
            assert not list((tmp_path / name).rglob("*.tmp"))


def archive_client(zip_bytes):
    import hashlib

    def handler(req):
        if req.url.path.endswith(".CHECKSUM"):
            return httpx.Response(
                200, content=f"{hashlib.sha256(zip_bytes).hexdigest()}  x\n".encode()
            )
        return httpx.Response(200, content=zip_bytes)

    return httpx.Client(transport=httpx.MockTransport(handler))


def run_fetch_into(tmp_path, csv):
    z = make_zip(tmp_path / "src.zip", csv)
    dl = cli.ArchiveDownloader(tmp_path / "arch")
    dl._client = archive_client(z.read_bytes())
    with Store(tmp_path / "d", tmp_path / "d" / "a.duckdb") as s:
        n = dl.fetch_into(s, "BTCUSDT", DAY)
        return n, s.read_trades("BTCUSDT", 0, 4_000_000_000_000_000), s._read_index()


def test_fetch_into_streams_and_cleans_up(tmp_path):
    csv = csv_rows()
    n, got, idx = run_fetch_into(tmp_path, csv)
    want, _ = old_path(tmp_path, make_zip(tmp_path / "o.zip", csv))
    assert n == 50 and got.equals(want) and idx["n_rows"].to_list() == [50]
    assert list((tmp_path / "arch").rglob("*")) == [tmp_path / "arch" / "BTCUSDT"]


def test_fetch_into_falls_back_for_unsorted_day(tmp_path):
    csv = "".join(reversed(csv_rows(20).splitlines(keepends=True)))
    n, got, idx = run_fetch_into(tmp_path, csv)
    assert n == 20 and got["trade_id"].is_sorted() and idx["n_rows"].to_list() == [20]
    assert list((tmp_path / "arch").rglob("*")) == [tmp_path / "arch" / "BTCUSDT"]


def test_ingest_skips_a_stored_day_and_counts_empty_as_missing(tmp_path):
    class Spy:
        def __init__(self):
            self.calls = []

        def fetch_into(self, store, symbol, day):
            self.calls.append(day)
            return 0

        def close(self):
            pass

    with Store(tmp_path / "d", tmp_path / "d" / "a.duckdb") as s:
        s.write_trades(read_trades_zip(make_zip(tmp_path / "a.zip", csv_rows()), "BTCUSDT"))
        spy = Spy()
        c = cli.ingest(s, spy, [("BTCUSDT", DAY), ("BTCUSDT", date(2025, 3, 2))])
    assert spy.calls == [date(2025, 3, 2)]
    assert (c.skipped, c.missing, c.downloaded) == (1, 1, 0)
