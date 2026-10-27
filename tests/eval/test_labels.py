import hashlib
from datetime import UTC, datetime

import polars as pl
import pytest

from idsai.contracts.frames import LABELS, TRADES
from idsai.data.store import Store
from idsai.eval.labels import (
    label_alignment,
    load_hu,
    load_sapienza,
    merge_dedup,
    sapienza_csv_path,
)

MIN_US = 60_000_000
T0 = int(datetime(2019, 6, 1, 16, 0, tzinfo=UTC).timestamp()) * 1_000_000


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


SAP = """symbol,group,date,hour,exchange
ABC,G1,2019-06-01,16:00,binance
DEF,G1,2019-06-01,16:00,yobit
GHI,G2,2019-06-02,09:30,binance
JKL,G2,2019-06-02,09:30,cryptopia
"""

HU = (
    "channel_id\tsession_id\tcoin\texchange\tpair\ttimestamp\n"
    "1\t10\tABC\tbinance\tBTC\t2019-06-01 16:00:00+00:00\n"
    "1\t11\tDEF\tbinance\tUSDT\t2019-06-01 16:00:00+00:00\n"
    "1\t12\tGHI\tyobit\tBTC\t2019-06-01 16:00:00+00:00\n"
    "1\t13\tJKL\tbinance\tBTC\t2019-06-04 12:01:33+00:00\n"
)


def _frame(rows):
    """rows: (source, symbol, t_us)"""
    return pl.DataFrame(
        [(f"{s}{t}", s, sym, t, t, "MM-T001", "telegram_pump", None, "REAL") for s, sym, t in rows],
        schema=LABELS,
        orient="row",
    )


# ---- loaders ----


def test_load_sapienza_binance_only_suffix_and_exact_us(tmp_path):
    df = load_sapienza(_write(tmp_path, "s.csv", SAP))
    assert df.schema == LABELS
    assert df["symbol"].to_list() == ["ABCBTC", "GHIBTC"]
    assert df["t_start_us"][0] == T0 == df["t_end_us"][0]
    assert df["t_start_us"][1] == T0 + 24 * 3600 * 1_000_000 + (9 * 60 + 30 - 16 * 60) * MIN_US
    assert set(df["source"]) == {"sapienza"}
    assert set(df["technique_id"]) == {"MM-T001"}
    assert set(df["family"]) == {"telegram_pump"}
    assert set(df["provenance"]) == {"REAL"}
    assert df["scenario_seed"].null_count() == df.height
    assert df["label_id"][0] == hashlib.sha1(f"ABCBTC|{T0}".encode()).hexdigest()[:12]


def test_load_sapienza_offset_applied(tmp_path):
    p = _write(tmp_path, "s.csv", SAP)
    base = load_sapienza(p)
    shifted = load_sapienza(p, offset_s=-90)
    assert (shifted["t_start_us"] - base["t_start_us"]).to_list() == [-90_000_000] * 2
    assert shifted["t_end_us"].to_list() == shifted["t_start_us"].to_list()
    assert shifted["label_id"].to_list() != base["label_id"].to_list()


def test_load_hu_keeps_only_binance_btc(tmp_path):
    df = load_hu(_write(tmp_path, "h.tsv", HU), offset_s=5)
    assert df.schema == LABELS
    assert df["symbol"].to_list() == ["ABCBTC", "JKLBTC"]
    assert df["t_start_us"][0] == T0 + 5_000_000
    expected = int(datetime(2019, 6, 4, 12, 1, 33, tzinfo=UTC).timestamp()) + 5
    assert df["t_start_us"][1] == expected * 1_000_000
    assert set(df["source"]) == {"hu"}
    assert set(df["provenance"]) == {"REAL"} and set(df["technique_id"]) == {"MM-T001"}


def test_load_hu_converts_offsets_to_utc(tmp_path):
    text = "channel_id\tsession_id\tcoin\texchange\tpair\ttimestamp\n"
    text += "1\t1\tABC\tbinance\tBTC\t2019-06-01 18:00:00+02:00\n"
    assert load_hu(_write(tmp_path, "h.tsv", text))["t_start_us"][0] == T0


def test_loaders_empty_result_has_schema(tmp_path):
    only_other = "symbol,group,date,hour,exchange\nA,G,2019-06-01,16:00,yobit\n"
    out = load_sapienza(_write(tmp_path, "s.csv", only_other))
    assert out.schema == LABELS and out.height == 0


def test_vendored_sapienza_file_has_520_binance_rows():
    df = load_sapienza(sapienza_csv_path())
    assert df.height == 520
    assert df["symbol"].str.ends_with("BTC").all()
    assert df["t_start_us"].min() > 1.5e15  # 2018+ in microseconds
    assert sapienza_csv_path().with_name("LICENSE-sapienza").exists()


# ---- merge_dedup ----


def test_merge_dedup_single_linkage_chain():
    # 16:00 -> 16:02 -> 16:06: consecutive gaps 120 s and 240 s (<= 300), so one cluster even
    # though 16:00..16:06 spans 360 s. 16:12 is 360 s after 16:06, so it starts a new cluster.
    f = _frame(
        [
            ("sapienza", "ABCBTC", T0),
            ("hu", "ABCBTC", T0 + 2 * MIN_US),
            ("hu", "ABCBTC", T0 + 6 * MIN_US),
            ("hu", "ABCBTC", T0 + 12 * MIN_US),
        ]
    )
    out = merge_dedup([f])
    assert out.schema == LABELS and out.height == 2
    assert out["t_start_us"].to_list() == [T0, T0 + 12 * MIN_US]
    assert out["t_end_us"].to_list() == out["t_start_us"].to_list()
    assert out["source"].to_list() == ["hu+sapienza", "hu"]


def test_merge_dedup_keeps_16_00_and_16_10_separate():
    f = _frame([("sapienza", "ABCBTC", T0), ("hu", "ABCBTC", T0 + 10 * MIN_US)])
    assert merge_dedup([f]).height == 2


def test_merge_dedup_tolerance_boundary_and_per_symbol_and_frames():
    a = _frame([("sapienza", "ABCBTC", T0), ("sapienza", "XYZBTC", T0)])
    b = _frame([("hu", "ABCBTC", T0 + 300_000_000)])  # exactly tol -> merged
    out = merge_dedup([a, b])
    assert out.height == 2
    abc = out.filter(pl.col("symbol") == "ABCBTC")
    assert abc["source"].to_list() == ["hu+sapienza"]
    xyz = out.filter(pl.col("symbol") == "XYZBTC")
    assert xyz["source"].to_list() == ["sapienza"]
    assert merge_dedup([a, b], tol_s=299).height == 3


def test_merge_dedup_label_id_and_order_independence():
    a = _frame([("sapienza", "ABCBTC", T0)])
    b = _frame([("hu", "ABCBTC", T0 + MIN_US)])
    out = merge_dedup([a, b])
    assert out["label_id"][0] == hashlib.sha1(f"ABCBTC|{T0}".encode()).hexdigest()[:12]
    assert out.equals(merge_dedup([b, a]))


def test_merge_dedup_empty():
    out = merge_dedup([pl.DataFrame(schema=LABELS)])
    assert out.schema == LABELS and out.height == 0
    assert merge_dedup([]).schema == LABELS


# ---- label_alignment ----


def _trade_frame(symbol, rows):
    """rows: (ts_us, quote_qty)"""
    n = len(rows)
    return pl.DataFrame(
        {
            "ts_us": [r[0] for r in rows],
            "recv_us": [r[0] for r in rows],
            "symbol": [symbol] * n,
            "price": [1.0] * n,
            "qty": [1.0] * n,
            "quote_qty": [r[1] for r in rows],
            "taker_side": ["buy"] * n,
            "trade_id": list(range(n)),
            "n_fills": [1] * n,
            "source": ["archive"] * n,
        },
        schema=TRADES,
    )


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "data", tmp_path / "data" / "app.duckdb") as s:
        yield s


def test_label_alignment_spike_60s_after_t0(store):
    rows = [(T0 - 30 * MIN_US + k * 20_000_000, 1.0) for k in range(200)]  # background
    rows += [(T0 + 70_000_000, 50.0), (T0 + 90_000_000, 50.0)]  # spike in minute T0 + 60 s
    store.write_trades(_trade_frame("ABCBTC", rows))
    labels = _frame([("sapienza", "ABCBTC", T0), ("sapienza", "NODATABTC", T0)])
    out = label_alignment(store, labels)
    assert out.height == 2
    hit = out.filter(pl.col("symbol") == "ABCBTC").row(0, named=True)
    assert hit["has_data"] is True and hit["lag_s"] == 60.0
    miss = out.filter(pl.col("symbol") == "NODATABTC").row(0, named=True)
    assert miss["has_data"] is False and miss["lag_s"] is None
    assert {"label_id", "symbol", "t_start_us"} <= set(out.columns)


def test_label_alignment_negative_lag_unaligned_t0_and_window(store):
    t0 = T0 + 17_000_000  # 16:00:17, not on a minute boundary
    rows = [(t0 - 3 * MIN_US + 5_000_000, 100.0), (t0 + 5 * MIN_US, 10.0)]
    rows.append((t0 + 4 * 3600 * 1_000_000, 1e6))  # bigger spike outside +-half_window: ignored
    store.write_trades(_trade_frame("ABCBTC", rows))
    labels = _frame([("sapienza", "ABCBTC", t0)])
    out = label_alignment(store, labels, half_window_s=3600)
    # peak minute starts at 15:57:00, which is 3 min 17 s before t0
    assert out["lag_s"][0] == -197.0


# ---- scripts/fetch_labels.py (offline) ----


def _load_fetch_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "fetch_labels.py"
    spec = importlib.util.spec_from_file_location("fetch_labels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_script_checks_header_listing_and_hosts(tmp_path):
    import httpx

    m = _load_fetch_script()
    listing = [{"name": m.FILE_NAME, "type": "file", "download_url": m.RAW_URL}]
    assert m.verify_listing(listing) == m.RAW_URL
    with pytest.raises(RuntimeError):
        m.verify_listing([{"name": "other.txt", "type": "file"}])
    with pytest.raises(ValueError):
        m.verify_listing(
            [{"name": m.FILE_NAME, "type": "file", "download_url": "https://evil.example/x"}]
        )
    m.check_header(HU)
    with pytest.raises(RuntimeError):
        m.check_header("a\tb\n")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            return httpx.Response(200, json=listing)
        assert request.url.host == "raw.githubusercontent.com"
        return httpx.Response(200, content=HU.encode())

    out = tmp_path / "labels" / "hu.tsv"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert m.fetch(out, client) == out
    assert load_hu(out).height == 2
