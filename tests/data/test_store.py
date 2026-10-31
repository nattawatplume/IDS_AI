import json
from datetime import date

import polars as pl
import pytest

from idsai.config import file_sha
from idsai.contracts.alert import Alert, DetectorRef
from idsai.contracts.frames import LABELS, TRADES
from idsai.data.store import Store

DAY_US = 86_400_000_000
D1 = date(2025, 3, 1)
D1_US = 1_740_787_200_000_000  # 2025-03-01T00:00:00Z
assert D1_US % DAY_US == 0


def trades(ids, ts, symbol="BTCUSDT", price=100.0, source="archive") -> pl.DataFrame:
    n = len(ids)
    return pl.DataFrame(
        {
            "ts_us": ts,
            "recv_us": ts,
            "symbol": [symbol] * n,
            "price": [price + i for i in range(n)],
            "qty": [1.0] * n,
            "quote_qty": [price + i for i in range(n)],
            "taker_side": ["buy" if i % 2 else "sell" for i in range(n)],
            "trade_id": ids,
            "n_fills": [1] * n,
            "source": [source] * n,
        },
        schema=TRADES,
    )


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "data", tmp_path / "data" / "app.duckdb") as s:
        yield s


def make_alert(alert_id="a1", ts=D1_US + 5_000_000, symbol="BTCUSDT", **over) -> Alert:
    kw = dict(
        alert_id=alert_id,
        ts_event_us=ts,
        symbol=symbol,
        technique_id="MM-T001",
        signature_id=3,
        rev=1,
        severity=2,
        status="open",
        fused_p=0.004,
        detectors=[DetectorRef(name="rules", layer="rule", p=0.001, score=4.2)],
        agreement=1,
        evidence=[("ret_60s", 3.5, 0.9), ("vol_10s", 7.0, 1.0)],
        window=(ts - 60_000_000, ts),
        dedup_key="BTCUSDT:MM-T001",
        suppress_for_s=300,
        config_hash="abc123def456",
        model_versions={"ae": "v1"},
        provenance="replay",
    )
    kw.update(over)
    return Alert(**kw)


# ---- trades ----


def test_roundtrip_and_layout(store, tmp_path):
    df = trades([1, 2, 3], [D1_US + 1, D1_US + 2, D1_US + 3])
    store.write_trades(df)
    out = store.read_trades("BTCUSDT", D1_US, D1_US + DAY_US)
    assert out.schema == TRADES
    assert out.equals(df)
    assert (tmp_path / "data/trades/symbol=BTCUSDT/date=2025-03-01/part.parquet").exists()
    idx = pl.read_parquet(tmp_path / "data/trades/_index.parquet")
    assert idx.columns == ["symbol", "date", "sha", "n_rows"]
    assert idx.row(0) == ("BTCUSDT", D1, store.partition_sha("BTCUSDT", D1), 3)


def test_same_day_written_twice_same_row_count(store):
    df = trades([1, 2, 3], [D1_US + 1, D1_US + 2, D1_US + 3])
    store.write_trades(df)
    store.write_trades(df)
    assert store.read_trades("BTCUSDT", D1_US, D1_US + DAY_US).height == 3
    idx = pl.read_parquet(store._index_path)
    assert idx.height == 1


def test_rewrite_replaces_partition(store):
    store.write_trades(trades([1, 2, 3], [D1_US + 1, D1_US + 2, D1_US + 3]))
    store.write_trades(trades([4], [D1_US + 9]))
    out = store.read_trades("BTCUSDT", D1_US, D1_US + DAY_US)
    assert out["trade_id"].to_list() == [4]


def test_duplicate_ids_deduped(store):
    ts = [D1_US + 1, D1_US + 2, D1_US + 2, D1_US + 3, D1_US + 1]
    store.write_trades(trades([1, 2, 2, 3, 1], ts))
    out = store.read_trades("BTCUSDT", D1_US, D1_US + DAY_US)
    assert out["trade_id"].to_list() == [1, 2, 3]
    # trades() gives row i the price 100 + i; keep-first means rows 0, 1, 3 survive (not 4, 2, 3)
    assert out["price"].to_list() == [100.0, 101.0, 103.0]


def test_out_of_order_read_back_sorted(store):
    store.write_trades(trades([3, 1, 2, 4], [D1_US + 5, D1_US + 5, D1_US + 1, D1_US + 5]))
    out = store.read_trades("BTCUSDT", D1_US, D1_US + DAY_US)
    assert out["trade_id"].to_list() == [2, 1, 3, 4]  # (ts_us, trade_id)


def test_half_open_boundaries(store):
    store.write_trades(trades([1, 2, 3], [D1_US + 10, D1_US + 20, D1_US + 30]))
    out = store.read_trades("BTCUSDT", D1_US + 10, D1_US + 30)
    assert out["trade_id"].to_list() == [1, 2]  # start inclusive, end exclusive
    assert store.read_trades("BTCUSDT", D1_US + 30, D1_US + 30).height == 0


def test_multi_day_and_multi_symbol_split(store):
    ts = [D1_US + DAY_US - 1, D1_US + DAY_US, D1_US + DAY_US + 1]
    df = pl.concat([trades([1, 2, 3], ts), trades([1, 2], [D1_US + 7, D1_US + 8], "ETHUSDT")])
    store.write_trades(df)
    assert store.has_day("BTCUSDT", D1) and store.has_day("BTCUSDT", date(2025, 3, 2))
    assert store.has_day("ETHUSDT", D1) and not store.has_day("ETHUSDT", date(2025, 3, 2))
    assert not store.has_day("SOLUSDT", D1)
    out = store.read_trades("BTCUSDT", D1_US, D1_US + 2 * DAY_US)
    assert out["trade_id"].to_list() == [1, 2, 3]
    assert store.read_trades("ETHUSDT", D1_US, D1_US + 2 * DAY_US).height == 2


def test_empty_read_has_trades_schema(store):
    out = store.read_trades("BTCUSDT", D1_US, D1_US + DAY_US)
    assert out.schema == TRADES and out.height == 0
    store.write_trades(trades([1], [D1_US + 1]))
    out = store.read_trades("BTCUSDT", D1_US + 100, D1_US + 200)
    assert out.schema == TRADES and out.height == 0


def test_write_empty_is_noop(store):
    store.write_trades(pl.DataFrame(schema=TRADES))
    assert not store.has_day("BTCUSDT", D1)


def test_partition_sha_changes_with_content(store):
    assert store.partition_sha("BTCUSDT", D1) is None
    store.write_trades(trades([1, 2], [D1_US + 1, D1_US + 2]))
    a = store.partition_sha("BTCUSDT", D1)
    assert a is not None and len(a) == 12
    store.write_trades(trades([1, 2], [D1_US + 1, D1_US + 2]))
    assert store.partition_sha("BTCUSDT", D1) == a  # same content, same sha
    store.write_trades(trades([1, 2, 3], [D1_US + 1, D1_US + 2, D1_US + 3]))
    assert store.partition_sha("BTCUSDT", D1) != a


def test_partition_sha_recomputed_when_file_changed_behind_index(store):
    store.write_trades(trades([1, 2], [D1_US + 1, D1_US + 2]))
    stale = store.partition_sha("BTCUSDT", D1)
    path = store._part_path("BTCUSDT", D1)
    trades([1, 2, 3], [D1_US + 1, D1_US + 2, D1_US + 3]).write_parquet(path)  # bypass the store
    fresh = store.partition_sha("BTCUSDT", D1)
    assert fresh != stale and fresh == file_sha(path)
    idx = pl.read_parquet(store._index_path).filter(pl.col("symbol") == "BTCUSDT")
    assert idx["sha"].to_list() == [fresh]  # index entry refreshed too


def test_partition_sha_without_index_entry(store):
    store.write_trades(trades([1], [D1_US + 1]))
    store._index_path.unlink()
    assert store.partition_sha("BTCUSDT", D1) == file_sha(store._part_path("BTCUSDT", D1))


def test_bad_symbol_writes_nothing(store, tmp_path):
    good = trades([1, 2], [D1_US + 1, D1_US + 2], "BTCUSDT")
    bad = trades([1], [D1_US + 3], "../evil")
    with pytest.raises(ValueError):
        store.write_trades(pl.concat([good, bad]))
    assert not store.has_day("BTCUSDT", D1)
    assert not store._index_path.exists()
    assert not any((tmp_path / "data" / "trades").rglob("*.parquet"))
    assert not (tmp_path / "data" / "evil").exists()


def test_last_trade_before(store):
    d2 = D1_US + DAY_US
    store.write_trades(trades([1, 2], [D1_US + 10, D1_US + 20]))
    store.write_trades(trades([3], [d2 + 5]))
    assert store.last_trade_before("BTCUSDT", D1_US + 10) is None  # strictly before
    assert store.last_trade_before("BTCUSDT", D1_US + 11)["trade_id"] == 1
    t = store.last_trade_before("BTCUSDT", d2 + 5)  # previous day's partition
    assert t["trade_id"] == 2 and t["ts_us"] == D1_US + 20 and t["symbol"] == "BTCUSDT"
    assert set(t) == set(TRADES.names())
    assert store.last_trade_before("BTCUSDT", d2 + 6)["trade_id"] == 3
    assert store.last_trade_before("BTCUSDT", d2 + 10 * DAY_US)["trade_id"] == 3
    assert store.last_trade_before("ETHUSDT", d2) is None


def test_rejects_unsafe_symbol(store):
    with pytest.raises(ValueError):
        store.read_trades("../x", 0, 1)


# ---- labels ----


def make_labels(ids=("l1", "l2")) -> pl.DataFrame:
    rows = [
        (i, "sapienza", "BTCUSDT", 1000 + k, 2000 + k, "MM-T001", "pump", None, "REAL")
        for k, i in enumerate(ids)
    ]
    rows.append(("l3", "sim", "ETHUSDT", 5, 9, "MM-T002", "wash", 7, "SYN"))
    return pl.DataFrame(rows, schema=LABELS, orient="row")


def test_labels_roundtrip_filter_and_idempotent(store):
    df = make_labels()
    store.write_labels(df)
    store.write_labels(df)
    out = store.read_labels()
    assert out.schema == LABELS and out.height == 3
    assert sorted(out["label_id"].to_list()) == ["l1", "l2", "l3"]
    assert out.filter(pl.col("label_id") == "l3")["scenario_seed"][0] == 7
    assert out.filter(pl.col("label_id") == "l1")["scenario_seed"][0] is None
    eth = store.read_labels("ETHUSDT")
    assert eth["label_id"].to_list() == ["l3"]
    assert store.read_labels("NOPE").schema == LABELS


def test_read_labels_empty(store):
    out = store.read_labels()
    assert out.schema == LABELS and out.height == 0


# ---- alerts / feedback ----


def test_alert_roundtrip_equality(store):
    a = make_alert()
    store.insert_alert(a)
    got = store.get_alert("a1")
    assert got == a
    assert isinstance(got.window, tuple) and isinstance(got.evidence[0], tuple)
    assert store.get_alert("missing") is None


def test_alert_upsert_and_listing(store):
    store.insert_alert(make_alert("a1", ts=D1_US + 1))
    store.insert_alert(make_alert("a2", ts=D1_US + 2, symbol="ETHUSDT"))
    updated = make_alert("a1", ts=D1_US + 1, status="updated", fused_p=0.0001)
    store.upsert_alert(updated)
    store.upsert_alert(make_alert("a3", ts=D1_US + 3))
    assert store.get_alert("a1") == updated
    assert [a.alert_id for a in store.list_alerts()] == ["a3", "a2", "a1"]  # newest first
    assert [a.alert_id for a in store.list_alerts(symbol="ETHUSDT")] == ["a2"]
    assert len(store.list_alerts(limit=2)) == 2


@pytest.mark.parametrize("write", ["insert_alert", "upsert_alert"])
def test_non_finite_evidence_is_dropped_on_write(store, write):
    bad = [("nan", float("nan"), 1.0), ("inf", 1.0, float("inf")), ("ok", 2.0, 0.5)]
    getattr(store, write)(make_alert("a1", evidence=bad))
    assert store.get_alert("a1").evidence == [("ok", 2.0, 0.5)]
    assert [a.alert_id for a in store.list_alerts()] == ["a1"]


def test_unreadable_alert_row_costs_only_itself(store, caplog):
    store.insert_alert(make_alert("good", ts=D1_US + 1))
    bad = make_alert("bad", ts=D1_US + 2).model_dump(mode="json")
    bad["evidence"] = [["ret_60s", None, 0.9]]
    store._db().execute(
        "INSERT INTO alerts(alert_id, ts_event_us, symbol, json) VALUES (?, ?, ?, ?)",
        ["bad", D1_US + 2, "BTCUSDT", json.dumps(bad)],
    )
    assert [a.alert_id for a in store.list_alerts()] == ["good"]
    assert store.get_alert("bad") is None
    assert "unreadable alert row" in caplog.text


def test_insert_duplicate_alert_raises(store):
    store.insert_alert(make_alert())
    with pytest.raises(Exception, match="(?i)constraint|duplicate"):
        store.insert_alert(make_alert())


def test_feedback(store):
    store.insert_alert(make_alert())
    store.insert_feedback("a1", "tp", "ann")
    store.insert_feedback("a1", "unsure", "bob")
    fb = store.list_feedback("a1")
    assert [(f["verdict"], f["analyst"]) for f in fb] == [("tp", "ann"), ("unsure", "bob")]
    assert fb[0]["alert_id"] == "a1" and fb[0]["ts_us"] > 1.7e15
    assert store.list_feedback("zzz") == []
    with pytest.raises(ValueError):
        store.insert_feedback("a1", "maybe", "ann")


# ---- runs ----


def test_runs_roundtrip(store):
    r1 = {
        "run_id": "r1",
        "experiment": "main",
        "variant": "fused",
        "seed": 0,
        "created_us": 10,
        "metrics_json": '{"fa_per_symbol_day": 0.5}',
        "git_sha": "abc",
        "provenance": "REAL",
    }
    r2 = {**r1, "run_id": "r2", "experiment": "baselines", "seed": 1, "created_us": 20}
    store.insert_run(r1)
    store.insert_run(r2)
    assert store.list_runs() == [r1, r2]
    assert store.list_runs("baselines") == [r2]
    assert store.list_runs("none") == []
    row = store._con.execute("select metrics_json from runs where run_id='r1'").fetchone()
    assert row[0] == '{"fa_per_symbol_day": 0.5}'


def test_runs_dict_metrics_serialized(store):
    store.insert_run(
        {"run_id": "r1", "experiment": "e", "variant": "v", "seed": 3, "metrics": {"x": 1.5}}
    )
    row = store._con.execute("select metrics_json, created_us from runs").fetchone()
    assert row[0] == '{"x": 1.5}' and row[1] > 1.7e15
    assert store.list_runs()[0]["metrics"] == {"x": 1.5}


# ---- lifecycle ----


def test_close_is_idempotent_and_persists(tmp_path):
    s = Store(tmp_path / "d", tmp_path / "d" / "app.duckdb")
    s.insert_alert(make_alert())
    s.close()
    s.close()
    with Store(tmp_path / "d", tmp_path / "d" / "app.duckdb") as s2:
        assert s2.get_alert("a1") == make_alert()


# ---- alert deletion (seek) ----


def _windowed(alert_id, end, provenance="replay"):
    return make_alert(alert_id, ts=end - 10, window=(end - 10, end), provenance=provenance)


def test_delete_alerts_from_uses_the_window_end_and_the_provenance(store):
    t = D1_US + 1_000
    for a in (
        _windowed("old", t - 1),
        _windowed("at", t),
        _windowed("new", t + 5),
        _windowed("sim", t + 5, "sim"),
        _windowed("live", t + 5, "live"),
    ):
        store.insert_alert(a)
    assert sorted(store.delete_alerts_from("replay", t)) == ["at", "new"]
    assert {a.alert_id for a in store.list_alerts()} == {"old", "sim", "live"}
    assert store.delete_alerts_from("replay", t) == []


def test_delete_alerts_by_id_only_within_the_provenance(store):
    for a in (_windowed("a", D1_US), _windowed("b", D1_US, "sim"), _windowed("c", D1_US, "live")):
        store.insert_alert(a)
    assert store.delete_alerts("replay", ["a", "b", "c", "missing"]) == ["a"]
    assert {a.alert_id for a in store.list_alerts()} == {"b", "c"}
    assert store.delete_alerts("replay", []) == []


@pytest.mark.parametrize("provenance", ["live", "LIVE", "other", ""])
def test_alert_deletion_refuses_live_and_unknown_provenance(store, provenance):
    store.insert_alert(_windowed("c", D1_US, "live"))
    with pytest.raises(ValueError):
        store.delete_alerts_from(provenance, 0)
    with pytest.raises(ValueError):
        store.delete_alerts(provenance, ["c"])
    assert store.get_alert("c") is not None
