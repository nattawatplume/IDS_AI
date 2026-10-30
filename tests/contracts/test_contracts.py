import polars as pl
import pytest
from pydantic import ValidationError

from idsai.contracts import frames
from idsai.contracts.alert import Alert, DetectorRef
from idsai.contracts.events import Trade


def valid_alert_kwargs(**over):
    kw = dict(
        alert_id="a1",
        ts_event_us=1_700_000_000_000_000,
        symbol="BTCUSDT",
        technique_id="MM-T001",
        signature_id=None,
        rev=None,
        severity=2,
        status="open",
        fused_p=0.004,
        detectors=[DetectorRef(name="rules", layer="rule", p=0.01, score=3.2)],
        agreement=1,
        evidence=[("ret_z_60", 5.1, 0.002)],
        window=(1_699_999_990_000_000, 1_700_000_000_000_000),
        dedup_key="BTCUSDT:MM-T001",
        suppress_for_s=300,
        config_hash="abc123abc123",
        model_versions={"ae": "v1"},
        provenance="replay",
    )
    kw.update(over)
    return kw


def valid_trade_kwargs(**over):
    kw = dict(
        ts_us=1,
        recv_us=2,
        symbol="BTCUSDT",
        price=1.0,
        qty=2.0,
        quote_qty=2.0,
        taker_side="buy",
        trade_id=7,
        n_fills=1,
        source="live",
    )
    kw.update(over)
    return kw


def test_alert_valid_and_defaults():
    a = Alert(**valid_alert_kwargs())
    assert a.schema_v == 1 and a.signature_id is None and a.rev is None
    assert a.detectors[0].layer == "rule"


def test_alert_bounds():
    with pytest.raises(ValidationError):
        Alert(**valid_alert_kwargs(severity=4))
    with pytest.raises(ValidationError):
        Alert(**valid_alert_kwargs(severity=0))
    with pytest.raises(ValidationError):
        Alert(**valid_alert_kwargs(agreement=4))
    with pytest.raises(ValidationError):
        Alert(**valid_alert_kwargs(status="pending"))
    with pytest.raises(ValidationError):
        Alert(**valid_alert_kwargs(provenance="REAL"))


def test_trades_schema_exact():
    assert dict(frames.TRADES) == {
        "ts_us": pl.Int64,
        "recv_us": pl.Int64,
        "symbol": pl.String,
        "price": pl.Float64,
        "qty": pl.Float64,
        "quote_qty": pl.Float64,
        "taker_side": pl.String,
        "trade_id": pl.Int64,
        "n_fills": pl.Int32,
        "source": pl.String,
    }


def test_labels_schema_exact():
    assert list(frames.LABELS) == [
        "label_id",
        "source",
        "symbol",
        "t_start_us",
        "t_end_us",
        "technique_id",
        "family",
        "scenario_seed",
        "provenance",
    ]
    assert frames.LABELS["scenario_seed"] == pl.Int64


def test_empty_has_schema():
    df = frames.empty(frames.TRADES)
    assert df.height == 0 and df.schema == frames.TRADES


def test_conform_casts_orders_and_keeps_nulls():
    df = pl.DataFrame(
        {
            "provenance": ["REAL"],
            "label_id": ["l1"],
            "source": ["sapienza"],
            "symbol": ["BRDBTC"],
            "t_start_us": [1],
            "t_end_us": [2],
            "technique_id": ["MM-T001"],
            "family": ["pump"],
            "scenario_seed": [None],
            "extra": [1],
        }
    )
    out = frames.conform(df, frames.LABELS)
    assert out.schema == frames.LABELS and out.columns == list(frames.LABELS)
    assert out["scenario_seed"].to_list() == [None]


def test_conform_rejects_missing_column():
    df = frames.empty(frames.TRADES).drop("price")
    with pytest.raises(frames.SchemaError, match="price"):
        frames.conform(df, frames.TRADES)


def test_trade_model_mirrors_trades():
    t = Trade(**valid_trade_kwargs())
    assert set(Trade.model_fields) == set(frames.TRADES)
    df = pl.DataFrame([t.model_dump()])
    assert frames.conform(df, frames.TRADES).schema == frames.TRADES
    with pytest.raises(ValidationError):
        Trade(**valid_trade_kwargs(taker_side="maker"))
    with pytest.raises(ValidationError):
        Trade(**valid_trade_kwargs(source="csv"))


def test_alert_without_peak_technique_id_still_loads():
    from idsai.contracts.alert import Alert

    data = {
        "alert_id": "a",
        "ts_event_us": 1,
        "symbol": "S",
        "technique_id": "MM-T001",
        "severity": 1,
        "status": "open",
        "fused_p": 0.5,
        "detectors": [],
        "agreement": 0,
        "evidence": [],
        "window": [1, 2],
        "dedup_key": "k",
        "suppress_for_s": 60,
        "config_hash": "h",
        "model_versions": {},
        "provenance": "sim",
        "schema_v": 1,
    }
    a = Alert.model_validate(data)
    assert a.peak_technique_id is None and a.schema_v == 1
