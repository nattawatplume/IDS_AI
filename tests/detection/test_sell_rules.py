"""M4 sell-side signatures, alert phase, explainer line (synthetic only)."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from idsai.ai.chief import Chief
from idsai.config import load_settings
from idsai.contracts.alert import Alert, phase_of
from idsai.contracts.frames import TRADES
from idsai.detection.features.bars import trades_to_bars
from idsai.detection.features.windows import compute_windows
from idsai.detection.registry import build_detectors, rule_signatures, sell_rule_detectors
from idsai.detection.rules import SELL_SIGNATURES_PATH, SignatureDetector, load_signatures
from idsai.detection.taxonomy import load_taxonomy, technique_ids
from idsai.pipeline.batch import score_windows
from idsai.product.explainer.template import explain, numbers_in
from tests.synth import day_us, make_trades

S = 1_000_000
SYM = "TESTBTC"
D = date(2025, 3, 1)
T0 = day_us(D)
UNIT = 1e-4
CFG = load_settings()
ON = load_settings(overrides={"features": {"sell_side": True}})
EVENT_US = T0 + 26 * 3600 * S
DUMP, RAID = 100104, 100601


def _fills(ts, price, qty, side, first_id) -> pl.DataFrame:
    n = len(ts)
    price = np.broadcast_to(np.asarray(price, dtype=float), (n,))
    return pl.DataFrame(
        {
            "ts_us": np.asarray(ts, dtype=np.int64),
            "recv_us": np.asarray(ts, dtype=np.int64),
            "symbol": [SYM] * n,
            "price": price,
            "qty": np.full(n, float(qty)),
            "quote_qty": price * qty,
            "taker_side": [side] * n,
            "trade_id": np.arange(first_id, first_id + n, dtype=np.int64),
            "n_fills": np.ones(n, dtype=np.int32),
            "source": ["archive"] * n,
        },
        schema=TRADES,
    )


def _move(t_start, seconds, p0, p1, side, first_id, k=3, qty=2000.0) -> pl.DataFrame:
    """`k` fills sharing one millisecond every second, price walking from p0 to p1."""
    t = t_start + np.arange(seconds, dtype=np.int64) * S
    ts = (np.repeat(t, k) + np.tile(np.arange(k) * 10, seconds)).astype(np.int64)
    price = np.round(np.repeat(np.linspace(p0, p1, seconds), k), 7)
    return _fills(ts, price, qty, side, first_id)


def _day(rise: bool) -> pl.DataFrame:
    """27 h of quiet trades; at 26 h an optional 5 min rise of 10 %, then a 60 s crash of 8 %."""
    base = make_trades(D, 2, symbol=SYM, seed=7, rate=0.3)
    base = base.filter(pl.col("ts_us") < T0 + 27 * 3600 * S)
    # later quiet trades must continue from the moved price: drop them after the event
    base = base.filter(
        (pl.col("ts_us") < EVENT_US - 600 * S) | (pl.col("ts_us") > EVENT_US + 600 * S)
    )
    p0 = float(base.filter(pl.col("ts_us") < EVENT_US - 600 * S)["price"][-1])
    parts = [base.filter(pl.col("ts_us") < EVENT_US - 600 * S)]
    top = p0
    t = EVENT_US - 600 * S
    if rise:
        top = p0 * 1.10
        parts.append(_move(t, 300, p0, top, "buy", 50_000_000))
    crash_start = EVENT_US
    parts.append(_move(crash_start, 60, top, top * 0.92, "sell", 60_000_000))
    return pl.concat(parts).sort(["ts_us", "trade_id"])


def _windows(trades: pl.DataFrame, cfg=ON) -> pl.DataFrame:
    end = T0 + 27 * 3600 * S
    return compute_windows(trades_to_bars(trades, SYM, T0, end), cfg.features, UNIT)


@pytest.fixture(scope="module")
def dump_windows() -> pl.DataFrame:
    return _windows(_day(rise=True))


@pytest.fixture(scope="module")
def raid_windows() -> pl.DataFrame:
    return _windows(_day(rise=False))


def _fired(windows: pl.DataFrame, sid: int) -> pl.DataFrame:
    (det,) = [d for d in sell_rule_detectors() if d.signature_id == sid]
    return windows.filter(pl.Series(det.fired(det.score(windows))))


def test_sell_signature_files_and_taxonomy():
    sell = load_signatures(SELL_SIGNATURES_PATH)
    assert {s.sid: s.technique for s in sell} == {DUMP: "MM-T001.003", RAID: "MM-T006.001"}
    assert all(s.rev == 1 and s.layer == "rule" for s in sell)
    ids = technique_ids()
    assert {"MM-T001.003", "MM-T006", "MM-T006.001"} <= ids
    assert load_taxonomy()["MM-T006"]["name_th"] == "ทุบราคา (bear raid)"
    # off: only signatures.yaml; on: both files, sell rules last
    off = [s.sid for s in rule_signatures(CFG)]
    on = [s.sid for s in rule_signatures(ON)]
    assert DUMP not in off and on == [*off, DUMP, RAID]
    assert [d.name for d in build_detectors(["rules"], ON)][-2:] == ["rule.100104", "rule.100601"]
    assert SignatureDetector(sell[0]).technique == "MM-T001.003"


def test_dump_rule_fires_after_a_rise_and_not_on_quiet_rows(dump_windows):
    hit = _fired(dump_windows, DUMP)
    assert hit.height > 0
    t = hit["t_end_us"].to_numpy()
    assert ((t > EVENT_US) & (t <= EVENT_US + 120 * S)).all()
    assert _fired(dump_windows, RAID).height == 0  # there was a rise: not a bear raid
    quiet = dump_windows.filter(pl.col("t_end_us") < EVENT_US - 700 * S)
    assert _fired(quiet, DUMP).height == 0 and _fired(quiet, RAID).height == 0


def test_bear_raid_rule_fires_on_a_crash_without_a_prior_rise(raid_windows):
    hit = _fired(raid_windows, RAID)
    assert hit.height > 0
    t = hit["t_end_us"].to_numpy()
    assert ((t > EVENT_US) & (t <= EVENT_US + 120 * S)).all()
    assert _fired(raid_windows, DUMP).height == 0


def test_pump_rule_unchanged_by_the_flag(dump_windows):
    off = _windows(_day(rise=True), CFG)
    (burst,) = [d for d in build_detectors(["rules"], CFG) if d.signature_id == 100101]
    a, b = burst.score(off), burst.score(dump_windows)
    assert np.array_equal(a, b, equal_nan=True)


def test_phase_of():
    assert phase_of("MM-T001.001") == phase_of("MM-T001.002") == "pump"
    assert phase_of("MM-T001.003") == phase_of("MM-T006.001") == phase_of("MM-T006") == "sell"
    assert (
        phase_of("MM-T001") is None and phase_of("MM-T005.001") is None and phase_of(None) is None
    )


def _chief_alerts(windows: pl.DataFrame, cfg) -> list[Alert]:
    # no stat.cusum: its MM-T001 rows would out-rank the rule rows of a merged episode
    dets = build_detectors(["rules", "stat.robust_z"], cfg)
    scored = score_windows(windows, dets)
    cols = ["symbol", "t_end_us", *[f"score__{d.name}" for d in dets]]
    cal = scored.filter(pl.col("t_end_us") < EVENT_US - 3600 * S)
    chief = Chief(dets, cfg.chief, "h", {})
    chief.calibrate(cal.select(cols))
    chief.fit_threshold(chief.fuse(cal.select(cols)), 10.0)
    ev = scored.filter(pl.col("t_end_us") >= EVENT_US - 3600 * S)
    return chief.episodes(chief.fuse(ev))


SIG_TECH = {DUMP: "MM-T001.003", RAID: "MM-T006.001", 100101: "MM-T001.001", 100102: "MM-T001.002"}


def test_chief_sets_phase_only_with_sell_rules(dump_windows, raid_windows):
    raid = _chief_alerts(raid_windows, ON)
    crash = [a for a in raid if a.window[0] <= EVENT_US + 120 * S and a.window[1] >= EVENT_US]
    assert crash and crash[0].phase == "sell" and crash[0].signature_id == RAID
    assert crash[0].technique_id == "MM-T006"
    # rise then crash in one episode: the episode names its highest-F classified row (here the
    # stealth pump rule), so its phase is that row's, "pump"
    dump = _chief_alerts(dump_windows, ON)
    assert dump
    for a in [*raid, *dump]:
        assert a.phase == phase_of(SIG_TECH.get(a.signature_id))
    off = _chief_alerts(_windows(_day(rise=True), CFG), CFG)
    assert off and all(a.phase is None for a in off)
    assert all('"phase"' not in a.model_dump_json() for a in off)


def _alert(**kw) -> Alert:
    base = {
        "alert_id": "replay-TESTBTC-1",
        "ts_event_us": 3_600_000_000,
        "symbol": "TESTBTC",
        "technique_id": "MM-T001",
        "peak_technique_id": "MM-T001",
        "signature_id": DUMP,
        "rev": 1,
        "severity": 2,
        "status": "closed",
        "fused_p": 0.001,
        "detectors": [],
        "agreement": 2,
        "evidence": [("z_sell_rush_60", 7.5, 7.5), ("drop_60", -0.08, 1.0)],
        "window": (3_600_000_000, 3_660_000_000),
        "dedup_key": "TESTBTC|1",
        "suppress_for_s": 300,
        "config_hash": "abc",
        "model_versions": {},
        "provenance": "replay",
    }
    return Alert(**{**base, **kw})


def test_old_alert_json_without_phase_still_loads_and_round_trips():
    old = _alert().model_dump_json()
    assert '"phase"' not in old
    a = Alert.model_validate_json(old)
    assert a.phase is None and a.model_dump_json() == old
    sell = _alert(phase="sell")
    assert '"phase":"sell"' in sell.model_dump_json()
    assert Alert.model_validate_json(sell.model_dump_json()).phase == "sell"


def test_explainer_phase_line():
    assert "ช่วง:" not in explain(_alert()).text_th
    t = explain(_alert(phase="sell")).text_th
    assert t.endswith("; ช่วง: กำลังเทขาย")
    assert explain(_alert(phase="pump")).text_th.endswith("; ช่วง: กำลังพุ่ง")
    assert "ราคาต่ำสุดในหน้าต่าง" in t and "(ช่วง 60 วินาที)" in t
    assert numbers_in("ช่วง: กำลังเทขาย") == []


def test_sell_signature_features_exist_only_with_the_flag():
    from idsai.detection.features.registry import feature_columns

    on, off = set(feature_columns(ON.features)), set(feature_columns(CFG.features))
    needs = {c.feature for s in load_signatures(SELL_SIGNATURES_PATH) for c in s.conditions}
    assert needs <= on and not needs <= off
