"""M4 sell-phase labels: sell windows from synthetic measurements and the per-phase recall."""

import math
from datetime import date

import numpy as np
import polars as pl

from idsai.contracts.alert import Alert
from idsai.contracts.frames import LABELS
from idsai.data.store import Store
from idsai.detection.simulator import Scenario, base_quote_rate, inject
from idsai.eval.pump_measure import measure_label_store
from idsai.eval.sell_labels import SellPhase, derive_sell_labels, sell_window
from tests.synth import S, day_us, make_trades

D = date(2025, 3, 1)


def test_sell_window_from_a_measurement():
    m = {"status": "measured", "t_peak_us": 1_000 * S, "dump_tau_s": 300.0}
    assert sell_window(m) == (1_000 * S, 1_300 * S)
    assert sell_window({**m, "dump_tau_s": 5.0}) == (1_000 * S, 1_060 * S)  # at least 60 s
    assert sell_window({**m, "dump_tau_s": 9_000.0}) == (1_000 * S, 4_600 * S)  # at most 1 h
    assert sell_window({**m, "dump_tau_s": None}) == (1_000 * S, 4_600 * S)  # censored: 1 h
    assert sell_window({"status": "no_rise", "t_peak_us": 5}) is None


def _alert(symbol, lo, hi, phase=None) -> Alert:
    return Alert(
        alert_id=f"a{lo}",
        ts_event_us=lo,
        symbol=symbol,
        technique_id="MM-T001",
        severity=1,
        status="closed",
        fused_p=0.01,
        detectors=[],
        agreement=1,
        evidence=[],
        window=(lo, hi),
        dedup_key="k",
        suppress_for_s=300,
        config_hash="h",
        model_versions={},
        provenance="replay",
        phase=phase,
    )


def test_sell_phase_per_event_and_summary():
    labels = pl.DataFrame(
        [
            ("A", "hu", "X", 0, 0, "MM-T001", "p", None, "REAL"),
            ("B", "hu", "X", 0, 0, "MM-T001", "p", None, "REAL"),
            ("C", "hu", "Y", 0, 0, "MM-T001", "p", None, "REAL"),
        ],
        schema=LABELS,
        orient="row",
    )
    # windows in µs: credit only rows with t_end in (t_peak + 60 s, hi]
    sell = {
        "A": {"status": "measured", "sell_window": [0, 300 * S], "dump_censored": False},
        "B": {"status": "measured", "sell_window": [1000 * S, 2000 * S], "dump_censored": True},
        "C": {"status": "no_rise", "sell_window": None, "dump_censored": False},
    }
    ph = SellPhase(sell)
    alerts = [
        _alert("X", 70 * S, 200 * S, "sell"),  # fires during A's fall
        _alert("X", 900 * S, 1050 * S),  # B: its rows end at t_peak + 50 s: a pump tail
        _alert("Y", 0, 10**15),
    ]
    exceed = {"X": np.array([70, 200, 900, 1050], dtype=np.int64) * S, "Y": np.array([0])}
    pe = ph.per_event(alerts, labels, exceed)
    assert pe["A"]["detected_sell_phase"] and pe["A"]["detected_sell_phase_correct"]
    assert not pe["B"]["detected_sell_phase"] and not pe["B"]["detected_sell_phase_correct"]
    assert pe["B"]["dump_censored"] is True and pe["B"]["sell_window_short"] is False
    assert pe["C"]["detected_sell_phase"] is None and pe["C"]["sell_status"] == "no_rise"
    # the same B with a row inside its fall is credited (any technique), but not as "correct"
    pe2 = ph.per_event(alerts[:1] + [_alert("X", 900 * S, 1100 * S)], labels,
                       {"X": np.array([70, 1100]) * S})  # fmt: skip
    assert pe2["B"]["detected_sell_phase"] and not pe2["B"]["detected_sell_phase_correct"]
    pe = pe2
    full = {k: {**v, "detected_any": k != "B"} for k, v in pe.items()}
    m = ph.summarize(full)
    assert m["recall_pump_phase"] == 2 / 3
    assert m["recall_sell_phase"] == 1.0 and m["recall_sell_phase_correct"] == 0.5
    short = SellPhase({"A": {"status": "measured", "sell_window": [0, 60 * S]}})
    got = short.per_event(alerts[:1], labels.head(1), {"X": np.array([30, 60]) * S})["A"]
    assert got["sell_window_short"] is True and got["detected_sell_phase"] is False
    assert m["n_sell_events"] == 2
    assert m["sell_status_counts"] == {"measured": 2, "no_rise": 1}
    assert math.isnan(ph.summarize({})["recall_sell_phase"])


def test_derive_sell_labels_from_a_synthetic_measured_pump(tmp_path):
    store = Store(tmp_path / "data", tmp_path / "data" / "app.duckdb")
    try:
        store.write_trades(make_trades(D, 3, symbol="AAABTC", seed=3, rate=0.2))
        t0 = day_us(date(2025, 3, 2)) + 12 * 3600 * S
        sc = Scenario("pump_burst", "MM-T001", "MM-T001.001", t0, 120.0, 0.2, 30.0, 5, 300.0, 1,
                      "train")  # fmt: skip
        day = store.read_trades("AAABTC", day_us(date(2025, 3, 2)), day_us(date(2025, 3, 3)))
        prev = store.read_trades("AAABTC", day_us(D), day_us(date(2025, 3, 2)))
        new, _ = inject(day, sc, base_quote_rate(prev), mode="replay")
        store.write_trades(new)
        labels = pl.DataFrame(
            [
                ("P", "hu", "AAABTC", t0, t0, "MM-T001", "p", None, "REAL"),
                ("Q", "hu", "AAABTC", day_us(D) + 3600 * S, 0, "MM-T001", "p", None, "REAL"),
            ],
            schema=LABELS,
            orient="row",
        )
        out = derive_sell_labels(store, labels)
        assert out == derive_sell_labels(store, labels)  # deterministic
        assert out["Q"] == {"status": "no_baseline", "sell_window": None, "dump_censored": False}
        p = out["P"]
        m = measure_label_store(store, "AAABTC", t0)
        assert p["status"] == m["status"] == "measured"
        lo, hi = p["sell_window"]
        assert lo == m["t_peak_us"]
        assert abs(lo - (t0 + 120 * S)) <= 2 * S  # the injected peak
        assert 60 * S <= hi - lo <= 3600 * S
        assert np.isclose((hi - lo) / S, max(60.0, m["dump_tau_s"]))
    finally:
        store.close()
