"""Auditor health checks: every ok / warn (/ fail) boundary on both sides of its threshold."""

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from scipy.stats import poisson

from idsai.ai.auditor import Auditor, Check, HealthReport
from idsai.config import load_settings
from idsai.contracts.frames import TRADES

DAY_US = 86_400 * 1_000_000
NOW_US = int(datetime(2026, 10, 28, tzinfo=UTC).timestamp() * 1_000_000)


@pytest.fixture
def aud():
    return Auditor(load_settings())


def _json(c: Check) -> dict:
    """The check's dump survives a strict JSON round trip (no NaN, no numpy types)."""
    d = c.model_dump()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    return d


def _trades(rows: list[tuple[str, int]]) -> pl.DataFrame:
    n = len(rows)
    return pl.DataFrame(
        {
            "ts_us": list(range(n)),
            "recv_us": list(range(n)),
            "symbol": [s for s, _ in rows],
            "price": [1.0] * n,
            "qty": [1.0] * n,
            "quote_qty": [1.0] * n,
            "taker_side": ["buy"] * n,
            "trade_id": [t for _, t in rows],
            "n_fills": [1] * n,
            "source": ["archive"] * n,
        },
        schema=TRADES,
    )


# ---- models


def test_health_report_overall_is_the_worst_status():
    ok = Check(name="a", status="ok", value=None, detail_th="ปกติ")
    warn = Check(name="b", status="warn", value=1, detail_th="เตือน")
    fail = Check(name="c", status="fail", value=None, detail_th="ล้มเหลว")
    assert HealthReport(checks=[], ts_us=1).overall == "ok"
    assert HealthReport(checks=[ok], ts_us=1).overall == "ok"
    assert HealthReport(checks=[ok, warn], ts_us=1).overall == "warn"
    assert HealthReport(checks=[fail, ok, warn], ts_us=1).overall == "fail"
    dumped = HealthReport(checks=[ok, warn], ts_us=5).model_dump()
    assert dumped["overall"] == "warn" and dumped["ts_us"] == 5 and len(dumped["checks"]) == 2


# ---- alert_rate: Poisson upper tail p < 0.01 => warn


def test_alert_rate_poisson_tail_boundary(aud):
    # budget 2 x 10 symbol-days: mu = 20; P(X >= 31) = 0.0135, P(X >= 32) = 0.0081
    below = aud.alert_rate(31, 10, 2.0)
    above = aud.alert_rate(32, 10, 2.0)
    assert below.status == "ok" and above.status == "warn"
    assert below.name == "alert_rate"
    v = _json(above)
    assert v["value"]["n_alerts"] == 32 and v["value"]["expected"] == pytest.approx(20.0)
    assert v["value"]["p"] == pytest.approx(poisson.sf(31, 20.0))
    assert below.value["p"] == pytest.approx(poisson.sf(30, 20.0)) and below.value["p"] > 0.01
    assert above.detail_th and below.detail_th


def test_alert_rate_zero_alerts_and_default_budget(aud):
    c = aud.alert_rate(0, 10)
    assert c.status == "ok" and c.value["p"] == pytest.approx(1.0)
    assert c.value["expected"] == pytest.approx(2.0 * 10)  # cfg.chief.budget_per_symbol_day


@pytest.mark.parametrize("symbol_days", [0, -1, 0.0])
def test_alert_rate_without_symbol_days_is_ok_with_note(aud, symbol_days):
    c = aud.alert_rate(50, symbol_days, 2.0)
    assert c.status == "ok" and "ข้อมูลไม่พอ" in c.detail_th
    _json(c)


# ---- calibration: KS vs uniform p < 0.01 => warn; < 200 finite values => ok with note


def test_calibration_uniform_is_ok(aud):
    c = aud.calibration((np.arange(200) + 0.5) / 200)
    assert c.status == "ok" and c.name == "calibration"
    assert c.value["n"] == 200 and c.value["p"] > 0.01
    _json(c)


def test_calibration_skewed_is_warn_with_recalibrate(aud):
    c = aud.calibration(np.full(200, 0.001))
    assert c.status == "warn" and "recalibrate" in c.detail_th
    assert c.value["p"] < 0.01
    _json(c)


def test_calibration_needs_200_finite_values(aud):
    skewed = np.concatenate([np.full(199, 0.001), [np.nan, np.inf, -np.inf]])
    c = aud.calibration(skewed)
    assert c.status == "ok" and c.value["n"] == 199 and "200" in c.detail_th
    assert aud.calibration([]).status == "ok"
    assert aud.calibration(np.full(200, 0.001)).status == "warn"  # one more finite value
    _json(c)


# ---- calibration_resolution: rate = budget / rows; warn < 10 D/(n+1), fail < D/(n+1)


@pytest.mark.parametrize(
    ("budget", "status"),
    [
        (5.0, "ok"),  # rate 0.05 == 10 * 5 / 1000: not strictly below
        (4.99, "warn"),
        (0.5, "warn"),  # rate 0.005 == 5 / 1000: not strictly below the fail floor
        (0.49, "fail"),
    ],
)
def test_calibration_resolution_boundaries(aud, budget, status):
    c = aud.calibration_resolution(999, 5, 100, budget)
    assert c.status == status and c.name == "calibration_resolution"
    v = _json(c)["value"]
    assert v["rate"] == pytest.approx(budget / 100)
    assert v["warn_below"] == pytest.approx(0.05) and v["fail_below"] == pytest.approx(0.005)


def test_calibration_resolution_default_budget(aud):
    # cfg budget 2.0 / 8640 rows (10 s stride) = 2.3e-4; 10 * 4 / 100_001 = 4e-4 => warn
    assert aud.calibration_resolution(100_000, 4, 8640).status == "warn"
    assert aud.calibration_resolution(1_000_000, 4, 8640).status == "ok"


# ---- latency: p99 > 2000 ms => warn


def test_latency_boundary(aud):
    at = aud.latency([2000.0] * 100)
    over = aud.latency([2000.001] * 100)
    assert at.status == "ok" and over.status == "warn" and at.name == "latency"
    v = _json(over)["value"]
    assert v["n"] == 100 and v["p99_ms"] == pytest.approx(2000.001)
    assert v["p50_ms"] == pytest.approx(2000.001)


def test_latency_p99_not_max(aud):
    # one slow sample in 1000 leaves p99 under the threshold
    samples = [100.0] * 999 + [10_000.0]
    c = aud.latency(samples)
    assert c.status == "ok" and c.value["p50_ms"] == pytest.approx(100.0)
    assert c.value["p99_ms"] == pytest.approx(float(np.percentile(samples, 99)))


def test_latency_empty_is_ok_with_note(aud):
    c = aud.latency([])
    assert c.status == "ok" and c.detail_th
    _json(c)


# ---- trade_gaps: non-sim trade_id jumps > 1 per symbol


def test_trade_gaps_consecutive_is_ok(aud):
    c = aud.trade_gaps(_trades([("AAABTC", i) for i in (3, 1, 2, 2, 4)]))  # dup + unsorted
    assert c.status == "ok" and c.name == "trade_gaps"
    assert c.value["n_gaps"] == 0 and c.value["n_missing"] == 0
    _json(c)


def test_trade_gaps_one_missing_id_warns(aud):
    c = aud.trade_gaps(_trades([("AAABTC", i) for i in (1, 2, 4, 5, 9)]))
    assert c.status == "warn"
    v = _json(c)["value"]
    assert v["n_gaps"] == 2 and v["n_missing"] == 1 + 3
    assert v["by_symbol"]["AAABTC"] == {"n_gaps": 2, "n_missing": 4}


def test_trade_gaps_ignores_sim_trades_and_separates_symbols(aud):
    rows = [("AAABTC", i) for i in (10, 11, 12)] + [("AAABTC", -1), ("AAABTC", -5)]
    rows += [("BBBBTC", i) for i in (11, 12, 13)]  # overlaps AAA's ids: still no gap
    c = aud.trade_gaps(_trades(rows))
    assert c.status == "ok" and c.value["n_gaps"] == 0
    rows.append(("BBBBTC", 15))
    c = aud.trade_gaps(_trades(rows))
    assert c.status == "warn" and c.value["by_symbol"]["BBBBTC"]["n_missing"] == 1
    assert c.value["by_symbol"]["AAABTC"]["n_gaps"] == 0


def test_trade_gaps_empty_is_ok(aud):
    assert aud.trade_gaps(_trades([])).status == "ok"
    assert aud.trade_gaps(_trades([("AAABTC", -1), ("AAABTC", -3)])).status == "ok"


# ---- exclusion: frac > 0.02 => warn


@pytest.mark.parametrize(("frac", "status"), [(0.0, "ok"), (0.02, "ok"), (0.0201, "warn")])
def test_exclusion_boundary(aud, frac, status):
    c = aud.exclusion(frac)
    assert c.status == status and c.name == "exclusion" and c.detail_th
    assert _json(c)["value"]["frac"] == pytest.approx(frac)


# ---- staleness: age > max_age_days or config_hash mismatch => warn; unparsable => warn


def _meta(created_at, config_hash="cfg"):
    return {"created_at": created_at, "config_hash": config_hash, "version": "v2"}


def test_staleness_age_boundary_int_us(aud):
    at = aud.staleness(_meta(NOW_US - 30 * DAY_US), NOW_US)
    over = aud.staleness(_meta(NOW_US - 30 * DAY_US - 1), NOW_US)
    assert at.status == "ok" and over.status == "warn" and at.name == "staleness"
    assert _json(at)["value"]["age_days"] == pytest.approx(30.0)


def test_staleness_accepts_registry_iso_string(aud):
    fresh = aud.staleness(_meta("2026-10-27T00:00:00Z"), NOW_US)
    assert fresh.status == "ok" and fresh.value["age_days"] == pytest.approx(1.0)
    assert aud.staleness(_meta("2026-09-28T00:00:00Z"), NOW_US).status == "ok"  # exactly 30
    assert aud.staleness(_meta("2026-09-27T23:59:59Z"), NOW_US).status == "warn"
    naive = aud.staleness(_meta("2026-10-27T00:00:00"), NOW_US)  # naive = UTC
    assert naive.value["age_days"] == pytest.approx(1.0)
    assert aud.staleness(_meta("2026-10-01T00:00:00Z"), NOW_US, max_age_days=7).status == "warn"


def test_staleness_config_hash_mismatch(aud):
    m = _meta("2026-10-27T00:00:00Z", "cfg")
    assert aud.staleness(m, NOW_US, config_hash="cfg").status == "ok"
    c = aud.staleness(m, NOW_US, config_hash="other")
    assert c.status == "warn" and c.value["config_match"] is False and "config" in c.detail_th
    assert aud.staleness(m, NOW_US).value["config_match"] is None
    _json(c)


@pytest.mark.parametrize("created_at", ["yesterday", None, 1.5e15, True, [1]])
def test_staleness_unparsable_is_warn(aud, created_at):
    c = aud.staleness(_meta(created_at), NOW_US)
    assert c.status == "warn" and c.value["age_days"] is None
    _json(c)


def test_staleness_missing_created_at_is_warn(aud):
    assert aud.staleness({"config_hash": "cfg"}, NOW_US).status == "warn"


# ---- warmup: valid >= min => ok; else warn with hours remaining (60 s spacing)


def test_warmup_boundary(aud):
    assert aud.warmup(360, 360).status == "ok"
    c = aud.warmup(359, 360)
    assert c.status == "warn" and c.name == "warmup"
    assert c.value["hours_remaining"] == pytest.approx(60 / 3600)
    assert "0 ชั่วโมง 1 นาที" in c.detail_th
    _json(c)


def test_warmup_from_zero_and_default_min(aud):
    c = aud.warmup(0)  # cfg.features.min_baseline_samples = 360 (6 h)
    assert c.status == "warn" and c.value["hours_remaining"] == pytest.approx(6.0)
    assert "6 ชั่วโมง 0 นาที" in c.detail_th
    assert aud.warmup(400).status == "ok"


# ---- report


def test_report_runs_only_supplied_checks(aud):
    r = aud.report(
        ts_us=123,
        exclusion={"frac": 0.5},
        latency={"samples_ms": [10.0]},
    )
    assert isinstance(r, HealthReport) and r.ts_us == 123
    assert {c.name for c in r.checks} == {"exclusion", "latency"}
    assert r.overall == "warn"
    d = r.model_dump()
    assert json.loads(json.dumps(d, allow_nan=False)) == d


def test_report_all_checks_and_fail_wins(aud):
    r = aud.report(
        ts_us=1,
        alert_rate={"n_alerts": 1, "symbol_days": 1},
        calibration={"pvals": []},
        calibration_resolution={"n_cal": 9, "n_detectors": 5, "rows_per_symbol_day": 100},
        latency={"samples_ms": []},
        trade_gaps={"trades": _trades([])},
        exclusion={"frac": 0.0},
        staleness={"meta": _meta(NOW_US), "now_us": NOW_US},
        warmup={"valid_samples": 360},
    )
    assert len(r.checks) == 8 and r.overall == "fail"


def test_report_without_inputs_is_ok_with_current_time(aud):
    r = aud.report()
    assert r.checks == [] and r.overall == "ok" and r.ts_us > 1_600_000_000 * 1_000_000


def test_report_unknown_check_raises(aud):
    with pytest.raises(ValueError, match="nope"):
        aud.report(nope={})


# ---- unreadable inputs: a monitor that cannot measure never reports health

NAN, INF = float("nan"), float("inf")


def _assert_unreadable(c: Check, *names: str) -> dict:
    assert c.status == "warn", c
    assert c.detail_th.startswith("อ่านค่าไม่ได้") and all(n in c.detail_th for n in names)
    v = _json(c)["value"]
    assert v["unreadable"] == list(names) and all(v[n] is None for n in names)
    return v


@pytest.mark.parametrize(
    ("args", "bad"),
    [
        ((10, NAN, 2.0), "symbol_days"),
        ((10, 5, NAN), "budget_per_symbol_day"),
        ((NAN, 5, 2.0), "n_alerts"),
        ((10, INF, 2.0), "symbol_days"),
        ((10, 5, None), None),  # None budget = the cfg default, not unreadable
    ],
)
def test_alert_rate_non_finite_input_is_warn(aud, args, bad):
    c = aud.alert_rate(*args)
    if bad is None:
        assert c.status == "ok"
        return
    v = _assert_unreadable(c, bad)
    assert sum(x is not None for k, x in v.items() if k != "unreadable") == 2  # raw kept


def test_calibration_drops_non_finite_and_counts_them(aud):
    # D3: non-finite p-values are dropped (not a health claim: "ยังไม่ประเมิน")
    c = aud.calibration([NAN] * 500)
    assert c.status == "ok" and c.value["n"] == 0 and c.value["n_dropped"] == 500
    assert "ยังไม่ประเมิน" in c.detail_th and "500" in c.detail_th
    c = aud.calibration(np.concatenate([(np.arange(200) + 0.5) / 200, [NAN, INF]]))
    assert c.value["n"] == 200 and c.value["n_dropped"] == 2
    _json(c)


@pytest.mark.parametrize("bad", ["n_cal", "n_detectors", "rows_per_symbol_day", "budget"])
def test_calibration_resolution_non_finite_input_is_warn(aud, bad):
    kw = {"n_cal": 999, "n_detectors": 5, "rows_per_symbol_day": 100.0, "budget": 5.0, bad: NAN}
    v = _assert_unreadable(aud.calibration_resolution(**kw), bad)
    assert v["n_cal" if bad != "n_cal" else "n_detectors"] is not None


@pytest.mark.parametrize(("n_cal", "n_detectors"), [(-1, 5), (999, 0), (999, -2)])
def test_calibration_resolution_invalid_counts_raise(aud, n_cal, n_detectors):
    with pytest.raises(ValueError):
        aud.calibration_resolution(n_cal, n_detectors, 100)


def test_calibration_resolution_zero_rows_stays_ok_with_note(aud):
    c = aud.calibration_resolution(999, 5, 0)
    assert c.status == "ok" and "ข้อมูลไม่พอ" in c.detail_th


def test_latency_inf_counts_and_nan_is_dropped(aud):
    c = aud.latency([100.0] * 98 + [INF] * 2)  # p99 interpolates into the inf tail
    assert c.status == "warn" and "inf" in c.detail_th
    v = _json(c)["value"]
    assert v["p99_ms"] is None and v["n_inf"] == 2 and v["p50_ms"] == pytest.approx(100.0)
    c = aud.latency([100.0] * 999 + [INF])  # p99 stays among the finite samples
    assert c.status == "ok" and c.value["p99_ms"] == pytest.approx(100.0)
    c = aud.latency([100.0] * 10 + [NAN] * 50)
    assert c.status == "ok" and c.value["n"] == 10
    c = aud.latency([NAN, NAN])
    assert c.status == "ok" and c.value["n"] == 0


def test_trade_gaps_null_ids_are_warn(aud):
    df = _trades([("AAABTC", 1), ("AAABTC", 2), ("AAABTC", 3)])
    df = df.with_columns(pl.Series("trade_id", [1, None, 3], dtype=pl.Int64))
    c = aud.trade_gaps(df)
    assert c.status == "warn" and c.detail_th.startswith("อ่านค่าไม่ได้")
    assert _json(c)["value"]["n_unreadable"] == 1


@pytest.mark.parametrize("frac", [NAN, INF, -INF, None, "x"])
def test_exclusion_non_finite_is_warn(aud, frac):
    _assert_unreadable(aud.exclusion(frac), "frac")


@pytest.mark.parametrize("bad", ["now_us", "max_age_days"])
def test_staleness_non_finite_input_is_warn(aud, bad):
    kw = {"now_us": NOW_US, "max_age_days": 30, bad: NAN}
    _assert_unreadable(aud.staleness(_meta(NOW_US), **kw), bad)


def test_staleness_future_created_at_is_warn(aud):
    assert aud.staleness(_meta(NOW_US), NOW_US).status == "ok"  # age 0
    c = aud.staleness(_meta(NOW_US + 1), NOW_US)
    assert c.status == "warn" and "อนาคต" in c.detail_th and c.value["age_days"] < 0
    c = aud.staleness(_meta("2026-11-05T00:00:00Z"), NOW_US)
    assert c.status == "warn" and "อนาคต" in c.detail_th


@pytest.mark.parametrize(
    ("valid", "minimum", "bad"), [(NAN, 360, "valid_samples"), (10, INF, "min_samples")]
)
def test_warmup_non_finite_input_is_warn(aud, valid, minimum, bad):
    _assert_unreadable(aud.warmup(valid, minimum), bad)


def test_report_with_unreadable_input_is_not_ok(aud):
    r = aud.report(ts_us=1, exclusion={"frac": NAN}, latency={"samples_ms": [1.0]})
    assert r.overall == "warn"
