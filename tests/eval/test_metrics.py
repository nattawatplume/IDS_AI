import math

import numpy as np
import polars as pl
import pytest

from idsai.contracts.alert import Alert
from idsai.contracts.frames import LABELS
from idsai.eval.metrics import (
    bootstrap_ci,
    match_events,
    paired_bootstrap,
    ratio_bootstrap_ci,
    significant,
    summarize,
    window_positive,
    window_pr_auc,
)

S = 1_000_000
T0 = 1_600_000_000 * S
SYM = "ABCBTC"


def alert(first_us, last_us, technique="MM-T001", symbol=SYM) -> Alert:
    return Alert(
        alert_id=f"{symbol}-{first_us}",
        ts_event_us=first_us,
        symbol=symbol,
        technique_id=technique,
        severity=1,
        status="closed",
        fused_p=0.001,
        detectors=[],
        agreement=1,
        evidence=[],
        window=(first_us, last_us),
        dedup_key="k",
        suppress_for_s=300,
        config_hash="c",
        model_versions={},
        provenance="replay",
    )


def labels(*rows) -> pl.DataFrame:
    """rows: (label_id, t_start_us, t_end_us, technique[, symbol])"""
    out = []
    for r in rows:
        lid, t0, t1, tech = r[:4]
        sym = r[4] if len(r) > 4 else SYM
        out.append((lid, "hu", sym, t0, t1, tech, "telegram_pump", None, "REAL"))
    return pl.DataFrame(out, schema=LABELS, orient="row")


ONE = labels(("e1", T0, T0, "MM-T001"))


def match(alerts, labs=ONE):
    return match_events(alerts, labs, before_s=120, after_s=300)


# ---------------------------------------------------------------- overlap edges
def test_episode_ending_121s_before_misses():
    mr = match([alert(T0 - 200 * S, T0 - 121 * S)])
    assert mr.detected_any == {"e1": False} and mr.n_alerts_matched == 0
    assert mr.ttd_s == {}


def test_episode_ending_119s_before_hits():
    mr = match([alert(T0 - 200 * S, T0 - 119 * S)])
    assert mr.detected_any == {"e1": True} and mr.n_alerts_matched == 1


def test_closed_interval_edges():
    assert match([alert(T0 - 300 * S, T0 - 120 * S)]).detected_any["e1"]
    assert match([alert(T0 + 300 * S, T0 + 900 * S)]).detected_any["e1"]
    assert not match([alert(T0 + 301 * S, T0 + 900 * S)]).detected_any["e1"]


def test_long_early_episode_hits_and_is_not_a_false_alert():
    mr = match([alert(T0 - 600 * S, T0 + 30 * S)])
    assert mr.detected_any["e1"] and mr.n_alerts == 1 and mr.n_alerts_matched == 1
    m = summarize(mr, symbol_days=1, n_windows_pos=1, n_windows=10)
    assert m["fa_per_symbol_day"] == 0.0 and m["precision"] == 1.0
    assert mr.ttd_s["e1"] == -600.0


def test_other_symbol_never_matches():
    mr = match([alert(T0, T0, symbol="XYZBTC")])
    assert not mr.detected_any["e1"] and mr.n_alerts_matched == 0


def test_one_episode_matching_two_labels_counts_once():
    labs = labels(("a", T0, T0, "MM-T001"), ("b", T0 + 60 * S, T0 + 60 * S, "MM-T001"))
    mr = match([alert(T0, T0 + 10 * S)], labs)
    assert mr.detected_any == {"a": True, "b": True}
    assert mr.n_alerts == 1 and mr.n_alerts_matched == 1 and mr.n_events == 2


# ---------------------------------------------------------------- correct vs any
def test_unclassified_counts_only_for_any():
    mr = match([alert(T0, T0 + 10 * S, technique="MM-T000")])
    assert mr.detected_any["e1"] and not mr.detected_correct["e1"]


def test_correct_uses_label_parent_technique():
    far = T0 + 10_000 * S
    labs = labels(("p", T0, T0, "MM-T001.002"), ("w", far, far, "MM-T005"))
    mr = match([alert(T0, T0, "MM-T001"), alert(far, far, "MM-T001")], labs)
    assert mr.detected_correct == {"p": True, "w": False}
    assert mr.detected_any == {"p": True, "w": True}


# ---------------------------------------------------------------- summary
def test_fa_per_symbol_day_units_and_precision_with_prevalence():
    alerts = [alert(T0, T0)] + [alert(T0 + k * 3600 * S, T0 + k * 3600 * S) for k in (2, 3, 4)]
    m = summarize(match(alerts), symbol_days=2.0, n_windows_pos=5, n_windows=1000)
    assert m["fa_per_symbol_day"] == pytest.approx(3 / 2)  # 3 unmatched episodes over 2 days
    assert m["precision"] == pytest.approx(1 / 4)
    assert m["prevalence"] == pytest.approx(5 / 1000)
    assert m["recall_any"] == 1.0 and m["recall_correct"] == 1.0
    assert m["f1"] == pytest.approx(2 * 0.25 / 1.25)
    assert set(m) == {
        "recall_any",
        "recall_correct",
        "precision",
        "prevalence",
        "fa_per_symbol_day",
        "ttd_median_s",
        "ttd_p90_s",
        "f1",
    }


def test_negative_ttd_from_earliest_matching_episode():
    mr = match([alert(T0 + 60 * S, T0 + 70 * S), alert(T0 - 60 * S, T0 - 50 * S)])
    assert mr.ttd_s["e1"] == -60.0
    m = summarize(mr, 1, 1, 1)
    assert m["ttd_median_s"] == -60.0 and m["ttd_p90_s"] == -60.0


def test_ttd_quantiles_over_detected_events_only():
    ts = [T0 + i * 10_000 * S for i in range(4)]
    labs = labels(*[(f"e{i}", t, t, "MM-T001") for i, t in enumerate(ts)])
    alerts = [alert(ts[i] + d * S, ts[i] + d * S) for i, d in [(0, 0), (1, 10), (2, 20)]]
    m = summarize(match(alerts, labs), 1, 1, 1)
    assert m["recall_any"] == 0.75
    assert m["ttd_median_s"] == 10.0 and m["ttd_p90_s"] == pytest.approx(18.0)


def test_empty_cases_are_nan_not_errors():
    m = summarize(match([]), symbol_days=1, n_windows_pos=0, n_windows=0)
    assert m["recall_any"] == 0.0 and m["fa_per_symbol_day"] == 0.0
    assert math.isnan(m["precision"]) and math.isnan(m["prevalence"])
    assert math.isnan(m["ttd_median_s"]) and math.isnan(m["f1"])
    none = summarize(match([alert(T0, T0)], labels()), symbol_days=0, n_windows_pos=0, n_windows=5)
    assert math.isnan(none["recall_any"]) and math.isnan(none["fa_per_symbol_day"])
    assert none["precision"] == 0.0 and none["prevalence"] == 0.0


# ---------------------------------------------------------------- windows
def test_window_positive_uses_matching_interval():
    t = np.array([T0 - 121 * S, T0 - 120 * S, T0, T0 + 300 * S, T0 + 301 * S, T0])
    sym = np.array([SYM] * 5 + ["XYZBTC"], dtype=object)
    y = window_positive(sym, t, ONE, 120, 300)
    assert y.tolist() == [False, True, True, True, False, False]


def test_pr_auc_toy():
    F = np.array([0.1, 0.4, 0.35, 0.8])
    y = np.array([0, 0, 1, 1], dtype=bool)
    assert window_pr_auc(F, y) == pytest.approx(0.8333333333)
    assert math.isnan(window_pr_auc(F, np.zeros(4, dtype=bool)))
    assert math.isnan(window_pr_auc(F, np.ones(4, dtype=bool)))
    assert window_pr_auc(np.array([np.nan, 1.0]), np.array([False, True])) == 1.0  # NaN lowest


# ---------------------------------------------------------------- bootstrap
def test_paired_bootstrap_equal_arrays_gives_zero_ci():
    rng = np.random.default_rng(1)
    a = (rng.random((5, 40)) < 0.5).astype(float)
    diff, ci = paired_bootstrap(a, a.copy(), n_boot=1000, seed=0)
    assert diff == 0.0 and ci == (0.0, 0.0)


def test_paired_bootstrap_uses_identical_indices():
    b = np.random.default_rng(2).random((3, 50))
    diff, ci = paired_bootstrap(b + 0.25, b, n_boot=500, seed=7)
    # independent resampling of the two arms would spread the CI around 0.25
    assert diff == pytest.approx(0.25)
    assert ci == pytest.approx((0.25, 0.25))


def test_paired_bootstrap_clear_gain_and_shape_check():
    b = np.zeros((5, 100))
    a = b.copy()
    a[:, :30] = 1.0
    diff, ci = paired_bootstrap(a, b, n_boot=1000, seed=0)
    assert diff == pytest.approx(0.3) and 0.15 < ci[0] < 0.3 < ci[1] < 0.45
    assert significant(diff, ci, 0.02)
    assert not significant(0.01, (0.005, 0.015), 0.02)
    assert not significant(0.1, (-0.01, 0.2), 0.02)
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros((2, 3)), np.zeros((2, 4)), 10, 0)


def test_bootstrap_ci_mean_over_seeds_and_determinism():
    per = np.array([[1, 0, 1, 1], [1, 0, 0, 1]], dtype=float)
    mean, ci = bootstrap_ci(per, n_boot=200, seed=3)
    assert mean == pytest.approx(0.625)
    assert ci == bootstrap_ci(per, n_boot=200, seed=3)[1]
    assert ci[0] <= mean <= ci[1]
    assert bootstrap_ci(np.ones(10), 100, 0) == (1.0, (1.0, 1.0))
    m, (lo, hi) = bootstrap_ci(np.array([]), 100, 0)
    assert math.isnan(m) and math.isnan(lo) and math.isnan(hi)


def test_ratio_bootstrap_ci():
    hits, n = np.array([3.0, 0.0, 1.0]), np.array([3.0, 5.0, 2.0])
    mean, (lo, hi) = ratio_bootstrap_ci(hits, n, 500, 0)
    assert mean == pytest.approx(0.4) and lo <= mean <= hi
    assert ratio_bootstrap_ci(hits, n, 500, 0) == (mean, (lo, hi))
    # equal cluster sizes: identical to the event bootstrap of cluster means
    per = np.array([[1, 0, 1, 1], [1, 0, 0, 1]], dtype=float)
    r = ratio_bootstrap_ci(per.sum(axis=0), np.full(4, 2.0), 200, 3)
    b = bootstrap_ci(per, 200, 3)
    assert r[0] == pytest.approx(b[0]) and r[1] == pytest.approx(b[1])
    m, (a, z) = ratio_bootstrap_ci(np.array([]), np.array([]), 10, 0)
    assert math.isnan(m) and math.isnan(a) and math.isnan(z)
    with pytest.raises(ValueError):
        ratio_bootstrap_ci(np.zeros(2), np.ones(3), 10, 0)


def test_significant_accepts_the_exact_min_effect_despite_float_rounding():
    assert 0.12 - 0.1 < 0.02  # a "true" 0.02 that rounds just below it
    assert significant(0.12 - 0.1, (0.01, 0.03), 0.02)
    assert significant(0.02, (0.02, 0.02), 0.02)
    assert not significant(0.0199, (0.01, 0.03), 0.02)  # a real shortfall is still rejected
    assert not significant(0.02, (0.0, 0.04), 0.02)  # CI lower bound must be strictly > 0
    assert not significant(0.02, (-1e-13, 0.04), 0.02)
