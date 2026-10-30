"""Cheap unit tests of experiment helpers (no window computation)."""

import re
from datetime import date

import numpy as np
import polars as pl
import pytest

from idsai.ai.chief import Chief
from idsai.config import load_settings
from idsai.contracts.detector import Detector, Evidence
from idsai.contracts.frames import LABELS
from idsai.data.store import Store
from idsai.detection.simulator import EthicsError, load_scenario_ranges
from idsai.eval.experiments import (
    FAMILIES,
    SATURATION_WARNING,
    SupervisedRFDetector,
    _Ctx,
    _evaluate,
    _Part,
    git_sha,
    model_version,
    run_experiment,
    stable_int,
)
from idsai.eval.splits import SymDay, day_start_us
from tests.synth import S, day_us, make_trades

D0 = date(2025, 1, 1)
DAY = date(2025, 1, 20)


def toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = x[:, 0] + 0.1 * rng.normal(size=n) > 1.0
    return pl.DataFrame({"a": x[:, 0], "b": x[:, 1]}), y


def test_rf_scores_probability_and_nan_rows():
    rows, y = toy()
    rf = SupervisedRFDetector(["a", "b"], seed=3)
    rf.fit_supervised(rows, y)
    test = pl.DataFrame({"a": [3.0, -3.0, np.nan, np.inf], "b": [0.0, 0.0, 0.0, 0.0]})
    s = rf.score(test)
    assert s[0] > 0.5 > s[1] and np.isnan(s[2]) and np.isfinite(s[3])
    assert rf.layer == "baseline" and rf.technique_ids == ["ANY"] and rf.name == "ref.supervised_rf"
    again = SupervisedRFDetector(["a", "b"], seed=3)
    again.fit_supervised(rows, y)
    assert np.array_equal(again.score(test), s, equal_nan=True)
    assert model_version(again) == model_version(rf) and model_version(rf).startswith("eval-")
    ev = rf.evidence({"a": 3.0, "b": 0.0})
    assert ev.top[0][0] == "a"


def test_rf_drops_nan_rows_and_needs_both_classes():
    rows, y = toy()
    rows = rows.with_columns(pl.when(pl.Series(y)).then(None).otherwise(pl.col("a")).alias("a"))
    with pytest.raises(ValueError, match="positive and negative"):
        SupervisedRFDetector(["a", "b"]).fit_supervised(rows, y)  # all positives had NaN
    with pytest.raises(RuntimeError, match="not fitted"):
        SupervisedRFDetector(["a"]).score(rows)


def test_stable_int_is_deterministic_and_key_sensitive():
    assert stable_int("ABCBTC", DAY) == stable_int("ABCBTC", DAY)
    assert stable_int("ABCBTC", DAY) != stable_int("ABDBTC", DAY)
    assert stable_int("ABCBTC", DAY) != stable_int("ABCBTC", date(2025, 1, 21))
    assert 0 <= stable_int("ABCBTC", DAY) < 2**64


def test_git_sha_shape():
    sha = git_sha()
    assert sha == "unknown" or len(sha.removesuffix("-dirty")) == 40


def test_unknown_experiment_and_live_injection_refused(tmp_path):
    cfg = load_settings(overrides={"mode": "live", "paths": {"data_dir": str(tmp_path)}})
    with Store(tmp_path, tmp_path / "app.duckdb") as s:
        with pytest.raises(KeyError, match="unknown experiment"):
            run_experiment("nope", cfg, s, 0)
        for name in ("semisyn", "ablation"):
            with pytest.raises(EthicsError):
                run_experiment(name, cfg, s, 0)
        assert s.list_runs() == []


SPLITS = {
    "train": ["2025-01-02", "2025-01-08"],
    "cal": ["2025-01-09", "2025-01-12"],
    "accept": ["2025-01-13", "2025-01-16"],
    "test": ["2025-01-17", "2025-01-24"],
    "drift_v1": None,
    "drift_v2": None,
}


def ctx_for(tmp_path, trades, seed=0):
    cfg = load_settings(overrides={"paths": {"data_dir": str(tmp_path)}, "splits": SPLITS})
    s = Store(tmp_path, tmp_path / "app.duckdb")
    s.write_trades(trades)
    t = day_us(D0) + 12 * 3600 * S
    s.write_labels(
        pl.DataFrame(
            [("l", "hu", "ABCBTC", t, t, "MM-T001", "telegram_pump", None, "REAL")],
            schema=LABELS,
            orient="row",
        )
    )
    return _Ctx(cfg, s, seed), s


def test_scenario_placement_is_seeded_and_inside_the_day(tmp_path):
    ctx, s = ctx_for(tmp_path, make_trades(DAY, 2, symbol="ABCBTC", rate=0.01))
    try:
        ranges = load_scenario_ranges()
        sd = SymDay("ABCBTC", DAY)
        d0 = day_start_us(DAY)
        for k, family in enumerate(FAMILIES):
            sc = ctx.scenario(sd, k, ranges)
            assert sc is not None and sc.family == family and sc.region == "test"
            assert sc == ctx.scenario(sd, k, ranges)
            start_s = (sc.t_start_us - d0) // S
            assert (sc.t_start_us - d0) % S == 0
            assert 3600 <= start_s <= 86400 - sc.duration_s - 600
        other = _Ctx(ctx.cfg, s, seed=1)
        assert other.scenario(sd, 0, ranges) != ctx.scenario(sd, 0, ranges)
    finally:
        s.close()


def test_low_tick_pumps_are_skipped(tmp_path):
    tr = make_trades(date(2025, 1, 19), 2, symbol="ABCBTC", rate=0.01)
    coarse = np.where(np.arange(tr.height) % 2 == 0, 3e-7, 4e-7)  # price / tick ~ 3-4
    tr = tr.with_columns(pl.Series("price", coarse), (pl.lit(1.0) * coarse).alias("quote_qty"))
    ctx, s = ctx_for(tmp_path, tr)
    try:
        ranges = load_scenario_ranges()
        sd = SymDay("ABCBTC", DAY)
        assert ctx.scenario(sd, FAMILIES.index("pump_burst"), ranges) is None
        assert ctx.scenario(sd, FAMILIES.index("pump_stealth"), ranges) is None
        assert ctx.scenario(sd, FAMILIES.index("wash_pingpong"), ranges) is not None
    finally:
        s.close()


@pytest.mark.parametrize("name", ["ref_supervised", "real_t001", "retrain_drift"])
def test_missing_training_data_is_recorded_not_raised(tmp_path, name):
    # the store only covers the test period: nothing to train on, nothing to calibrate
    ctx, s = ctx_for(tmp_path, make_trades(DAY, 2, symbol="ABCBTC", rate=0.01))
    research = Store(tmp_path, tmp_path / "research.duckdb")
    try:
        recs = run_experiment(name, ctx.cfg, s, 0, research=research)
        budgets = ctx.cfg.eval.budgets_per_symbol_day
        assert len(recs) == len(budgets) * (4 if name == "retrain_drift" else 1)
        assert all(r["metrics"] is None and r["error"] for r in recs)
        # error records describe their detector set like OK records do
        for r in recs:
            assert r["detectors"] and r["deterministic"] is False
            assert r["classifies"] is (name == "real_t001" or r["variant"] in ("v1", "v2"))
        if name == "retrain_drift":
            ai = [r for r in recs if r["variant"].endswith(":ai")]
            assert {r["variant"] for r in ai} == {"v1:ai", "v2:ai"}
            assert all(r["detectors"] == ["ai.autoencoder_trade"] for r in ai)
        assert len(research.list_runs(name)) == len(recs) and s.list_runs() == []
    finally:
        s.close()
        research.close()


# ---------------------------------------------------------------- FA on normal days (values)
class _Col(Detector):
    """Scores are read by the Chief from `score__stat.test`; this object only names them."""

    name = "stat.test"
    layer = "stat"
    technique_ids = ["ANY"]
    requires: list[str] = []

    def score(self, windows):
        return windows["score__stat.test"].to_numpy()

    def evidence(self, row):
        return Evidence()


E, N1, N2, H1, H2 = (date(2020, 7, k) for k in (1, 5, 6, 10, 11))


def at_h(day, h, m=0):
    return day_us(day) + (h * 3600 + m * 60) * S


def frame(rows):
    """rows: (day, hour, minute, score)"""
    return pl.DataFrame(
        {
            "symbol": ["ABCBTC"] * len(rows),
            "t_end_us": [at_h(d, h, m) for d, h, m, _ in rows],
            "score__stat.test": [float(v) for *_, v in rows],
        },
        schema={"symbol": pl.String, "t_end_us": pl.Int64, "score__stat.test": pl.Float64},
    )


def calibrated_chief():
    rng = np.random.default_rng(0)
    n = 10_000
    cal = pl.DataFrame(
        {
            "symbol": ["ABCBTC"] * n,
            "t_end_us": day_us(date(2020, 6, 1)) + np.arange(n, dtype=np.int64) * 8 * S,
            "score__stat.test": rng.random(n),
        }
    )
    chief = Chief([_Col()], load_settings().chief, "cfg", {})
    chief.calibrate(cal)
    return chief, cal


def test_fa_counts_unmatched_episodes_on_normal_days_only():
    chief, _ = calibrated_chief()  # score 5 -> p = 1/10001 -> F ~ 4; score 0 -> F ~ 0
    t0 = at_h(E, 12)
    labels = pl.DataFrame(
        [("ev", "hu", "ABCBTC", t0, t0, "MM-T001", "telegram_pump", None, "REAL")],
        schema=LABELS,
        orient="row",
    )
    real = _Part(
        frame(
            [
                (E, 0, 0, 0),
                (E, 12, 1, 5),  # matched episode
                (E, 18, 0, 5),  # unmatched, but on the event day: not a false alert
                (N1, 0, 0, 0),
                (N1, 6, 0, 5),
                (N1, 15, 0, 5),  # two false alerts
                (N2, 0, 0, 0),  # quiet normal day
            ]
        ),
        labels,
        days=[SymDay("ABCBTC", E), SymDay("ABCBTC", N1), SymDay("ABCBTC", N2)],
        fa_days=[SymDay("ABCBTC", N1), SymDay("ABCBTC", N2)],
    )
    hard = _Part(
        frame([(H1, 3, 0, 5), (H1, 9, 0, 5), (H2, 0, 0, 0), (H2, 23, 0, 5)]),
        labels.head(0),
        days=[SymDay("ABCBTC", H1), SymDay("ABCBTC", H2)],
    )
    fused = chief.fuse(real.frame)
    metrics, per_event = _evaluate(
        chief, 3.0, [fused], [real], (chief.fuse(hard.frame), hard), 120, 300
    )
    assert metrics["n_alerts"] == 4 and metrics["n_alerts_matched"] == 1
    assert metrics["precision"] == 0.25 and metrics["recall_any"] == 1.0
    assert metrics["fa_per_symbol_day"] == 1.0  # 2 unmatched episodes / 2 normal days
    assert metrics["fa_per_symbol_day_hard"] == 1.5  # 3 episodes / 2 hard-negative days
    assert per_event["ev"]["day"] == E.isoformat() and per_event["ev"]["ttd_s"] == 60.0


def test_only_the_saturation_warning_is_silenced():
    chief, cal = calibrated_chief()
    with pytest.warns(RuntimeWarning) as caught:
        chief.fit_threshold(chief.fuse(cal), 1e6)
    assert chief.tau_saturated
    assert any(re.match(SATURATION_WARNING, str(w.message)) for w in caught)
    assert not re.match(SATURATION_WARNING, "invalid value encountered in divide")


# ---------------------------------------------------------------- labels of a set's event days
def _label(lid, t0, t1=None):
    return (lid, "hu", "ABCBTC", t0, t1 or t0, "MM-T001", "telegram_pump", None, "REAL")


def _ctx_with_labels(tmp_path, first_day, n_days, rows):
    cfg = load_settings(overrides={"paths": {"data_dir": str(tmp_path)}, "splits": SPLITS})
    s = Store(tmp_path, tmp_path / "app.duckdb")
    s.write_trades(make_trades(first_day, n_days, symbol="ABCBTC", rate=0.01))
    s.write_labels(pl.DataFrame(rows, schema=LABELS, orient="row"))
    return _Ctx(cfg, s, 0), s


def test_label_past_midnight_in_a_missing_day_is_not_an_event(tmp_path):
    a, b = at_h(DAY, 23, 55), at_h(date(2025, 1, 21), 0, 3)
    ctx, s = _ctx_with_labels(
        tmp_path, date(2025, 1, 19), 2, [_label("a", a), _label("dropped", b)]
    )
    try:
        assert ctx.split.labels_dropped == {"test": {"missing_day": 1}}
        assert ctx.split.test_events == [SymDay("ABCBTC", DAY, 10)]
        part = ctx.real_parts(ctx.split.test_events, [], [])
        assert part.labels["label_id"].to_list() == ["a"]
    finally:
        s.close()


def test_label_of_a_neighbouring_period_is_not_an_event_of_this_set(tmp_path):
    last_accept, first_test = date(2025, 1, 16), date(2025, 1, 17)
    a, b = at_h(last_accept, 23, 55), at_h(first_test, 0, 3)
    ctx, s = _ctx_with_labels(
        tmp_path,
        date(2025, 1, 15),
        3,
        # "a" runs past midnight, so it overlaps the first test day too
        [_label("a", a, at_h(first_test, 0, 2)), _label("b", b)],
    )
    try:
        assert ctx.split.accept_events == [SymDay("ABCBTC", last_accept, 10)]
        assert ctx.split.test_events == [SymDay("ABCBTC", first_test)]
        accept = ctx.real_parts(ctx.split.accept_events, [], [])
        assert accept.labels["label_id"].to_list() == ["a"]
        test = ctx.real_parts(ctx.split.test_events, [], [])
        assert test.labels["label_id"].to_list() == ["b"]
    finally:
        s.close()
