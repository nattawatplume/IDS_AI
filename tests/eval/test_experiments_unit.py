"""Cheap unit tests of experiment helpers (no window computation)."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from idsai.config import load_settings
from idsai.contracts.frames import LABELS
from idsai.data.store import Store
from idsai.detection.simulator import EthicsError, load_scenario_ranges
from idsai.eval.experiments import (
    FAMILIES,
    SupervisedRFDetector,
    _Ctx,
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
        assert len(recs) == len(budgets) * (2 if name == "retrain_drift" else 1)
        assert all(r["metrics"] is None and r["error"] for r in recs)
        assert len(research.list_runs(name)) == len(recs) and s.list_runs() == []
    finally:
        s.close()
        research.close()
