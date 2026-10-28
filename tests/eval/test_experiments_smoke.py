"""End-to-end smoke test of the experiment runner on a synthetic research store.

3 BTC-quoted symbols x 16 days with overridden splits (1 normal day per symbol and period).
AAABTC carries REAL-provenance labels with a pump injected into the stored trades (train,
accept and test events); every symbol has labels in the cal period and the first days so the
normal-day candidates are pinned: BBBBTC's train normal (day 2) lies only in v1's drift range,
AAABTC's (day 3) in both, so v1 and v2 train on different days. Clean days are
cached on disk, so later experiments in this module reuse them.
"""

import json
import math
from datetime import date, timedelta

import polars as pl
import pytest

from idsai.config import config_hash, load_settings
from idsai.contracts.frames import LABELS
from idsai.data.store import Store
from idsai.detection.simulator import Scenario, base_quote_rate, inject
from idsai.eval.experiments import run_experiment
from idsai.eval.report import build_report
from tests.synth import S, day_us, make_trades

D0 = date(2025, 3, 1)
SYMS = ("AAABTC", "BBBBTC", "CCCBTC")
FAMILIES = {"pump_burst", "pump_stealth", "wash_pingpong"}
BUDGETS = [0.1, 10.0]
FINITE = ("recall_any", "recall_correct", "fa_per_symbol_day", "prevalence")
RECORD_KEYS = {
    "run_id",
    "experiment",
    "variant",
    "budget",
    "seed",
    "provenance",
    "git_sha",
    "config_hash",
    "signatures_sha",
    "model_versions",
    "split_def",
    "metrics",
    "per_event",
    "tau",
    "tau_saturated",
    "n_cal",
    "n_events",
    "n_symbol_days",
    "n_skipped",
    "error",
}


def d(k: int) -> date:
    return D0 + timedelta(days=k)


def noon(k: int) -> int:
    return day_us(d(k)) + 12 * 3600 * S


# REAL-provenance labels: (symbol, day index, pump injected into the stored trades?)
LABEL_ROWS = [
    ("AAABTC", 1, True),  # train event
    ("AAABTC", 5, False),  # cal period: blocks days 4 and 6
    ("AAABTC", 9, True),  # accept event
    ("AAABTC", 13, True),  # test event
    *[(s, k, False) for s in ("BBBBTC", "CCCBTC") for k in (0, 5)],
    ("BBBBTC", 4, False),  # pins BBBBTC's only train-normal day to day 2 (inside v1 only)
]


def make_cfg(tmp):
    return load_settings(
        overrides={
            "paths": {"data_dir": str(tmp / "data")},
            "splits": {
                "train": [str(d(1)), str(d(4))],
                "cal": [str(d(5)), str(d(7))],
                "accept": [str(d(8)), str(d(11))],
                "test": [str(d(12)), str(d(15))],
                "normal_days_per_symbol": 1,
                "min_gap_days": 2,
                "drift_v1": [str(d(1)), str(d(3))],
                "drift_v2": [str(d(3)), str(d(4))],
            },
            "eval": {"budgets_per_symbol_day": BUDGETS, "hard_negative_k": 2, "n_boot": 200},
            "ae": {"epochs": 2, "hidden": 16, "bottleneck": 4, "patience": 1},
        }
    )


def pump(t0: int, seed: int) -> Scenario:
    return Scenario(
        family="pump_burst",
        technique_id="MM-T001",
        sub_technique="MM-T001.001",
        t_start_us=t0,
        duration_s=120.0,
        impact=0.2,
        vol_mult=30.0,
        rush_size=5,
        dump_tau_s=300.0,
        seed=seed,
        region="train",
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    cfg = make_cfg(tmp)
    app = Store(tmp / "data", tmp / "data" / "app.duckdb")
    research = Store(tmp / "data", tmp / "data" / "research.duckdb")
    for k, sym in enumerate(SYMS):
        app.write_trades(make_trades(D0, 16, symbol=sym, seed=k, rate=0.2))
    rows = []
    for i, (sym, k, pumped) in enumerate(LABEL_ROWS):
        t0 = noon(k)
        rows.append((f"L{i}", "hu", sym, t0, t0, "MM-T001", "telegram_pump", None, "REAL"))
        if pumped:
            day = app.read_trades(sym, day_us(d(k)), day_us(d(k + 1)))
            prev = app.read_trades(sym, day_us(d(k - 1)), day_us(d(k)))
            new, _ = inject(day, pump(t0, seed=i), base_quote_rate(prev), mode="replay")
            app.write_trades(new)
    app.write_labels(pl.DataFrame(rows, schema=LABELS, orient="row"))
    yield cfg, app, research, tmp
    app.close()
    research.close()


@pytest.fixture(scope="module")
def semisyn(world):
    cfg, app, research, _ = world
    return run_experiment("semisyn", cfg, app, seed=0, research=research)


@pytest.fixture(scope="module")
def drift(world):
    cfg, app, research, _ = world
    return run_experiment("retrain_drift", cfg, app, seed=0, research=research)


def check_common(records, experiment, provenance, variants, cfg):
    assert {(r["variant"], r["budget"]) for r in records} == {
        (v, b) for v in variants for b in BUDGETS
    }
    for r in records:
        assert RECORD_KEYS <= set(r), RECORD_KEYS - set(r)
        assert r["experiment"] == experiment and r["provenance"] == provenance
        assert r["seed"] == 0 and r["config_hash"] == config_hash(cfg)
        assert r["run_id"] == f"{experiment}|{r['variant']}|b{r['budget']:g}|s0|{config_hash(cfg)}"
        assert len(r["signatures_sha"]) == 12 and r["git_sha"]
        assert r["split_def"]["counts"]["cal_normal"] == 3
        assert r["n_cal"] is None or r["n_cal"] > 0
        if r["budget"] == 0.1:  # 3 cal symbol-days are too few for 0.1 alerts per symbol-day
            assert r["metrics"] is None and "small" in r["error"].lower()
        else:
            assert r["error"] is None and r["tau"] is not None
            assert isinstance(r["tau_saturated"], bool)
            for k in FINITE:
                assert math.isfinite(r["metrics"][k]), (r["variant"], k)
            assert "accuracy" not in json.dumps(r["metrics"])


def test_semisyn_records(semisyn, world):
    cfg, app, research, _ = world
    check_common(semisyn, "semisyn", "SEMI-SYN", FAMILIES, cfg)
    for r in semisyn:
        assert r["n_events"] == 1 and r["n_symbol_days"] == 1  # one test-normal day per family
        assert r["n_skipped_tick"] == 0
        assert len(r["per_event"]) == (0 if r["error"] else 1)
        for pe in r["per_event"].values():  # cluster key of the report's bootstrap
            assert pe["symbol"] in SYMS and pe["family"] == r["variant"]
            assert date.fromisoformat(pe["day"]) >= d(12)
    burst = [r for r in semisyn if r["variant"] == "pump_burst" and r["budget"] == 10.0]
    assert burst[0]["metrics"]["recall_any"] == 1.0  # a 3-8 % burst on a quiet pair
    runs = research.list_runs("semisyn")
    assert len(runs) == len(semisyn) and app.list_runs() == []


def test_retrain_drift_records(drift, world):
    cfg, app, research, _ = world
    check_common(drift, "retrain_drift", "REAL", {"v1", "v2", "v1:ai", "v2:ai"}, cfg)
    accept_label = "L2"
    by = {(r["variant"], r["budget"]): r for r in drift}
    v1, v2 = by[("v1", 10.0)], by[("v2", 10.0)]
    assert v1["train_days"] == 3 and 1 <= v2["train_days"] < 3
    ae1, ae2 = (r["model_versions"]["ai.autoencoder_trade"] for r in (v1, v2))
    assert ae1 != ae2
    for v in ("v1", "v2"):  # the ai-only Chief reuses the same fitted AE (no extra fit)
        ai = by[(f"{v}:ai", 10.0)]
        assert ai["detectors"] == ["ai.autoencoder_trade"] and ai["classifies"] is False
        assert ai["model_versions"] == {
            "ai.autoencoder_trade": by[(v, 10.0)]["model_versions"]["ai.autoencoder_trade"]
        }
        assert by[(v, 10.0)]["classifies"] is True
    for r in drift:
        assert set(r["per_event"]) == (set() if r["error"] else {accept_label})
        assert r["n_events"] == 1 and r["n_symbol_days"] == 4  # 1 accept event + 3 normal days
        assert r["model_versions"]["ai.autoencoder_trade"].startswith("eval-")
        assert r["split_def"]["counts"]["accept_events"] == 1
    assert len(research.list_runs("retrain_drift")) == len(drift)


def test_same_seed_gives_identical_records(drift, world):
    cfg, app, research, _ = world
    again = run_experiment("retrain_drift", cfg, app, seed=0, research=research)
    assert json.dumps(again, sort_keys=True) == json.dumps(drift, sort_keys=True)
    assert len(research.list_runs("retrain_drift")) == len(drift)  # same run_ids replaced


def test_build_report(semisyn, drift, world):
    cfg, app, research, tmp = world
    md, js = tmp / "report.md", tmp / "report.json"
    build_report(research, md, js, cfg=cfg)
    text = md.read_text(encoding="utf-8")
    raw = js.read_text(encoding="utf-8")
    n = len(research.list_runs())
    assert f"runs: {n}" in text and "runs: " in raw
    assert f"signatures_sha: {semisyn[0]['signatures_sha']}" in text
    assert "SEMI-SYN" in text and "REAL" in text
    assert "accuracy" not in text.lower() and "accuracy" not in raw.lower()
    data = json.loads(raw)
    assert data["runs"] == n
    cell = data["experiments"]["semisyn"]["pump_burst"]["10"]
    assert cell["provenance"] == "SEMI-SYN" and cell["n_seeds"] == 1
    assert cell["mean"]["recall_any"] == 1.0
    err = data["experiments"]["retrain_drift"]["v1"]["0.1"]
    assert err["errors"] and err["mean"] is None


@pytest.mark.slow
def test_other_experiments_run(world):
    """real_t001, ablation, baselines and ref_supervised on the same store (slow)."""
    from idsai.eval.experiments import ABLATION, BASELINES, RF_NAME, SUPERVISED_NOTE

    cfg, app, research, _ = world
    expected = {
        "real_t001": ({"full"}, {"REAL"}),
        "ablation": (
            {f"{s}:{c}" for s in ("real_t001", "semisyn") for c in ABLATION},
            {"REAL", "SEMI-SYN"},
        ),
        "baselines": (set(BASELINES), {"REAL"}),
        "ref_supervised": ({RF_NAME}, {"REAL"}),
    }
    for name, (variants, provs) in expected.items():
        recs = run_experiment(name, cfg, app, seed=0, research=research)
        assert {r["variant"] for r in recs} == variants, name
        assert {r["provenance"] for r in recs} == provs, name
        for r in recs:
            assert RECORD_KEYS <= set(r)
            if r["budget"] == 10.0:
                assert r["error"] is None, (name, r["variant"], r["error"])
                assert math.isfinite(r["metrics"]["fa_per_symbol_day"])
                if r["provenance"] == "REAL":
                    assert r["n_events"] == 1 and set(r["per_event"]) == {"L3"}
                    assert math.isfinite(r["metrics"]["fa_per_symbol_day_hard"])
        if name == "baselines":
            assert all(r["classifies"] is False for r in recs)
        if name == "real_t001":
            assert all(r["classifies"] is True for r in recs)
        if name == "ref_supervised":
            assert all(r["note"] == SUPERVISED_NOTE for r in recs)
            assert all(r["train_positive_rows"] > 0 for r in recs)
        if name == "ablation":
            det = {r["variant"]: r["deterministic"] for r in recs}
            assert det["real_t001:rules"] and not det["real_t001:all"]
            assert not det["semisyn:rules"]
