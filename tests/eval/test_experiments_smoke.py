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

from idsai.ai.chief import AiFusion
from idsai.config import REPO_ROOT, config_hash, load_settings
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
    # the experiments' own config (AE detector set, fusion "fuse"), as the real eval runs use;
    # configs/default.yaml now holds the live AI layer (ai.iforest, shift 1)
    return load_settings(
        REPO_ROOT / "configs" / "eval_prev_live.yaml",
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
        },
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
            assert pe["techniques"] == sorted(set(pe["techniques"]))
            assert bool(pe["techniques"]) == pe["detected_any"]
        if r["error"] is None:  # FA on every loaded day, zero days included
            normal = r["fa_by_day"]["normal"]
            assert len(normal) == r["metrics"]["n_fa_days"] == r["n_symbol_days"]
            fa = sum(normal.values()) / len(normal)
            assert r["metrics"]["fa_per_symbol_day"] == pytest.approx(fa)
            assert r["fa_by_day"]["hard"] == {}
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
                    fd, m = r["fa_by_day"], r["metrics"]
                    assert len(fd["normal"]) == m["n_fa_days"]
                    assert len(fd["hard"]) == m["n_hard_days"] > 0
                    for which, k in (("normal", ""), ("hard", "_hard")):
                        fa = sum(fd[which].values()) / len(fd[which])
                        assert m[f"fa_per_symbol_day{k}"] == pytest.approx(fa), (name, which)
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


def test_m1_fusion_fits_and_scores_once_and_refuses_per_variant(world, monkeypatch):
    """m1_fusion: detectors are fit once and every (symbol-day, scenario) is scored once per
    seed; the five variants only re-fuse those scores."""
    from idsai.ai.autoencoder import AutoencoderDetector
    from idsai.eval import experiments as ex

    cfg, app, research, tmp = world
    fits, scores, loads = [], [], []
    real_fit, real_score, real_load = (
        AutoencoderDetector.fit,
        AutoencoderDetector.score,
        ex._Ctx.load,
    )

    def fit(self, normal):
        fits.append(normal.height)
        return real_fit(self, normal)

    def score(self, windows):
        scores.append(windows.height)
        return real_score(self, windows)

    def load(self, items, detectors, **kw):
        items = list(items)
        if detectors:
            loads.extend((sd, None if sc is None else sc.family) for sd, sc in items)
        return real_load(self, items, detectors, **kw)

    monkeypatch.setattr(AutoencoderDetector, "fit", fit)
    monkeypatch.setattr(AutoencoderDetector, "score", score)
    monkeypatch.setattr(ex._Ctx, "load", load)
    recs = run_experiment("m1_fusion", cfg, app, seed=0, research=research)

    variants = set(ex.m1_variants(cfg))
    assert variants == {"all", "no_ai", "ai_mondrian", "ai_shift1", "ai_shift2"}
    assert {r["variant"] for r in recs} == {f"{s}:{v}" for s in ex.M1_SPLITS for v in variants}
    assert len(fits) == 1  # one AE fit per seed
    assert len(loads) == len(set(loads))  # no symbol-day (or injected day) scored twice
    assert len(scores) <= len(loads)  # at most one AE score call per loaded day
    sp = ex._Ctx(cfg, app, 0).split
    n_days = len(sp.cal_normal) + len(sp.accept_events) + len(sp.accept_normal)
    n_days += len(sp.test_events) + len(sp.test_normal) + len(sp.test_hard_negative)
    assert len(loads) == n_days + len(ex.FAMILIES) * len(sp.test_normal)

    by = {(r["variant"], r["budget"]): r for r in recs}
    for r in recs:
        assert RECORD_KEYS <= set(r)
        assert r["experiment"] == "m1_fusion"
        assert r["provenance"] == ("SEMI-SYN" if r["variant"].startswith("semisyn:") else "REAL")
        assert r["regime_feature"] == "jump_1800"
        v = r["variant"].split(":", 1)[1]
        assert r["ai_fusion"] == ex.m1_variants(cfg)[v].as_dict()
        if r["budget"] == 10.0:
            assert r["error"] is None, (r["variant"], r["error"])
            assert math.isfinite(r["metrics"]["fa_per_symbol_day"])
    acc = by[("accept:all", 10.0)]
    k = ex.m1_hard_k(cfg, len(sp.accept_normal))
    assert len(acc["accept_hard_days"]) == acc["metrics"]["n_hard_days"] == k
    assert set(acc["per_event"]) == {"L2"} and set(by[("test:all", 10.0)]["per_event"]) == {"L3"}
    # the same fitted models in every variant
    assert len({json.dumps(r["model_versions"], sort_keys=True) for r in recs}) == 1
    # semisyn parts share their days: the multiplicity travels with the counts
    semi = by[("semisyn:all", 10.0)]["fa_by_day"]
    assert set(semi["normal_mult"].values()) == {len(ex.FAMILIES)}

    md, js = tmp / "m1.md", tmp / "m1.json"
    data = build_report(research, md, js, cfg=cfg)
    text = md.read_text(encoding="utf-8")
    assert "### m1_fusion" in text and "accuracy" not in text.lower()
    assert data["m1_fusion"] is None  # the smoke budgets (0.1, 10) have no budget-2 records


def test_m4_sell_two_variants_from_one_fit_with_per_phase_recall(world, monkeypatch):
    """m4_sell: one fit and one scoring pass; `base` equals real_t001; the REAL label gets a
    measured sell window; bear_raid is SEMI-SYN."""
    from idsai.ai.autoencoder import AutoencoderDetector

    cfg, app, research, tmp = world
    real = {
        r["budget"]: r for r in run_experiment("real_t001", cfg, app, seed=0, research=research)
    }
    fits = []
    real_fit = AutoencoderDetector.fit

    def fit(self, normal):
        fits.append(normal.height)
        return real_fit(self, normal)

    monkeypatch.setattr(AutoencoderDetector, "fit", fit)
    recs = run_experiment("m4_sell", cfg, app, seed=0, research=research)
    assert len(fits) == 1
    assert {r["variant"] for r in recs} == {
        f"{s}:{v}" for s in ("test", "bear_raid") for v in ("base", "sell")
    }
    by = {(r["variant"], r["budget"]): r for r in recs}
    for r in recs:
        assert RECORD_KEYS <= set(r)
        assert r["experiment"] == "m4_sell" and r["config_hash"] == config_hash(cfg)
        assert r["provenance"] == ("SEMI-SYN" if r["variant"].startswith("bear_raid") else "REAL")
        assert r["sell_side"] == r["variant"].endswith(":sell")
        assert len(r["signatures_sell_sha"]) == 12
        names = set(r["detectors"])
        assert ("rule.100104" in names) == r["sell_side"] and ("rule.100601" in names) == (
            r["sell_side"]
        )
        if r["budget"] == 10.0:
            assert r["error"] is None, (r["variant"], r["error"])
            assert "accuracy" not in json.dumps(r["metrics"])
    # same fitted models in both variants, and base reproduces real_t001 exactly
    assert len({json.dumps(r["model_versions"], sort_keys=True) for r in recs}) == 1
    base, sell = by[("test:base", 10.0)], by[("test:sell", 10.0)]
    for k in ("recall_any", "recall_correct", "fa_per_symbol_day", "fa_per_symbol_day_hard"):
        assert base["metrics"][k] == real[10.0]["metrics"][k], k
    assert base["fa_by_day"] == real[10.0]["fa_by_day"]
    sell_keys = {"sell_status", "dump_censored", "sell_window_short", "detected_sell_phase",
                 "detected_sell_phase_correct"}  # fmt: skip
    assert {
        lid: {k: v for k, v in pe.items() if k not in sell_keys}
        for lid, pe in base["per_event"].items()
    } == real[10.0]["per_event"]
    assert base["tau"] == real[10.0]["tau"]
    # per-phase recall on the REAL test label (pumped at noon of day 13, 120 s ramp)
    for r in (base, sell):
        pe = r["per_event"]["L3"]
        assert pe["sell_status"] == "measured"
        assert isinstance(pe["detected_sell_phase"], bool)
        m = r["metrics"]
        assert m["recall_pump_phase"] == m["recall_any"]
        assert m["n_sell_events"] == 1 and m["sell_status_counts"] == {"measured": 1}
        assert math.isfinite(m["recall_sell_phase"])
    assert base["metrics"]["recall_sell_phase_correct"] == 0.0  # no sell rule, no sell phase
    raid = by[("bear_raid:sell", 10.0)]
    assert raid["n_events"] == 1 and raid["n_skipped_tick"] == 0
    (pe,) = raid["per_event"].values()
    assert pe["family"] == "bear_raid"
    assert math.isfinite(raid["metrics"]["recall_any"])

    md, js = tmp / "m4.md", tmp / "m4.json"
    data = build_report(research, md, js, cfg=cfg)
    text = md.read_text(encoding="utf-8")
    assert "## m4_sell" in text and "accuracy" not in text.lower()
    assert "m4_sell" in data["experiments"]


def test_m2_ai_fits_each_model_once_and_no_ai_equals_m1(world, monkeypatch):
    """m2_ai: the four AI models are fit once per seed and every (symbol-day, scenario) is
    scored once; the 16 fusion variants (and the two diagnostics `test:<m>:alone`) only re-fuse
    those scores; `no_ai` reproduces m1_fusion's `no_ai` exactly."""
    from idsai.ai.autoencoder import AutoencoderDetector
    from idsai.ai.baselines import IsolationForestDetector
    from idsai.eval import experiments as ex

    cfg, app, research, tmp = world
    m1 = {
        (r["variant"], r["budget"]): r
        for r in run_experiment("m1_fusion", cfg, app, seed=0, research=research)
    }
    fits, loads = [], []
    real_ae_fit, real_if_fit, real_load = (
        AutoencoderDetector.fit,
        IsolationForestDetector.fit,
        ex._Ctx.load,
    )

    def ae_fit(self, normal):
        fits.append((self.name, self.cfg.hidden, self.cfg.bottleneck))
        return real_ae_fit(self, normal)

    def if_fit(self, normal):
        fits.append((self.name, self.layer))
        return real_if_fit(self, normal)

    def load(self, items, detectors, **kw):
        items = list(items)
        if detectors:
            loads.extend((sd, None if sc is None else sc.family) for sd, sc in items)
        return real_load(self, items, detectors, **kw)

    monkeypatch.setattr(AutoencoderDetector, "fit", ae_fit)
    monkeypatch.setattr(IsolationForestDetector, "fit", if_fit)
    monkeypatch.setattr(ex._Ctx, "load", load)
    recs = run_experiment("m2_ai", cfg, app, seed=0, research=research)

    # one fit per model per seed: today's AE, the two grid points, and iforest on the ai layer
    assert sorted(fits, key=str) == sorted(
        [
            ("ai.autoencoder_trade", cfg.ae.hidden, cfg.ae.bottleneck),
            ("ai.autoencoder_wide", 128, 16),
            ("ai.autoencoder_narrow", 64, 8),
            ("ai.iforest", "ai"),
        ],
        key=str,
    )
    assert len(loads) == len(set(loads))  # no symbol-day (or injected day) scored twice
    sp = ex._Ctx(cfg, app, 0).split
    n_days = len(sp.cal_normal) + len(sp.accept_events) + len(sp.accept_normal)
    n_days += len(sp.test_events) + len(sp.test_normal) + len(sp.test_hard_negative)
    assert len(loads) == n_days + len(ex.FAMILIES) * len(sp.test_normal)

    want = {f"{s}:{m}:{f}" for s in ex.M2_SPLITS for m in ex.M2_MODELS for f in ex.M2_FUSIONS}
    want |= {f"{s}:no_ai" for s in ex.M2_SPLITS}
    want |= {f"test:{m}:alone" for m in ex.M2_ALONE}
    assert len(ex.M2_MODELS) * len(ex.M2_FUSIONS) + 1 == 16
    assert {r["variant"] for r in recs} == want
    by = {(r["variant"], r["budget"]): r for r in recs}
    mv_all = {}
    for r in recs:
        assert RECORD_KEYS <= set(r)
        assert r["experiment"] == "m2_ai"
        assert r["deterministic"] is False  # every variant holds a seeded AI model
        assert r["provenance"] == ("SEMI-SYN" if r["variant"].startswith("semisyn:") else "REAL")
        mv_all.update(r["model_versions"])
        parts = r["variant"].split(":")
        if parts[1] == "no_ai":
            assert r["ai_fusion"] == AiFusion("exclude").as_dict() and r["ai_model"] is None
        elif parts[2] == "alone":
            assert r["ai_model"] == parts[1] and all(n.startswith("ai.") for n in r["detectors"])
        else:
            assert r["ai_model"] == parts[1]
            assert r["ai_fusion"] == ex.M2_FUSIONS[parts[2]].as_dict()
            ai = {n for n in r["detectors"] if n.startswith("ai.")}
            assert ai == {ex.M2_AI_NAMES[x] for x in ex.M2_MODELS[parts[1]]}
        if r["budget"] == 10.0:
            assert r["error"] is None, (r["variant"], r["error"])
            assert "accuracy" not in json.dumps(r["metrics"])
    assert set(mv_all) == set(ex.M2_AI_NAMES.values())  # four distinct fitted models
    assert by[("accept:ae+iforest:shift2", 10.0)]["model_versions"].keys() == {
        "ai.autoencoder_trade",
        "ai.iforest",
    }
    k = ex.m1_hard_k(cfg, len(sp.accept_normal))
    assert by[("accept:no_ai", 10.0)]["metrics"]["n_hard_days"] == k
    # no_ai is M1's no_ai: the same detectors, fitted AE, calibration, tau and outcomes
    for split in ex.M2_SPLITS:
        for b in BUDGETS:
            a, m = by[(f"{split}:no_ai", b)], m1[(f"{split}:no_ai", b)]
            for key in ("metrics", "per_event", "fa_by_day", "tau", "tau_saturated", "n_cal",
                        "detectors", "model_versions", "error", "n_events"):  # fmt: skip
                assert a.get(key) == m.get(key), (split, b, key)

    # the report's M2 section built from these real records (the smoke run has no budget 2)
    from idsai.eval import report

    monkeypatch.setattr(report, "PAIRED_BUDGET", 10.0)
    md, js = tmp / "m2.md", tmp / "m2.json"
    data = build_report(research, md, js, cfg=cfg)
    text = md.read_text(encoding="utf-8")
    assert "### m2_ai" in text and "accuracy" not in text.lower()
    m2 = data["m2_ai"]
    assert m2 is not None and m2["selection"] is not None
    accept = {r["variant"] for r in m2["splits"]["accept"]["rows"]}
    assert accept == set(ex.m2_variants()) and len(accept) == 16
    assert set(m2["splits"]) == set(ex.M2_SPLITS)
    # one accept event: no paired gain can be significant, so the pick stays no_ai
    assert m2["selection"]["chosen"] == "no_ai"
    assert m2["ae_vs_iforest"] is not None or any("iforest" in n for n in m2["notes"])


def test_m3_attr_fits_once_and_naming_changes_only_names(world, monkeypatch):
    """m3_attr: the live stack is built inside the experiment (the eval config keeps its hash),
    ai.iforest is fit once per seed, every (symbol-day, scenario) is scored once; `attr` keeps
    base's tau, FA and recall_any exactly (only names move), and wash2 swaps one rule."""
    from idsai.ai.baselines import IsolationForestDetector
    from idsai.eval import experiments as ex

    cfg, app, research, tmp = world
    fits, loads = [], []
    real_fit, real_load = IsolationForestDetector.fit, ex._Ctx.load

    def if_fit(self, normal):
        fits.append(self.name)
        return real_fit(self, normal)

    def load(self, items, detectors, **kw):
        items = list(items)
        if detectors:
            loads.extend((sd, None if sc is None else sc.family) for sd, sc in items)
        return real_load(self, items, detectors, **kw)

    monkeypatch.setattr(IsolationForestDetector, "fit", if_fit)
    monkeypatch.setattr(ex._Ctx, "load", load)
    recs = run_experiment("m3_attr", cfg, app, seed=0, research=research)
    assert fits == ["ai.iforest"]
    assert len(loads) == len(set(loads))  # no symbol-day (or injected day) scored twice
    assert {r["variant"] for r in recs} == {
        f"{s}:{v}" for s in ex.M3_SPLITS for v in ex.M3_VARIANTS
    }
    by = {(r["variant"], r["budget"]): r for r in recs}
    for r in recs:
        assert RECORD_KEYS <= set(r)
        assert r["experiment"] == "m3_attr" and r["config_hash"] == config_hash(cfg)
        on, rev = ex.M3_VARIANTS[r["variant"].split(":")[1]]
        assert (r["attr_rules"], r["wash_rev"]) == (on, rev)
        assert r["ai_fusion"] == AiFusion("shift", 1.0).as_dict()
        assert len(r["signatures_attr_sha"]) == 12 and len(r["signatures_wash2_sha"]) == 12
        names = r["detectors"]
        assert "ai.iforest" in names and "ai.autoencoder_trade" not in names
        assert ("rule.100501.rev2" in names) == (rev == 2)
        assert ("rule.100501" in names) == (rev == 1)
        assert not any(n.startswith("attr.") for n in names)  # naming rules never fused
        if r["budget"] == 10.0:
            assert r["error"] is None, (r["variant"], r["error"])
            assert {"n_fa_episodes", "n_fa_named", "n_fa_named_by_rule"} <= set(r["metrics"])
            if not on:
                assert r["metrics"]["n_fa_named_by_rule"] == 0
    for split in ex.M3_SPLITS:
        for b in BUDGETS:
            base, attr = by[(f"{split}:base", b)], by[(f"{split}:attr", b)]
            for key in ("tau", "tau_saturated", "n_cal", "fa_by_day", "detectors", "error"):
                assert attr.get(key) == base.get(key), (split, b, key)
            if base["metrics"] is None:
                continue
            for k in ("recall_any", "fa_per_symbol_day", "fa_per_symbol_day_hard", "n_alerts",
                      "n_fa_episodes", "pr_auc"):  # fmt: skip
                assert attr["metrics"].get(k) == base["metrics"].get(k), (split, b, k)
            for lid, pe in base["per_event"].items():
                assert attr["per_event"][lid]["detected_any"] == pe["detected_any"]
                assert pe["detected_correct"] <= attr["per_event"][lid]["detected_correct"]

    from idsai.eval import report

    monkeypatch.setattr(report, "PAIRED_BUDGET", 10.0)
    md, js = tmp / "m3.md", tmp / "m3.json"
    data = build_report(research, md, js, cfg=cfg)
    text = md.read_text(encoding="utf-8")
    assert "### m3_attr" in text and "accuracy" not in text.lower()
    assert "#### confusion semisyn:attr" in text
    m3 = data["m3_attr"]
    assert set(m3["splits"]) == set(ex.M3_SPLITS)
    assert {r["variant"] for r in m3["splits"]["accept"]["rows"]} == set(ex.M3_VARIANTS)
    # one accept event: no paired gain can be significant, so the pick stays base
    assert m3["selection"]["chosen"] == "base"
    assert m3["confusion"]["semisyn"]["base"]["rows"]


def test_m5_rules_fits_once_and_swaps_only_the_pump_rules(world, monkeypatch):
    """m5_rules: the live stack (sell side, naming rules, ai.iforest shift 1) is built inside the
    experiment (the eval config keeps its hash), ai.iforest is fit once per seed, every
    (symbol-day, scenario) is scored once; each variant swaps only its rev 2 pump rules in."""
    from idsai.ai.baselines import IsolationForestDetector
    from idsai.eval import experiments as ex

    cfg, app, research, tmp = world
    fits, loads = [], []
    real_fit, real_load = IsolationForestDetector.fit, ex._Ctx.load

    def if_fit(self, normal):
        fits.append(self.name)
        return real_fit(self, normal)

    def load(self, items, detectors, **kw):
        items = list(items)
        if detectors:
            loads.extend((sd, None if sc is None else sc.family) for sd, sc in items)
        return real_load(self, items, detectors, **kw)

    monkeypatch.setattr(IsolationForestDetector, "fit", if_fit)
    monkeypatch.setattr(ex._Ctx, "load", load)
    recs = run_experiment("m5_rules", cfg, app, seed=0, research=research)
    assert fits == ["ai.iforest"]
    assert len(loads) == len(set(loads))  # no symbol-day (or injected day) scored twice
    assert {r["variant"] for r in recs} == {
        f"{s}:{v}" for s in ex.M5_SPLITS for v in ex.M5_VARIANTS
    }
    swapped = {"r101": {"rule.100101": "rule.100101.rev2"},
               "r101_25": {"rule.100101": "rule.100101.rev2_25"},
               "r102": {"rule.100102": "rule.100102.rev2"},
               "r101+r102": {"rule.100101": "rule.100101.rev2",
                             "rule.100102": "rule.100102.rev2"}}  # fmt: skip
    by = {(r["variant"], r["budget"]): r for r in recs}
    for r in recs:
        assert RECORD_KEYS <= set(r)
        assert r["experiment"] == "m5_rules" and r["config_hash"] == config_hash(cfg)
        v = r["variant"].split(":")[1]
        assert r["rule_rev2"] == sorted(ex.M5_CANDIDATES[k][1] for k in ex.M5_VARIANTS[v])
        assert (r["sell_side"], r["attr_rules"]) == (True, True)
        assert r["ai_fusion"] == AiFusion("shift", 1.0).as_dict()
        for k in ("signatures_r2_sha", "signatures_r2_alt_sha", "signatures_sell_sha",
                  "signatures_attr_sha"):  # fmt: skip
            assert len(r[k]) == 12
        names = r["detectors"]
        base = by[(f"{r['variant'].split(':')[0]}:base", r["budget"])]["detectors"]
        want = [swapped.get(v, {}).get(n, n) for n in base]
        assert names == want
        assert "ai.iforest" in names and "ai.autoencoder_trade" not in names
        assert "rule.100104" in names and "rule.100601" in names  # the live sell-side rules
        assert not any(n.startswith("attr.") for n in names)  # naming rules never fused
        if r["budget"] == 10.0:
            assert r["error"] is None, (r["variant"], r["error"])
            assert {"n_fa_episodes", "n_fa_named", "n_fa_named_by_rule"} <= set(r["metrics"])
    # the base variant is the live stack at rev 1
    assert {n for n in by[("test:base", 10.0)]["detectors"] if ".rev2" in n} == set()

    from idsai.eval import report

    monkeypatch.setattr(report, "PAIRED_BUDGET", 10.0)
    md, js = tmp / "m5.md", tmp / "m5.json"
    data = build_report(research, md, js, cfg=cfg)
    text = md.read_text(encoding="utf-8")
    assert "### m5_rules" in text and "accuracy" not in text.lower()
    assert "#### confusion semisyn:r101+r102" in text
    m5 = data["m5_rules"]
    assert set(m5["splits"]) == set(ex.M5_SPLITS)
    assert [r["variant"] for r in m5["splits"]["accept"]["rows"]] == list(ex.M5_VARIANTS)
    # one accept event: no paired gain can be significant, so the pick stays base
    assert m5["selection"]["chosen"] == "base"
    assert m5["confusion"]["semisyn"]["base"]["rows"]


def test_m8_stealth_fits_once_and_changes_only_rule_100102(world, monkeypatch):
    """m8_stealth: the live stack is built inside the experiment (the eval config keeps its hash),
    ai.iforest is fit once per seed, every (symbol-day, scenario) is scored once; stealth_attr
    drops rule.100102 from the fused detectors and adds attr.100102 to the naming rules,
    stealth_act puts rule.100102.rev3 in its place."""
    from idsai.ai.baselines import IsolationForestDetector
    from idsai.eval import experiments as ex

    cfg, app, research, tmp = world
    fits, loads, namings = [], [], {}
    real_fit, real_load, real_chief = IsolationForestDetector.fit, ex._Ctx.load, ex.Chief

    def if_fit(self, normal):
        fits.append(self.name)
        return real_fit(self, normal)

    def load(self, items, detectors, **kw):
        items = list(items)
        if detectors:
            loads.extend((sd, None if sc is None else sc.family) for sd, sc in items)
        return real_load(self, items, detectors, **kw)

    def chief(detectors, *a, naming=None, **kw):
        key = tuple(d.name for d in detectors)
        namings.setdefault(key, set()).add(tuple(d.name for d in naming or []))
        return real_chief(detectors, *a, naming=naming, **kw)

    monkeypatch.setattr(IsolationForestDetector, "fit", if_fit)
    monkeypatch.setattr(ex._Ctx, "load", load)
    monkeypatch.setattr(ex, "Chief", chief)
    recs = run_experiment("m8_stealth", cfg, app, seed=0, research=research)
    assert fits == ["ai.iforest"]
    assert len(loads) == len(set(loads))  # no symbol-day (or injected day) scored twice
    assert {r["variant"] for r in recs} == {
        f"{s}:{v}" for s in ex.M8_SPLITS for v in ex.M8_VARIANTS
    }
    by = {(r["variant"], r["budget"]): r for r in recs}
    m3_naming = ("attr.100191", "attr.100591")
    for r in recs:
        assert RECORD_KEYS <= set(r)
        assert r["experiment"] == "m8_stealth" and r["config_hash"] == config_hash(cfg)
        split, v = r["variant"].split(":")
        assert r["stealth"] == ex.M8_VARIANTS[v]
        assert (r["sell_side"], r["attr_rules"]) == (True, True)
        assert r["ai_fusion"] == AiFusion("shift", 1.0).as_dict()
        for k in ("signatures_r3_sha", "signatures_sell_sha", "signatures_attr_sha"):
            assert len(r[k]) == 12
        names = r["detectors"]
        base = by[(f"{split}:base", r["budget"])]["detectors"]
        if v == "base":
            want, naming = base, m3_naming
        elif v == "stealth_attr":
            want, naming = [n for n in base if n != "rule.100102"], (*m3_naming, "attr.100102")
        else:
            want = ["rule.100102.rev3" if n == "rule.100102" else n for n in base]
            naming = m3_naming
        assert names == want
        assert namings[tuple(names)] == {naming}
        assert "ai.iforest" in names and "rule.100104" in names and "rule.100601" in names
        assert not any(n.startswith("attr.") for n in names)  # naming rules never fused
        if r["budget"] == 10.0:
            assert r["error"] is None, (r["variant"], r["error"])
            assert {"n_fa_episodes", "n_fa_named", "n_fa_named_by_rule"} <= set(r["metrics"])
    assert "rule.100102" in by[("test:base", 10.0)]["detectors"]

    from idsai.eval import report

    monkeypatch.setattr(report, "PAIRED_BUDGET", 10.0)
    md, js = tmp / "m8.md", tmp / "m8.json"
    data = build_report(research, md, js, cfg=cfg)
    text = md.read_text(encoding="utf-8")
    assert "### m8_stealth" in text and "accuracy" not in text.lower()
    assert "#### confusion semisyn:stealth_act" in text
    m8 = data["m8_stealth"]
    assert set(m8["splits"]) == set(ex.M8_SPLITS)
    assert [r["variant"] for r in m8["splits"]["accept"]["rows"]] == list(ex.M8_VARIANTS)
    for r in m8["splits"]["accept"]["rows"]:
        if r["variant"] != "base":
            assert r["vs_ref_fa_hard"] is not None and r["vs_ref_fa_hard"]["n_seeds"] == 1
    # one accept event: no paired recall comparison is decidable, so the pick stays base
    assert m8["selection"]["chosen"] == "base"
    assert set(m8["selection"]["excluded"]) == {"stealth_attr", "stealth_act"}
    assert m8["confusion"]["semisyn"]["base"]["rows"]
