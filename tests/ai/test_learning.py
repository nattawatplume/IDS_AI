"""Champion/challenger decision: paired event bootstrap and the report's retrain section."""

import json
import pickle
import re

import numpy as np
import pytest

from idsai.ai.learning import Decision, compare
from idsai.config import load_settings
from idsai.data.store import Store
from idsai.eval.metrics import MIN_EVENTS, ratio_bootstrap_ci
from idsai.eval.report import MIXED_CONFIG_NOTE, build_report

N_BOOT = 1000
SEED = 0


def _arrays(n_seeds: int, n_events: int, champ_hits: int, chall_hits: int):
    champ = np.zeros((n_seeds, n_events), dtype=np.int8)
    chall = np.zeros((n_seeds, n_events), dtype=np.int8)
    champ[:, :champ_hits] = 1
    chall[:, :chall_hits] = 1
    return champ, chall


def test_clear_improvement_is_accepted():
    # challenger detects 30 more of 100 events in every seed
    champ, chall = _arrays(5, 100, 50, 80)
    d = compare(champ, chall, N_BOOT, SEED)
    assert isinstance(d, Decision)
    assert d.accept is True
    assert d.diff_mean == pytest.approx(0.30)
    assert 0 < d.ci[0] < 0.30 < d.ci[1]
    assert d.reason_th.startswith("ยอมรับ challenger")
    assert "+0.300" in d.reason_th and "0.02" in d.reason_th


def test_identical_arms_are_rejected_with_zero_ci():
    champ, _ = _arrays(5, 100, 40, 40)
    d = compare(champ, champ.copy(), N_BOOT, SEED)
    assert d.accept is False
    assert d.diff_mean == 0.0
    assert d.ci == (0.0, 0.0)
    assert d.reason_th.startswith("ไม่ยอมรับ challenger") and "เหมือนกัน" in d.reason_th


def test_consistent_small_gain_is_rejected_by_min_effect():
    # every event gains exactly one detection out of 100 seeds: diff +0.01 with CI [0.01, 0.01]
    champ = np.zeros((100, 100), dtype=bool)
    chall = np.eye(100, dtype=bool)
    d = compare(champ, chall, N_BOOT, SEED, min_effect=0.02)
    assert d.diff_mean == pytest.approx(0.01)
    assert d.ci[0] > 0  # significant, but too small to matter
    assert d.accept is False
    assert "ต่ำกว่าเกณฑ์ขั้นต่ำ" in d.reason_th and "+0.010" in d.reason_th
    # the same gain passes once the minimum effect is lowered
    assert compare(champ, chall, N_BOOT, SEED, min_effect=0.005).accept is True


def test_noisy_mixed_arms_are_rejected():
    rng = np.random.default_rng(7)
    champ = rng.integers(0, 2, size=(5, 60))
    chall = rng.integers(0, 2, size=(5, 60))
    d = compare(champ, chall, N_BOOT, SEED)
    assert d.ci[0] <= 0 <= d.ci[1]
    assert d.accept is False
    assert "คร่อม 0" in d.reason_th


def test_worse_challenger_is_rejected():
    champ, chall = _arrays(5, 100, 80, 40)
    d = compare(champ, chall, N_BOOT, SEED)
    assert d.diff_mean == pytest.approx(-0.40) and d.ci[1] < 0
    assert d.accept is False and d.reason_th.startswith("ไม่ยอมรับ challenger")


def test_compare_is_deterministic():
    rng = np.random.default_rng(3)
    champ = rng.integers(0, 2, size=(5, 40))
    chall = rng.integers(0, 2, size=(5, 40))
    assert compare(champ, chall, 500, 11) == compare(champ, chall, 500, 11)


def _two_per_event(n: int = 100, k: int = 2):
    """`n` seeds x `n` events; the challenger detects each event in exactly `k` seeds."""
    champ = np.zeros((n, n), dtype=bool)
    chall = np.zeros((n, n), dtype=bool)
    for j in range(n):
        chall[(j + np.arange(k)) % n, j] = True
    return champ, chall


def test_exact_min_effect_is_accepted_despite_float_rounding():
    # diff is a true 0.02 that the float mean rounds to 0.019999999999999997
    champ, chall = _two_per_event()
    d = compare(champ, chall, N_BOOT, SEED, min_effect=0.02)
    assert d.diff_mean == pytest.approx(0.02) and d.diff_mean < 0.02
    assert d.ci[0] > 0
    assert d.accept is True
    assert "ถึงเกณฑ์ขั้นต่ำ" in d.reason_th and "เกิน" not in d.reason_th


def _col(j: int, shape: tuple[int, int] = (5, 100)) -> np.ndarray:
    """Detections only at event `j`, on every seed."""
    a = np.zeros(shape, dtype=bool)
    a[:, j] = True
    return a


def test_single_event_is_rejected_with_degenerate_ci():
    # the bootstrap of one event cannot vary: CI (1, 1) must not read as certainty
    d = compare(np.zeros((5, 1)), np.ones((5, 1)), N_BOOT, SEED)
    assert d.accept is False and d.ci == (1.0, 1.0) and d.diff_mean == pytest.approx(1.0)
    assert "เพียง 1 ครั้ง" in d.reason_th and f"อย่างน้อย {MIN_EVENTS}" in d.reason_th
    n = MIN_EVENTS
    assert compare(np.zeros((5, n)), np.ones((5, n)), N_BOOT, SEED).accept is True


def test_too_few_events_cannot_be_decided_even_with_a_clear_gain():
    assert MIN_EVENTS == 10
    n = MIN_EVENTS - 1
    d = compare(np.zeros((5, n)), np.ones((5, n)), N_BOOT, SEED)
    assert d.accept is False and d.diff_mean == pytest.approx(1.0)
    assert "ตัดสินไม่ได้" in d.reason_th
    assert f"เพียง {n} ครั้ง" in d.reason_th and f"อย่างน้อย {MIN_EVENTS}" in d.reason_th


def test_ci_low_exactly_zero_says_touches_not_straddles():
    d = compare(np.zeros((5, 100)), _col(50), N_BOOT, SEED)
    assert d.ci[0] == 0.0 and d.accept is False
    assert "CI แตะ 0" in d.reason_th and "คร่อม" not in d.reason_th


@pytest.mark.parametrize("n_boot", [0, -1])
def test_non_positive_n_boot_raises(n_boot):
    with pytest.raises(ValueError, match="n_boot"):
        compare(np.zeros((5, 3)), np.ones((5, 3)), n_boot, SEED)


def test_reason_never_claims_a_probability():
    rng = np.random.default_rng(7)
    small = (np.zeros((100, 100), dtype=bool), np.eye(100, dtype=bool))
    branches = {
        "accept": _arrays(5, 100, 50, 80),
        "ci_below_0": _arrays(5, 100, 80, 40),
        "ci_straddles_0": (rng.integers(0, 2, size=(5, 60)), rng.integers(0, 2, size=(5, 60))),
        "below_min_effect": small,
        "zero_events": (np.zeros((5, 0)), np.zeros((5, 0))),
        "one_event": (np.zeros((5, 1)), np.ones((5, 1))),
        "identical": (np.eye(5, 100, dtype=bool), np.eye(5, 100, dtype=bool)),
        "ci_touches_0": (np.zeros((5, 100)), _col(50)),
    }
    expected = {
        "accept": "ถึงเกณฑ์ขั้นต่ำ",
        "ci_below_0": "ต่ำกว่า 0 ทั้งช่วง",
        "ci_straddles_0": "คร่อม 0",
        "below_min_effect": "ต่ำกว่าเกณฑ์ขั้นต่ำ",
        "zero_events": "ไม่มีเหตุการณ์",
        "one_event": "เพียง 1 ครั้ง",
        "identical": "เหมือนกัน",
        "ci_touches_0": "CI แตะ 0",
    }
    banned = ("ความน่าจะเป็น", "probability", "โอกาส", "แน่นอน")
    for name, (champ, chall) in branches.items():
        d = compare(champ, chall, N_BOOT, SEED)
        reason = d.reason_th
        # "ยอมรับ" is a substring of "ไม่ยอมรับ": the accept branch is identified by the prefix
        assert d.accept is (name == "accept"), (name, reason)
        assert reason.startswith("ยอมรับ" if name == "accept" else "ไม่ยอมรับ"), (name, reason)
        assert expected[name] in reason, (name, reason)
        assert not any(b in reason.lower() for b in banned), (name, reason)


@pytest.mark.parametrize(
    ("champ", "chall"),
    [
        (np.zeros((5, 10)), np.zeros((5, 11))),  # n_events differ
        (np.zeros((5, 10)), np.zeros((4, 10))),  # n_seeds differ
        (np.zeros(10), np.zeros(10)),  # 1-D
        (np.zeros((2, 5, 10)), np.zeros((2, 5, 10))),  # 3-D
        (np.zeros((0, 10)), np.zeros((0, 10))),  # no seeds
        (np.full((5, 10), 0.5), np.zeros((5, 10))),  # not 0/1
        (np.full((5, 10), 2), np.zeros((5, 10))),
        (np.zeros((5, 10)), np.full((5, 10), np.nan)),
    ],
)
def test_compare_validates_shape_and_values(champ, chall):
    with pytest.raises(ValueError):
        compare(champ, chall, N_BOOT, SEED)


def test_zero_events_is_rejected_with_reason():
    d = compare(np.zeros((5, 0)), np.zeros((5, 0)), N_BOOT, SEED)
    assert d.accept is False and d.diff_mean == 0.0 and d.ci == (0.0, 0.0)
    assert "ไม่มีเหตุการณ์" in d.reason_th


# ---- report: retrain_drift decision section


EVENTS = [f"lab{i:02d}" for i in range(40)]


def _rec(variant, seed, detected, budget=2.0, **kw):
    per_event = {
        lid: {
            "symbol": "ABCBTC",
            "day": "2021-03-01",
            "family": "pump",
            "detected_any": bool(d),
            "detected_correct": bool(d),
            "ttd_s": 1.0 if d else None,
        }
        for lid, d in zip(EVENTS, detected, strict=False)
    }
    base = {
        "run_id": f"retrain_drift|{variant}|b{budget:g}|s{seed}|cfg",
        "experiment": "retrain_drift",
        "variant": variant,
        "budget": float(budget),
        "seed": seed,
        "provenance": "REAL",
        "git_sha": "abc",
        "config_hash": "cfg",
        "signatures_sha": "sig",
        "metrics": {"recall_any": sum(detected) / len(detected)} if detected else None,
        "per_event": per_event,
        "deterministic": False,
        "classifies": True,
        "error": None,
        "n_events": len(per_event),
    }
    base.update(kw)
    return base


@pytest.fixture
def research(tmp_path):
    with Store(tmp_path / "data", tmp_path / "research.duckdb") as s:
        yield s


def _build(store, tmp_path):
    cfg = load_settings(overrides={"eval": {"n_boot": 300}})
    md, js = tmp_path / "out" / "report.md", tmp_path / "out" / "report.json"
    data = build_report(store, md, js, cfg=cfg)
    return data, md.read_text(encoding="utf-8"), json.loads(js.read_text(encoding="utf-8"))


V1 = [1] * 10 + [0] * 30  # recall 0.25
V2 = [1] * 30 + [0] * 10  # recall 0.75


def test_report_retrain_decision_md_and_json(research, tmp_path):
    for seed in range(3):
        research.insert_run(_rec("v1", seed, V1))
        research.insert_run(_rec("v2", seed, V2))
        research.insert_run(_rec("v1:ai", seed, V1))
        research.insert_run(_rec("v2:ai", seed, V1))  # AE alone: no change
        # other budgets never enter the decision
        research.insert_run(_rec("v1", seed, [1] * 40, budget=10))
    # seed 3 has only the champion arm: dropped with a note
    research.insert_run(_rec("v1", 3, V1))
    # a CalibrationError record at budget 2 is excluded
    research.insert_run(_rec("v2", 4, [], metrics=None, per_event={}, error="cal too small"))
    research.insert_run(_rec("v1", 4, V1))
    data, text, raw = _build(research, tmp_path)
    assert raw == data
    assert "retrain_decision" in raw

    rd = raw["retrain_decision"]
    assert rd["champion"] == "v1" and rd["challenger"] == "v2"
    assert rd["budget"] == 2 and rd["provenance"] == "REAL"
    assert rd["n_seeds"] == 3 and rd["n_events"] == 40
    assert rd["diff_mean"] == pytest.approx(0.5)
    lo, hi = rd["ci"]
    assert 0 < lo < 0.5 < hi
    assert rd["accept"] is True and rd["reason_th"].startswith("ยอมรับ challenger")
    # seeds 3 and 4 lack an OK challenger record: dropped, and the report says so
    assert any("3" in n and "4" in n and "seed" in n for n in rd["notes"])
    assert any("null" in n for n in rd["notes"])
    # series for the chart: the cluster ratio estimator, unit = event
    s1, s2 = rd["series"]["v1"], rd["series"]["v2"]
    assert s1["mean"] == pytest.approx(0.25) and s2["mean"] == pytest.approx(0.75)
    hits = np.array(V1, dtype=float) * 3
    mean, ci = ratio_bootstrap_ci(hits, np.full(40, 3.0), 300, 0)
    assert s1["mean"] == pytest.approx(mean) and s1["ci"] == pytest.approx(list(ci))
    assert s1["ci"][0] < s1["mean"] < s1["ci"][1]

    ai = rd["ai_only"]
    assert ai["champion"] == "v1:ai" and ai["challenger"] == "v2:ai"
    assert ai["n_seeds"] == 3 and ai["diff_mean"] == 0.0 and ai["ci"] == [0.0, 0.0]
    assert ai["accept"] is False and set(ai["series"]) == {"v1:ai", "v2:ai"}

    assert "## retrain_drift: champion v1 vs challenger v2" in text
    assert "AE alone (diagnostic)" in text
    assert rd["reason_th"] in text and ai["reason_th"] in text
    assert "ยอมรับ challenger" in text


def test_report_without_retrain_records(research, tmp_path):
    data, text, raw = _build(research, tmp_path)
    assert "retrain_decision" in raw and raw["retrain_decision"] is None
    assert "ยังไม่มีผล retrain_drift" in text


def test_report_retrain_without_both_ok_arms(research, tmp_path):
    # v2 failed calibration on every seed; the AE-only pair is missing entirely
    for seed in range(2):
        research.insert_run(_rec("v1", seed, V1))
        research.insert_run(_rec("v2", seed, [], metrics=None, per_event={}, error="cal"))
    _, text, raw = _build(research, tmp_path)
    assert "retrain_decision" in raw
    rd = raw["retrain_decision"]
    assert rd is not None and rd["accept"] is None and rd["diff_mean"] is None
    assert rd["series"] is None and rd["n_seeds"] == 0
    assert rd["reason_th"] and rd["reason_th"] in text
    assert rd["ai_only"] is None
    assert "AE alone (diagnostic): ยังไม่มีผล" in text


def test_report_retrain_without_shared_events(research, tmp_path):
    # both arms OK but on disjoint events: nothing to compare, undecided with a Thai reason
    research.insert_run(_rec("v1", 0, V1))
    v2 = _rec("v2", 0, V2)
    v2["per_event"] = {f"other-{k}": pe for k, pe in v2["per_event"].items()}
    research.insert_run(v2)
    _, text, raw = _build(research, tmp_path)
    rd = raw["retrain_decision"]
    assert rd["n_seeds"] == 1 and rd["n_events"] == 0
    assert rd["accept"] is None and rd["ci"] == [0.0, 0.0]
    assert "ไม่มีเหตุการณ์" in rd["reason_th"]
    assert rd["series"]["v1"] == {"mean": None, "ci": None}
    assert any("0 จาก 80" in n for n in rd["notes"])
    (row,) = [ln for ln in text.splitlines() if ln.startswith("| full stack (deployed) |")]
    assert not re.search(r"\bnan\b", row), row
    assert row.endswith("| ยังตัดสินไม่ได้ |")


def test_report_retrain_uses_only_the_latest_config_hash(research, tmp_path):
    t = 1_000
    for seed in range(3):  # an older config where the challenger looked worse
        for variant, det in (("v1", V2), ("v2", V1), ("v1:ai", V1), ("v2:ai", V1)):
            t += 1
            research.insert_run(
                _rec(
                    variant,
                    seed,
                    det,
                    config_hash="old",
                    created_us=t,
                    run_id=f"retrain_drift|{variant}|b2|s{seed}|old",
                )
            )
    for seed in range(2):
        for variant, det in (("v1", V1), ("v2", V2)):
            t += 1
            research.insert_run(_rec(variant, seed, det, created_us=t))
    _, text, raw = _build(research, tmp_path)
    rd = raw["retrain_decision"]
    assert rd["n_seeds"] == 2 and rd["seeds"] == [0, 1]
    assert rd["diff_mean"] == pytest.approx(0.5) and rd["accept"] is True
    ignored = "ไม่นับ 6 run ของ config_hash เก่า (ใช้เฉพาะ config_hash ล่าสุด cfg)"
    assert ignored in rd["notes"]
    # the AE-only pair only exists under the old config: its block uses those records alone,
    # and both blocks say that they come from different configs
    ai = rd["ai_only"]
    assert ai["n_seeds"] == 3 and ai["config_hash"] == "old" and rd["config_hash"] == "cfg"
    assert not any(n.startswith("ไม่นับ") and "config_hash" in n for n in ai["notes"])
    mismatch = (
        "config_hash ไม่ตรงกัน: full stack ใช้ cfg แต่ AE alone (diagnostic) ใช้ old "
        "จึงเทียบสองบล็อกนี้กันตรง ๆ ไม่ได้"
    )
    assert mismatch in rd["notes"] and mismatch in ai["notes"]
    assert f"  - {ignored}" in text.splitlines() and f"  - {mismatch}" in text.splitlines()
    # the tables pool both configs while the decision uses the latest: the report warns
    assert raw["notes"][0] == MIXED_CONFIG_NOTE
    assert f"**คำเตือน:** {MIXED_CONFIG_NOTE}" in text.splitlines()


def test_report_retrain_same_config_has_no_mismatch_note(research, tmp_path):
    for seed in range(2):
        for variant, det in (("v1", V1), ("v2", V2), ("v1:ai", V1), ("v2:ai", V2)):
            research.insert_run(_rec(variant, seed, det))
    rd = _build(research, tmp_path)[2]["retrain_decision"]
    assert rd["config_hash"] == rd["ai_only"]["config_hash"] == "cfg"
    assert not any("config_hash" in n for n in rd["notes"] + rd["ai_only"]["notes"])
    data, text, _ = _build(research, tmp_path)
    assert MIXED_CONFIG_NOTE not in data["notes"] and MIXED_CONFIG_NOTE not in text


def test_report_retrain_says_it_cannot_decide_on_few_events(research, tmp_path):
    n = MIN_EVENTS - 1
    for seed in range(2):
        research.insert_run(_rec("v1", seed, [0] * n))
        research.insert_run(_rec("v2", seed, [1] * n))
    _, text, raw = _build(research, tmp_path)
    rd = raw["retrain_decision"]
    assert rd["n_events"] == n and rd["diff_mean"] == pytest.approx(1.0)
    assert rd["accept"] is None
    assert "ตัดสินไม่ได้" in rd["reason_th"] and f"เพียง {n} ครั้ง" in rd["reason_th"]
    (row,) = [ln for ln in text.splitlines() if ln.startswith("| full stack (deployed) |")]
    assert row.endswith("| ยังตัดสินไม่ได้ |")
    assert rd["reason_th"] in text


def test_report_retrain_decides_at_the_minimum_event_count(research, tmp_path):
    for seed in range(2):
        research.insert_run(_rec("v1", seed, [0] * MIN_EVENTS))
        research.insert_run(_rec("v2", seed, [1] * MIN_EVENTS))
    assert _build(research, tmp_path)[2]["retrain_decision"]["accept"] is True


_BUNDLE_SCRIPT = r"""
import json, pickle, sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from idsai.ai.chief import AiFusion, Chief
from idsai.ai.learning import load_bundle, save_bundle
from idsai.config import config_hash, load_settings
from idsai.detection.features.bars import trades_to_bars
from idsai.detection.features.windows import compute_windows
from idsai.detection.registry import build_detectors
from idsai.pipeline.batch import score_windows
from tests.synth import day_us, make_trades

mode, root = sys.argv[1], Path(sys.argv[2])
cfg = load_settings()  # configs/default.yaml: rules, robust_z, cusum, ai.iforest; shift 1
D = date(2025, 3, 1)
t0 = day_us(D)
bars = trades_to_bars(make_trades(D, 2, symbol="TESTBTC", seed=4, rate=0.3), "TESTBTC", t0,
                      t0 + 2 * 86_400_000_000)
win = compute_windows(bars, cfg.features, 1e-4).filter(pl.col("t_end_us") >= t0 + 86_400_000_000)
if mode == "save":
    dets = build_detectors(cfg.detectors, cfg, 0)
    for d in dets:
        if d.layer == "ai":
            d.fit(win)
    scored = score_windows(win, dets)
    cols = ["symbol", "t_end_us", *[f"score__{d.name}" for d in dets]]
    chief = Chief(dets, cfg.chief, config_hash(cfg), {"ai.iforest": "test-1"},
                  ai=AiFusion.from_cfg(cfg.chief))
    chief.calibrate(scored.select(cols))
    chief.fit_threshold(chief.fuse(scored.select(cols)), 10.0)
    meta = {"data_hash": "d", "config_hash": config_hash(cfg), "seed": 0, "created_at": "t",
            "train_range": ["2025-03-02", "2025-03-02"]}
    save_bundle(root, list(cfg.detectors), dets, chief, meta)
else:
    dets, chief, _ = load_bundle(root, cfg)
    scored = score_windows(win, dets)
fused = chief.fuse(scored)
out = {"F": fused["F"].to_list(), "iforest": scored["score__ai.iforest"].to_list(),
       "tau": chief.tau, "ai": chief.ai.as_dict()}
(root.parent / f"{mode}.json").write_text(json.dumps(out), encoding="utf-8")
print("ok")
"""


def test_bundle_saved_in_one_process_loads_in_another(tmp_path):
    # the Docker build saves the bundle, serve loads it in a fresh interpreter: the sklearn
    # model's pickle is not byte-stable across processes, so the bundle check must not depend
    # on re-pickling (it hashes the stored file bytes)
    import subprocess
    import sys

    from idsai.config import REPO_ROOT

    root = tmp_path / "bundle"
    for mode in ("save", "load"):
        r = subprocess.run(
            [sys.executable, "-c", _BUNDLE_SCRIPT, mode, str(root)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=REPO_ROOT,
            timeout=600,
        )
        assert r.returncode == 0, r.stderr
    saved = json.loads((tmp_path / "save.json").read_text(encoding="utf-8"))
    loaded = json.loads((tmp_path / "load.json").read_text(encoding="utf-8"))
    assert loaded["ai"] == {"mode": "shift", "shift": 1.0, "regime_feature": None}
    assert saved == loaded  # same scores, F and tau after the round trip
    assert max(saved["F"]) > 0
    # a changed value in the stored model is still refused
    from idsai.ai.learning import BundleError, load_bundle
    from idsai.config import load_settings

    (path,) = (root / "registry" / "ai.iforest").glob("v*/state.pkl")
    state = pickle.loads(path.read_bytes())
    state["mu"] = state["mu"] + 1e-9
    path.write_bytes(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    with pytest.raises(BundleError, match="ไม่ตรงกับที่บันทึกไว้"):
        load_bundle(root, load_settings())


def test_ae_bundle_sha_is_unchanged_by_hashing_the_file():
    # bundles built before: sha = sha256(pickle.dumps(state_dict)); the registry file holds
    # exactly those bytes, so hashing the file keeps their shas
    import hashlib
    import io

    from idsai.ai.autoencoder import AutoencoderDetector
    from idsai.ai.learning import _bytes_sha
    from idsai.config import AECfg

    rng = np.random.default_rng(0)

    def f32(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    det = AutoencoderDetector(["a", "b"], AECfg(hidden=4, bottleneck=2))
    det.load_state_dict(
        {
            "features": ["a", "b"],
            "cfg": AECfg(hidden=4, bottleneck=2).model_dump(),
            "seed": 0,
            "center": np.zeros(2),
            "scale": np.ones(2),
            "net": {
                "0.weight": f32(4, 2),
                "0.bias": f32(4),
                "2.weight": f32(2, 4),
                "2.bias": f32(2),
                "3.weight": f32(4, 2),
                "3.bias": f32(4),
                "5.weight": f32(2, 4),
                "5.bias": f32(2),
            },  # fmt: skip
        }
    )
    old = hashlib.sha256(pickle.dumps(det.state_dict(), protocol=pickle.HIGHEST_PROTOCOL))
    buf = io.BytesIO()
    pickle.dump(det.state_dict(), buf, protocol=pickle.HIGHEST_PROTOCOL)  # as the registry does
    assert _bytes_sha(buf.getvalue()) == old.hexdigest()[:12]


def test_bundle_loads_the_naming_rules_only_with_attr_rules(tmp_path):
    # M3: naming rules hold no state, so a bundle stores nothing of them; load_bundle rebuilds
    # them from the config (the flag changes config_hash, so an off bundle never loads on)
    from datetime import date

    import polars as pl

    from idsai.ai.chief import AiFusion, Chief
    from idsai.ai.learning import load_bundle, save_bundle
    from idsai.config import config_hash
    from idsai.detection.features.bars import trades_to_bars
    from idsai.detection.features.windows import compute_windows
    from idsai.detection.registry import build_detectors
    from idsai.pipeline.batch import score_windows
    from tests.synth import S, day_us, make_trades

    on = load_settings(overrides={"chief": {"attr_rules": True}})
    d0 = date(2025, 3, 1)
    t0 = day_us(d0)
    bars = trades_to_bars(
        make_trades(d0, 2, symbol="TESTBTC", seed=4, rate=0.3), "TESTBTC", t0, t0 + 2 * 86_400 * S
    )
    win = compute_windows(bars, on.features, 1e-4).filter(pl.col("t_end_us") >= t0 + 86_400 * S)
    names = ["rules", "stat.robust_z", "stat.cusum"]
    dets = build_detectors(names, on, 0)
    scored = score_windows(win, dets)
    chief = Chief(dets, on.chief, config_hash(on), {}, ai=AiFusion.from_cfg(on.chief))
    chief.calibrate(scored)
    chief.fit_threshold(chief.fuse(scored), 10.0)
    meta = {"data_hash": "d", "config_hash": config_hash(on), "seed": 0, "created_at": "t",
            "train_range": ["2025-03-02", "2025-03-02"]}  # fmt: skip
    save_bundle(tmp_path / "b", names, dets, chief, meta)
    _, loaded, _ = load_bundle(tmp_path / "b", on)
    assert [d.name for d in loaded.naming] == ["attr.100191", "attr.100591"]
    assert loaded.state_dict()["cal"].keys() == chief.state_dict()["cal"].keys()


def test_bundle_round_trip_with_rule_rev2(tmp_path):
    # M5: rules hold no state, so load_bundle rebuilds rev 2 from the config (`chief.rule_rev2`);
    # the Chief's calibration of `rule.<sid>` round-trips, and the flag changes config_hash, so a
    # flag-off bundle never loads under rev 2 (and the other way round)
    from datetime import date

    import polars as pl

    from idsai.ai.chief import AiFusion, Chief
    from idsai.ai.learning import BundleError, load_bundle, save_bundle
    from idsai.config import config_hash
    from idsai.detection.features.bars import trades_to_bars
    from idsai.detection.features.windows import compute_windows
    from idsai.detection.registry import build_detectors
    from idsai.pipeline.batch import score_windows
    from tests.synth import S, day_us, make_trades

    on = load_settings(overrides={"chief": {"rule_rev2": [100101, 100102]}})
    d0 = date(2025, 3, 1)
    t0 = day_us(d0)
    bars = trades_to_bars(
        make_trades(d0, 2, symbol="TESTBTC", seed=4, rate=0.3), "TESTBTC", t0, t0 + 2 * 86_400 * S
    )
    win = compute_windows(bars, on.features, 1e-4).filter(pl.col("t_end_us") >= t0 + 86_400 * S)
    names = ["rules", "stat.robust_z", "stat.cusum"]
    dets = build_detectors(names, on, 0)
    scored = score_windows(win, dets)
    chief = Chief(dets, on.chief, config_hash(on), {}, ai=AiFusion.from_cfg(on.chief))
    chief.calibrate(scored)
    chief.fit_threshold(chief.fuse(scored), 10.0)
    meta = {"data_hash": "d", "config_hash": config_hash(on), "seed": 0, "created_at": "t",
            "train_range": ["2025-03-02", "2025-03-02"]}  # fmt: skip
    save_bundle(tmp_path / "b", names, dets, chief, meta)
    got, loaded, _ = load_bundle(tmp_path / "b", on)
    revs = {d.signature_id: d.rev for d in got if d.layer == "rule"}
    assert revs[100101] == 2 and revs[100102] == 2 and revs[100501] == 1
    assert pickle.dumps(loaded.state_dict()) == pickle.dumps(chief.state_dict())
    fused = loaded.fuse(score_windows(win, got))
    assert fused["F"].to_list() == chief.fuse(scored)["F"].to_list()
    with pytest.raises(BundleError):
        load_bundle(tmp_path / "b", load_settings())
