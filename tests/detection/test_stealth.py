"""M8 `chief.stealth`: rule 100102 as a naming-only rule ("attr") or at rev 3 with a
trading-activity gate ("act", `configs/signatures_r3.yaml`), through the registry (synthetic
only)."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from idsai.ai.chief import NAMING_SOURCE, Chief, ChiefStream
from idsai.config import load_settings
from idsai.detection.registry import (
    build_detectors,
    naming_detectors,
    rule_signatures,
    stealth_naming_signature,
    stealth_rev3_signature,
)
from idsai.detection.rules import (
    R3_SIGNATURES_PATH,
    STEALTH_SID,
    SignatureDetector,
    load_signatures,
)
from tests.ai.test_chief import CFG, MODES, VERSIONS, _mixed_world

OFF = load_settings()
ATTR = load_settings(overrides={"chief": {"stealth": "attr"}})
ACT = load_settings(overrides={"chief": {"stealth": "act"}})
(REV1,) = [s for s in load_signatures() if s.sid == STEALTH_SID]
GATE = ("z_n_trades_60", ">=", 3.0)


def test_rev3_file_is_rev1_plus_the_activity_gate():
    (r3,) = load_signatures(R3_SIGNATURES_PATH)
    assert r3 == stealth_rev3_signature()
    assert (r3.sid, r3.layer, r3.rev, r3.technique, r3.name) == (
        STEALTH_SID, "rule", 3, REV1.technique, REV1.name,
    )  # fmt: skip
    cond = [(c.feature, c.op, c.thr) for c in r3.conditions]
    assert cond == [*((c.feature, c.op, c.thr) for c in REV1.conditions), GATE]


# a row that meets every rev 1 condition
REV1_OK = {"cusum_imb": 9.0, "ret_1800": 0.05, "buy_imb_1800": 0.5}


def test_rev3_fires_only_with_rev1_and_the_gate():
    xs = [2.999, 3.0, 5.0, float("nan")]
    w = pl.DataFrame({"z_n_trades_60": xs, **{k: [v] * len(xs) for k, v in REV1_OK.items()}})
    r1, r3 = SignatureDetector(REV1), SignatureDetector(stealth_rev3_signature())
    assert r1.fired(r1.score(w)).tolist() == [True] * 4
    assert r3.fired(r3.score(w)).tolist() == [False, True, True, False]
    # the rev 1 conditions still bind
    for k in REV1_OK:
        bad = w.with_columns(pl.lit(-1.0).alias(k))
        assert not np.any(r3.fired(r3.score(bad))), k


def test_off_keeps_rev1_in_f_and_no_stealth_naming():
    revs = [(s.sid, s.rev) for s in rule_signatures(OFF) if s.sid == STEALTH_SID]
    assert revs == [(STEALTH_SID, 1)]
    assert "attr.100102" not in [d.name for d in naming_detectors(OFF)]


def test_act_swaps_rev3_in_place_under_the_same_name():
    off, act = rule_signatures(OFF), rule_signatures(ACT)
    assert [s.sid for s in act] == [s.sid for s in off]
    assert [s for s in act if s.sid != STEALTH_SID] == [s for s in off if s.sid != STEALTH_SID]
    assert {s.sid: s.rev for s in act}[STEALTH_SID] == 3
    assert [d.name for d in build_detectors(["rules"], ACT)] == [
        d.name for d in build_detectors(["rules"], OFF)
    ]
    assert [d.name for d in naming_detectors(ACT)] == [d.name for d in naming_detectors(OFF)]


def test_attr_moves_the_rule_from_f_to_the_naming_layer_last():
    off, attr = rule_signatures(OFF), rule_signatures(ATTR)
    assert [s for s in attr] == [s for s in off if s.sid != STEALTH_SID]
    assert "rule.100102" not in [d.name for d in build_detectors(["rules"], ATTR)]
    names = [d.name for d in naming_detectors(ATTR)]
    assert names == [*(d.name for d in naming_detectors(OFF)), "attr.100102"]
    sig = stealth_naming_signature()
    assert sig.layer == "attr"
    assert sig.model_copy(update={"layer": "rule"}) == REV1  # same conditions, sid, technique, rev
    (namer,) = [d for d in naming_detectors(ATTR) if d.name == "attr.100102"]
    assert namer.technique == "MM-T001.002" and namer.signature_id == STEALTH_SID


def test_act_changes_only_the_stealth_rule_score_and_attr_drops_it():
    from idsai.detection.features.bars import trades_to_bars
    from idsai.detection.features.windows import compute_windows
    from idsai.pipeline.batch import score_windows
    from tests.synth import S, day_us, make_trades

    d0 = date(2025, 3, 1)
    t0 = day_us(d0)
    bars = trades_to_bars(
        make_trades(d0, 2, symbol="TESTBTC", seed=4, rate=0.3), "TESTBTC", t0, t0 + 2 * 86_400 * S
    )
    win = compute_windows(bars, OFF.features, 1e-4)
    names = ["rules", "stat.robust_z", "stat.cusum"]
    a = score_windows(win, build_detectors(names, OFF))
    b = score_windows(win, build_detectors(names, ACT))
    c = score_windows(win, build_detectors(names, ATTR))
    assert a.columns == b.columns
    assert c.columns == [k for k in a.columns if k != "score__rule.100102"]
    for k in a.columns:
        x, y = a[k].to_numpy(), b[k].to_numpy()
        same = np.array_equal(x, y, equal_nan=x.dtype.kind == "f")
        assert same == (k != "score__rule.100102"), k
        if k in c.columns:
            assert np.array_equal(x, c[k].to_numpy(), equal_nan=x.dtype.kind == "f"), k
    # rev 3 fires on a subset of rev 1's rows
    r1, r3 = a["score__rule.100102"].to_numpy(), b["score__rule.100102"].to_numpy()
    with np.errstate(invalid="ignore"):
        assert not np.any((r3 >= 0) & ~(r1 >= 0))


def _run(ch: Chief, sc: pl.DataFrame):
    ch.calibrate(sc)
    fused = ch.fuse(sc)
    ch.fit_threshold(fused, 15.0)
    batch = ch.episodes(fused)
    st = ChiefStream(ch, provenance="replay")
    emitted = [a for r in sc.to_dicts() for a in st.step(r)] + st.flush()
    return fused, batch, emitted


@pytest.mark.parametrize("mode", ["fuse", "shift"])
def test_attr_stealth_never_adds_to_f_but_names_mm_t000_episodes(mode):
    dets, sc = _mixed_world(seed=11)
    rng = np.random.default_rng(3)
    sc = sc.with_columns(
        pl.Series("score__attr.100102", np.where(rng.random(sc.height) < 0.05, 0.5, -0.5))
    )
    (namer,) = [d for d in naming_detectors(ATTR) if d.name == "attr.100102"]
    f0, b0, _ = _run(Chief(dets, CFG, "cfg123", VERSIONS, ai=MODES[mode]), sc)
    on = Chief(dets, CFG, "cfg123", VERSIONS, ai=MODES[mode], naming=[namer])
    f1, b1, e1 = _run(on, sc)
    assert f1.equals(f0)  # F, every p, agreement, row technique: identical
    assert "p__attr.100102" not in f1.columns
    assert len(b1) == len(b0)
    named = 0
    for a, b in zip(b0, b1, strict=True):
        assert (a.alert_id, a.window, a.fused_p) == (b.alert_id, b.window, b.fused_p)
        if a.technique_id != "MM-T000":
            assert (b.technique_id, b.signature_id, b.technique_source) == (
                a.technique_id, a.signature_id, None,
            )  # fmt: skip
        elif b.technique_id != "MM-T000":
            named += 1
            assert (b.technique_id, b.signature_id, b.rev) == ("MM-T001", STEALTH_SID, 1)
            assert b.technique_source == NAMING_SOURCE
    assert named > 0
    closed = [a for a in e1 if a.status == "closed"]
    assert sorted(a.model_dump_json() for a in closed) == sorted(a.model_dump_json() for a in b1)
