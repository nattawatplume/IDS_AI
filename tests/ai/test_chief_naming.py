"""M3 naming-only rules in the Chief: they name an MM-T000 episode and change nothing else."""

import hashlib
import json
import pickle

import numpy as np
import polars as pl
import pytest

from idsai.ai.chief import NAMING_SOURCE, Chief, ChiefStream
from idsai.config import load_settings
from idsai.detection.registry import naming_detectors
from idsai.detection.rules import Condition, Signature, SignatureDetector
from tests.ai.test_chief import (
    CFG,
    MODES,
    NAN,
    S_US,
    T0_US,
    VERSIONS,
    Stub,
    _mixed_world,
    ai_chief,
    cal_with,
    frame,
    rule,
)

# sha256 of the canonical outputs below, computed on the Chief before M3 (commit 77d78b2)
GOLDEN_OFF = "0945b9f300cf56d2f3738f34eed79235b31c6b822961f3b59aa8d29dc790a906"


def namer(sid: int, technique: str, feature: str = "na") -> SignatureDetector:
    sig = Signature(
        sid=sid,
        technique=technique,
        layer="attr",
        rev=1,
        name="n",
        conditions=[Condition(feature=feature, op=">=", thr=1.0)],
    )
    return SignatureDetector(sig)


CFG_OFF = load_settings()
BURST, WASH = namer(191, "MM-T001.001", "na"), namer(591, "MM-T005.001", "nb")


def _canon(x):
    if isinstance(x, float):
        return float(f"{x:.10g}")
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_canon(v) for v in x]
    return x


def _run(ch: Chief, sc: pl.DataFrame):
    ch.calibrate(sc)
    fused = ch.fuse(sc)
    ch.fit_threshold(fused, 15.0)
    batch = ch.episodes(fused)
    st = ChiefStream(ch, provenance="replay")
    emitted = [a for r in sc.to_dicts() for a in st.step(r)] + st.flush()
    return fused, batch, emitted


def _with_naming_scores(sc: pl.DataFrame, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = sc.height
    return sc.with_columns(
        pl.Series("score__attr.191", np.where(rng.random(n) < 0.03, 0.5, -0.5)),
        pl.Series("score__attr.591", np.where(rng.random(n) < 0.03, 0.5, -0.5)),
    )


# ---------------------------------------------------------------- flag off: bit-identical


def test_flag_off_outputs_equal_the_pre_m3_golden():
    assert naming_detectors(load_settings()) == []
    h = hashlib.sha256()
    for mode in ("fuse", "shift"):
        dets, sc = _mixed_world(seed=11)
        ch = Chief(dets, CFG, "cfg123", VERSIONS, ai=MODES[mode], naming=naming_detectors(CFG_OFF))
        fused, batch, emitted = _run(ch, sc)
        assert all("technique_source" not in a.model_dump() for a in [*batch, *emitted])
        doc = {
            "tau": ch.tau,
            "F": fused["F"].to_list(),
            "technique": fused["technique"].to_list(),
            "agreement": fused["agreement"].to_list(),
            "alerts": [a.model_dump(mode="json") for a in [*batch, *emitted]],
        }
        h.update(json.dumps(_canon(doc), sort_keys=True).encode())
    assert h.hexdigest() == GOLDEN_OFF


# ---------------------------------------------------------------- flag on: only names change


@pytest.mark.parametrize("mode", ["fuse", "shift"])
def test_naming_changes_only_the_names_of_mm_t000_episodes(mode):
    dets, sc = _mixed_world(seed=11)
    sc = _with_naming_scores(sc, 5)
    f0, b0, e0 = _run(ai_chief(MODES[mode], *dets), sc)
    on = Chief(dets, CFG, "cfg123", VERSIONS, ai=MODES[mode], naming=[BURST, WASH])
    f1, b1, e1 = _run(on, sc)
    assert f1.equals(f0)  # F, every p, agreement, row technique: identical
    assert len(b1) == len(b0)
    named = 0
    for a, b in zip(b0, b1, strict=True):
        da, db = a.model_dump(), b.model_dump()
        for k in ("technique_id", "signature_id", "rev", "technique_source"):
            da.pop(k, None)
            db.pop(k, None)
        assert da == db  # ids, windows, fused_p, detectors, agreement, evidence, severity
        if a.technique_id != "MM-T000":
            assert (b.technique_id, b.signature_id, b.rev) == (
                a.technique_id,
                a.signature_id,
                a.rev,
            )
            assert b.technique_source is None
        elif b.technique_id != "MM-T000":
            named += 1
            assert b.technique_source == NAMING_SOURCE
            assert (b.technique_id, b.signature_id) in {("MM-T001", 191), ("MM-T005", 591)}
    assert named > 0
    # stream: the same alert ids and statuses' episodes; the closed alerts equal the batch
    closed = [a for a in e1 if a.status == "closed"]
    assert sorted(a.model_dump_json() for a in closed) == sorted(a.model_dump_json() for a in b1)
    assert {a.alert_id for a in e1} == {a.alert_id for a in e0}


def _calibrated(*namers: SignatureDetector) -> Chief:
    ch = Chief(
        [Stub("stat.a", "stat", requires=("fa",)), rule(501, "MM-T005.001")],
        CFG,
        "cfg123",
        VERSIONS,
        naming=list(namers),
    )
    fired = [-1.0] * 990 + [1.0] * 5 + [2.0] * 5
    ch.calibrate(cal_with("stat.a", range(1000), extra={"rule.501": fired}))
    ch.tau = 1.0
    return ch


def _both(ch: Chief, sc: pl.DataFrame):
    (batch,) = ch.episodes(ch.fuse(sc))
    st = ChiefStream(ch, provenance="replay")
    emitted = [a for r in sc.to_dicts() for a in st.step(r)] + st.flush()
    assert emitted[-1].status == "closed"
    assert emitted[-1].model_dump_json() == batch.model_dump_json()
    return batch, emitted


def test_an_f_rule_name_wins_over_a_naming_rule():
    ch = _calibrated(BURST)
    sc = frame("A", [0, 100], stat_a=[5000.0, -1.0], rule_501=[-1.0, 0.5], attr_191=[1.0, 1.0])
    batch, _ = _both(ch, sc)
    assert (batch.technique_id, batch.signature_id, batch.technique_source) == (
        "MM-T005",
        501,
        None,
    )


def test_naming_rule_names_an_unclassified_episode():
    ch = _calibrated(BURST)
    sc = frame("A", [0, 100], stat_a=[5000.0, 998.0], rule_501=[-1.0, -1.0], attr_191=[-1.0, 0.5])
    batch, emitted = _both(ch, sc)
    assert (batch.technique_id, batch.signature_id, batch.rev) == ("MM-T001", 191, 1)
    assert batch.technique_source == NAMING_SOURCE
    assert batch.peak_technique_id == "MM-T000"  # the peak row's own (detection) technique
    assert [(a.status, a.technique_id) for a in emitted] == [
        ("open", "MM-T000"),
        ("updated", "MM-T001"),
        ("closed", "MM-T001"),
    ]
    assert "technique_source" in json.loads(batch.model_dump_json())


def test_naming_follows_the_priority_order_not_the_time_order():
    ch = _calibrated(BURST, WASH)
    sc = frame(
        "A",
        [0, 100, 200],
        stat_a=[5000.0, 998.0, 998.0],
        rule_501=[-1.0, -1.0, -1.0],
        attr_591=[0.5, -1.0, -1.0],  # wash fires first in time
        attr_191=[-1.0, -1.0, 0.5],  # burst later, but it comes first in the list
    )
    batch, emitted = _both(ch, sc)
    assert (batch.technique_id, batch.signature_id) == ("MM-T001", 191)
    assert [a.signature_id for a in emitted] == [591, 191, 191]
    ch2 = _calibrated(WASH, BURST)
    assert ch2.episodes(ch2.fuse(sc))[0].signature_id == 591


def test_naming_reads_every_row_the_episode_spans_and_no_other():
    ch = _calibrated(BURST)
    # exceedances at 0 and 200; row 100 lies inside the span (F below tau), 300 after it
    inside = frame(
        "A",
        [0, 100, 200],
        stat_a=[5000.0, -1.0, 5000.0],
        rule_501=[-1.0] * 3,
        attr_191=[-1.0, 0.5, -1.0],
    )
    assert ch.fuse(inside)["F"][1] < ch.tau
    batch, _ = _both(ch, inside)
    assert batch.technique_id == "MM-T001" and batch.window == (T0_US, T0_US + 200 * S_US)
    for t_s, f_s in (([-100, 0, 200], [0.5, -1.0, -1.0]), ([0, 200, 300], [-1.0, -1.0, 0.5])):
        outside = frame(
            "A",
            t_s,
            stat_a=[-1.0, 5000.0, 5000.0] if t_s[0] < 0 else [5000.0, 5000.0, -1.0],
            rule_501=[-1.0] * 3,
            attr_191=f_s,
        )
        batch, _ = _both(ch, outside)
        assert batch.technique_id == "MM-T000" and batch.technique_source is None


def test_naming_rows_of_another_symbol_never_count():
    ch = _calibrated(BURST)
    sc = frame(
        ["A", "B", "A"],
        [0, 100, 200],
        stat_a=[5000.0, -1.0, 5000.0],
        rule_501=[-1.0] * 3,
        attr_191=[-1.0, 0.5, -1.0],
    )
    alerts = ch.episodes(ch.fuse(sc))
    assert [a.technique_id for a in alerts] == ["MM-T000"]


def test_pending_naming_hits_are_dropped_when_the_episode_closes():
    ch = _calibrated(BURST)
    sc = frame(
        "A",
        [0, 100, 1000],  # 100: inside the suppress gap after the last exceedance; 1000: new episode
        stat_a=[5000.0, -1.0, 5000.0],
        rule_501=[-1.0] * 3,
        attr_191=[-1.0, 0.5, -1.0],
    )
    batch = ch.episodes(ch.fuse(sc))
    st = ChiefStream(ch, provenance="replay")
    emitted = [a for r in sc.to_dicts() for a in st.step(r)] + st.flush()
    closed = [a for a in emitted if a.status == "closed"]
    assert [a.technique_id for a in batch] == ["MM-T000", "MM-T000"]
    assert [a.model_dump_json() for a in closed] == [a.model_dump_json() for a in batch]


# ---------------------------------------------------------------- parity and state


@pytest.mark.parametrize("mode", list(MODES))
def test_stream_matches_batch_with_naming_in_every_ai_mode(mode):
    dets, sc = _mixed_world(seed=7)
    sc = _with_naming_scores(sc, 9)
    ch = Chief(dets, CFG, "cfg123", VERSIONS, ai=MODES[mode], naming=[BURST, WASH])
    fused, batch, emitted = _run(ch, sc)
    assert len(batch) > 10
    assert any(a.technique_source == NAMING_SOURCE for a in batch)
    closed = [a for a in emitted if a.status == "closed"]
    assert sorted(a.model_dump_json() for a in closed) == sorted(a.model_dump_json() for a in batch)


def test_state_is_unchanged_by_naming_and_round_trips():
    dets, sc = _mixed_world(seed=3)
    sc = _with_naming_scores(sc, 2)
    plain = ai_chief(MODES["shift"], *dets)
    on = Chief(dets, CFG, "cfg123", VERSIONS, ai=MODES["shift"], naming=[BURST, WASH])
    for ch in (plain, on):
        ch.calibrate(sc)
        ch.fit_threshold(ch.fuse(sc), 15.0)
    s_plain, s_on = plain.state_dict(), on.state_dict()
    assert pickle.dumps(s_plain) == pickle.dumps(s_on)  # naming holds no calibration
    again = Chief(dets, CFG, "cfg123", VERSIONS, ai=MODES["shift"], naming=[BURST, WASH])
    again.load_state_dict(pickle.loads(pickle.dumps(s_on)))
    fused = again.fuse(sc)
    assert [a.model_dump_json() for a in again.episodes(fused)] == [
        a.model_dump_json() for a in on.episodes(on.fuse(sc))
    ]


# ---------------------------------------------------------------- guards


def test_naming_detectors_must_be_attr_layer_and_uniquely_named():
    dets = [Stub("stat.a", "stat")]
    with pytest.raises(ValueError, match="attr"):
        Chief(dets, CFG, "c", {}, naming=[rule(101, "MM-T001.001")])
    with pytest.raises(ValueError, match="duplicate"):
        Chief(dets, CFG, "c", {}, naming=[BURST, BURST])
    ch = Chief(dets, CFG, "c", {}, naming=[BURST])
    assert [d.name for d in ch.score_detectors] == ["stat.a", "attr.191"]
    assert [d.name for d in ch.detectors] == ["stat.a"]  # never fused


def test_missing_naming_score_column_raises():
    ch = _calibrated(BURST)
    sc = frame("A", [0], stat_a=[5000.0], rule_501=[-1.0])
    with pytest.raises(KeyError, match="attr.191"):
        ch.episodes(ch.fuse(sc))
    with pytest.raises(KeyError):
        ChiefStream(ch, provenance="replay").step(sc.to_dicts()[0])


def test_nan_naming_score_never_names():
    ch = _calibrated(BURST)
    sc = frame("A", [0], stat_a=[5000.0], rule_501=[-1.0], attr_191=[NAN])
    batch, _ = _both(ch, sc)
    assert batch.technique_id == "MM-T000"
