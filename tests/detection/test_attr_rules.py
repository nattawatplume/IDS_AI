"""M3 naming-only signatures (`configs/signatures_attr.yaml`) and 100501 rev 2 (synthetic only)."""

import numpy as np
import polars as pl

from idsai.config import load_settings
from idsai.detection.registry import build_detectors, naming_detectors, rule_signatures
from idsai.detection.rules import (
    NAMING_SIGNATURES_PATH,
    WASH2_SIGNATURES_PATH,
    SignatureDetector,
    load_signatures,
)

OFF = load_settings(overrides={"chief": {"attr_rules": False}})
ON = load_settings(overrides={"chief": {"attr_rules": True}})


def test_naming_file_holds_only_attr_layer_rules_in_priority_order():
    sigs = load_signatures(NAMING_SIGNATURES_PATH)
    assert [s.sid for s in sigs] == [100191, 100591]  # priority: pump burst, then wash
    assert all(s.layer == "attr" for s in sigs)
    assert [s.technique for s in sigs] == ["MM-T001.001", "MM-T005.001"]
    burst, wash = sigs
    assert {(c.feature, c.op, c.thr) for c in burst.conditions} == {
        ("ret_60", ">=", 0.02),
        ("z_rush_60", ">=", 4.0),
    }
    assert {(c.feature, c.op, c.thr) for c in wash.conditions} == {
        ("z_pingpong_300", ">=", 5.0),
        ("ret_300", "abs<=", 0.002),
    }
    # sids never collide with the F-rules
    assert not {s.sid for s in sigs} & {s.sid for s in load_signatures()}


def test_naming_detectors_only_with_the_flag_and_never_in_the_rule_layer():
    assert naming_detectors(OFF) == []
    dets = naming_detectors(ON)
    assert [d.name for d in dets] == ["attr.100191", "attr.100591"]
    assert all(d.layer == "attr" for d in dets)
    assert [d.technique_ids for d in dets] == [["MM-T001"], ["MM-T005"]]
    # the flag adds nothing to the detectors that enter F
    names_off = [d.name for d in build_detectors(["rules"], OFF)]
    assert [d.name for d in build_detectors(["rules"], ON)] == names_off
    assert [s.sid for s in rule_signatures(ON)] == [s.sid for s in rule_signatures(OFF)]


def test_naming_rule_fires_like_a_signature():
    burst = naming_detectors(ON)[0]
    w = pl.DataFrame(
        {"ret_60": [0.03, 0.01, 0.03, float("nan")], "z_rush_60": [5.0, 5.0, 3.0, 5.0]}
    )
    assert burst.fired(burst.score(w)).tolist() == [True, False, False, False]


def test_wash_rev2_drops_only_the_volume_condition():
    (rev1,) = [s for s in load_signatures() if s.sid == 100501]
    (rev2,) = load_signatures(WASH2_SIGNATURES_PATH)
    assert (rev1.rev, rev2.rev) == (1, 2)
    assert (rev2.sid, rev2.layer, rev2.technique) == (100501, "rule", rev1.technique)
    keep = [c for c in rev1.conditions if c.feature != "z_quote_vol_300"]
    assert rev2.conditions == keep
    w = pl.DataFrame({"z_pingpong_300": [6.0], "ret_300": [0.001], "z_quote_vol_300": [1.0]})
    assert SignatureDetector(rev2).fired(SignatureDetector(rev2).score(w)).tolist() == [True]
    assert not np.any(SignatureDetector(rev1).fired(SignatureDetector(rev1).score(w)))
