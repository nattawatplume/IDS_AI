"""M5 rev 2 of the pump rules 100101 / 100102 (`configs/signatures_r2.yaml`, the eval-only
alternative `signatures_r2_alt.yaml`) and the `chief.rule_rev2` flag (synthetic only)."""

import numpy as np
import polars as pl
import pytest

from idsai.config import load_settings
from idsai.detection.registry import build_detectors, rev2_signatures, rule_signatures
from idsai.detection.rules import (
    R2_ALT_SIGNATURES_PATH,
    R2_SIGNATURES_PATH,
    SignatureDetector,
    load_signatures,
)

OFF = load_settings()
BOTH = load_settings(overrides={"chief": {"rule_rev2": [100101, 100102]}})
REV1 = {s.sid: s for s in load_signatures()}
# sid -> (the one condition rev 2 changes, rev 1 threshold, rev 2 threshold)
CHANGED = {100101: ("z_quote_vol_60", 6.0, 3.0), 100102: ("cusum_imb", 8.0, 6.0)}


def _only_threshold_changed(rev2, feature, thr):
    rev1 = REV1[rev2.sid]
    assert (rev1.rev, rev2.rev) == (1, 2)
    assert (rev2.layer, rev2.technique, rev2.name) == (rev1.layer, rev1.technique, rev1.name)
    assert [(c.feature, c.op) for c in rev2.conditions] == [
        (c.feature, c.op) for c in rev1.conditions
    ]
    for c1, c2 in zip(rev1.conditions, rev2.conditions, strict=True):
        assert c2.thr == (thr if c1.feature == feature else c1.thr)


def test_rev2_file_changes_one_threshold_per_rule():
    sigs = load_signatures(R2_SIGNATURES_PATH)
    assert [s.sid for s in sigs] == [100101, 100102]
    for s in sigs:
        feature, _, thr = CHANGED[s.sid]
        _only_threshold_changed(s, feature, thr)


def test_alt_file_holds_100101_at_2_5():
    (alt,) = load_signatures(R2_ALT_SIGNATURES_PATH)
    _only_threshold_changed(alt, "z_quote_vol_60", 2.5)


# a row that meets every other condition of both rules
OTHER = {"ret_60": 0.05, "z_rush_60": 5.0, "ret_1800": 0.05, "buy_imb_1800": 0.5}


@pytest.mark.parametrize("sid", [100101, 100102])
def test_rev2_fires_at_its_threshold_and_not_below(sid):
    feature, thr1, thr2 = CHANGED[sid]
    xs = [thr2 - 1e-6, thr2, (thr2 + thr1) / 2, thr1, float("nan")]
    w = pl.DataFrame({feature: xs, **{k: [v] * len(xs) for k, v in OTHER.items()}})
    rev1 = SignatureDetector(REV1[sid])
    rev2 = SignatureDetector(rev2_signatures([sid])[sid])
    assert rev2.fired(rev2.score(w)).tolist() == [False, True, True, True, False]
    assert rev1.fired(rev1.score(w)).tolist() == [False, False, False, True, False]
    # the other conditions still bind: rev 2 alone never fires
    bad = w.with_columns(pl.lit(-1.0).alias(next(iter(c.feature for c in REV1[sid].conditions
                                                     if c.feature != feature))))  # fmt: skip
    assert not np.any(rev2.fired(rev2.score(bad)))


def test_alt_fires_from_2_5():
    (alt,) = load_signatures(R2_ALT_SIGNATURES_PATH)
    d = SignatureDetector(alt)
    w = pl.DataFrame({"z_quote_vol_60": [2.4999, 2.5, 3.0], "ret_60": [0.05] * 3,
                      "z_rush_60": [5.0] * 3})  # fmt: skip
    assert d.fired(d.score(w)).tolist() == [False, True, True]


def test_flag_off_keeps_rev1_and_on_swaps_in_place_under_the_same_name():
    off = rule_signatures(OFF)
    assert [s for s in off if s.sid in REV1] == [s for s in load_signatures() if s.layer == "rule"]
    assert all(s.rev == 1 for s in off if s.sid in CHANGED)
    for flag, want in (([100101], {100101: 2, 100102: 1}), ([100102], {100101: 1, 100102: 2}),
                       ([100101, 100102], {100101: 2, 100102: 2})):  # fmt: skip
        cfg = load_settings(overrides={"chief": {"rule_rev2": flag}})
        on = rule_signatures(cfg)
        assert [s.sid for s in on] == [s.sid for s in off]  # same order, same rules
        assert {s.sid: s.rev for s in on if s.sid in CHANGED} == want
        assert [s for s in on if s.sid not in flag] == [s for s in off if s.sid not in flag]
        names = [d.name for d in build_detectors(["rules"], cfg)]
        assert names == [d.name for d in build_detectors(["rules"], OFF)]


def test_flag_rejects_unknown_or_repeated_rules():
    with pytest.raises(ValueError, match="rule rev 2"):
        build_detectors(["rules"], load_settings(overrides={"chief": {"rule_rev2": [100501]}}))
    with pytest.raises(ValueError, match="twice"):
        load_settings(overrides={"chief": {"rule_rev2": [100101, 100101]}})


def test_flag_changes_only_the_listed_rules_scores():
    from datetime import date

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
    b = score_windows(win, build_detectors(names, BOTH))
    assert a.columns == b.columns
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        same = np.array_equal(x, y, equal_nan=x.dtype.kind == "f")
        assert same == (c not in ("score__rule.100101", "score__rule.100102")), c
