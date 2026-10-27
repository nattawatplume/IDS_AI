import math
import pickle

import numpy as np
import polars as pl
import pytest

from idsai.ai.baselines import IsolationForestDetector, PCADetector, RandomDetector
from idsai.config import file_sha, load_settings
from idsai.contracts.detector import Detector, Evidence
from idsai.detection.features.registry import model_features
from idsai.detection.registry import build_detectors, register, valid_names
from idsai.detection.rules import (
    SIGNATURES_PATH,
    Condition,
    Signature,
    SignatureDetector,
    load_signatures,
)
from idsai.detection.stats import CusumDetector, RobustZDetector
from idsai.detection.taxonomy import load_taxonomy, technique_ids

CFG = load_settings()
NAN = float("nan")


def frame(**cols) -> pl.DataFrame:
    return pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float64) for k, v in cols.items()})


def sig(conds, layer="rule", sid=999001, technique="MM-T001.001") -> Signature:
    return Signature(
        sid=sid,
        technique=technique,
        layer=layer,
        rev=1,
        name="t",
        conditions=[Condition(feature=f, op=op, thr=t) for f, op, t in conds],
    )


# ---------------------------------------------------------------- signatures


def test_margin_is_zero_at_threshold_and_fires():
    d = SignatureDetector(sig([("a", ">=", 6.0)]))
    s = d.score(frame(a=[6.0, 12.0, 3.0]))
    assert s.dtype == np.float64
    np.testing.assert_allclose(s, [0.0, 1.0, -0.5])
    assert d.fired(s).tolist() == [True, True, False]


def test_margin_le_direction():
    d = SignatureDetector(sig([("a", "<=", 2.0)]))
    np.testing.assert_allclose(d.score(frame(a=[2.0, 1.0, 3.0])), [0.0, 0.5, -0.5])


def test_margin_abs_le_is_symmetric():
    d = SignatureDetector(sig([("r", "abs<=", 0.002)]))
    s = d.score(frame(r=[0.002, -0.002, 0.001, -0.004, 0.0]))
    np.testing.assert_allclose(s, [0.0, 0.0, 0.5, -1.0, 1.0], atol=1e-12)
    assert d.fired(s).tolist() == [True, True, True, False, True]


def test_zero_threshold_uses_floor_not_division_by_zero():
    d = SignatureDetector(sig([("a", ">=", 0.0)]))
    s = d.score(frame(a=[0.0, 1e-9]))
    assert np.isfinite(s).all()
    assert s[0] == 0.0 and s[1] > 0


def test_score_is_min_over_conditions():
    d = SignatureDetector(sig([("a", ">=", 1.0), ("b", ">=", 10.0)]))
    s = d.score(frame(a=[3.0, 3.0], b=[10.0, 5.0]))
    np.testing.assert_allclose(s, [0.0, -0.5])
    assert d.fired(s).tolist() == [True, False]


def test_nan_required_gives_nan_score_and_not_fired():
    d = SignatureDetector(sig([("a", ">=", 1.0), ("b", ">=", 1.0)]))
    s = d.score(frame(a=[NAN, 5.0], b=[5.0, NAN]))
    assert np.isnan(s).all()
    assert d.fired(s).tolist() == [False, False]


def test_signature_metadata_from_yaml():
    by_sid = {d.signature_id: d for d in map(SignatureDetector, load_signatures())}
    assert set(by_sid) == {100101, 100102, 100103, 100501}
    burst = by_sid[100101]
    assert burst.name == "rule.100101"
    assert burst.layer == "rule"
    assert burst.technique_ids == ["MM-T001"]
    assert burst.rev == 1
    assert burst.requires == ["z_quote_vol_60", "ret_60", "z_rush_60"]
    kk = by_sid[100103]
    assert kk.name == "baseline.kk" and kk.layer == "baseline"
    assert kk.technique_ids == ["MM-T001"]
    assert by_sid[100501].technique_ids == ["MM-T005"]


def test_signature_features_are_real_columns():
    from idsai.detection.features.registry import feature_columns

    cols = set(feature_columns(CFG.features))
    for s in load_signatures():
        for c in s.conditions:
            assert c.feature in cols, (s.sid, c.feature)


def test_signatures_sha_is_stable_hex():
    sha = file_sha(SIGNATURES_PATH)
    assert len(sha) == 12 and int(sha, 16) >= 0


def test_signature_evidence_per_condition_sorted_and_bounded():
    d = SignatureDetector(sig([("a", ">=", 1.0), ("b", ">=", 10.0), ("c", "abs<=", 0.5)]))
    ev = d.evidence({"a": 3.0, "b": 10.0, "c": -0.25})
    assert ev.top == [("a", 3.0, 2.0), ("c", -0.25, 0.5), ("b", 10.0, 0.0)]
    assert len(d.evidence({"a": NAN, "b": 11.0, "c": 0.0}).top) == 2  # NaN dropped


# --------------------------------------------------------------------- stats


def test_robust_z_is_row_max_not_abs():
    d = RobustZDetector(["z_a", "z_b", "z_c"])
    s = d.score(frame(z_a=[1.0, -9.0], z_b=[4.0, -8.0], z_c=[-2.0, -7.0]))
    np.testing.assert_allclose(s, [4.0, -7.0])
    assert d.fired(s).all()


def test_robust_z_nan_if_any_required_nan():
    d = RobustZDetector(["z_a", "z_b"])
    s = d.score(frame(z_a=[NAN, 1.0], z_b=[2.0, 2.0]))
    assert np.isnan(s[0]) and s[1] == 2.0


def test_robust_z_evidence_by_abs_z():
    d = RobustZDetector(["z_a", "z_b", "z_c"])
    ev = d.evidence({"z_a": 1.0, "z_b": -5.0, "z_c": 3.0})
    assert [t[0] for t in ev.top] == ["z_b", "z_c", "z_a"]
    assert ev.top[0] == ("z_b", -5.0, 5.0)


def test_cusum_max_of_two_and_metadata():
    d = CusumDetector()
    assert d.name == "stat.cusum" and d.layer == "stat" and d.technique_ids == ["MM-T001"]
    s = d.score(frame(cusum_ret=[1.0, 9.0, NAN], cusum_imb=[3.0, 2.0, 1.0]))
    np.testing.assert_allclose(s[:2], [3.0, 9.0])
    assert np.isnan(s[2])
    ev = d.evidence({"cusum_ret": 1.0, "cusum_imb": 3.0})
    assert [t[0] for t in ev.top] == ["cusum_imb", "cusum_ret"]


# ----------------------------------------------------------------- baselines


def gaussian_frame(n, d, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    cov = np.full((d, d), 0.6) + 0.4 * np.eye(d)
    x = rng.multivariate_normal(np.zeros(d), cov, size=n) + shift
    return pl.DataFrame({f"f{i}": x[:, i] for i in range(d)})


FEATS = [f"f{i}" for i in range(6)]


FIT_MAKERS = [lambda: IsolationForestDetector(FEATS, seed=0), lambda: PCADetector(FEATS)]


def outlier_row() -> pl.DataFrame:
    # breaks the correlation structure and is far out
    return pl.DataFrame({f"f{i}": [8.0 if i % 2 == 0 else -8.0] for i in range(6)})


@pytest.mark.parametrize("make", FIT_MAKERS)
def test_outlier_beats_p99_of_normal(make):
    det = make()
    normal = gaussian_frame(3000, 6, seed=1)
    det.fit(normal)
    p99 = np.quantile(det.score(normal), 0.99)
    assert det.score(outlier_row())[0] > p99


@pytest.mark.parametrize("make", FIT_MAKERS)
def test_score_before_fit_raises(make):
    with pytest.raises(RuntimeError):
        make().score(gaussian_frame(5, 6, seed=1))


@pytest.mark.parametrize("make", FIT_MAKERS)
def test_fit_drops_nan_rows_and_score_is_nan_for_nan_rows(make):
    det = make()
    normal = gaussian_frame(500, 6, seed=2)
    dirty = pl.concat([normal, normal.head(10).with_columns(pl.lit(NAN).alias("f3"))])
    det.fit(dirty)
    probe = pl.concat([normal.head(3), normal.head(2).with_columns(pl.lit(NAN).alias("f0"))])
    s = det.score(probe)
    assert np.isfinite(s[:3]).all() and np.isnan(s[3:]).all()
    assert s.dtype == np.float64


@pytest.mark.parametrize("make", FIT_MAKERS)
def test_state_dict_round_trips(make):
    det = make()
    normal = gaussian_frame(400, 6, seed=3)
    det.fit(normal)
    state = pickle.loads(pickle.dumps(det.state_dict()))
    clone = make()
    clone.load_state_dict(state)
    probe = gaussian_frame(50, 6, seed=4, shift=1.0)
    np.testing.assert_array_equal(det.score(probe), clone.score(probe))


def test_iforest_deterministic_per_seed():
    normal = gaussian_frame(400, 6, seed=3)
    probe = gaussian_frame(30, 6, seed=4, shift=1.0)
    a, b = IsolationForestDetector(FEATS, seed=7), IsolationForestDetector(FEATS, seed=7)
    a.fit(normal)
    b.fit(normal)
    np.testing.assert_array_equal(a.score(probe), b.score(probe))


def test_pca_reconstruction_error_is_nonnegative_and_zero_in_subspace():
    det = PCADetector(FEATS, n_components=6)  # keeps everything -> no residual
    det.fit(gaussian_frame(500, 6, seed=5))
    s = det.score(gaussian_frame(20, 6, seed=6))
    assert (s >= -1e-12).all() and s.max() < 1e-9


def test_baseline_evidence_top_standardized_features():
    det = PCADetector(FEATS)
    det.fit(gaussian_frame(500, 6, seed=5))
    row = {f: 0.0 for f in FEATS} | {"f2": 10.0, "f4": -6.0}
    ev = det.evidence(row)
    assert len(ev.top) == 5
    assert [t[0] for t in ev.top[:2]] == ["f2", "f4"]
    assert [t[2] for t in ev.top] == sorted((t[2] for t in ev.top), reverse=True)


def test_random_reproducible_per_seed_and_in_unit_interval():
    w = pl.DataFrame({"symbol": ["A", "A", "B"], "t_end_us": [10, 20, 10]})
    a = RandomDetector(seed=0).score(w)
    np.testing.assert_array_equal(a, RandomDetector(seed=0).score(w))
    assert not np.array_equal(a, RandomDetector(seed=1).score(w))
    assert ((a >= 0) & (a < 1)).all() and len(set(a)) == 3
    assert a.dtype == np.float64


def test_random_state_round_trip_and_empty_evidence():
    d = RandomDetector(seed=3)
    e = RandomDetector(seed=0)
    e.load_state_dict(d.state_dict())
    assert e.seed == 3
    assert d.evidence({}).top == []


# ------------------------------------------------------------------ registry


def test_build_rules_excludes_kk():
    dets = build_detectors(["rules"], CFG)
    assert {d.signature_id for d in dets} == {100101, 100102, 100501}
    assert all(d.layer == "rule" for d in dets)


def test_build_baseline_kk_layer():
    (d,) = build_detectors(["baseline.kk"], CFG)
    assert d.layer == "baseline" and d.signature_id == 100103


def test_build_all_known_names():
    names = ["rules", "stat.robust_z", "stat.cusum", "baseline.iforest", "baseline.pca",
             "baseline.random", "baseline.kk"]  # fmt: skip
    dets = build_detectors(names, CFG)
    assert all(isinstance(d, Detector) for d in dets)
    rz = next(d for d in dets if d.name == "stat.robust_z")
    assert rz.requires and all(c.startswith("z_") for c in rz.requires)
    ifo = next(d for d in dets if d.name == "baseline.iforest")
    assert ifo.requires == model_features(CFG.features)


def test_unknown_name_raises_keyerror_listing_valid_names():
    with pytest.raises(KeyError) as ei:
        build_detectors(["nope"], CFG)
    msg = str(ei.value)
    assert "rules" in msg and "stat.cusum" in msg


def test_default_detectors_fail_until_autoencoder_registered():
    assert "ai.autoencoder_trade" not in valid_names()
    with pytest.raises(KeyError):
        build_detectors(CFG.detectors, CFG)


def test_register_adds_a_name():
    register("test.dummy", lambda cfg, seed: RandomDetector(seed))
    try:
        (d,) = build_detectors(["test.dummy"], CFG, seed=4)
        assert d.seed == 4
    finally:
        from idsai.detection import registry

        registry._FACTORIES.pop("test.dummy")


# ------------------------------------------------------------------ evidence


def test_evidence_validator_rejects_unsorted_or_oversized():
    with pytest.raises(ValueError):
        Evidence(top=[("a", 1.0, 1.0), ("b", 1.0, 2.0)])
    with pytest.raises(ValueError):
        Evidence(top=[("a", 1.0, 1.0)] * 6)


def test_evidence_from_items_sorts_truncates_and_drops_nan():
    items = [(f"f{i}", float(i), float(i)) for i in range(8)] + [("bad", NAN, 99.0)]
    ev = Evidence.from_items(items)
    assert [t[0] for t in ev.top] == ["f7", "f6", "f5", "f4", "f3"]
    assert all(not math.isnan(t[1]) for t in ev.top)


# ------------------------------------------------------------------ taxonomy


def test_taxonomy_has_expected_ids_and_fields():
    tax = load_taxonomy()
    assert list(tax) == [f"MM-T00{i}" for i in range(6)]
    for entry in tax.values():
        for k in ("name_th", "definition_th", "reference", "observability_th", "sub_techniques"):
            assert k in entry
    assert {s["id"] for s in tax["MM-T001"]["sub_techniques"]} == {"MM-T001.001", "MM-T001.002"}
    assert {s["id"] for s in tax["MM-T005"]["sub_techniques"]} == {"MM-T005.001"}


def test_every_signature_technique_is_in_taxonomy():
    ids = technique_ids()
    for s in load_signatures():
        assert s.technique in ids, s.sid
        assert s.technique.split(".")[0] in ids
