import json
import pickle
import time

import numpy as np
import polars as pl
import pytest
import torch
from sklearn.metrics import roc_auc_score

from idsai.ai.autoencoder import AutoencoderDetector
from idsai.ai.learning import ModelRegistry
from idsai.config import AECfg, load_settings
from idsai.detection.features.registry import model_features
from idsai.detection.registry import build_detectors

NAN = float("nan")
T0_US = 1_699_999_980 * 1_000_000  # multiple of 60 s
STEP_US = 60_000_000
SMALL = AECfg(hidden=16, bottleneck=4, epochs=5, batch_size=256)


def feats(d: int) -> list[str]:
    return [f"f{i}" for i in range(d)]


def correlated(n: int, d: int, seed: int, shift: np.ndarray | float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # rank-1 shared factor + isotropic noise: correlation 0.6 between every pair
    common = rng.standard_normal((n, 1))
    return np.sqrt(0.6) * common + np.sqrt(0.4) * rng.standard_normal((n, d)) + shift


def windows(x: np.ndarray, t0_us: int = T0_US, step_us: int = STEP_US) -> pl.DataFrame:
    """Two symbols alternating; each symbol advances `step_us` every two rows."""
    n, d = x.shape
    i = np.arange(n)
    return pl.DataFrame(
        {
            "symbol": np.where(i % 2 == 0, "AAAUSDT", "BBBUSDT"),
            "t_end_us": t0_us + (i // 2) * step_us,
            **{f"f{j}": x[:, j] for j in range(d)},
        }
    )


@pytest.fixture(scope="module")
def small_fit() -> tuple[AutoencoderDetector, pl.DataFrame]:
    det = AutoencoderDetector(feats(8), SMALL, seed=0)
    w = windows(correlated(2000, 8, seed=1))
    det.fit(w)
    return det, w


def test_metadata():
    det = AutoencoderDetector(feats(3), SMALL, seed=0)
    assert det.name == "ai.autoencoder_trade"
    assert det.layer == "ai" and det.track == "T"
    assert det.technique_ids == ["ANY"]
    assert det.requires == feats(3)


def test_roc_auc_on_shifted_rows():
    d = 32
    det = AutoencoderDetector(feats(d), AECfg(epochs=8), seed=0)
    t = time.perf_counter()
    det.fit(windows(correlated(20_000, d, seed=1)))
    fit_s = time.perf_counter() - t
    # shift with alternating signs: breaks the shared-factor correlation structure
    shift = np.where(np.arange(d) % 2 == 0, 1.5, -1.5)
    x = np.vstack([correlated(5000, d, seed=2), correlated(200, d, seed=3, shift=shift)])
    y = np.r_[np.zeros(5000), np.ones(200)]
    s = det.score(windows(x))
    auc = roc_auc_score(y, s)
    print(f"AE ROC-AUC={auc:.4f} fit_s={fit_s:.2f} info={det.fit_info}")
    assert auc > 0.9


def test_same_seed_same_scores_and_other_seed_differs():
    w = windows(correlated(1500, 8, seed=1))
    probe = windows(correlated(300, 8, seed=2, shift=0.5))
    a, b, c = (AutoencoderDetector(feats(8), SMALL, seed=s) for s in (5, 5, 6))
    for det in (a, b, c):
        det.fit(w)
    sa = a.score(probe)
    assert np.array_equal(sa, b.score(probe))
    assert not np.array_equal(sa, c.score(probe))


def test_score_dtype_shape_nonnegative(small_fit):
    det, w = small_fit
    s = det.score(w)
    assert s.dtype == np.float64 and s.shape == (w.height,)
    assert np.isfinite(s).all() and (s >= 0).all()
    assert det.score(w.head(0)).shape == (0,)


def test_nan_row_scores_nan_and_inf_is_a_value(small_fit):
    det, w = small_fit
    probe = pl.concat(
        [
            w.head(2),
            w.head(1).with_columns(pl.lit(NAN).alias("f3")),
            w.head(1).with_columns(pl.lit(None, dtype=pl.Float64).alias("f0")),
            w.head(1).with_columns(pl.lit(np.inf).alias("f5")),
        ]
    )
    s = det.score(probe)
    assert np.isfinite(s[:2]).all()
    assert np.isnan(s[2]) and np.isnan(s[3])
    assert np.isfinite(s[4]) and s[4] > s[0]  # inf is clipped to +clip, not missing


def test_score_before_fit_raises():
    det = AutoencoderDetector(feats(4), SMALL, seed=0)
    with pytest.raises(RuntimeError):
        det.score(windows(correlated(5, 4, seed=0)))
    with pytest.raises(RuntimeError):
        det.evidence({f: 0.0 for f in feats(4)})


def test_fit_without_usable_rows_raises():
    det = AutoencoderDetector(feats(4), SMALL, seed=0)
    off_grid = windows(correlated(50, 4, seed=0), t0_us=T0_US + 10_000_000)
    with pytest.raises(ValueError, match="no usable"):
        det.fit(off_grid)
    all_nan = windows(np.full((50, 4), NAN))
    with pytest.raises(ValueError, match="no usable"):
        det.fit(all_nan)
    with pytest.raises(RuntimeError):  # still unfitted
        det.score(all_nan)


def test_fit_uses_only_train_stride_rows_in_any_input_order():
    base = windows(correlated(1200, 6, seed=1))
    # 10 s-grid rows between the 60 s samples carry junk: fit must ignore them
    junk = windows(correlated(1200, 6, seed=9, shift=50.0), t0_us=T0_US + 10_000_000)
    a = AutoencoderDetector(feats(6), SMALL, seed=0)
    b = AutoencoderDetector(feats(6), SMALL, seed=0)
    a.fit(base)
    b.fit(pl.concat([junk, base]).sample(fraction=1.0, shuffle=True, seed=3))
    probe = windows(correlated(100, 6, seed=2))
    assert np.array_equal(a.score(probe), b.score(probe))
    assert a.fit_info["n_rows"] == 1200


def test_fit_drops_nan_rows_and_holds_out_last_tenth():
    base = windows(correlated(1000, 6, seed=1))
    dirty = pl.concat([base, base.head(20).with_columns(pl.lit(NAN).alias("f2"))])
    det = AutoencoderDetector(feats(6), SMALL, seed=0)
    det.fit(dirty)
    assert det.fit_info["n_rows"] == 1000
    assert det.fit_info["n_val"] == 100  # last 10 % in time order
    assert det.fit_info["n_train"] == 900


def test_evidence_top5_sorted_and_points_at_perturbed_feature(small_fit):
    det, w = small_fit
    row = w.row(0, named=True) | {"f6": 9.0}
    ev = det.evidence(row)
    assert len(ev.top) == 5
    assert ev.top[0][0] == "f6" and ev.top[0][1] == 9.0
    contribs = [c for _, _, c in ev.top]
    assert contribs == sorted(contribs, reverse=True) and min(contribs) >= 0
    nan_row = w.row(0, named=True) | {"f1": NAN}
    assert det.evidence(nan_row).top == []


def test_global_torch_state_is_restored():
    threads = torch.get_num_threads()
    det_flag = torch.are_deterministic_algorithms_enabled()
    rng_state = torch.get_rng_state()
    det = AutoencoderDetector(feats(4), SMALL, seed=0)
    det.fit(windows(correlated(300, 4, seed=1)))
    det.score(windows(correlated(10, 4, seed=2)))
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == det_flag
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_state_dict_is_plain_and_round_trips(small_fit):
    det, _ = small_fit
    state = det.state_dict()
    assert not any(isinstance(v, torch.Tensor) for v in state["net"].values())
    clone = AutoencoderDetector(feats(8), AECfg(), seed=99)
    clone.load_state_dict(pickle.loads(pickle.dumps(state)))
    assert clone.cfg == SMALL and clone.seed == 0
    probe = windows(correlated(200, 8, seed=4, shift=1.0))
    assert np.array_equal(det.score(probe), clone.score(probe))


# ------------------------------------------------------------------ registry


def meta(**over):
    return {
        "data_hash": "abc123",
        "config_hash": "def456",
        "seed": 0,
        "created_at": "2026-10-28T00:00:00Z",
        "train_range": ["2024-01-01", "2024-02-01"],
    } | over


def test_registry_save_load_identical_scores(tmp_path, small_fit):
    det, _ = small_fit
    reg = ModelRegistry(tmp_path / "models")
    assert reg.save(det, meta()) == "v1"
    vdir = tmp_path / "models" / det.name / "v1"
    assert (vdir / "state.pkl").is_file()
    on_disk = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
    assert list(on_disk) == sorted(on_disk)
    state, m = reg.load(det.name)
    assert m["data_hash"] == "abc123" and m["version"] == "v1" and m["name"] == det.name
    clone = AutoencoderDetector(feats(8), SMALL, seed=1)
    clone.load_state_dict(state)
    probe = windows(correlated(200, 8, seed=5))
    assert np.array_equal(det.score(probe), clone.score(probe))


def test_registry_versions_increment_and_sort_numerically(tmp_path, small_fit):
    det, _ = small_fit
    reg = ModelRegistry(tmp_path)
    assert reg.versions(det.name) == []
    assert [reg.save(det, meta(seed=i)) for i in range(3)] == ["v1", "v2", "v3"]
    # a v9 already on disk: the next save is v10, and v10 sorts after v9
    v9 = tmp_path / det.name / "v9"
    v9.mkdir()
    (v9 / "meta.json").write_text(json.dumps(meta(seed=9)), encoding="utf-8")
    (v9 / "state.pkl").write_bytes(pickle.dumps({}))
    assert reg.save(det, meta(seed=10)) == "v10"
    assert reg.versions(det.name) == ["v1", "v2", "v3", "v9", "v10"]
    _, latest = reg.load(det.name)
    assert latest["seed"] == 10
    _, v2 = reg.load(det.name, "v2")
    assert v2["seed"] == 1


def test_registry_rejects_incomplete_meta_and_unknown_versions(tmp_path, small_fit):
    det, _ = small_fit
    reg = ModelRegistry(tmp_path)
    for k in ("data_hash", "config_hash", "seed", "created_at", "train_range"):
        m = meta()
        del m[k]
        with pytest.raises(ValueError, match=k):
            reg.save(det, m)
    assert reg.versions(det.name) == []  # nothing written
    with pytest.raises(KeyError):
        reg.load("nope")
    reg.save(det, meta())
    with pytest.raises(KeyError):
        reg.load(det.name, "v7")


def test_registry_never_overwrites(tmp_path, small_fit):
    det, _ = small_fit
    reg = ModelRegistry(tmp_path)
    reg.save(det, meta(seed=1))
    before = (tmp_path / det.name / "v1" / "meta.json").read_bytes()
    assert reg.save(det, meta(seed=2)) == "v2"
    assert (tmp_path / det.name / "v1" / "meta.json").read_bytes() == before


# --------------------------------------------------------- detector registry


def test_default_detectors_build_with_autoencoder():
    settings = load_settings()
    dets = build_detectors(settings.detectors, settings, seed=3)
    assert {d.layer for d in dets} == {"rule", "stat", "ai"}
    (ae,) = [d for d in dets if d.name == "ai.autoencoder_trade"]
    assert isinstance(ae, AutoencoderDetector)
    assert ae.requires == model_features(settings.features)
    assert ae.cfg == settings.ae and ae.seed == 3


@pytest.mark.slow
def test_fit_100k_by_40_under_120s():
    d = 40
    det = AutoencoderDetector(feats(d), AECfg(), seed=0)
    w = windows(correlated(100_000, d, seed=1))
    t = time.perf_counter()
    det.fit(w)
    elapsed = time.perf_counter() - t
    print(f"AE fit 100k x 40: {elapsed:.1f}s info={det.fit_info}")
    assert elapsed < 120
