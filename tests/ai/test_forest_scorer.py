"""`ForestScorer`: the NumPy IsolationForest scorer equals sklearn's `score_samples`.

Synthetic data only. The scorer replays sklearn step for step, so the scores are expected to be
bit-identical; the asserted bound is the 1e-12 the eval allows.
"""

import hashlib
import pickle
import time

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import IsolationForest

from idsai.ai.baselines import (
    AiIsolationForestDetector,
    ForestScorer,
    IsolationForestDetector,
)

TOL = 1e-12


def _state_bytes_sha(det) -> str:
    blob = pickle.dumps(det.state_dict(), protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(blob).hexdigest()


def _queries(rng: np.random.Generator, train: np.ndarray) -> np.ndarray:
    d = train.shape[1]
    q = np.vstack(
        [
            rng.normal(size=(3000, d)) * 3,  # spread out: many isolated rows
            train[:50],  # rows the trees were grown on
            np.full((1, d), 1e6),  # the Z_CAP edges (what ±inf inputs become)
            np.full((1, d), -1e6),
            np.zeros((1, d)),
            train[:1] + np.float32(1e-7),  # within float32 rounding of a training row
        ]
    )
    q[7, 0] = np.nan  # sklearn routes NaN by missing_go_to_left; so must the scorer
    return q


@pytest.mark.parametrize(
    ("max_features", "n", "d"),
    [(1.0, 4000, 60), (0.5, 3000, 7), (3, 300, 5), (1.0, 2, 3), (1.0, 1, 3)],
)
def test_scorer_equals_sklearn_score_samples(max_features, n, d):
    rng = np.random.default_rng(n + d)
    train = rng.normal(size=(n, d))
    model = IsolationForest(n_estimators=200, random_state=0, max_features=max_features)
    model.fit(train)
    q = _queries(rng, train)
    want = -model.score_samples(q)
    got = ForestScorer(model).score(q)
    assert got.shape == want.shape and np.isfinite(got).all()
    assert np.max(np.abs(got - want)) <= TOL


def test_chunking_does_not_change_scores(monkeypatch):
    rng = np.random.default_rng(3)
    model = IsolationForest(n_estimators=50, random_state=1).fit(rng.normal(size=(800, 6)))
    q = rng.normal(size=(1000, 6)) * 2
    whole = ForestScorer(model).score(q)
    monkeypatch.setattr(ForestScorer, "CHUNK_ROWS", 7)
    assert np.array_equal(ForestScorer(model).score(q), whole)
    assert np.array_equal(ForestScorer(model).score(np.empty((0, 6))), np.empty(0))


def _frame(rng: np.random.Generator, n: int) -> pl.DataFrame:
    return pl.DataFrame({c: rng.normal(size=n) for c in ("a", "b", "c")})


@pytest.mark.parametrize("cls", [IsolationForestDetector, AiIsolationForestDetector])
def test_detector_scores_like_sklearn_and_keeps_nan_masking(cls):
    rng = np.random.default_rng(5)
    det = cls(["a", "b", "c"], seed=0)
    det.fit(_frame(rng, 1500))
    q = _frame(rng, 400).with_columns(
        pl.when(pl.int_range(pl.len()) == 3).then(None).otherwise(pl.col("a")).alias("a"),
        pl.when(pl.int_range(pl.len()) == 4).then(float("inf")).otherwise(pl.col("b")).alias("b"),
    )
    got = det.score(q)
    x = q.select("a", "b", "c").to_numpy().astype(np.float64)
    ok = ~np.isnan(x).any(axis=1)
    want = np.full(x.shape[0], np.nan)
    want[ok] = -det._model.score_samples(det._std(x[ok]))
    assert np.isnan(got[3]) and np.isfinite(got[4])  # NaN row -> NaN, inf row -> clipped, finite
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.nanmax(np.abs(got - want)) <= TOL


def test_one_row_at_a_time_equals_the_batch():
    rng = np.random.default_rng(6)
    det = AiIsolationForestDetector(["a", "b", "c"], seed=2)
    det.fit(_frame(rng, 1000))
    q = _frame(rng, 60)
    batch = det.score(q)
    rows = np.concatenate([det.score(q.slice(i, 1)) for i in range(q.height)])
    assert np.array_equal(rows, batch)  # stream (one row) and batch: bit-identical


def test_state_round_trip_rebuilds_the_scorer_and_scoring_leaves_the_state_alone():
    rng = np.random.default_rng(7)
    det = AiIsolationForestDetector(["a", "b", "c"], seed=0)
    det.fit(_frame(rng, 1000))
    state = det.state_dict()
    assert set(state) == {"features", "seed", "mu", "sd", "model"}  # nothing derived is stored
    before = _state_bytes_sha(det)
    blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    back = AiIsolationForestDetector(["a", "b", "c"], seed=0)
    back.load_state_dict(pickle.loads(blob))
    assert back._scorer is not None and back._scorer is not det._scorer
    loaded = _state_bytes_sha(back)
    q = _frame(rng, 200)
    assert np.array_equal(back.score(q), det.score(q))
    # scoring touches no stored state (the bundle hashes the registry file bytes, see learning)
    assert _state_bytes_sha(det) == before and _state_bytes_sha(back) == loaded


@pytest.mark.slow
def test_single_row_scoring_is_well_under_1_ms():
    rng = np.random.default_rng(8)
    model = IsolationForest(n_estimators=200, random_state=0).fit(rng.normal(size=(20000, 60)))
    scorer = ForestScorer(model)
    row = rng.normal(size=(1, 60))
    scorer.score(row)
    n = 300
    t = time.perf_counter()
    for _ in range(n):
        scorer.score(row)
    per_row_ms = (time.perf_counter() - t) / n * 1000
    assert per_row_ms < 1.0, per_row_ms
