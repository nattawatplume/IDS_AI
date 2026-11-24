"""Non-AI-layer baselines (IsolationForest, PCA, random) for the baselines experiment.

Baselines run alone and never take part in fusion. The one exception is
`AiIsolationForestDetector` (M2): the same IsolationForest as `baseline.iforest`, named
`ai.iforest` on the `ai` layer so the Chief fuses it like the autoencoder (its raw score goes
through the same split-conformal p-value path; the Chief is score-agnostic).

IsolationForest scoring goes through `ForestScorer`, a NumPy re-implementation of sklearn's
`score_samples` over the fitted trees (all trees at once, row chunks), because sklearn's per-call
overhead is ~24 ms for one row (the live stream scores one row at a time). It reproduces sklearn
step for step (float32 input, `x <= threshold` with NaN to `missing_go_to_left`, per-leaf
`depth + c(n_leaf) - 1`, summed over trees in estimator order, `2 ** (-depth / (T * c(psi)))`),
so its scores equal `-model.score_samples` bit for bit; a test pins that. The derived arrays are
rebuilt from the model at fit and load and never stored, so the state (and its sha) is unchanged.

Rows with any NaN feature score NaN (NaN-only masking, as in rules/stats: ±inf is an extreme
value, its standardized value is clipped to ±`Z_CAP` so it scores finite and high). `fit` uses
only fully finite rows, since mean/std and the sklearn models cannot be fit on inf.
"""

import hashlib
from typing import Any

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length

from idsai.contracts.detector import Detector, Evidence, feature_matrix

STD_FLOOR = 1e-9
Z_CAP = 1e6  # bound on |standardized value|; only ever reached by ±inf inputs
N_ESTIMATORS = 200


class _FeatureModel(Detector):
    """Shared plumbing: feature list, standardization stats, NaN handling, evidence."""

    layer = "baseline"
    track = "T"

    def __init__(self, features: list[str], seed: int = 0):
        if not features:
            raise ValueError(f"{type(self).__name__} needs at least one feature")
        self.technique_ids = ["ANY"]
        self.requires = list(features)
        self.seed = seed
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    def _need_fit(self) -> None:
        if self._mu is None:
            raise RuntimeError(f"{self.name} is not fitted; call fit() first")

    def fit(self, normal: pl.DataFrame) -> None:
        x = feature_matrix(normal, self.requires)
        x = x[np.isfinite(x).all(axis=1)]
        if x.shape[0] < 2:
            raise ValueError(f"{self.name}: fewer than 2 complete normal rows to fit on")
        mu = x.mean(axis=0)
        sd = np.maximum(x.std(axis=0), STD_FLOOR)
        self._fit_std((x - mu) / sd)
        # set last: a failed model fit leaves the detector unfitted (score raises RuntimeError)
        self._mu, self._sd = mu, sd

    def _fit_std(self, z: np.ndarray) -> None:
        raise NotImplementedError

    def _std(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self._mu) / self._sd, -Z_CAP, Z_CAP)

    def score(self, windows: pl.DataFrame) -> np.ndarray:
        self._need_fit()
        x = feature_matrix(windows, self.requires)
        out = np.full(x.shape[0], np.nan)
        ok = ~np.isnan(x).any(axis=1)
        if ok.any():
            out[ok] = self._score_std(self._std(x[ok]))
        return out

    def _score_std(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evidence(self, row: dict[str, Any]) -> Evidence:
        """Top features by magnitude of the standardized input value."""
        self._need_fit()
        x = np.array([float(row[f]) for f in self.requires])
        z = self._std(x)
        return Evidence.from_items(
            (f, float(x[i]), float(abs(z[i]))) for i, f in enumerate(self.requires)
        )


class ForestScorer:
    """`-IsolationForest.score_samples(x)` (higher = more anomalous) for a fitted forest, in NumPy.

    Built once from the model: per tree (padded to the largest tree) the children, the split
    column in the full input (the estimator's feature subset already applied), the threshold,
    the NaN direction and the per-leaf `depth + c(n_leaf) - 1` sklearn adds up.
    """

    CHUNK_ROWS = 2048  # bounds the (rows, trees) temporaries to a few MB

    def __init__(self, model: IsolationForest):
        trees = [e.tree_ for e in model.estimators_]
        n_trees, width = len(trees), max(t.node_count for t in trees)
        subsample = model._max_features != model.n_features_in_  # as sklearn decides it
        self.left = np.zeros((n_trees, width), dtype=np.int64)
        self.right = np.zeros((n_trees, width), dtype=np.int64)
        self.column = np.zeros((n_trees, width), dtype=np.int64)
        self.threshold = np.zeros((n_trees, width), dtype=np.float64)
        self.nan_left = np.zeros((n_trees, width), dtype=bool)
        self.leaf_value = np.zeros((n_trees, width), dtype=np.float64)
        for i, (t, feats) in enumerate(zip(trees, model.estimators_features_, strict=True)):
            n = t.node_count
            leaf = t.children_left == -1
            nodes = np.arange(n)
            # a leaf points to itself, so extra steps leave it where it is
            self.left[i, :n] = np.where(leaf, nodes, t.children_left)
            self.right[i, :n] = np.where(leaf, nodes, t.children_right)
            f = np.where(leaf, 0, t.feature)
            self.column[i, :n] = np.asarray(feats)[f] if subsample else f
            self.threshold[i, :n] = t.threshold
            self.nan_left[i, :n] = np.asarray(t.missing_go_to_left, dtype=bool)
            # sklearn: depths += decision_path_lengths[leaf] + avg_path_length[leaf] - 1.0
            self.leaf_value[i, :n] = (
                model._decision_path_lengths[i] + model._average_path_length_per_tree[i] - 1.0
            )
        self.steps = max(t.max_depth for t in trees)
        self.denominator = float(n_trees * _average_path_length([model._max_samples])[0])
        self._trees = np.arange(n_trees)

    def score(self, x: np.ndarray) -> np.ndarray:
        x32 = np.asarray(x, dtype=np.float32)  # sklearn validates to float32
        out = np.empty(x32.shape[0], dtype=np.float64)
        for lo in range(0, x32.shape[0], self.CHUNK_ROWS):
            out[lo : lo + self.CHUNK_ROWS] = self._score(x32[lo : lo + self.CHUNK_ROWS])
        return out

    def _score(self, x: np.ndarray) -> np.ndarray:
        rows = np.arange(x.shape[0])[:, None]
        trees = self._trees[None, :]
        node = np.zeros((x.shape[0], self._trees.size), dtype=np.int64)
        for _ in range(self.steps):
            v = x[rows, self.column[trees, node]].astype(np.float64)
            go_left = (v <= self.threshold[trees, node]) | (
                np.isnan(v) & self.nan_left[trees, node]
            )
            node = np.where(go_left, self.left[trees, node], self.right[trees, node])
        # cumsum adds tree by tree from 0, in estimator order, like sklearn's `depths +=`
        depths = np.cumsum(self.leaf_value[trees, node], axis=1)[:, -1]
        if self.denominator == 0:  # one training sample: sklearn sets depth / denominator to 1
            return 2 ** (-np.ones_like(depths))
        return 2 ** (-np.divide(depths, self.denominator))


class IsolationForestDetector(_FeatureModel):
    name = "baseline.iforest"

    def __init__(self, features: list[str], seed: int = 0):
        super().__init__(features, seed)
        self._model: IsolationForest | None = None
        self._scorer: ForestScorer | None = None

    def _fit_std(self, z: np.ndarray) -> None:
        model = IsolationForest(n_estimators=N_ESTIMATORS, random_state=self.seed)
        model.fit(z)
        self._model, self._scorer = model, ForestScorer(model)

    def _score_std(self, z: np.ndarray) -> np.ndarray:
        return self._scorer.score(z)

    def state_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.requires),
            "seed": self.seed,
            "mu": self._mu,
            "sd": self._sd,
            "model": self._model,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.requires = list(d["features"])
        self.seed = d["seed"]
        self._mu, self._sd, self._model = d["mu"], d["sd"], d["model"]
        self._scorer = None if self._model is None else ForestScorer(self._model)


class AiIsolationForestDetector(IsolationForestDetector):
    """`baseline.iforest` as an AI-layer detector (M2): identical model and raw scores."""

    name = "ai.iforest"
    layer = "ai"


class PCADetector(_FeatureModel):
    """Score = squared reconstruction error of the standardized row (PCA keeps `n_components`)."""

    name = "baseline.pca"

    def __init__(self, features: list[str], n_components: float | int = 0.95):
        super().__init__(features, seed=0)
        self.n_components = n_components
        self._components: np.ndarray | None = None  # (k, d)
        self._center: np.ndarray | None = None

    def _fit_std(self, z: np.ndarray) -> None:
        pca = PCA(n_components=self.n_components, svd_solver="full").fit(z)
        self._components = pca.components_
        self._center = pca.mean_

    def _score_std(self, z: np.ndarray) -> np.ndarray:
        zc = z - self._center
        resid = zc - (zc @ self._components.T) @ self._components
        return (resid**2).sum(axis=1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.requires),
            "n_components": self.n_components,
            "mu": self._mu,
            "sd": self._sd,
            "components": self._components,
            "center": self._center,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.requires = list(d["features"])
        self.n_components = d["n_components"]
        self._mu, self._sd = d["mu"], d["sd"]
        self._components, self._center = d["components"], d["center"]


class RandomDetector(Detector):
    """Uniform [0, 1) score from a blake2b hash of `(symbol, t_end_us, seed)` (control baseline).

    It reads no features, so its evidence is empty.
    """

    name = "baseline.random"
    layer = "baseline"
    track = "T"

    def __init__(self, seed: int = 0):
        self.technique_ids = ["ANY"]
        self.requires = []
        self.seed = seed

    def _u(self, symbol: str, t_end_us: int) -> float:
        h = hashlib.blake2b(f"{symbol}|{t_end_us}|{self.seed}".encode(), digest_size=8)
        # top 53 bits / 2**53 is exactly representable and < 1 (n / 2**64 can round up to 1.0)
        return (int.from_bytes(h.digest(), "big") >> 11) / 2.0**53

    def score(self, windows: pl.DataFrame) -> np.ndarray:
        syms = windows["symbol"].to_list()
        ts = windows["t_end_us"].to_list()
        return np.array([self._u(s, t) for s, t in zip(syms, ts, strict=True)], dtype=np.float64)

    def evidence(self, row: dict[str, Any]) -> Evidence:
        return Evidence(top=[])

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.seed = d["seed"]
