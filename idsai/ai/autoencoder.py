"""AI layer: a trade-feature autoencoder (`ai.autoencoder_trade`).

A symmetric MLP `in -> hidden -> bottleneck -> hidden -> in` (ReLU after the hidden layers;
linear bottleneck and output) is trained with Adam + MSE to reconstruct normal-period window
rows. The score of a row is its mean squared reconstruction error in scaled space.

Fit pipeline: keep rows whose `t_end_us` lies on the `train_stride_s` grid (the windows grid is
epoch-anchored at a 10 s stride, so this keeps one row per symbol every `train_stride_s`), drop
rows with any non-finite feature, sort by `(t_end_us, symbol)`, hold out the last 10 % (time
order) for early stopping and restore the best-validation weights. A `RobustScaler` is fit on
the training part only; scaled inputs are clipped to `±clip` at fit and score time.

Scoring masks NaN only: a NaN (or null) feature gives a NaN score, while ±inf is a value that
scales to ±inf and is clipped to `±clip`. Fit and score run single-threaded with deterministic
algorithms, so the same seed and data give bit-identical scores; torch's global thread count,
deterministic flag and CPU RNG state are restored afterwards.
"""

import contextlib
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn

from idsai.config import AECfg
from idsai.contracts.detector import Detector, Evidence, feature_matrix

VAL_FRACTION = 0.1
SCORE_CHUNK = 65_536  # rows per forward pass at score time (bounds memory)


@contextlib.contextmanager
def _torch_determinism(seed: int | None = None) -> Iterator[None]:
    """Single thread + deterministic algorithms (+ seeded CPU RNG); restores global state."""
    threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            yield
    finally:
        torch.use_deterministic_algorithms(det, warn_only=warn_only)
        torch.set_num_threads(threads)


def _build_net(d: int, hidden: int, bottleneck: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d, hidden),
        nn.ReLU(),
        nn.Linear(hidden, bottleneck),
        nn.Linear(bottleneck, hidden),
        nn.ReLU(),
        nn.Linear(hidden, d),
    )


class AutoencoderDetector(Detector):
    name = "ai.autoencoder_trade"
    layer = "ai"
    track = "T"

    def __init__(self, features: list[str], cfg: AECfg, seed: int = 0):
        if not features:
            raise ValueError("AutoencoderDetector needs at least one feature")
        self.technique_ids = ["ANY"]
        self.requires = list(features)
        self.cfg = cfg
        self.seed = seed
        self._net: nn.Sequential | None = None
        self._center: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self.fit_info: dict[str, Any] = {}

    def _need_fit(self) -> None:
        if self._net is None:
            raise RuntimeError(f"{self.name} is not fitted; call fit() first")

    def _prep(self, x: np.ndarray) -> np.ndarray:
        """Robust-scale and clip to `±clip`; float32 network input."""
        z = (x - self._center) / self._scale
        return np.clip(z, -self.cfg.clip, self.cfg.clip).astype(np.float32)

    def _training_rows(self, normal: pl.DataFrame) -> np.ndarray:
        step_us = self.cfg.train_stride_s * 1_000_000
        rows = normal.filter(pl.col("t_end_us") % step_us == 0).sort(["t_end_us", "symbol"])
        x = feature_matrix(rows, self.requires)
        # RobustScaler and the loss need finite inputs, so fit rows must be fully finite
        return x[np.isfinite(x).all(axis=1)]

    def fit(self, normal: pl.DataFrame) -> None:
        x = self._training_rows(normal)
        n = x.shape[0]
        if n == 0:
            raise ValueError(
                f"{self.name}: no usable rows to fit on (need finite features on the "
                f"{self.cfg.train_stride_s} s train grid)"
            )
        n_val = int(n * VAL_FRACTION)
        x_train, x_val = x[: n - n_val], x[n - n_val :]
        scaler = RobustScaler().fit(x_train)
        center, scale = scaler.center_.copy(), scaler.scale_.copy()

        def prep(a: np.ndarray) -> torch.Tensor:
            z = np.clip((a - center) / scale, -self.cfg.clip, self.cfg.clip)
            return torch.from_numpy(z.astype(np.float32))

        xt, xv = prep(x_train), prep(x_val)
        c = self.cfg
        with _torch_determinism(self.seed):
            net = _build_net(x.shape[1], c.hidden, c.bottleneck)
            opt = torch.optim.Adam(net.parameters(), lr=c.lr, weight_decay=c.weight_decay)
            rng = np.random.default_rng(self.seed)
            best_loss, best_epoch, best_state, bad = math.inf, -1, None, 0
            epoch = -1
            for epoch in range(c.epochs):
                net.train()
                perm = torch.from_numpy(rng.permutation(xt.shape[0]))
                train_sum = 0.0
                for i in range(0, xt.shape[0], c.batch_size):
                    xb = xt[perm[i : i + c.batch_size]]
                    loss = nn.functional.mse_loss(net(xb), xb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    train_sum += loss.item() * xb.shape[0]
                # early stopping on the held-out tail (train loss when it is empty)
                if n_val:
                    net.eval()
                    with torch.no_grad():
                        monitor = nn.functional.mse_loss(net(xv), xv).item()
                else:
                    monitor = train_sum / xt.shape[0]
                if monitor < best_loss:
                    best_loss, best_epoch, bad = monitor, epoch, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= c.patience:
                        break
            if best_state is not None:  # epochs == 0 keeps the initial weights
                net.load_state_dict(best_state)
            net.eval()
        # set last: a failed fit leaves the detector unfitted
        self._net, self._center, self._scale = net, center, scale
        self.fit_info = {
            "n_rows": n,
            "n_train": n - n_val,
            "n_val": n_val,
            "epochs_run": epoch + 1,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
        }

    def _sq_err(self, z: np.ndarray) -> np.ndarray:
        """Per-feature squared reconstruction error (float64) of prepared rows."""
        out = np.empty(z.shape, dtype=np.float64)
        with _torch_determinism(), torch.no_grad():
            for i in range(0, z.shape[0], SCORE_CHUNK):
                zb = torch.from_numpy(z[i : i + SCORE_CHUNK])
                recon = self._net(zb).numpy().astype(np.float64)
                out[i : i + SCORE_CHUNK] = (recon - zb.numpy().astype(np.float64)) ** 2
        return out

    def score(self, windows: pl.DataFrame) -> np.ndarray:
        self._need_fit()
        x = feature_matrix(windows, self.requires)
        out = np.full(x.shape[0], np.nan)
        ok = ~np.isnan(x).any(axis=1)
        if ok.any():
            out[ok] = self._sq_err(self._prep(x[ok])).mean(axis=1)
        return out

    def evidence(self, row: dict[str, Any]) -> Evidence:
        """Top 5 features by squared reconstruction error; empty when any feature is NaN."""
        self._need_fit()
        x = np.array([[float(row[f]) for f in self.requires]])
        if np.isnan(x).any():
            return Evidence(top=[])
        err = self._sq_err(self._prep(x))[0]
        return Evidence.from_items(
            (f, float(x[0, i]), float(err[i])) for i, f in enumerate(self.requires)
        )

    def state_dict(self) -> dict[str, Any]:
        """Plain-Python/NumPy state (no tensors): weights, scaler, features, config, seed."""
        self._need_fit()
        return {
            "features": list(self.requires),
            "cfg": self.cfg.model_dump(),
            "seed": self.seed,
            "center": self._center.copy(),
            "scale": self._scale.copy(),
            "net": {k: v.detach().cpu().numpy().copy() for k, v in self._net.state_dict().items()},
            "fit_info": dict(self.fit_info),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.requires = list(d["features"])
        self.cfg = AECfg.model_validate(d["cfg"])
        self.seed = d["seed"]
        net = _build_net(len(self.requires), self.cfg.hidden, self.cfg.bottleneck)
        net.load_state_dict({k: torch.from_numpy(np.array(v)) for k, v in d["net"].items()})
        net.eval()
        self._net = net
        self._center = np.asarray(d["center"], dtype=np.float64)
        self._scale = np.asarray(d["scale"], dtype=np.float64)
        self.fit_info = dict(d.get("fit_info", {}))
