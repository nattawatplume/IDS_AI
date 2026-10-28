"""Model lifecycle: label-only training mask and the versioned on-disk model registry.

This is the one `ai/` module allowed to read labels: it uses them only to keep labeled periods
(± a margin) out of the normal training data, never as training targets. Alerts are never used
to mask training data (no feedback loop from the detector's own output).
"""

import json
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from idsai.contracts.detector import Detector

REQUIRED_META = ("data_hash", "config_hash", "seed", "created_at", "train_range")
MAX_SAVE_ATTEMPTS = 100
_VERSION_RE = re.compile(r"^v([1-9]\d*)$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def training_mask(
    windows: pl.DataFrame, labels: pl.DataFrame, margin_s: int = 3600
) -> tuple[pl.Series, float]:
    """Keep-mask for training rows and the fraction excluded.

    A window row (`symbol`, `t_end_us`) is excluded iff some label of the same symbol has
    `t_start_us - margin <= t_end_us <= t_end_us_label + margin` (inclusive). `labels` follows
    the LABELS frame. Returns `(pl.Series[bool] "keep", excluded_fraction)`; 0.0 when empty.
    """
    n = windows.height
    keep = np.ones(n, dtype=bool)
    if n and labels.height:
        margin_us = int(margin_s) * 1_000_000
        sym = windows["symbol"].to_numpy()
        t = windows["t_end_us"].to_numpy().astype(np.int64)
        for s, lab in labels.partition_by("symbol", as_dict=True).items():
            rows = np.flatnonzero(sym == s[0])
            if not rows.size:
                continue
            lab = lab.sort("t_start_us")
            lo = lab["t_start_us"].to_numpy().astype(np.int64) - margin_us
            # running max of interval ends: a short label nested in a long one can't hide it
            hi = np.maximum.accumulate(lab["t_end_us"].to_numpy().astype(np.int64) + margin_us)
            # last interval starting at or before t; excluded iff t is within the max end so far
            i = np.searchsorted(lo, t[rows], side="right") - 1
            hit = (i >= 0) & (t[rows] <= hi[np.maximum(i, 0)])
            keep[rows[hit]] = False
    frac = float((~keep).mean()) if n else 0.0
    return pl.Series("keep", keep, dtype=pl.Boolean), frac


class ModelRegistry:
    """Versioned models on disk: `{models_dir}/{name}/v{n}/state.pkl` + `meta.json`.

    `state.pkl` is a pickle of `Detector.state_dict()`. Pickle is acceptable here because the
    registry only ever loads models this project produced locally; never load a registry
    directory from an untrusted source. Versions start at `v1`, increase by one past the highest
    existing version, and an existing version directory is never overwritten.
    """

    def __init__(self, models_dir: str | Path):
        self.root = Path(models_dir)

    def _dir(self, name: str) -> Path:
        if not _NAME_RE.match(name) or ".." in name:
            raise ValueError(f"invalid model name {name!r}")
        return self.root / name

    @staticmethod
    def _numbers(d: Path, complete_only: bool) -> list[int]:
        if not d.is_dir():
            return []
        out = []
        for p in d.iterdir():
            m = _VERSION_RE.match(p.name)
            # any path named v{n} (even a stray file) reserves n; only complete dirs are versions
            if m and (not complete_only or (p / "meta.json").is_file()):
                out.append(int(m[1]))
        return sorted(out)

    def versions(self, name: str) -> list[str]:
        """Complete versions of `name`, oldest first, sorted numerically (`v10` after `v9`)."""
        return [f"v{n}" for n in self._numbers(self._dir(name), complete_only=True)]

    def save(self, det: Detector, meta: dict[str, Any]) -> str:
        """Write `det.state_dict()` + `meta` as the next version; returns e.g. `"v3"`.

        `meta` must hold `data_hash, config_hash, seed, created_at, train_range`; `name` and
        `version` are added. `meta.json` is written with sorted keys (non-JSON values via `str`).
        """
        missing = [k for k in REQUIRED_META if k not in meta]
        if missing:
            raise ValueError(f"model meta is missing required keys: {missing}")
        d = self._dir(det.name)
        d.mkdir(parents=True, exist_ok=True)
        state = det.state_dict()
        # write into a temp dir, then rename: a version dir is either complete or absent, and
        # rename refuses to replace an existing (non-empty) version dir
        tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=d))
        try:
            with open(tmp / "state.pkl", "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            for _ in range(MAX_SAVE_ATTEMPTS):
                existing = self._numbers(d, complete_only=False)
                version = f"v{(existing[-1] if existing else 0) + 1}"
                full = {**meta, "name": det.name, "version": version}
                (tmp / "meta.json").write_text(
                    json.dumps(full, sort_keys=True, indent=2, default=str), encoding="utf-8"
                )
                try:
                    os.rename(tmp, d / version)
                    return version
                except FileExistsError:  # a concurrent save took this number: retry
                    continue
                except OSError:
                    if (d / version).exists():
                        continue
                    raise
            raise RuntimeError(
                f"could not claim a new version of {det.name!r} after {MAX_SAVE_ATTEMPTS} tries"
            )
        finally:
            if tmp.exists():
                for p in tmp.iterdir():
                    p.unlink()
                tmp.rmdir()

    def load(self, name: str, version: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """`(state, meta)` of `version` (latest when None); `KeyError` if unknown."""
        known = self.versions(name)
        if not known:
            raise KeyError(f"no saved versions of model {name!r} under {self.root}")
        v = known[-1] if version is None else version
        if v not in known:
            raise KeyError(f"model {name!r} has no version {v!r}; known: {known}")
        vdir = self._dir(name) / v
        with open(vdir / "state.pkl", "rb") as f:
            state = pickle.load(f)
        meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
        return state, meta
