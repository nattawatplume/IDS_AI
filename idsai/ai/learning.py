"""Model lifecycle: label-only training mask, the versioned on-disk model registry, and the
champion/challenger decision.

This is the one `ai/` module allowed to read labels: it uses them only to keep labeled periods
(± a margin) out of the normal training data, never as training targets. Alerts are never used
to mask training data (no feedback loop from the detector's own output).
"""

import json
import os
import pickle
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from idsai.contracts.detector import Detector
from idsai.eval.metrics import paired_bootstrap, significant

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


MIN_EVENTS = 2  # fewer events: the paired bootstrap has no variance to measure


@dataclass(frozen=True)
class Decision:
    """Champion/challenger verdict: `diff_mean` and `ci` are challenger - champion recall."""

    accept: bool
    diff_mean: float
    ci: tuple[float, float]
    reason_th: str


def _detections(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 2 or a.shape[0] == 0:
        raise ValueError(f"{name} must be (n_seeds, n_events) with n_seeds >= 1, got {a.shape}")
    a = a.astype(np.float64)
    if not np.isin(a, (0.0, 1.0)).all():
        raise ValueError(f"{name} must hold 0/1 detections only")
    return a


def compare(
    champion: np.ndarray,
    challenger: np.ndarray,
    n_boot: int,
    seed: int,
    min_effect: float = 0.02,
) -> Decision:
    """Accept the challenger iff it detects more events, by a paired event bootstrap.

    `champion`, `challenger`: `(n_seeds, n_events)` 0/1 (or bool) detections on the same events
    and seeds. Each arm is averaged over seeds per event, then events are resampled with the
    same indices for both arms (`paired_bootstrap`). Accept iff the 95 % CI lower bound of
    `challenger - champion` is > 0 and the mean difference is >= `min_effect` (`significant`).
    No events gives a reject with diff 0 and CI (0, 0); a single event is a reject too (its
    bootstrap has no variance, so the CI would collapse to one point and claim certainty).
    """
    if int(n_boot) < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    a = _detections(champion, "champion")
    b = _detections(challenger, "challenger")
    if a.shape != b.shape:
        raise ValueError(f"champion and challenger shapes differ: {a.shape} {b.shape}")
    if a.shape[1] == 0:
        return Decision(False, 0.0, (0.0, 0.0), "ไม่ยอมรับ challenger: ไม่มีเหตุการณ์ให้เปรียบเทียบ")
    diff, ci = paired_bootstrap(b, a, n_boot, seed)
    nums = f"{diff:+.3f} (95% CI {ci[0]:.3f} ถึง {ci[1]:.3f})"
    if a.shape[1] < MIN_EVENTS:
        reason = (
            f"ไม่ยอมรับ challenger: มีเหตุการณ์เพียง {a.shape[1]} ครั้ง (ผลต่าง {nums}) "
            f"ต้องมีอย่างน้อย {MIN_EVENTS} ครั้งจึงประเมินช่วงความเชื่อมั่นได้"
        )
        return Decision(False, diff, ci, reason)
    accept = significant(diff, ci, min_effect)
    if accept:
        reason = f"ยอมรับ challenger: recall ดีขึ้น {nums} ถึงเกณฑ์ขั้นต่ำ {min_effect:g}"
    elif ci[1] < 0:
        reason = f"ไม่ยอมรับ challenger: recall ลดลง {nums} CI ต่ำกว่า 0 ทั้งช่วง"
    elif ci[0] < 0:
        reason = f"ไม่ยอมรับ challenger: ผลต่าง recall {nums} CI คร่อม 0"
    elif ci[0] == ci[1] == 0:
        reason = f"ไม่ยอมรับ challenger: ผลต่าง recall {nums} ทั้งสองฝั่งตรวจพบเหตุการณ์เหมือนกัน"
    elif ci[0] == 0:
        reason = f"ไม่ยอมรับ challenger: ผลต่าง recall {nums} CI แตะ 0 (ขอบล่างต้องมากกว่า 0)"
    else:
        reason = f"ไม่ยอมรับ challenger: recall ดีขึ้น {nums} แต่ต่ำกว่าเกณฑ์ขั้นต่ำ {min_effect:g}"
    return Decision(accept, diff, ci, reason)
