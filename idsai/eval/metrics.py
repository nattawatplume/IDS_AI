"""Event-level metrics: alert-to-label matching, summary rates, window PR-AUC and bootstraps.

Matching (Global Constraints): an alert episode `[first_us, last_us]` matches a label of the same
symbol when it overlaps `[t_start - before_s, t_end + after_s]` (closed intervals). An episode
that starts early but lasts into that interval matches, and is never a false alert. A detection
is "correct" only when the episode's technique equals the label technique's parent (`MM-T000`,
unclassified, is never correct). A window row is positive when its `t_end_us` lies in a label's
matching interval.

There is no accuracy and no point-adjusted F1 here; F1 is stored for completeness only and
precision is always reported next to the prevalence of positive windows.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from idsai.contracts.alert import Alert
from idsai.contracts.frames import LABELS, conform

S_US = 1_000_000
UNCLASSIFIED = "MM-T000"
NAN = float("nan")


@dataclass(frozen=True)
class MatchResult:
    n_events: int
    detected_any: dict[str, bool]
    detected_correct: dict[str, bool]
    n_alerts: int
    n_alerts_matched: int
    ttd_s: dict[str, float]  # detected (any) events only; first_us - t_start, may be negative


def _parent(technique_id: str) -> str:
    return technique_id.split(".")[0]


def match_events(
    alerts: Sequence[Alert], labels: pl.DataFrame, before_s: float, after_s: float
) -> MatchResult:
    """Match alert episodes to labels (see the module docstring)."""
    labels = conform(labels, LABELS)
    before, after = int(round(before_s * S_US)), int(round(after_s * S_US))
    by_symbol: dict[str, list[tuple[int, int, str]]] = {}
    for a in alerts:
        by_symbol.setdefault(a.symbol, []).append((a.window[0], a.window[1], a.technique_id))
    matched: set[tuple[str, int]] = set()  # (symbol, index into by_symbol[symbol])
    det_any: dict[str, bool] = {}
    det_ok: dict[str, bool] = {}
    ttd: dict[str, float] = {}
    for lid, sym, t0, t1, tech in labels.select(
        "label_id", "symbol", "t_start_us", "t_end_us", "technique_id"
    ).iter_rows():
        lo, hi = t0 - before, t1 + after
        want = _parent(tech)
        hits = [
            (i, first, t)
            for i, (first, last, t) in enumerate(by_symbol.get(sym, []))
            if first <= hi and last >= lo
        ]
        det_any[lid] = bool(hits)
        det_ok[lid] = any(t != UNCLASSIFIED and _parent(t) == want for _, _, t in hits)
        if hits:
            ttd[lid] = (min(first for _, first, _ in hits) - t0) / S_US
        matched.update((sym, i) for i, _, _ in hits)
    return MatchResult(
        n_events=labels.height,
        detected_any=det_any,
        detected_correct=det_ok,
        n_alerts=len(alerts),
        n_alerts_matched=len(matched),
        ttd_s=ttd,
    )


def _mean(values: list[bool] | list[float]) -> float:
    return float(np.mean(values)) if len(values) else NAN


def summarize(mr: MatchResult, symbol_days: float, n_windows_pos: int, n_windows: int) -> dict:
    """Headline rates of one match result.

    `fa_per_symbol_day` = unmatched episodes / `symbol_days`; callers that evaluate events and
    normal days together pass the normal-day episodes and symbol-days (see experiments). NaN
    marks an undefined rate (no events, no alerts, no detected events, no windows).
    """
    recall_any = _mean(list(mr.detected_any.values()))
    recall_correct = _mean(list(mr.detected_correct.values()))
    precision = mr.n_alerts_matched / mr.n_alerts if mr.n_alerts else NAN
    unmatched = mr.n_alerts - mr.n_alerts_matched
    fa = unmatched / symbol_days if symbol_days > 0 else NAN
    ttd = np.array(list(mr.ttd_s.values()), dtype=np.float64)
    if math.isnan(precision) or math.isnan(recall_correct):
        f1 = NAN
    elif precision + recall_correct == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall_correct / (precision + recall_correct)
    return {
        "recall_any": recall_any,
        "recall_correct": recall_correct,
        "precision": precision,
        "prevalence": n_windows_pos / n_windows if n_windows else NAN,
        "fa_per_symbol_day": fa,
        "ttd_median_s": float(np.median(ttd)) if ttd.size else NAN,
        "ttd_p90_s": float(np.percentile(ttd, 90)) if ttd.size else NAN,
        "f1": f1,
    }


def window_positive(
    symbols: np.ndarray, t_end_us: np.ndarray, labels: pl.DataFrame, before_s: float, after_s: float
) -> np.ndarray:
    """Bool per window row: `t_end_us` inside a same-symbol label's matching interval."""
    labels = conform(labels, LABELS)
    y = np.zeros(len(t_end_us), dtype=bool)
    before, after = int(round(before_s * S_US)), int(round(after_s * S_US))
    t = np.asarray(t_end_us, dtype=np.int64)
    sym = np.asarray(symbols, dtype=object)
    for s, t0, t1 in labels.select("symbol", "t_start_us", "t_end_us").iter_rows():
        y |= (sym == s) & (t >= t0 - before) & (t <= t1 + after)
    return y


def window_pr_auc(F: np.ndarray, y: np.ndarray) -> float:
    """Average precision of score `F` for window labels `y`; NaN when `y` has only one class.

    NaN scores rank lowest.
    """
    F = np.asarray(F, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    if y.size == 0 or y.all() or not y.any():
        return NAN
    finite = F[np.isfinite(F)]
    low = (finite.min() if finite.size else 0.0) - 1.0
    F = np.where(np.isnan(F), low, F)
    F = np.clip(F, low, finite.max() + 1.0 if finite.size else 1.0)
    return float(average_precision_score(y, F))


def _event_means(per_event: np.ndarray) -> np.ndarray:
    a = np.asarray(per_event, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        return a.mean(axis=0)
    raise ValueError(f"per_event must be (n_events,) or (n_seeds, n_events), got {a.shape}")


def _ci(stats: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci(
    per_event: np.ndarray, n_boot: int, seed: int
) -> tuple[float, tuple[float, float]]:
    """`(mean, 95 % CI)` of the event mean by resampling events with replacement.

    `per_event` is `(n_events,)` or `(n_seeds, n_events)` (averaged over seeds first). No
    events gives NaNs.
    """
    m = _event_means(per_event)
    if m.size == 0:
        return NAN, (NAN, NAN)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m.size, size=(n_boot, m.size))
    return float(m.mean()), _ci(m[idx].mean(axis=1))


def ratio_bootstrap_ci(
    hits: np.ndarray, n: np.ndarray, n_boot: int, seed: int
) -> tuple[float, tuple[float, float]]:
    """`(sum(hits) / sum(n), 95 % CI)` resampling clusters with replacement.

    `hits[i]`, `n[i]`: detections and values in cluster `i`. Each resample recomputes the ratio,
    so the point estimate and the CI come from one estimator even with unequal cluster sizes.
    No clusters gives NaNs.
    """
    hits = np.asarray(hits, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    if hits.shape != n.shape or hits.ndim != 1:
        raise ValueError(f"hits and n must be equal-length 1-D arrays: {hits.shape} {n.shape}")
    if n.size == 0 or n.sum() <= 0:
        return NAN, (NAN, NAN)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n.size, size=(n_boot, n.size))
    return float(hits.sum() / n.sum()), _ci(hits[idx].sum(axis=1) / n[idx].sum(axis=1))


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int
) -> tuple[float, tuple[float, float]]:
    """`(mean diff a - b, 95 % CI)` with the same event resample indices for both arms.

    `a`, `b`: `(n_seeds, n_events)`, events aligned; each arm is averaged over its seeds first.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"need (n_seeds, n_events) arrays, equal n_events: {a.shape} {b.shape}")
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    n = ma.size
    if n == 0:
        return NAN, (NAN, NAN)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = ma[idx].mean(axis=1) - mb[idx].mean(axis=1)
    return float(ma.mean() - mb.mean()), _ci(diffs)


def significant(diff_mean: float, ci: tuple[float, float], min_effect: float) -> bool:
    """Accept rule: 95 % CI lower bound > 0 and mean difference >= `min_effect`."""
    return bool(ci[0] > 0 and diff_mean >= min_effect)
