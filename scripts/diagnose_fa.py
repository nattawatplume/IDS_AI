"""M7: what the remaining false alarms and missed pumps look like. TRAIN range only.

Read-only diagnosis (`docs/superpowers/specs/2026-11-26-m7-fa-diagnosis-design.md`). The split,
stack, iforest fit, Chief calibration on `cal_normal` and tau at `--budget` (2) are exactly
`diagnose_market.fit_stack` (the eval's setup); accept, test and SEMI-SYN are never read.

Sets:
- FA: unmatched episodes on `train_normal` days (`unmatched_onsets`, the eval's rule).
- FA_hard: the FA on the `m1_hard_k(cfg, n)` most volatile train-normal days (the
  `_most_volatile` rule: highest `realized_vol_1m`, ties by `(symbol, day)`).
- DET: train REAL pumps with a matched episode; the episode is the first matched one.
- MISS: train REAL pumps with no matched episode.

Per set: the layer at the peak-F row (smallest p, ties in the Chief's detector order; and the layer
whose shifted -log10 p is F), how many layers are at or over tau there (the ai layer after its
shift), episode length, peak F / tau, feature values at the onset row, UTC hour, day volatility
rank among train-normal days, and the 30-minute direction (`ret_1800` 30 min after onset). MISS
has no episode: its peak is the highest-F row in the label's matching window (below tau by
definition) and its onset is the label start. FA vs DET and MISS vs DET are compared feature by
feature with a rank AUC (descriptive, not a selection). p and F are statistical anomaly scores,
never probabilities.

Nothing is written under `data/`; the DuckDB is copied to a temp dir and only read.

    uv run python scripts/diagnose_fa.py --out <scratch>/m7_diag.json
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from diagnose_market import Stack, fit_stack, on_db_copy

from idsai.ai.chief import Chief, score_column
from idsai.config import load_settings
from idsai.data.store import DAY_US
from idsai.eval.experiments import _episodes, _fa_by_day, _normal_rows, _unmatched, m1_hard_k
from idsai.eval.metrics import match_events, unmatched_onsets
from idsai.eval.splits import day_of_us, day_start_us, realized_vol_1m

S_US = 1_000_000
MIN_US = 60 * S_US
HOUR_US = 3600 * S_US
DIR_S = 1800
FEATURES = (
    "z_quote_vol_60",
    "z_n_trades_60",
    "z_rush_60",
    "z_buy_imb_60",
    "ret_60",
    "jump_1800",
    "cusum_imb",
    "cusum_ret",
    "ret_1800",  # with cusum_imb and buy_imb_1800: the inputs of the stealth rule 100102
    "buy_imb_1800",
)
DIR_COL = "ret_1800"  # log return of the 30 min ending at the row
KEEP = tuple(dict.fromkeys((*FEATURES, DIR_COL)))
LAYERS = ("rules", "robust_z", "cusum", "iforest")
AUC_FLAG = 0.15  # |AUC - 0.5| >= 0.15, i.e. AUC >= 0.65 or <= 0.35
TOP_SYMBOLS = 5
CF_SID = 100102  # the counterfactual drops this rule from F (train only, same calibration and tau)


# ------------------------------------------------------------------ pure helpers
def rank_auc(a, b) -> float:
    """P(a > b) + 0.5 P(a == b) over finite values (Mann-Whitney U / (n_a n_b)); NaN if empty."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    x = np.concatenate([a, b])
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, first, counts = np.unique(xs, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0  # 1-based average rank of each tie group
    ranks = np.empty(x.size)
    ranks[order] = np.repeat(avg, counts)
    u = ranks[: a.size].sum() - a.size * (a.size + 1) / 2.0
    return float(u / (a.size * b.size))


def gini(values) -> float:
    """Gini of non-negative counts (0 = even, -> 1 = all in one); NaN for an empty or zero sum."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.size == 0 or x.sum() <= 0:
        return float("nan")
    i = np.arange(1, x.size + 1)
    return float(2.0 * (i * x).sum() / (x.size * x.sum()) - (x.size + 1) / x.size)


def top_share(values, k: int) -> float:
    """Share of the total held by the `k` largest values; NaN for a zero sum."""
    x = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    return float(x[:k].sum() / x.sum()) if x.sum() > 0 else float("nan")


def shares(items, keys) -> dict:
    """`{key: {"n", "share"}}` over `items` (None items left out of the denominator)."""
    known = [x for x in items if x is not None]
    c = Counter(known)
    n = len(known)
    return {k: {"n": c[k], "share": (c[k] / n if n else float("nan"))} for k in keys}


def med_iqr(values) -> dict:
    """`{n, median, q25, q75}` over the finite values."""
    x = np.asarray([v for v in values if v is not None], dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None}
    q25, med, q75 = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "median": float(med), "q25": float(q25), "q75": float(q75)}


def pct_rank(v: float, ref: np.ndarray) -> float | None:
    """Mid-rank percentile of `v` among `ref` (0 = calmest, 1 = most volatile)."""
    if v is None or not math.isfinite(v) or ref.size == 0:
        return None
    return float(((ref < v).sum() + 0.5 * (ref == v).sum()) / ref.size)


def most_volatile(vols: dict, k: int) -> set:
    """The `k` keys with the highest vol, ties by key order: `_most_volatile`'s rule on
    precomputed vols (the eval reads the same `realized_vol_1m` per day)."""
    days = sorted(vols)
    order = sorted(range(len(days)), key=lambda i: (-vols[days[i]], days[i]))
    return {days[i] for i in order[:k]}


# ------------------------------------------------------------------ row lookup
def layer_of(det) -> str:
    if det.layer == "rule":
        return "rules"
    if det.layer == "ai":
        return "iforest"
    return det.name.split(".", 1)[-1]


class Rows:
    """The fused train rows sorted by `(symbol, t_end_us)`, with per-layer p-values."""

    def __init__(self, st: Stack):
        df = st.fused.sort(["symbol", "t_end_us"])
        sym = df["symbol"].to_numpy()
        names, first = np.unique(sym, return_index=True)
        ends = np.append(first[1:], sym.size)
        self.span = {s: (int(a), int(b)) for s, a, b in zip(names, first, ends, strict=True)}
        self.t = df["t_end_us"].to_numpy().astype(np.int64)
        self.F = df["F"].to_numpy().astype(np.float64)
        by_layer: dict[str, list[str]] = {k: [] for k in LAYERS}
        for d in st.chief.fused_detectors:
            by_layer[layer_of(d)].append(f"p__{d.name}")
        self.p = np.column_stack(
            [df.select(pl.min_horizontal(by_layer[k])).to_series().to_numpy() for k in LAYERS]
        ).astype(np.float64)
        # per row: the rule with the smallest p (None when no rule fired, p = 1 for every rule)
        rule_dets = [d for d in st.chief.fused_detectors if d.layer == "rule"]
        rp = np.column_stack([df[f"p__{d.name}"].to_numpy() for d in rule_dets])
        rnames = np.array([str(d.signature_id) for d in rule_dets] + [None], dtype=object)
        best = np.argmin(rp, axis=1)
        self.rule = rnames[np.where(rp.min(axis=1) < 1.0, best, len(rule_dets))]
        self.rule_names = [str(d.signature_id) for d in rule_dets]
        nlp = -np.log10(self.p)
        if st.ai.mode == "shift":
            j = LAYERS.index("iforest")
            nlp[:, j] = np.maximum(0.0, nlp[:, j] - st.ai.shift)
        self.nlp = nlp
        self.feat = np.column_stack([df[c].to_numpy() for c in FEATURES]).astype(np.float64)
        self.dir = df[DIR_COL].to_numpy().astype(np.float64)

    def peak(self, symbol: str, lo_us: int, hi_us: int) -> int | None:
        """Row of the highest F with `t_end_us` in `[lo, hi]` (earliest on ties)."""
        a, b = self.span.get(symbol, (0, 0))
        i = a + np.searchsorted(self.t[a:b], lo_us, side="left")
        j = a + np.searchsorted(self.t[a:b], hi_us, side="right")
        if j <= i:
            return None
        return int(i + np.argmax(self.F[i:j]))

    def at(self, symbol: str, t_us: int, ahead_us: int = 0) -> int | None:
        """First row with `t_end_us` in `[t, t + ahead]`."""
        a, b = self.span.get(symbol, (0, 0))
        i = a + int(np.searchsorted(self.t[a:b], t_us, side="left"))
        return i if i < b and self.t[i] <= t_us + ahead_us else None


# ------------------------------------------------------------------ profiles
def record(
    rows: Rows,
    symbol: str,
    onset_us: int,
    peak: int | None,
    onset_row: int | None,
    length_min: float | None,
    vol_rank: float | None,
) -> dict:
    later = rows.at(symbol, onset_us + DIR_S * S_US, ahead_us=10 * S_US)
    r = rows.dir[later] if later is not None else float("nan")
    return {
        "symbol": symbol,
        "onset_us": int(onset_us),
        "peak": peak,
        "onset_row": onset_row,
        "length_min": length_min,
        "vol_rank": vol_rank,
        "direction": None
        if not math.isfinite(r)
        else ("up" if r > 0 else "down" if r < 0 else "flat"),
    }


def profile(rows: Rows, recs: list[dict], tau: float) -> dict:
    by_min_p, by_F, n_over, f_tau, rule_F = [], [], [], [], []
    for r in recs:
        i = r["peak"]
        if i is None:
            by_min_p.append(None)
            by_F.append(None)
            continue
        n_over.append(int((rows.nlp[i] >= tau).sum()))
        f_tau.append(rows.F[i] / tau)
        if not rows.F[i] > 0:  # every p = 1: no layer drives anything
            by_min_p.append(None)
            by_F.append(None)
            continue
        by_min_p.append(LAYERS[int(np.argmin(rows.p[i]))])  # ties -> Chief detector order
        by_F.append(LAYERS[int(np.argmax(rows.nlp[i]))])
        if by_F[-1] == "rules":
            rule_F.append(rows.rule[i])
    feats = {}
    for k, c in enumerate(FEATURES):
        feats[c] = med_iqr(rows.feat[r["onset_row"], k] for r in recs if r["onset_row"] is not None)
    hours = [int((r["onset_us"] // HOUR_US) % 24) for r in recs]
    blocks = [f"{6 * (h // 6):02d}-{6 * (h // 6) + 5:02d}" for h in hours]
    multi = sum(1 for n in n_over if n >= 2)
    return {
        "n": len(recs),
        "layer_min_p": shares(by_min_p, LAYERS),
        "layer_F": shares(by_F, LAYERS),
        "rule_when_rules_drive_F": shares(rule_F, rows.rule_names),
        "layers_over_tau": {str(k): v for k, v in sorted(Counter(n_over).items())},
        "multi_layer": {"n": multi, "share": multi / len(n_over) if n_over else float("nan")},
        "length_min": med_iqr(r["length_min"] for r in recs),
        "peak_F_over_tau": med_iqr(f_tau),
        "features_onset": feats,
        "hour_hist": [hours.count(h) for h in range(24)],
        "hour_blocks": shares(blocks, ["00-05", "06-11", "12-17", "18-23"]),
        "vol_rank": med_iqr(r["vol_rank"] for r in recs),
        "direction_30m": shares([r["direction"] for r in recs], ["up", "down", "flat"]),
        "direction_unknown": sum(1 for r in recs if r["direction"] is None),
    }


def auc_table(rows: Rows, a: list[int], b: list[int]) -> dict:
    """Per feature: AUC of rows `a` vs rows `b` (> 0.5: `a` tends higher)."""
    out = {}
    for k, c in enumerate(FEATURES):
        va, vb = rows.feat[a, k], rows.feat[b, k]
        auc = rank_auc(va, vb)
        out[c] = {
            "auc": auc,
            "n_a": int(np.isfinite(va).sum()),
            "n_b": int(np.isfinite(vb).sum()),
            "flag": bool(math.isfinite(auc) and abs(auc - 0.5) >= AUC_FLAG),
        }
    return out


def unique_rows(xs) -> tuple[list[int], int]:
    """Distinct non-None rows in first-seen order, and how many duplicates were dropped (labels
    that share an episode onset or a start row count once in an AUC)."""
    known = [i for i in xs if i is not None]
    out = list(dict.fromkeys(known))
    return out, len(known) - len(out)


def overlaps(windows: dict, symbol: str, lo: int, hi: int) -> bool:
    return any(a <= hi and b >= lo for a, b in windows.get(symbol, ()))


def first_matches(alerts: list, labels: pl.DataFrame, before: int, after: int) -> dict:
    """Per label_id: `(first_us, last_us, technique_id)` of its first matched episode, or None."""
    by_sym: dict[str, list] = {}
    for al in alerts:
        by_sym.setdefault(al.symbol, []).append((al.window[0], al.window[1], al.technique_id))
    out = {}
    for lid, sym, ls, le in labels.select(
        "label_id", "symbol", "t_start_us", "t_end_us"
    ).iter_rows():
        lo, hi = ls - before, le + after
        hits = [w for w in by_sym.get(sym, ()) if w[0] <= hi and w[1] >= lo]
        out[lid] = min(hits, key=lambda w: w[0]) if hits else None
    return out


def counterfactual(st: Stack, cfg, fa_alerts: list, fa_hard: list, hard_of, every_old) -> dict:
    """F without rule `CF_SID` (same calibration of the other detectors, same tau): which train
    FA episodes vanish and which detected pumps are lost or change their first matched episode."""
    train, tau = st.train, st.tau
    b_s, a_s = cfg.eval.before_s, cfg.eval.after_s
    before, after = int(b_s * S_US), int(a_s * S_US)
    dets = [d for d in st.dets if getattr(d, "signature_id", None) != CF_SID]
    assert len(dets) == len(st.dets) - 1, f"rule {CF_SID} not in the stack"
    chief = Chief(dets, cfg.chief, st.ctx.config_hash, st.chief.model_versions, ai=st.ai)
    cols = ["symbol", "t_end_us", *[score_column(d.name) for d in chief.score_detectors]]
    chief.calibrate(st.cal.frame.select(cols))
    fused = chief.fuse(train.frame.select(cols))
    normal = _episodes(chief, _normal_rows(fused, train.fa_days), tau)
    every = _episodes(chief, fused, tau)
    del fused
    fa_on = unmatched_onsets(normal, train.labels, b_s, a_s)
    win: dict[str, list] = {}
    for al in normal:
        win.setdefault(al.symbol, []).append(al.window)
    gone = [not overlaps(win, al.symbol, *al.window) for al in fa_alerts]
    gone_hard = [g for g, h in zip(gone, fa_hard, strict=True) if h]
    old = first_matches(every_old, train.labels, before, after)
    new = first_matches(every, train.labels, before, after)
    mr_old = match_events(every_old, train.labels, b_s, a_s)
    mr_new = match_events(every, train.labels, b_s, a_s)
    det = [lid for lid, w in old.items() if w is not None]
    lost = [lid for lid in det if new[lid] is None]
    kept = [lid for lid in det if new[lid] is not None]
    return {
        "dropped_rule": CF_SID,
        "note": "same tau and calibration (tau not refit); naming rules are not in this stack",
        "n_fa_before": len(fa_alerts),
        "n_fa_after": len(fa_on),
        "fa_vanished": int(sum(gone)),
        "fa_vanished_share": sum(gone) / len(gone) if gone else None,
        "n_fa_hard_before": len(gone_hard),
        "n_fa_hard_after": sum(1 for sym, t in fa_on if hard_of(sym, t)),
        "fa_hard_vanished": int(sum(gone_hard)),
        "fa_hard_vanished_share": sum(gone_hard) / len(gone_hard) if gone_hard else None,
        "fa_per_symbol_day_after": len(fa_on) / len(train.fa_days) if train.fa_days else None,
        "n_det_before": len(det),
        "n_det_after": int(sum(mr_new.detected_any.values())),
        "det_lost": lost,
        "det_first_episode_changed": sum(1 for lid in kept if new[lid][:2] != old[lid][:2]),
        "det_onset_changed": sum(1 for lid in kept if new[lid][0] != old[lid][0]),
        "det_technique_changed": sum(1 for lid in kept if new[lid][2] != old[lid][2]),
        "n_correct_before": int(sum(mr_old.detected_correct.values())),
        "n_correct_after": int(sum(mr_new.detected_correct.values())),
    }


# ------------------------------------------------------------------ run
def day_vols(store, days) -> dict:
    out = {}
    for sd in days:
        d0 = day_start_us(sd.day)
        out[(sd.symbol, sd.day)] = realized_vol_1m(store.read_trades(sd.symbol, d0, d0 + DAY_US))
    return out


def fa_day(symbol: str, onset_us: int, listed: dict) -> tuple:
    """`_fa_by_day`'s day of an onset: its own listed day, else the latest earlier listed day of
    the symbol, else the earliest."""
    d = day_of_us(onset_us)
    days = listed[symbol]
    if d in days:
        return symbol, d
    earlier = [x for x in days if x <= d]
    return symbol, (earlier[-1] if earlier else days[0])


def run(cfg, store, budget: float, seed: int) -> dict:
    st = fit_stack(cfg, store, budget, seed, keep=KEEP)
    sp, chief, tau, train = st.ctx.split, st.chief, st.tau, st.train
    b_s, a_s = cfg.eval.before_s, cfg.eval.after_s
    before, after = int(b_s * S_US), int(a_s * S_US)
    rows = Rows(st)
    t0 = time.time()
    vols = day_vols(store, train.days)
    print(f"day vols {len(vols)} in {time.time() - t0:.0f} s", flush=True)
    normal_keys = [(sd.symbol, sd.day) for sd in train.fa_days]
    ref = np.array([vols[k] for k in normal_keys], dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    k_hard = m1_hard_k(cfg, len(train.fa_days))
    hard = most_volatile({k: vols[k] for k in normal_keys}, k_hard)

    # FA
    normal = _episodes(chief, _normal_rows(st.fused, train.fa_days), tau)
    fa_on = unmatched_onsets(normal, train.labels, b_s, a_s)
    fa_alerts = _unmatched(normal, fa_on)
    listed: dict[str, list] = {}
    for s, d in sorted(normal_keys):
        listed.setdefault(s, []).append(d)
    fa_recs, fa_hard_recs = [], []
    for al in fa_alerts:
        f0, f1 = al.window
        key = fa_day(al.symbol, f0, listed)
        rec = record(
            rows,
            al.symbol,
            f0,
            rows.peak(al.symbol, f0, f1),
            rows.at(al.symbol, f0),
            (f1 - f0) / MIN_US,
            pct_rank(vols.get(key, float("nan")), ref),
        )
        rec["hard"] = key in hard
        fa_recs.append(rec)
        if rec["hard"]:
            fa_hard_recs.append(rec)

    # DET / MISS
    every = _episodes(chief, st.fused, tau)
    mr = match_events(every, train.labels, b_s, a_s)

    def hard_of(sym: str, t_us: int) -> bool:
        return fa_day(sym, t_us, listed) in hard

    st.fused = None  # free the full fused frame before the counterfactual re-fuse
    t0 = time.time()
    cf = counterfactual(st, cfg, fa_alerts, [r["hard"] for r in fa_recs], hard_of, every)
    print(f"counterfactual without {CF_SID} in {time.time() - t0:.0f} s", flush=True)
    by_sym: dict[str, list] = {}
    for al in every:
        by_sym.setdefault(al.symbol, []).append(al.window)
    det_recs, miss_recs = [], []
    det_start, miss_start, det_wpeak, miss_wpeak = [], [], [], []
    for lid, sym, ls, le in train.labels.select(
        "label_id", "symbol", "t_start_us", "t_end_us"
    ).iter_rows():
        lo, hi = ls - before, le + after
        vr = pct_rank(vols.get((sym, day_of_us(ls)), float("nan")), ref)
        start_row = rows.at(sym, ls, ahead_us=MIN_US)
        wpeak = rows.peak(sym, lo, hi)
        if mr.detected_any[lid]:
            f0, f1 = min((w for w in by_sym[sym] if w[0] <= hi and w[1] >= lo), key=lambda w: w[0])
            det_recs.append(
                record(
                    rows, sym, f0, rows.peak(sym, f0, f1), rows.at(sym, f0), (f1 - f0) / MIN_US, vr
                )
            )
            det_start.append(start_row)
            det_wpeak.append(wpeak)
        else:
            miss_recs.append(record(rows, sym, ls, wpeak, start_row, None, vr))
            miss_start.append(start_row)
            miss_wpeak.append(wpeak)

    fa_rows, _ = unique_rows(r["onset_row"] for r in fa_recs)
    fa_hard_rows, _ = unique_rows(r["onset_row"] for r in fa_hard_recs)
    det_rows, dup_det = unique_rows(r["onset_row"] for r in det_recs)
    det_s, dup_ds = unique_rows(det_start)
    miss_s, dup_ms = unique_rows(miss_start)
    det_p, dup_dp = unique_rows(det_wpeak)
    miss_p, dup_mp = unique_rows(miss_wpeak)

    def feats_at(ix: list) -> dict:
        return {c: med_iqr(rows.feat[ix, k]) for k, c in enumerate(FEATURES)}

    # concentration
    per_sym = Counter(r["symbol"] for r in fa_recs)
    syms = sorted({s for s, _ in normal_keys})
    counts = [per_sym.get(s, 0) for s in syms]
    per_day = _fa_by_day(fa_on, train.fa_days)
    day_counts = np.array(list(per_day.values()), dtype=np.float64)
    top_syms = sorted(per_sym.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_SYMBOLS]
    n_days_sym = Counter(s for s, _ in normal_keys)
    hard_sym = Counter(s for s, _ in hard)
    fa_hard_sym = Counter(r["symbol"] for r in fa_hard_recs)
    top_names = {s for s, _ in top_syms}
    top_detail = [
        {
            "symbol": s,
            "n": n,
            "normal_days": n_days_sym[s],
            "fa_per_normal_day": n / n_days_sym[s],
            "hard_days": hard_sym[s],
            "fa_on_hard_days": fa_hard_sym[s],
        }
        for s, n in top_syms
    ]
    n_rest = len(train.fa_days) - len(hard)
    return {
        "config_hash": st.ctx.config_hash,
        "split_sha12": sp.sha12(),
        "seed": seed,
        "budget": budget,
        "tau": tau,
        "tau_saturated": bool(chief.tau_saturated),
        "ai_fusion": st.ai.as_dict(),
        "detectors": [d.name for d in st.dets],
        "layers": list(LAYERS),
        "features": list(FEATURES),
        "n_train_normal_days": len(train.fa_days),
        "n_train_event_days": len(sp.train_events),
        "n_train_labels": train.labels.height,
        "n_fa": len(fa_recs),
        "n_det": len(det_recs),
        "n_miss": len(miss_recs),
        "hard": {
            "k": k_hard,
            "days": sorted(f"{s}|{d.isoformat()}" for s, d in hard),
            "n_fa_hard": len(fa_hard_recs),
            "fa_per_day_hard": len(fa_hard_recs) / len(hard) if hard else None,
            "fa_per_day_rest": (len(fa_recs) - len(fa_hard_recs)) / n_rest if n_rest else None,
        },
        "profiles": {
            "FA": profile(rows, fa_recs, tau),
            "FA_hard": profile(rows, fa_hard_recs, tau),
            "DET": profile(rows, det_recs, tau),
            "MISS": profile(rows, miss_recs, tau),
        },
        "concentration": {
            "n_symbols": len(syms),
            "n_symbols_with_fa": sum(1 for c in counts if c),
            "top5": top_detail,
            "top5_share": top_share(counts, TOP_SYMBOLS),
            "top10pct_symbols": max(1, len(syms) // 10),
            "top10pct_symbols_share": top_share(counts, max(1, len(syms) // 10)),
            "fa_per_normal_day_all": len(fa_recs) / len(normal_keys) if normal_keys else None,
            "hard_days_of_top5": sum(hard_sym[s] for s in top_names),
            "top5_fa_on_hard_days": sum(fa_hard_sym[s] for s in top_names),
            "gini_symbols": gini(counts),
            "fa_per_day": {
                "days": int(day_counts.size),
                "share_days_with_fa": float((day_counts > 0).mean()) if day_counts.size else None,
                "median": float(np.median(day_counts)) if day_counts.size else None,
                "p90": float(np.percentile(day_counts, 90)) if day_counts.size else None,
                "max": int(day_counts.max()) if day_counts.size else None,
                "top10pct_days_share": top_share(day_counts, max(1, day_counts.size // 10)),
                "gini_days": gini(day_counts),
                "hist": {str(k): v for k, v in sorted(Counter(day_counts.astype(int)).items())},
            },
        },
        "auc": {
            "FA_vs_DET_onset": auc_table(rows, fa_rows, det_rows),
            "FA_hard_vs_DET_onset": auc_table(rows, fa_hard_rows, det_rows),
            "MISS_vs_DET_label_start": auc_table(rows, miss_s, det_s),
            "MISS_vs_DET_window_peak": auc_table(rows, miss_p, det_p),
        },
        "auc_duplicates_dropped": {
            "DET_onset": dup_det,
            "DET_label_start": dup_ds,
            "MISS_label_start": dup_ms,
            "DET_window_peak": dup_dp,
            "MISS_window_peak": dup_mp,
        },
        "features_label_anchor": {
            "MISS_label_start": feats_at(miss_s),
            "DET_label_start": feats_at(det_s),
            "MISS_window_peak": feats_at(miss_p),
            "DET_window_peak": feats_at(det_p),
        },
        "counterfactual": cf,
        "notes": {
            "auc": "AUC = P(a > b) + 0.5 P(a = b) on finite values; flag at >= 0.65 or <= 0.35",
            "miss": "MISS peak = highest-F row in [t_start - before_s, t_end + after_s]; onset = "
            "label start (first row within 60 s)",
            "direction": f"sign of {DIR_COL} at onset + {DIR_S} s",
            "vol_rank": "mid-rank percentile of the day's realized_vol_1m among train-normal "
            "days; event days include the pump itself",
            "length": "last - first exceedance row of the episode (0 = one 10 s row)",
            "peak_F_ceiling": "F is capped at -log10(1 / (n_cal + 1)); DET peaks sit at that "
            "ceiling, so the FA vs DET peak F/tau gap is understated",
        },
        "skipped": {k: dict(v) for k, v in (("train", train.skipped), ("cal", st.cal.skipped))},
    }


# ------------------------------------------------------------------ output
def _fmt(m: dict, nd: int = 2) -> str:
    if not m["n"]:
        return "-"
    return f"{m['median']:.{nd}f} [{m['q25']:.{nd}f}, {m['q75']:.{nd}f}]"


def print_table(res: dict) -> None:
    sets = ("FA", "FA_hard", "DET", "MISS")
    P = res["profiles"]
    print(
        f"tau {res['tau']:.3f} (budget {res['budget']:g}); FA {res['n_fa']} on "
        f"{res['n_train_normal_days']} train-normal days (hard k={res['hard']['k']}: "
        f"{res['hard']['n_fa_hard']}); DET {res['n_det']}, MISS {res['n_miss']} of "
        f"{res['n_train_labels']} train pumps"
    )
    print(
        "set | n | "
        + " | ".join(f"minp:{k}" for k in LAYERS)
        + " | "
        + " | ".join(f"F:{k}" for k in LAYERS)
        + " | >=2 layers | len min | peak F/tau"
    )
    for s in sets:
        p = P[s]
        print(
            f"{s} | {p['n']} | "
            + " | ".join(f"{p['layer_min_p'][k]['share']:.2f}" for k in LAYERS)
            + " | "
            + " | ".join(f"{p['layer_F'][k]['share']:.2f}" for k in LAYERS)
            + f" | {p['multi_layer']['share']:.2f} | {_fmt(p['length_min'], 1)} | "
            f"{_fmt(p['peak_F_over_tau'])}"
        )
    print("feature at onset (median [IQR]) | " + " | ".join(sets))
    for c in res["features"]:
        print(f"{c} | " + " | ".join(_fmt(P[s]["features_onset"][c]) for s in sets))
    print(
        "set | rule (when rules drive F): "
        + ", ".join(res["profiles"]["FA"]["rule_when_rules_drive_F"])
    )
    for s in sets:
        rw = P[s]["rule_when_rules_drive_F"]
        print(f"{s} | " + " ".join(f"{k}:{v['n']}" for k, v in rw.items()))
    print("set | vol rank | up/down (30 min) | hours 00-05/06-11/12-17/18-23")
    for s in sets:
        p = P[s]
        hb = "/".join(f"{p['hour_blocks'][k]['share']:.2f}" for k in p["hour_blocks"])
        d = p["direction_30m"]
        print(
            f"{s} | {_fmt(p['vol_rank'])} | {d['up']['share']:.2f}/{d['down']['share']:.2f} | {hb}"
        )
    c = res["concentration"]
    fd = c["fa_per_day"]
    print(
        f"symbols {c['n_symbols_with_fa']}/{c['n_symbols']} with FA; top5 share "
        f"{c['top5_share']:.3f}, gini {c['gini_symbols']:.3f}; days with FA "
        f"{fd['share_days_with_fa']:.3f}, top10% days hold {fd['top10pct_days_share']:.3f}, "
        f"max {fd['max']}/day; FA/day hard {res['hard']['fa_per_day_hard']:.2f} vs rest "
        f"{res['hard']['fa_per_day_rest']:.2f}"
    )
    print("label-anchored features (median) | " + " | ".join(res["features_label_anchor"]))
    for f in res["features"]:
        cells = [_fmt(v[f]) for v in res["features_label_anchor"].values()]
        print(f"{f} | " + " | ".join(cells))
    cf = res["counterfactual"]
    print(
        f"without {cf['dropped_rule']}: FA {cf['n_fa_before']} -> {cf['n_fa_after']} "
        f"(vanished {cf['fa_vanished']}), FA_hard {cf['n_fa_hard_before']} -> "
        f"{cf['n_fa_hard_after']} (vanished {cf['fa_hard_vanished']}); DET {cf['n_det_before']} "
        f"-> {cf['n_det_after']} (lost {len(cf['det_lost'])}, first episode changed "
        f"{cf['det_first_episode_changed']}, technique changed {cf['det_technique_changed']}); "
        f"correct {cf['n_correct_before']} -> {cf['n_correct_after']}"
    )
    print(f"AUC duplicates dropped: {res['auc_duplicates_dropped']}")
    print("AUC | " + " | ".join(res["auc"]))
    for f in res["features"]:
        cells = []
        for t in res["auc"].values():
            x = t[f]
            cells.append(f"{x['auc']:.3f}{'*' if x['flag'] else ''}")
        print(f"{f} | " + " | ".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    cfg = load_settings(args.config) if args.config else load_settings()
    if cfg.paths.data_dir.resolve() in out_path.parents:
        print("--out must be outside data/", file=sys.stderr)
        return 1
    t_start = time.time()
    res = on_db_copy(cfg, "m7diag_", lambda store: run(cfg, store, args.budget, args.seed))
    res["wall_s"] = round(time.time() - t_start, 1)
    out_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print_table(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
