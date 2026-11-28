"""M6 step A: are false alarms clustered around BTC moves? TRAIN range only.

Read-only diagnosis for the M6 gate
(`docs/superpowers/specs/2026-11-26-m6-market-context-design.md`).

- Split: `make_split` on the REAL labels (the eval's split, `cfg.eval.split_seed`). Only
  `train_normal`, `train_events` and `cal_normal` are scored; accept and test are never read.
- Stack: the live config (`configs/default.yaml` unless `--config`): rules (with the sell-side
  rules), stat.robust_z, stat.cusum, ai.iforest under `AiFusion.from_cfg` (shift 1). The iforest
  is fit on the masked, grid-thinned train-normal rows (`_Ctx.fit`, as the eval does), the Chief is
  calibrated on `cal_normal` and tau is fit there at `--budget` (2) per symbol-day, exactly as the
  eval's `_variant` does. Naming rules never change episodes, so they are left out.
- FA episodes: unmatched episodes on train-normal days (`unmatched_onsets`, the eval's rule).
  Pump detections: train REAL labels with a matched episode; the detection onset is the first
  matched episode's first row (`t_start + ttd`).
- Market: `mkt_absret_z` of BTCUSDT 1m klines (`idsai.data.market`, window 60 s = `ret_60`). An
  onset is "near a BTC move" at threshold k when some kline whose close lies in
  [onset - 10 min, onset + 10 min] has z >= k.
- Base rates (per train-normal symbol-day minute, klines with a finite z): `base_minute` = share
  of minutes with z >= k (the spec's base rate), `base_near` = share of minutes that are themselves
  within +-10 min of such a minute (the like-for-like chance level of the "near" test).

Nothing is written under `data/`: the DuckDB file is copied to a temp dir and only read; the JSON
goes to `--out` (keep it outside `data/`).

    uv run python scripts/diagnose_market.py --out <scratch>/m6_diag.json
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from idsai.ai.chief import AiFusion, Chief
from idsai.config import load_settings
from idsai.data.market import asof, load_klines, market_dir, mkt_absret_z
from idsai.data.store import Store
from idsai.eval.experiments import (
    SATURATION_WARNING,
    _Ctx,
    _episodes,
    _normal_rows,
    _Part,
)
from idsai.eval.metrics import match_events, unmatched_onsets
from idsai.eval.splits import day_start_us

S_US = 1_000_000
MIN_US = 60 * S_US
NEAR_S = 600
THRESHOLDS = (2.0, 3.0, 4.0)
GATE_K = 3.0


def near_flags(onsets_us: np.ndarray, t_known: np.ndarray, z: np.ndarray, k: float) -> list:
    """Per onset: True if a kline closing within +-NEAR_S has z >= k; None if no finite z there."""
    out = []
    for t in onsets_us:
        lo = np.searchsorted(t_known, t - NEAR_S * S_US, side="left")
        hi = np.searchsorted(t_known, t + NEAR_S * S_US, side="right")
        zs = z[lo:hi]
        zs = zs[np.isfinite(zs)]
        out.append(bool((zs >= k).any()) if zs.size else None)
    return out


def share(flags: list) -> tuple[int, int, float]:
    known = [f for f in flags if f is not None]
    k = sum(known)
    return k, len(known), (k / len(known) if known else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    cfg = load_settings(args.config) if args.config else load_settings()
    data_dir = cfg.paths.data_dir.resolve()
    if data_dir in out_path.parents:
        print("--out must be outside data/", file=sys.stderr)
        return 1
    t_start = time.time()
    res = on_db_copy(cfg, "m6diag_", lambda store: run(cfg, store, args.budget, args.seed))
    res["wall_s"] = round(time.time() - t_start, 1)
    out_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print_table(res)
    return 0


def on_db_copy(cfg, prefix: str, fn: Callable[[Store], dict]) -> dict:
    """`fn(store)` on a temp copy of the app DuckDB (the real database is never opened)."""
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    db = tmp / "app.duckdb"
    shutil.copy2(cfg.paths.app_db, db)
    try:
        with Store(cfg.paths.data_dir, db) as store:
            return fn(store)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@dataclass
class Stack:
    """The live stack fit, calibrated and scored on the TRAIN range (shared by the diagnoses)."""

    ctx: _Ctx
    dets: list
    ai: AiFusion
    chief: Chief
    tau: float
    cal: _Part
    train: _Part
    fused: pl.DataFrame  # train rows: symbol, t_end_us, `keep` columns, scores, p__*, F, ...


def fit_stack(cfg, store: Store, budget: float, seed: int, keep: tuple[str, ...] = ()) -> Stack:
    """Split, fit the iforest on train-normal, calibrate the Chief on cal-normal, tau at `budget`,
    then score and fuse every train row (events + normal), keeping the feature columns `keep`."""
    ctx = _Ctx(cfg, store, seed)
    sp = ctx.split
    print(f"split {sp.counts()}", flush=True)
    dets = [d for ds in ctx.build(cfg.detectors).values() for d in ds]
    mv, info = ctx.fit(dets, sp.train_normal)
    print(f"fit {sorted(mv)} on {info.get('train_rows')} rows", flush=True)
    cal = ctx.clean(sp.cal_normal, dets)
    train = ctx.real_parts(sp.train_events, sp.train_normal, dets, keep=keep)
    print(f"scored cal {cal.frame.height} rows, train {train.frame.height} rows", flush=True)

    ai = AiFusion.from_cfg(cfg.chief)
    chief = Chief(dets, cfg.chief, ctx.config_hash, mv, ai=ai)
    cols = ["symbol", "t_end_us", *[f"score__{d.name}" for d in chief.score_detectors]]
    chief.calibrate(cal.frame.select(cols))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=SATURATION_WARNING, category=RuntimeWarning)
        tau = chief.fit_threshold(chief.fuse(cal.frame.select(cols)), budget)
    fused = chief.fuse(train.frame.select(*cols, *keep))
    return Stack(ctx, dets, ai, chief, tau, cal, train, fused)


def run(cfg, store: Store, budget: float, seed: int) -> dict:
    st = fit_stack(cfg, store, budget, seed)
    ctx, sp, dets, ai, chief = st.ctx, st.ctx.split, st.dets, st.ai, st.chief
    tau, cal, train, fused = st.tau, st.cal, st.train, st.fused
    b, a = cfg.eval.before_s, cfg.eval.after_s

    normal = _episodes(chief, _normal_rows(fused, train.fa_days), tau)
    fa = unmatched_onsets(normal, train.labels, b, a)
    every = _episodes(chief, fused, tau)
    mr = match_events(every, train.labels, b, a)
    t0 = dict(train.labels.select("label_id", "t_start_us").iter_rows())
    det = sorted(lid for lid, ok in mr.detected_any.items() if ok)
    det_onsets = np.array([t0[lid] + int(round(mr.ttd_s[lid] * S_US)) for lid in det], np.int64)
    lab_starts = np.array([t0[lid] for lid in det], np.int64)

    days = sorted({sd.day for sd in train.days})
    first, last = days[0] - timedelta(days=1), days[-1] + timedelta(days=1)
    kl = load_klines(market_dir(cfg.paths.data_dir), first, last)
    tk, _, z = mkt_absret_z(kl["open_time_us"].to_numpy(), kl["close"].to_numpy())
    print(f"klines {kl.height}, finite z {int(np.isfinite(z).sum())}", flush=True)

    fa_on = np.array([t for _, t in fa], np.int64)
    # base rates over every train-normal symbol-day minute (minute end = a kline close time)
    grid = (np.arange(1440, dtype=np.int64) + 1) * MIN_US
    mins = np.concatenate([day_start_us(sd.day) + grid for sd in train.fa_days])
    zmin = asof(mins, tk, z, max_age_s=0)
    fin = np.isfinite(zmin)
    rows = []
    for k in THRESHOLDS:
        fa_k = share(near_flags(fa_on, tk, z, k))
        pd_k = share(near_flags(det_onsets, tk, z, k))
        pl_k = share(near_flags(lab_starts, tk, z, k))
        base_min = float((zmin[fin] >= k).mean())
        # minute is within +-10 min of a hit minute (like-for-like chance level)
        base_near = share(near_flags(mins[fin], tk, z, k))
        rows.append(
            {
                "k": k,
                "fa_near": fa_k[0],
                "fa_known": fa_k[1],
                "fa_share": fa_k[2],
                "base_minute_hits": int((zmin[fin] >= k).sum()),
                "base_minutes": int(fin.sum()),
                "base_minute": base_min,
                "base_near": base_near[2],
                "pump_near": pd_k[0],
                "pump_known": pd_k[1],
                "pump_share": pd_k[2],
                "pump_label_start_near": pl_k[0],
                "pump_label_start_share": pl_k[2],
            }
        )
    g = next(r for r in rows if r["k"] == GATE_K)
    gate = {
        "fa_ge_2x_base": g["fa_share"] >= 2 * g["base_minute"],
        "fa_ge_0.10": g["fa_share"] >= 0.10,
        "pump_lt_half_fa": g["pump_share"] < 0.5 * g["fa_share"],
    }
    gate["pass"] = all(gate.values())
    return {
        "config_hash": ctx.config_hash,
        "split_sha12": sp.sha12(),
        "seed": seed,
        "budget": budget,
        "tau": tau,
        "tau_saturated": bool(chief.tau_saturated),
        "detectors": [d.name for d in dets],
        "ai_fusion": ai.as_dict(),
        "n_train_normal_days": len(train.fa_days),
        "n_train_event_days": len(sp.train_events),
        "n_train_labels": train.labels.height,
        "n_detected": len(det),
        "n_fa_episodes": len(fa),
        "fa_per_symbol_day": len(fa) / len(train.fa_days) if train.fa_days else None,
        "near_s": NEAR_S,
        "rows": rows,
        "gate_k": GATE_K,
        "gate": gate,
        "skipped": {k: dict(v) for k, v in (("train", train.skipped), ("cal", cal.skipped))},
    }


def print_table(res: dict) -> None:
    print(
        f"tau {res['tau']:.3f} (budget {res['budget']:g}); FA episodes {res['n_fa_episodes']} on "
        f"{res['n_train_normal_days']} train-normal days; detected {res['n_detected']}/"
        f"{res['n_train_labels']} train pumps"
    )
    print("k | FA near/known (share) | base minute | base near | pump near/known (share)")
    for r in res["rows"]:
        print(
            f"{r['k']:g} | {r['fa_near']}/{r['fa_known']} ({r['fa_share']:.3f}) | "
            f"{r['base_minute']:.4f} ({r['base_minute_hits']}/{r['base_minutes']}) | "
            f"{r['base_near']:.3f} | {r['pump_near']}/{r['pump_known']} ({r['pump_share']:.3f})"
        )
    print(f"gate (k={res['gate_k']:g}): {res['gate']}")


if __name__ == "__main__":
    sys.exit(main())
