"""Set the M4 sell-side rule thresholds (`configs/signatures_sell.yaml`) from TRAIN data only.

Read-only and offline, like `scripts/measure_pumps.py`: labels are built from the label files
(`idsai labels` logic; no DuckDB is opened), trades are read straight from the Parquet partitions
`data_dir/trades/symbol=S/date=D/part.parquet`, and nothing is written under `data_dir`. Only
REAL labels whose UTC day is inside the train split (<= train end, 2019-06-30) are used, and the
normal days are train-period days of the same symbols at least `GAP_DAYS` from any label.

For every sampled train label measured as a pump (`idsai.eval.pump_measure`) the day's window
features are computed with `features.sell_side` on, and for every grid point of the shared
falling conditions (`ret_60 <=`, `z_sell_rush_60 >=`, `z_quote_vol_60 >=`) it reports:
- `hit_sell`: share of measured labels where the condition fires inside the sell window
  `[t_peak, t_peak + clip(dump_tau, 60 s, 1 h)]` (`sell_labels.sell_window`), split by
  `jump_1800 >= 0.03` (dump phase) or not (bear raid);
- `fa_per_day`: fired episodes (rows grouped by 300 s gaps) per normal day, the cost.
The grid is small on purpose (the machine is shared). It prints a table and writes JSON to
`--out` (required, keep it outside `data/`). It never edits the signatures file.

    uv run python scripts/tune_sell_rules.py --data-dir ../IdsAI/data --out /tmp/tune.json
"""

import argparse
import itertools
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from idsai.config import load_settings, quote_unit_for
from idsai.contracts.frames import TRADES, conform, empty
from idsai.detection.features.bars import trades_to_bars
from idsai.detection.features.windows import compute_windows
from idsai.eval.pump_measure import measure_trades, read_span
from idsai.eval.sell_labels import sell_window

S_US = 1_000_000
DAY_US = 86_400 * S_US
_EPOCH = date(1970, 1, 1)
GAP_DAYS = 3
EPISODE_GAP_S = 300
JUMP_THR = 0.03
GRID = {
    "ret_60": [-0.01, -0.02, -0.03],
    "z_sell_rush_60": [2.0, 3.0, 4.0],
    "z_quote_vol_60": [2.0, 3.0, 4.0, 6.0],
}


def day_of_us(ts_us: int) -> date:
    return _EPOCH + timedelta(days=int(ts_us) // DAY_US)


def day_start_us(day: date) -> int:
    return (day - _EPOCH).days * DAY_US


def part_path(data_dir: Path, symbol: str, day: date) -> Path:
    return data_dir / "trades" / f"symbol={symbol}" / f"date={day.isoformat()}" / "part.parquet"


def read_trades(data_dir: Path, symbol: str, start_us: int, end_us: int) -> pl.DataFrame:
    first, last = day_of_us(start_us), day_of_us(end_us - 1)
    paths = [
        p
        for n in range((last - first).days + 1)
        if (p := part_path(data_dir, symbol, first + timedelta(days=n))).is_file()
    ]
    if not paths:
        return empty(TRADES)
    out = (
        pl.scan_parquet(paths)
        .filter((pl.col("ts_us") >= start_us) & (pl.col("ts_us") < end_us))
        .sort(["ts_us", "trade_id"])
        .collect()
    )
    return conform(out, TRADES)


def day_windows(data_dir: Path, symbol: str, day: date, fcfg) -> pl.DataFrame | None:
    """Window rows of `day` (baseline from day - 1) with the sell features, or None."""
    if not (
        part_path(data_dir, symbol, day).is_file()
        and part_path(data_dir, symbol, day - timedelta(days=1)).is_file()
    ):
        return None
    d0 = day_start_us(day)
    trades = read_trades(data_dir, symbol, d0 - DAY_US, d0 + DAY_US)
    bars = trades_to_bars(trades, symbol, d0 - DAY_US, d0 + DAY_US)
    w = compute_windows(bars, fcfg, quote_unit_for(symbol, fcfg))
    return w.filter(pl.col("t_end_us") >= d0)


def fires(w: pl.DataFrame, ret: float, zs: float, zq: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (
            (w["ret_60"].to_numpy() <= ret)
            & (w["z_sell_rush_60"].to_numpy() >= zs)
            & (w["z_quote_vol_60"].to_numpy() >= zq)
        )


def episodes(t_us: np.ndarray) -> int:
    if t_us.size == 0:
        return 0
    return int(1 + np.count_nonzero(np.diff(np.sort(t_us)) > EPISODE_GAP_S * S_US))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--max-events", type=int, default=15)
    ap.add_argument("--max-normal", type=int, default=15)
    args = ap.parse_args(argv)
    data_dir = args.data_dir.resolve()
    if args.out.resolve().is_relative_to(data_dir):
        sys.exit("--out must be outside the data directory (this script never writes there)")

    cfg = load_settings(args.config, overrides={"paths": {"data_dir": str(data_dir)}})
    fcfg = cfg.features.model_copy(update={"sell_side": True})
    train_end = cfg.splits.train[1]

    from idsai.cli import build_labels  # label files only: no DuckDB

    labels = build_labels(cfg)[0].filter(pl.col("provenance") == "REAL").sort("t_start_us")
    days_by_sym: dict[str, set[date]] = {}
    for sym, t0 in labels.select("symbol", "t_start_us").iter_rows():
        days_by_sym.setdefault(sym, set()).add(day_of_us(t0))
    train = [
        (lid, sym, int(t0))
        for lid, sym, t0 in labels.select("label_id", "symbol", "t_start_us").iter_rows()
        if day_of_us(t0) <= train_end
    ]

    events, status = [], {}
    for lid, sym, t0 in train:
        if len(events) >= args.max_events:
            break
        day = day_of_us(t0)
        if not (part_path(data_dir, sym, day).is_file()):
            status["missing_day"] = status.get("missing_day", 0) + 1
            continue
        prev = read_trades(data_dir, sym, day_start_us(day) - DAY_US, day_start_us(day))
        lo, hi = read_span(t0)
        m = measure_trades(prev, read_trades(data_dir, sym, lo, hi), t0)
        status[m["status"]] = status.get(m["status"], 0) + 1
        win = sell_window(m)
        if win is None or hi > day_start_us(day) + DAY_US:  # keep the window inside day D
            continue
        w = day_windows(data_dir, sym, day, fcfg)
        if w is None or w.height == 0:
            continue
        events.append((lid, sym, win, w))
        print(f"event {lid} {sym} {day} measured; rows {w.height}", flush=True)

    normal = []
    for sym in sorted(days_by_sym):
        if len(normal) >= args.max_normal:
            break
        lab_days = sorted(days_by_sym[sym])
        sym_dir = data_dir / "trades" / f"symbol={sym}"
        stored = sorted(
            date.fromisoformat(p.name.removeprefix("date="))
            for p in (sym_dir.iterdir() if sym_dir.is_dir() else [])
            if p.name.startswith("date=")
        )
        for cand in stored:  # the first stored train day far from every label of the symbol
            if cand > train_end or any(abs((cand - x).days) < GAP_DAYS for x in lab_days):
                continue
            w = day_windows(data_dir, sym, cand, fcfg)
            if w is not None and w.height:
                normal.append((sym, cand, w))
                print(f"normal {sym} {cand}; rows {w.height}", flush=True)
                break

    rows = []
    for ret, zs, zq in itertools.product(*GRID.values()):
        dump = raid = 0
        for _lid, _sym, (lo, hi), w in events:
            t = w["t_end_us"].to_numpy()
            f = fires(w, ret, zs, zq) & (t > lo) & (t <= hi + 60 * S_US)
            if f.any():
                jump = w["jump_1800"].to_numpy()[f]
                dump += bool((jump >= JUMP_THR).any())
                raid += bool((jump < JUMP_THR).any())
        fa = [episodes(w["t_end_us"].to_numpy()[fires(w, ret, zs, zq)]) for _, _, w in normal]
        rows.append(
            {
                "ret_60": ret,
                "z_sell_rush_60": zs,
                "z_quote_vol_60": zq,
                "hit_sell_dump": dump / len(events) if events else None,
                "hit_sell_raid": raid / len(events) if events else None,
                "fa_per_day": float(np.mean(fa)) if fa else None,
            }
        )
    result = {
        "train_end": train_end.isoformat(),
        "n_train_labels": len(train),
        "n_events": len(events),
        "n_normal_days": len(normal),
        "measure_status": status,
        "grid": rows,
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"events {len(events)} normal days {len(normal)} status {status}")
    print("ret_60 z_sell_rush_60 z_quote_vol_60 | hit dump / raid | FA per normal day")
    for r in rows:
        print(
            f"{r['ret_60']:+.2f} {r['z_sell_rush_60']:.0f} {r['z_quote_vol_60']:.0f} | "
            f"{r['hit_sell_dump']} / {r['hit_sell_raid']} | {r['fa_per_day']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
