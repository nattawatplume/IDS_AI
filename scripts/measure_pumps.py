"""Measure real train-period pumps so `configs/scenarios.yaml` pump_burst train ranges fit them.

Task 20 Step 3 (ruling L3): only REAL labels whose `t_start_us` falls on a UTC day inside the
`splits.yaml` train period are measured; every other label is counted as excluded with a reason.
The script never edits `scenarios.yaml`: it prints the quantiles and a suggested `pump_burst.train`
block (10-90 % ranges) and writes everything to JSON (`data/pump_measurements.json` by default).

Read-only and offline: labels come from the app DuckDB opened with `read_only=True` (or, when that
fails, e.g. while a download holds the write lock, from the label files exactly as
`idsai labels` builds them); trades are read straight from the Parquet partitions
`data_dir/trades/symbol=S/date=D/part.parquet`. No `Store` is constructed (it would create tables
and directories). Same inputs give the same JSON byte for byte.

Each parameter is measured so that a pump injected by `idsai.detection.simulator.inject` with
that parameter reads back as (about) the same value. For a label at time `t0` on UTC day `D`:

- `base_quote_per_s`: `base_quote_rate(all trades of day D-1)`, exactly as the batch pipeline
  sizes an injection (`max(total_quote / 86400, median_trade_quote / 60)`).
- `p_ref` (pre-pump reference): median trade price in `[ws - REF_S, ws)` with
  `ws = t0 - SEARCH_BEFORE_S`; inject's `impact` multiplies the unmodified real price, which this
  stands for.
- `impact`: `max trade price in [ws, t0 + SEARCH_AFTER_S) / p_ref - 1`; the time of that trade is
  `t_peak` (inject's price peaks at `t_start + duration_s`). Labels with `impact < MIN_IMPACT`
  show no pump in the price and are excluded (`no_rise`).
- `duration_s`: inject ramps the price linearly from `p_ref` to `p_ref * (1 + impact)` over
  `duration_s`. `t10` is the last trade in `[ws, t_peak)` priced at or below
  `p_ref * (1 + RAMP_FRAC * impact)` (`ws` if there is none), and
  `duration_s = max((t_peak - t10) / (1 - RAMP_FRAC), MIN_DURATION_S)`, the linear-ramp
  extrapolation; the estimated pump start is `t_s = t_peak - duration_s`.
- `vol_mult`: inject adds `vol_mult * base` quote per second during the ramp on top of the real
  flow, so `vol_mult = (quote volume of trades in [t_s, t_peak] / duration_s - background) / base`
  where `background` is the quote volume per second of the reference window.
- `rush_size`: inject places taker-buy fills in clusters of `rush_size` sharing one millisecond.
  Taker buys in `[t_s, t_peak]` are grouped by millisecond (`ts_us // 1000`) and `rush_size` is
  the median group size.
- `dump_tau_s`: after the peak inject decays the excess as `impact * exp(-(t - t_peak) / tau)`.
  Trades in `(t_peak, t_peak + POST_S]` are bucketed into `DUMP_BUCKET_S` buckets from `t_peak`;
  `dump_tau_s` is the midpoint of the first bucket whose median price is at or below
  `p_ref * (1 + impact / e)`. If no bucket gets there, it is not measurable (None) for that label.

    uv run python scripts/measure_pumps.py [--config configs/default.yaml] [--out PATH]
"""

import argparse
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from idsai.config import Settings, load_settings
from idsai.contracts.frames import LABELS, TRADES, conform, empty
from idsai.detection.simulator.scenarios import base_quote_rate, load_scenario_ranges

S_US = 1_000_000
DAY_US = 86_400 * S_US
_EPOCH = date(1970, 1, 1)

SEARCH_BEFORE_S = 300  # the peak is searched from 5 min before the label time
SEARCH_AFTER_S = 1800  # ... to 30 min after it
REF_S = 3600  # reference window: the hour before the search window
POST_S = 3600  # dump search after the peak
DUMP_BUCKET_S = 10
RAMP_FRAC = 0.1  # price level (fraction of the impact) that marks the ramp start
MIN_IMPACT = 0.01  # below this the label shows no pump in the price
MIN_DURATION_S = 1.0
QUANTILES = (0.1, 0.5, 0.9)
PARAMS = ("duration_s", "impact", "vol_mult", "rush_size", "dump_tau_s")
# lower bounds of a suggested range, so every sampled scenario is a valid injection
FLOORS = {
    "duration_s": 1.0,
    "impact": MIN_IMPACT,
    "vol_mult": 1.0,
    "rush_size": 1,
    "dump_tau_s": 1.0,
}
DECIMALS = {"duration_s": 0, "impact": 4, "vol_mult": 2, "dump_tau_s": 0}
EXCLUDE_REASONS = (
    "not_train",
    "missing_day",
    "no_baseline",
    "no_base_volume",
    "no_reference_trades",
    "no_window_trades",
    "no_rise",
)


# ------------------------------------------------------------------ read-only inputs
def day_of_us(ts_us: int) -> date:
    return _EPOCH + timedelta(days=int(ts_us) // DAY_US)


def day_start_us(day: date) -> int:
    return (day - _EPOCH).days * DAY_US


def part_path(data_dir: Path, symbol: str, day: date) -> Path:
    return (
        Path(data_dir) / "trades" / f"symbol={symbol}" / f"date={day.isoformat()}" / "part.parquet"
    )


def read_trades(data_dir: Path, symbol: str, start_us: int, end_us: int) -> pl.DataFrame:
    """Trades with `start_us <= ts_us < end_us` from the Parquet partitions (read-only)."""
    if end_us <= start_us:
        return empty(TRADES)
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


def labels_from_db(db_path: Path) -> pl.DataFrame:
    """REAL labels from the app DuckDB, opened read-only (raises if it cannot be opened)."""
    import duckdb

    if not Path(db_path).is_file():
        raise FileNotFoundError(db_path)
    cols = ", ".join(LABELS.names())
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            f"SELECT {cols} FROM labels WHERE provenance = 'REAL' ORDER BY t_start_us, label_id"
        ).fetchall()
    finally:
        con.close()
    return pl.DataFrame(rows, schema=LABELS, orient="row")


def labels_from_files(cfg: Settings) -> pl.DataFrame:
    """The labels `idsai labels` would store (Sapienza + Hu if present, splits.yaml offsets)."""
    from idsai.cli import build_labels

    return build_labels(cfg)[0]


def load_labels(cfg: Settings, source: str, db_path: Path) -> tuple[pl.DataFrame, str]:
    """`(labels, where they came from)`; `source` is `db`, `files` or `auto` (db, else files)."""
    if source == "files":
        return labels_from_files(cfg), "files"
    try:
        return labels_from_db(db_path), "db"
    except Exception as e:  # locked by a writer, missing, or no labels table
        if source == "db":
            raise
        print(f"app db not readable ({type(e).__name__}); building labels from the files")
        return labels_from_files(cfg), "files"


# ------------------------------------------------------------------ measurement
def measure_window(
    trades: pl.DataFrame, t0: int, base: float, p_ref: float, background: float
) -> dict[str, Any]:
    """Measure one pump in `trades` (which must cover `[t0 - SEARCH_BEFORE_S, peak + POST_S]`).

    Returns the measured fields, or `{"status": reason}` when nothing can be measured.
    """
    ts = trades["ts_us"].to_numpy()
    price = trades["price"].to_numpy().astype(np.float64)
    ws = t0 - SEARCH_BEFORE_S * S_US
    we = t0 + SEARCH_AFTER_S * S_US
    lo, hi = np.searchsorted(ts, ws, "left"), np.searchsorted(ts, we, "left")
    if hi <= lo:
        return {"status": "no_window_trades"}
    i_peak = lo + int(np.argmax(price[lo:hi]))  # earliest trade at the maximum price
    t_peak = int(ts[i_peak])
    impact = float(price[i_peak] / p_ref - 1.0)
    out: dict[str, Any] = {"t_peak_us": t_peak, "impact": impact}
    if impact < MIN_IMPACT:
        return {"status": "no_rise", **out}

    level = p_ref * (1.0 + RAMP_FRAC * impact)
    below = np.nonzero(price[lo:i_peak] <= level)[0]
    t10 = int(ts[lo + below[-1]]) if below.size else ws
    duration_s = max((t_peak - t10) / S_US / (1.0 - RAMP_FRAC), MIN_DURATION_S)
    t_s = t_peak - int(round(duration_s * S_US))

    in_pump = (ts >= t_s) & (ts <= t_peak)
    rate = float(trades["quote_qty"].to_numpy()[in_pump].sum()) / duration_s
    vol_mult = (rate - background) / base

    buys = trades.filter(
        (pl.col("ts_us") >= t_s) & (pl.col("ts_us") <= t_peak) & (pl.col("taker_side") == "buy")
    )
    if buys.height:
        sizes = buys.group_by(pl.col("ts_us") // 1000).len()["len"].to_numpy()
        rush_size = float(np.median(sizes))
    else:
        rush_size = None

    post = trades.filter((pl.col("ts_us") > t_peak) & (pl.col("ts_us") <= t_peak + POST_S * S_US))
    dump_tau_s = None
    if post.height:
        bucket_us = DUMP_BUCKET_S * S_US
        med = (
            post.group_by(((pl.col("ts_us") - t_peak - 1) // bucket_us).alias("b"))
            .agg(pl.col("price").median())
            .sort("b")
        )
        hit = med.filter(pl.col("price") <= p_ref * (1.0 + impact / math.e))
        if hit.height:
            dump_tau_s = (int(hit["b"][0]) + 0.5) * DUMP_BUCKET_S

    return {
        "status": "measured",
        **out,
        "t_start_est_us": t_s,
        "duration_s": duration_s,
        "vol_mult": vol_mult,
        "rush_size": rush_size,
        "dump_tau_s": dump_tau_s,
    }


def measure_label(data_dir: Path, symbol: str, t0: int) -> dict[str, Any]:
    """Measure the label at `t0` from the stored trades (status `measured` or an exclusion)."""
    day = day_of_us(t0)
    if not part_path(data_dir, symbol, day).is_file():
        return {"status": "missing_day"}
    prev = day - timedelta(days=1)
    if not part_path(data_dir, symbol, prev).is_file():
        return {"status": "no_baseline"}
    prev_trades = read_trades(data_dir, symbol, day_start_us(prev), day_start_us(day))
    try:
        base = base_quote_rate(prev_trades)
    except ValueError:
        return {"status": "no_base_volume"}

    ws = t0 - SEARCH_BEFORE_S * S_US
    ref_lo = ws - REF_S * S_US
    trades = read_trades(data_dir, symbol, ref_lo, t0 + (SEARCH_AFTER_S + POST_S) * S_US)
    ref = trades.filter(pl.col("ts_us") < ws)
    if ref.height == 0:
        return {"status": "no_reference_trades", "base_quote_per_s": base}
    p_ref = float(ref["price"].median())
    background = float(ref["quote_qty"].sum()) / REF_S
    res = measure_window(trades.filter(pl.col("ts_us") >= ws), t0, base, p_ref, background)
    return {"base_quote_per_s": base, "p_ref": p_ref, "background_quote_per_s": background, **res}


# ------------------------------------------------------------------ summary
def quantiles(values: list[float]) -> dict[str, Any]:
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "q10": None, "q50": None, "q90": None}
    q = np.quantile(v, QUANTILES)
    return {"n": int(v.size), "q10": float(q[0]), "q50": float(q[1]), "q90": float(q[2])}


def _num(x: float, decimals: int) -> float | int:
    return int(round(x)) if decimals == 0 else round(float(x), decimals)


def suggest_train(qs: dict[str, dict[str, Any]], current: dict[str, Any]) -> tuple[dict, list]:
    """Suggested `pump_burst.train` block (10-90 % ranges) and the parameters kept as they are
    because nothing could be measured for them."""
    block: dict[str, list] = {}
    kept = []
    for p in PARAMS:
        q = qs[p]
        if q["n"] == 0:
            if p in current:
                block[p] = list(current[p])
            kept.append(p)
            continue
        floor = FLOORS[p]
        if p == "rush_size":
            lo = max(int(floor), int(round(q["q10"])))
            hi = max(lo, int(round(q["q90"])))
        else:
            d = DECIMALS[p]
            lo = _num(max(floor, q["q10"]), d)
            hi = _num(max(lo, q["q90"]), d)
        block[p] = [lo, hi]
    return block, kept


def run(cfg: Settings, labels: pl.DataFrame, label_source: str) -> dict[str, Any]:
    """Measure every REAL train-period label; the full result as a JSON-ready dict."""
    labels = conform(labels, LABELS).filter(pl.col("provenance") == "REAL")
    labels = labels.sort(["t_start_us", "symbol", "label_id"])
    train_lo, train_hi = cfg.splits.train
    data_dir = Path(cfg.paths.data_dir)
    excluded = dict.fromkeys(EXCLUDE_REASONS, 0)
    not_train_by_period: dict[str, int] = {}
    records = []
    for lab in labels.iter_rows(named=True):
        t0 = int(lab["t_start_us"])
        day = day_of_us(t0)
        if not (train_lo <= day <= train_hi):
            excluded["not_train"] += 1
            period = _period(day, cfg)
            not_train_by_period[period] = not_train_by_period.get(period, 0) + 1
            continue
        res = measure_label(data_dir, lab["symbol"], t0)
        if res["status"] != "measured":
            excluded[res["status"]] += 1
        records.append(
            {
                "label_id": lab["label_id"],
                "source": lab["source"],
                "symbol": lab["symbol"],
                "t_start_us": t0,
                "day": day.isoformat(),
                **res,
            }
        )
    measured = [r for r in records if r["status"] == "measured"]
    qs = {p: quantiles([r[p] for r in measured]) for p in PARAMS}
    current = load_scenario_ranges().get("pump_burst", {}).get("train", {})
    block, kept = suggest_train(qs, current)
    return {
        "generated_by": "scripts/measure_pumps.py",
        "label_source": label_source,
        "train_period": [train_lo.isoformat(), train_hi.isoformat()],
        "settings": {
            "search_before_s": SEARCH_BEFORE_S,
            "search_after_s": SEARCH_AFTER_S,
            "ref_s": REF_S,
            "post_s": POST_S,
            "dump_bucket_s": DUMP_BUCKET_S,
            "ramp_frac": RAMP_FRAC,
            "min_impact": MIN_IMPACT,
            "floors": FLOORS,
        },
        "n_labels": labels.height,
        "n_train_labels": len(records),
        "n_measured": len(measured),
        "excluded": excluded,
        "not_train_by_period": dict(sorted(not_train_by_period.items())),
        "quantiles": qs,
        "suggested_pump_burst_train": block,
        "kept_unmeasured": kept,
        "labels": records,
    }


def _period(day: date, cfg: Settings) -> str:
    for name in ("train", "cal", "accept", "test"):
        lo, hi = getattr(cfg.splits, name)
        if lo <= day <= hi:
            return name
    return "none"


def summary_lines(result: dict[str, Any]) -> list[str]:
    lines = [
        f"labels: {result['n_labels']} REAL ({result['label_source']}), "
        f"{result['n_train_labels']} in train {result['train_period']}, "
        f"{result['n_measured']} measured",
        "excluded: "
        + ", ".join(f"{k} {v}" for k, v in result["excluded"].items() if v)
        + (
            f" (not_train by period: {result['not_train_by_period']})"
            if result["excluded"]["not_train"]
            else ""
        ),
        "parameter       n      q10        q50        q90",
    ]
    for p, q in result["quantiles"].items():
        if q["n"]:
            lines.append(
                f"{p:<12} {q['n']:>4} {q['q10']:>10.4g} {q['q50']:>10.4g} {q['q90']:>10.4g}"
            )
        else:
            lines.append(f"{p:<12} {0:>4}  not measurable")
    lines.append("suggested scenarios.yaml block (not applied):")
    block = {"pump_burst": {"train": result["suggested_pump_burst_train"]}}
    lines.extend(yaml.safe_dump(block, default_flow_style=None, sort_keys=False).splitlines())
    if result["kept_unmeasured"]:
        lines.append(f"kept from scenarios.yaml (nothing measured): {result['kept_unmeasured']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="settings YAML (default configs/default.yaml)")
    ap.add_argument(
        "--out", default=None, help="JSON output (default data_dir/pump_measurements.json)"
    )
    ap.add_argument("--labels", choices=("auto", "db", "files"), default="auto")
    ap.add_argument("--db", default=None, help="app DuckDB (default paths.app_db), read-only")
    args = ap.parse_args(argv)
    cfg = load_settings(args.config)
    db = Path(args.db) if args.db else Path(cfg.paths.app_db)
    labels, label_source = load_labels(cfg, args.labels, db)
    result = run(cfg, labels, label_source)
    for line in summary_lines(result):
        print(line)
    out = Path(args.out) if args.out else Path(cfg.paths.data_dir) / "pump_measurements.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
