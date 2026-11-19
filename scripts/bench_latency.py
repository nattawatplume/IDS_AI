"""Compute latency of the stream pipeline on one stored symbol-day (Phase 0 M0, Part D).

Replays one symbol-day through the same code path as `idsai serve --mode replay` (`prepare_replay`
in `idsai/cli.py`): trades of D-1 and D from the trade store, a `Replayer` at max speed (speed 0)
with `start_us` = D 00:00 UTC and a 24 h pre-roll from D-1 00:00, into a `StreamPipeline` in mode
"replay" with the detectors and Chief of a model bundle (`load_bundle`). No HTTP API is started:
the pipeline coroutine is run directly with `asyncio.run`, and its `Bus` has no subscribers.

What is measured is `StreamPipeline.latency_ms`: for each bar that yields a window row after the
pre-roll (one per `stride_s`), the monotonic time from receiving the tick that closed the bar to
having the row scored by every detector and stepped through the Chief (bar push, window row,
float32 rounding, `score_windows`, `ChiefStream.step`). It does not include the series point and
the store / bus writes that follow, the network, waiting for the bar to close, or the stride.

The pipeline keeps the last `LATENCY_SAMPLES` samples in a bounded deque. To keep every sample,
the deque is swapped for an unbounded one on the pipeline object before the run (an attribute of
this instance only; `idsai/pipeline/stream.py` is not changed, and it only ever appends to it).

The bundle defaults to the demo pack (`data/demo`, the bundle `serve --mode replay` uses when no
`--models` is given); `models/research` or `models/live` can be passed with `--models`. The
default day is the demo pack's first pump day (STEEMBTC 2021-01-18, in the test split).

Writes only `--out` (default `data/latency.json`). Alerts go to a DuckDB file in a temporary
directory that is deleted afterwards; `data/app.duckdb` and `data/research.duckdb` are never
opened. Trades are read from `data_dir/trades` only.

    uv run python scripts/bench_latency.py [--symbol STEEMBTC] [--day 2021-01-18]
        [--models data/demo] [--config PATH] [--out data/latency.json]

The numbers are compute latency on this machine; they vary with the hardware and its load.
"""

import argparse
import asyncio
import json
import os
import platform
import sys
import tempfile
import time
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from idsai.config import Settings, load_settings

S_US = 1_000_000
DAY_US = 86_400 * S_US
DEFAULT_SYMBOL = "STEEMBTC"
DEFAULT_DAY = date(2021, 1, 18)
TARGET_P99_MS = 2000.0
NOTE = (
    "compute latency on this machine: tick received -> row scored by every detector and the "
    "Chief; excludes network, waiting for bar close, the stride, and store / bus writes"
)


def day_start_us(day: date) -> int:
    return (day - date(1970, 1, 1)).days * DAY_US


def collect_latency(
    cfg: Settings, store: Any, detectors: list[Any], chief: Any, trades: Any, day: date
) -> tuple[list[float], float]:
    """`(every latency_ms sample, wall seconds)` of one replay of `day` at max speed."""
    from idsai.bus import Bus
    from idsai.data.replay import Replayer
    from idsai.pipeline.stream import StreamPipeline

    d0 = day_start_us(day)
    replayer = Replayer(trades, speed=0, start_us=d0, preroll_from_us=d0 - DAY_US)
    pipeline = StreamPipeline(cfg, store, detectors, chief, Bus(), "replay")
    pipeline.latency_ms = deque()  # unbounded: keep every sample, not only the last ones
    t0 = time.perf_counter()
    asyncio.run(pipeline.run(replayer))
    return list(pipeline.latency_ms), time.perf_counter() - t0


def summarize(samples: list[float]) -> dict[str, Any]:
    """n, mean, median, p90, p99 and max in ms (linear percentiles), rounded to 3 decimals."""
    if not samples:
        return {"n": 0, **dict.fromkeys(["mean_ms", "median_ms", "p90_ms", "p99_ms", "max_ms"])}
    a = np.asarray(samples, dtype=np.float64)
    q = np.percentile(a, [50, 90, 99])
    return {
        "n": int(a.size),
        "mean_ms": round(float(a.mean()), 3),
        "median_ms": round(float(q[0]), 3),
        "p90_ms": round(float(q[1]), 3),
        "p99_ms": round(float(q[2]), 3),
        "max_ms": round(float(a.max()), 3),
    }


def machine_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def build_result(
    symbol: str, day: date, models: Path, stats: dict[str, Any], wall_s: float, n_trades: int
) -> dict[str, Any]:
    p99 = stats["p99_ms"]
    return {
        "symbol": symbol,
        "day": day.isoformat(),
        "models": models.as_posix(),
        "mode": "replay",
        "speed": 0,
        "n_trades": n_trades,
        "wall_s": round(wall_s, 3),
        "latency": stats,
        "target_p99_ms": TARGET_P99_MS,
        "meets_target": None if p99 is None else bool(p99 < TARGET_P99_MS),
        "machine": machine_info(),
        "note": NOTE,
    }


def format_text(r: dict[str, Any]) -> str:
    s, m = r["latency"], r["machine"]

    def ms(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f} ms"

    verdict = {True: "yes", False: "NO", None: "n/a"}[r["meets_target"]]
    return "\n".join(
        [
            f"stream compute latency: {r['symbol']} {r['day']} (mode replay, speed 0, "
            f"24 h pre-roll), models {r['models']}",
            f"trades replayed: {r['n_trades']}  wall: {r['wall_s']:.3f} s",
            f"n      {s['n']}",
            f"mean   {ms(s['mean_ms'])}",
            f"median {ms(s['median_ms'])}",
            f"p90    {ms(s['p90_ms'])}",
            f"p99    {ms(s['p99_ms'])}",
            f"max    {ms(s['max_ms'])}",
            f"target p99 < {r['target_p99_ms']:.0f} ms: {verdict}",
            f"machine: {m['platform']} | {m['machine']} | python {m['python']} | "
            f"cpu_count {m['cpu_count']}",
            f"note: {r['note']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL)
    ap.add_argument("--day", type=date.fromisoformat, default=DEFAULT_DAY)
    ap.add_argument("--models", default=None, help="bundle dir (default: data_dir/demo)")
    ap.add_argument("--out", default=None, help="JSON output (default: data_dir/latency.json)")
    args = ap.parse_args(argv)

    from idsai.ai.learning import load_bundle
    from idsai.data.store import Store

    cfg = load_settings(args.config)
    data_dir = cfg.paths.data_dir
    models = Path(args.models) if args.models else data_dir / "demo"
    out = Path(args.out) if args.out else data_dir / "latency.json"
    symbol, day = args.symbol, args.day

    with tempfile.TemporaryDirectory() as tmp, Store(data_dir, Path(tmp) / "bench.duckdb") as st:
        for d in (day - timedelta(days=1), day):
            if not st.has_day(symbol, d):
                print(f"missing trades for {symbol} {d} (D and D-1 are both needed)")
                return 1
        detectors, chief, _ = load_bundle(models, cfg)
        d0 = day_start_us(day)
        trades = st.read_trades(symbol, d0 - DAY_US, d0 + DAY_US)
        samples, wall_s = collect_latency(cfg, st, detectors, chief, trades, day)

    result = build_result(symbol, day, models, summarize(samples), wall_s, trades.height)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(format_text(result))
    print(f"wrote {out.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
