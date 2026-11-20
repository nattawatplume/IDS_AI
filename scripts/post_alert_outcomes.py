"""Alert-to-peak timing and the price after an alert for the REAL labels (Phase 0.1 + 0.2).

Read-only and offline. Run records come from the research DuckDB (`runs`, opened with
`read_only=True`): experiment `real_t001`, variant `full`, one budget (`--budget`, default 2),
every seed, only the most recent `config_hash` (as `report._latest_config`); records without
metrics are skipped. Label `t_start_us` comes from the research DB `labels` table when it holds
every needed label, else from `measure_pumps.load_labels` (app DuckDB read-only, else the label
files). Trades are read straight from `data_dir/trades/symbol=S/date=D/part.parquet`.

Per (label, seed) with `detected_any`: `alert_us = t0 + round(ttd_s * 1e6)` (alert-episode onset).
Per label (seed independent): `measure_pumps.measure_label`; only `status == "measured"` feeds
the peak fields, every other status is counted by its reason.

- `lead_to_peak_s = (t_peak_us - alert_us) / 1e6` (positive: the alert came before the peak)
- `p(t)`: price of the last trade at or before `t`; missing when there is no trade in
  `[t - 1 h, t]` (counted)
- `p_alert = p(alert_us)`, `p_peak` = the peak trade price, `ret_peak = p_peak / p_alert - 1`
- `ret_h = p(alert_us + h) / p_alert - 1` for h in 15m, 1h, 24h
- `drawdown_h` / `max_up_h` = min / max trade price in `(alert_us, alert_us + h] / p_alert - 1`
  (missing without trades there, or when `ret_h` is missing: the window end is not covered)

CI rule (written to the output too): an event a seed did not detect is left out of that seed.
Each label's value is the mean over the seeds that detected it, and the 95 % CI is the
event bootstrap (`idsai.eval.metrics.bootstrap_ci`, seed 0) over those per-label means.

    uv run python scripts/post_alert_outcomes.py [--config PATH] [--budget 2] [--out PATH]

Same inputs give the same JSON and Markdown byte for byte. Historical statistics, not advice.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from idsai.config import load_settings
from idsai.eval.metrics import bootstrap_ci
from idsai.eval.report import _latest_config

_mp_path = Path(__file__).resolve().parent / "measure_pumps.py"
_spec = importlib.util.spec_from_file_location("measure_pumps", _mp_path)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)

S_US = mp.S_US
EXPERIMENT = "real_t001"
VARIANT = "full"
LOOKBACK_S = 3600  # p(t) is missing without a trade in the hour before t
ONSET_MIN_S = -120.0  # sensitivity view: alert onset at most 2 min before the label
HORIZONS = {"15m": 900, "1h": 3600, "24h": 86_400}
QUANTILE_FIELDS = (
    "lead_to_peak_s",
    "ret_peak",
    "ret_15m",
    "ret_1h",
    "ret_24h",
    "drawdown_15m",
    "max_up_15m",
    "drawdown_1h",
    "max_up_1h",
    "drawdown_24h",
    "max_up_24h",
)
CI_RULE = (
    "per-label seed-mean over the seeds that detected the label (undetected seeds left out), "
    "then event bootstrap over labels (idsai.eval.metrics.bootstrap_ci, rng seed 0)"
)
CAVEATS = (
    "labels are announcement times (BTC pairs, 2018-2021), not the true pump start",
    "alert time ignores the 1 s bar close and the 10 s stride (same note as TTD)",
    "historical statistics of a statistical anomaly detector, not advice and not an accusation",
)


# ------------------------------------------------------------------ read-only inputs
def load_runs(research_db: Path, budget: float) -> tuple[dict[int, dict], str | None, str | None]:
    """`({seed: record}, config_hash, note)` for real_t001/full at `budget`, latest config."""
    con = duckdb.connect(str(research_db), read_only=True)
    try:
        rows = con.execute(
            "SELECT record_json FROM runs WHERE experiment = ? AND variant = ? "
            "ORDER BY created_us, run_id",
            [EXPERIMENT, VARIANT],
        ).fetchall()
    finally:
        con.close()
    recs = [json.loads(r[0]) for r in rows]
    recs, note = _latest_config([r for r in recs if float(r["budget"]) == float(budget)])
    by_seed = {int(r["seed"]): r for r in recs if r.get("metrics") is not None}
    config_hash = recs[-1].get("config_hash") if recs else None
    return dict(sorted(by_seed.items())), config_hash, note


def _research_labels(research_db: Path) -> dict[str, tuple[str, int]]:
    con = duckdb.connect(str(research_db), read_only=True)
    try:
        rows = con.execute("SELECT label_id, symbol, t_start_us FROM labels").fetchall()
    except duckdb.Error:
        rows = []
    finally:
        con.close()
    return {lid: (sym, int(t0)) for lid, sym, t0 in rows}


def label_starts(
    research_db: Path, ids: set[str], cfg: Any, source: str, app_db: Path
) -> tuple[dict[str, tuple[str, int]], str]:
    """`({label_id: (symbol, t_start_us)}, where they came from)` for `ids`."""
    got = _research_labels(research_db)
    if ids <= got.keys():
        return got, "research_db"
    labels, where = mp.load_labels(cfg, source, app_db)
    out = {
        r["label_id"]: (r["symbol"], int(r["t_start_us"]))
        for r in labels.select("label_id", "symbol", "t_start_us").iter_rows(named=True)
    }
    return out, where


# ------------------------------------------------------------------ price lookups
def price_at(ts: np.ndarray, price: np.ndarray, t: int) -> float | None:
    """Price of the last trade at or before `t`; None without a trade in `[t - 1 h, t]`."""
    i = int(np.searchsorted(ts, t, "right")) - 1
    if i < 0 or ts[i] < t - LOOKBACK_S * S_US:
        return None
    return float(price[i])


def extremes(ts: np.ndarray, price: np.ndarray, lo: int, hi: int) -> tuple[float, float] | None:
    """`(min, max)` trade price in `(lo, hi]`; None without trades there."""
    a, b = int(np.searchsorted(ts, lo, "right")), int(np.searchsorted(ts, hi, "right"))
    if b <= a:
        return None
    seg = price[a:b]
    return float(seg.min()), float(seg.max())


def alert_outcome(
    ts: np.ndarray, price: np.ndarray, alert_us: int, meas: dict[str, Any]
) -> dict[str, Any]:
    """Every field of one (label, seed) alert; missing values are None."""
    out: dict[str, Any] = {"alert_us": alert_us}
    p_alert = price_at(ts, price, alert_us)
    out["p_alert"] = p_alert
    measured = meas["status"] == "measured"
    t_peak = meas.get("t_peak_us") if measured else None
    p_peak = None
    if t_peak is not None:
        at = (ts == t_peak).nonzero()[0]
        p_peak = float(price[at].max()) if at.size else None
    out["lead_to_peak_s"] = (t_peak - alert_us) / S_US if t_peak is not None else None
    out["p_peak"] = p_peak
    ok = p_alert is not None and p_alert > 0
    out["ret_peak"] = p_peak / p_alert - 1.0 if ok and p_peak is not None else None
    for name, h in HORIZONS.items():
        t = alert_us + h * S_US
        p = price_at(ts, price, t) if ok else None
        out[f"ret_{name}"] = p / p_alert - 1.0 if p is not None else None
        ext = extremes(ts, price, alert_us, t) if p is not None else None  # window end covered
        out[f"drawdown_{name}"] = ext[0] / p_alert - 1.0 if ext else None
        out[f"max_up_{name}"] = ext[1] / p_alert - 1.0 if ext else None
    return out


# ------------------------------------------------------------------ summary helpers
def _clean(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    return x


def frac_ci(pairs: list[dict], pred, n_boot: int) -> dict[str, Any]:
    """Share of pairs where `pred(pair)` is true (pairs with None skipped), with the CI rule."""
    per_label: dict[str, list[float]] = {}
    for p in pairs:
        v = pred(p)
        if v is not None:
            per_label.setdefault(p["label_id"], []).append(float(v))
    flat = [v for vs in per_label.values() for v in vs]
    means = np.asarray([np.mean(per_label[k]) for k in sorted(per_label)], dtype=np.float64)
    label_mean, (lo, hi) = bootstrap_ci(means, n_boot, 0)
    return {
        "n_pairs": len(flat),
        "n_labels": int(means.size),
        "pair_mean": float(np.mean(flat)) if flat else None,
        "label_mean": label_mean,
        "ci95": [lo, hi],
    }


def _lead_positive(p: dict) -> bool | None:
    v = p["lead_to_peak_s"]
    return None if v is None else v > 0


def _ret_1h_negative(p: dict) -> bool | None:
    v = p["ret_1h"]
    return None if v is None else v < 0


def onset_near_label(pairs: list[dict], onset_min_s: float, n_boot: int) -> dict[str, Any]:
    """Sensitivity view on the pairs whose alert onset is at most `-onset_min_s` before the
    label (`ttd_s >= onset_min_s`): an earlier onset may be part of an episode that began
    before the label and inflates "before peak" and the lead."""
    near = [p for p in pairs if p["ttd_s"] >= onset_min_s]
    return {
        "onset_min_s": onset_min_s,
        "n_pairs_kept": len(near),
        "n_pairs_dropped": len(pairs) - len(near),
        "frac_alert_before_peak": frac_ci(near, _lead_positive, n_boot),
        "frac_ret_1h_negative": frac_ci(near, _ret_1h_negative, n_boot),
        "quantiles": {f: mp.quantiles([p[f] for p in near]) for f in ("lead_to_peak_s", "ret_1h")},
    }


def run(
    data_dir: Path,
    runs_by_seed: dict[int, dict],
    starts: dict[str, tuple[str, int]],
    n_boot: int,
    onset_min_s: float = ONSET_MIN_S,
) -> dict[str, Any]:
    """Every detected (label, seed) pair and the summary as a JSON-ready dict."""
    detected: dict[str, dict[int, float]] = {}
    all_ids: set[str] = set()
    n_no_ttd = 0
    for seed, rec in runs_by_seed.items():
        for lid, ev in rec["per_event"].items():
            all_ids.add(lid)
            if not ev.get("detected_any"):
                continue
            if ev.get("ttd_s") is None:
                n_no_ttd += 1
                continue
            detected.setdefault(lid, {})[seed] = float(ev["ttd_s"])

    status_counts: dict[str, int] = {}
    n_no_label = 0
    labels_out = []
    pairs = []
    for lid in sorted(detected):
        if lid not in starts:
            n_no_label += 1
            continue
        symbol, t0 = starts[lid]
        meas = mp.measure_label(data_dir, symbol, t0)
        status_counts[meas["status"]] = status_counts.get(meas["status"], 0) + 1
        alerts = {s: t0 + int(round(ttd * S_US)) for s, ttd in sorted(detected[lid].items())}
        lo = min(alerts.values()) - LOOKBACK_S * S_US
        hi = max(alerts.values()) + max(HORIZONS.values()) * S_US + 1
        trades = mp.read_trades(data_dir, symbol, lo, hi)
        ts = trades["ts_us"].to_numpy()
        price = trades["price"].to_numpy().astype(np.float64)
        labels_out.append(
            {
                "label_id": lid,
                "symbol": symbol,
                "t_start_us": t0,
                "status": meas["status"],
                "t_peak_us": meas.get("t_peak_us") if meas["status"] == "measured" else None,
                "seeds_detected": sorted(alerts),
            }
        )
        for seed, alert_us in alerts.items():
            pairs.append(
                {
                    "label_id": lid,
                    "seed": seed,
                    "ttd_s": detected[lid][seed],
                    **alert_outcome(ts, price, alert_us, meas),
                }
            )

    measured_ids = {r["label_id"] for r in labels_out if r["status"] == "measured"}
    qs = {f: mp.quantiles([p[f] for p in pairs]) for f in QUANTILE_FIELDS}
    missing = {
        f: sum(p[f] is None for p in pairs)
        for f in ("p_alert", "ret_15m", "ret_1h", "ret_24h", "drawdown_1h")
    }
    missing["peak_fields_not_measured"] = sum(p["lead_to_peak_s"] is None for p in pairs)
    per_seed = {
        str(s): mp.quantiles([p["lead_to_peak_s"] for p in pairs if p["seed"] == s])["q50"]
        for s in runs_by_seed
    }
    result = {
        "generated_by": "scripts/post_alert_outcomes.py",
        "experiment": EXPERIMENT,
        "variant": VARIANT,
        "seeds": sorted(runs_by_seed),
        "n_labels": len(all_ids),
        "n_detected": len(detected),
        "n_detected_pairs": len(pairs),
        "n_measured": len(measured_ids),
        "excluded": {
            "status": dict(sorted(status_counts.items())),
            "no_label": n_no_label,
            "detected_without_ttd_pairs": n_no_ttd,
        },
        "missing_pairs": missing,
        "frac_alert_before_peak": frac_ci(pairs, _lead_positive, n_boot),
        "frac_ret_1h_negative": frac_ci(pairs, _ret_1h_negative, n_boot),
        "quantiles": qs,
        "per_seed_median_lead_to_peak_s": per_seed,
        "onset_near_label": onset_near_label(pairs, onset_min_s, n_boot),
        "settings": {
            "lookback_s": LOOKBACK_S,
            "horizons_s": HORIZONS,
            "n_boot": n_boot,
            "bootstrap_seed": 0,
        },
        "ci_rule": CI_RULE,
        "caveats": list(CAVEATS),
        "labels": labels_out,
        "pairs": pairs,
    }
    return _clean(result)


# ------------------------------------------------------------------ output
def _f(x: float | None, spec: str = ".4g") -> str:
    return "n/a" if x is None else format(x, spec)


def _frac_line(name: str, d: dict) -> str:
    return (
        f"{name}: pair_mean {_f(d['pair_mean'], '.3f')} ({d['n_pairs']} pairs), "
        f"label_mean {_f(d['label_mean'], '.3f')} CI95 [{_f(d['ci95'][0], '.3f')}, "
        f"{_f(d['ci95'][1], '.3f')}] ({d['n_labels']} labels)"
    )


def summary_lines(r: dict[str, Any]) -> list[str]:
    ex = r["excluded"]
    lines = [
        f"runs: {r['experiment']}/{r['variant']} budget {r['budget']} config {r['config_hash']} "
        f"seeds {r['seeds']} labels from {r['label_source']}",
        f"labels: {r['n_labels']}, detected (any seed) {r['n_detected']} "
        f"({r['n_detected_pairs']} label-seed pairs), measured peak {r['n_measured']}",
        "excluded from peak stats: "
        + ", ".join(f"{k} {v}" for k, v in ex["status"].items() if k != "measured")
        + f"; no_label {ex['no_label']}; detected_without_ttd_pairs "
        f"{ex['detected_without_ttd_pairs']}",
        "missing (pairs): " + ", ".join(f"{k} {v}" for k, v in r["missing_pairs"].items()),
        _frac_line("frac_alert_before_peak", r["frac_alert_before_peak"]),
        _frac_line("frac(ret_1h < 0)", r["frac_ret_1h_negative"]),
        "field                n        q10        q50        q90",
    ]
    for f, q in r["quantiles"].items():
        lines.append(
            f"{f:<16} {q['n']:>5} {_f(q['q10']):>10} {_f(q['q50']):>10} {_f(q['q90']):>10}"
        )
    lines.append(
        "per-seed median lead_to_peak_s: "
        + ", ".join(f"s{s} {_f(v)}" for s, v in r["per_seed_median_lead_to_peak_s"].items())
    )
    lines.append(_note_24h(r))
    o = r["onset_near_label"]
    lines += [
        f"onset_near_label (ttd_s >= {o['onset_min_s']:g} s): {o['n_pairs_kept']} pairs kept, "
        f"{o['n_pairs_dropped']} dropped (onset may belong to an episode begun before the label)",
        "  " + _frac_line("frac_alert_before_peak", o["frac_alert_before_peak"]),
        "  " + _frac_line("frac(ret_1h < 0)", o["frac_ret_1h_negative"]),
    ]
    for f, q in o["quantiles"].items():
        lines.append(
            f"  {f:<16} {q['n']:>5} {_f(q['q10']):>10} {_f(q['q50']):>10} {_f(q['q90']):>10}"
        )
    if r.get("config_note"):
        lines.append(f"note: {r['config_note']}")
    return lines


def _note_24h(r: dict[str, Any]) -> str:
    n = r["quantiles"]["ret_24h"]["n"]
    return (
        f"note: the 24h fields rest on only {n} of {r['n_detected_pairs']} pairs "
        "(next-day trades are mostly not stored)"
    )


def markdown(r: dict[str, Any]) -> str:
    out = [
        "# Post-alert outcomes (REAL, real_t001 full)",
        "",
        "สถิติย้อนหลังของเวลาเตือนเทียบกับจุดสูงสุดของราคา และราคาหลังการเตือน "
        "เป็นความผิดปกติทางสถิติ ไม่ใช่การกล่าวหา และไม่ใช่คำแนะนำการลงทุน",
        "",
        f"- budget {r['budget']}, config_hash `{r['config_hash']}`, seeds {r['seeds']}, "
        f"labels from `{r['label_source']}`",
        f"- n_labels {r['n_labels']}, n_detected {r['n_detected']} "
        f"(n_detected_pairs {r['n_detected_pairs']}), n_measured {r['n_measured']}",
        f"- excluded status (peak fields): {r['excluded']['status']}, "
        f"no_label {r['excluded']['no_label']}, "
        f"detected_without_ttd_pairs {r['excluded']['detected_without_ttd_pairs']}",
        f"- missing_pairs: {r['missing_pairs']}",
        "",
        "## Fractions",
        "",
        "| field | pair_mean | label_mean | CI95 | n_pairs | n_labels |",
        "|---|---|---|---|---|---|",
    ]
    for name, key in (
        ("frac_alert_before_peak", "frac_alert_before_peak"),
        ("frac(ret_1h < 0)", "frac_ret_1h_negative"),
    ):
        d = r[key]
        out.append(
            f"| {name} | {_f(d['pair_mean'], '.3f')} | {_f(d['label_mean'], '.3f')} | "
            f"[{_f(d['ci95'][0], '.3f')}, {_f(d['ci95'][1], '.3f')}] | {d['n_pairs']} | "
            f"{d['n_labels']} |"
        )
    out += [
        "",
        f"CI rule: {r['ci_rule']}",
        "",
        "## Quantiles (pooled over label-seed pairs)",
        "",
        "| field | n | q10 | q50 | q90 |",
        "|---|---|---|---|---|",
    ]
    for f, q in r["quantiles"].items():
        out.append(f"| {f} | {q['n']} | {_f(q['q10'])} | {_f(q['q50'])} | {_f(q['q90'])} |")
    out += [
        "",
        "per-seed median lead_to_peak_s: "
        + ", ".join(f"s{s} {_f(v)}" for s, v in r["per_seed_median_lead_to_peak_s"].items()),
        "",
        f"ช่วง 24h อิงเพียง {r['quantiles']['ret_24h']['n']} จาก {r['n_detected_pairs']} คู่ "
        "(ส่วนใหญ่ไม่ได้เก็บ trade ของวันถัดไป)",
    ]
    o = r["onset_near_label"]
    out += [
        "",
        f"## Sensitivity: onset_near_label (ttd_s >= {o['onset_min_s']:g} s)",
        "",
        "การเตือนบางครั้งอาจเป็นส่วนหนึ่งของ episode ที่เริ่มก่อนเวลา label "
        "ซึ่งทำให้สัดส่วน “เตือนก่อนจุดสูงสุด” และ lead ดูสูงเกินจริง "
        f"ส่วนนี้นับเฉพาะคู่ที่ ttd_s >= {o['onset_min_s']:g} s "
        f"(เก็บ {o['n_pairs_kept']} คู่, ตัดออก {o['n_pairs_dropped']} คู่)",
        "",
        "| field | pair_mean | label_mean | CI95 | n_pairs | n_labels |",
        "|---|---|---|---|---|---|",
    ]
    for name, key in (
        ("frac_alert_before_peak", "frac_alert_before_peak"),
        ("frac(ret_1h < 0)", "frac_ret_1h_negative"),
    ):
        d = o[key]
        out.append(
            f"| {name} | {_f(d['pair_mean'], '.3f')} | {_f(d['label_mean'], '.3f')} | "
            f"[{_f(d['ci95'][0], '.3f')}, {_f(d['ci95'][1], '.3f')}] | {d['n_pairs']} | "
            f"{d['n_labels']} |"
        )
    out += ["", "| field | n | q10 | q50 | q90 |", "|---|---|---|---|---|"]
    for f, q in o["quantiles"].items():
        out.append(f"| {f} | {q['n']} | {_f(q['q10'])} | {_f(q['q50'])} | {_f(q['q90'])} |")
    out += [
        "",
        "## ข้อควรระวัง (caveats)",
        "",
        "- label คือเวลาประกาศ (คู่ BTC ปี 2018-2021) ไม่ใช่เวลาเริ่ม pump จริง",
        "- เวลาเตือนไม่รวมการปิด bar 1 วินาทีและ stride 10 วินาที (หมายเหตุเดียวกับ TTD)",
        "- เป็นสถิติย้อนหลัง ไม่ใช่คำแนะนำการลงทุน",
        "- `lead_to_peak_s > 0` แปลว่าเตือนก่อนจุดสูงสุด; `ret_peak` คือส่วนต่างสูงสุดที่ยังเหลือ "
        "ณ เวลาเตือน (ดู `drawdown_1h` คู่กับ `max_up_1h` เสมอ)",
    ]
    if r.get("config_note"):
        out += ["", f"หมายเหตุ: {r['config_note']}"]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="settings YAML (default configs/default.yaml)")
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--n-boot", type=int, default=None, help="default eval.n_boot")
    ap.add_argument("--research-db", default=None, help="default paths.research_db (read-only)")
    ap.add_argument("--app-db", default=None, help="default paths.app_db (read-only)")
    ap.add_argument("--labels", choices=("auto", "db", "files"), default="auto")
    ap.add_argument("--data-dir", default=None, help="default paths.data_dir")
    ap.add_argument(
        "--onset-min-s",
        type=float,
        default=ONSET_MIN_S,
        help="sensitivity view keeps pairs with ttd_s >= this (default -120)",
    )
    ap.add_argument(
        "--out", default=None, help="JSON output (default data_dir/post_alert_outcomes.json)"
    )
    args = ap.parse_args(argv)
    cfg = load_settings(args.config)
    research_db = Path(args.research_db) if args.research_db else Path(cfg.paths.research_db)
    app_db = Path(args.app_db) if args.app_db else Path(cfg.paths.app_db)
    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg.paths.data_dir)
    n_boot = args.n_boot if args.n_boot is not None else cfg.eval.n_boot

    runs_by_seed, config_hash, note = load_runs(research_db, args.budget)
    ids = {lid for r in runs_by_seed.values() for lid in r["per_event"]}
    starts, label_source = label_starts(research_db, ids, cfg, args.labels, app_db)
    result = run(data_dir, runs_by_seed, starts, n_boot, args.onset_min_s)
    result.update(
        budget=args.budget, config_hash=config_hash, config_note=note, label_source=label_source
    )
    for line in summary_lines(result):
        print(line)
    out = Path(args.out) if args.out else data_dir / "post_alert_outcomes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(markdown(result), encoding="utf-8")
    print(f"wrote {out} and {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
