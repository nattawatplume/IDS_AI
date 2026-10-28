"""Evaluation report built only from the `runs` table (every run is counted, none hand-picked).

`build_report` groups run records by experiment x variant x budget and writes a Markdown report
plus the same content as JSON (read by `serve`). Per cell: the mean over seeds of each metric
(recall: the cluster ratio estimator below), a 95 % bootstrap CI of recall, the provenance,
the seed count and a "tau saturated" flag. Recall and false alerts per symbol-day are the
headline; precision is always printed with prevalence.

Bootstrap unit: recall is a cluster ratio estimator, `sum(hits) / sum(n)` over every
`(seed, event)` value, and each bootstrap resample of clusters recomputes that ratio, so the
shown recall and its CI come from the same estimator. REAL events are their own clusters
(`event`, keyed by label_id). SEMI-SYN scenarios of every seed and family are injected into
the same test-normal days, so they are clustered by `(symbol, day)` (`symbol-day`); treating
`(seed, event)` pairs as independent would make the CI too narrow. Older SEMI-SYN records
without a `day` fall back to `event (legacy, optimistic)`.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from idsai.config import Settings, load_settings
from idsai.data.store import Store
from idsai.eval.experiments import AE, BASELINES, EXPERIMENTS
from idsai.eval.metrics import paired_bootstrap, ratio_bootstrap_ci, significant

CI_SEED = 0
PAIRED_BUDGET = 2.0
MEAN_METRICS = (
    "recall_any",
    "recall_correct",
    "precision",
    "prevalence",
    "fa_per_symbol_day",
    "fa_per_symbol_day_hard",
    "ttd_median_s",
    "ttd_p90_s",
    "pr_auc",
    "f1",
    "n_alerts",
)
CI_METRICS = {"recall_any": "detected_any", "recall_correct": "detected_correct"}
DETERMINISTIC_NOTE = "deterministic: CI from event bootstrap"
SATURATED_MARK = "τ saturated"
AGNOSTIC_NOTE = "technique-agnostic: recall_correct n/a"
UNIT_EVENT, UNIT_SYMBOL_DAY = "event", "symbol-day"
UNIT_LEGACY = "event (legacy, optimistic)"
DROP_REASONS = ("out_of_period", "cal_period", "missing_day", "no_baseline")
NOTES_TH = [
    "หน้าต่างเวลาที่ทับซ้อนกันมี autocorrelation สูง จึงทำให้การรับประกันของ conformal p-value "
    "อ่อนลง: อัตราการแจ้งเตือนจริงอาจสูงกว่างบที่ตั้งไว้",
    "เวลาตรวจจับ (TTD, latency) ไม่รวมเวลารอปิดแท่ง 1 วินาทีและเวลารอ stride 10 วินาที",
    "precision แสดงคู่กับ prevalence (สัดส่วนหน้าต่างที่เป็นบวก) เสมอ; F1 เก็บไว้ใน JSON แต่ไม่ใช่ตัวชี้วัดหลัก",
    "คะแนนรวม (fused) คือ anomaly p-value หรือ “ความหายากเทียบตลาดปกติ” ไม่ใช่ความน่าจะเป็นว่ามีการปั่นราคา",
    "ทุกการแจ้งเตือนคือความผิดปกติทางสถิติ ไม่ได้กล่าวหาบุคคลใด",
    f"{SATURATED_MARK}: งบการแจ้งเตือนสูงกว่าอัตราสูงสุดที่ข้อมูลสอบเทียบทำได้ τ จึงถูกตั้งที่ค่า F บวกที่เล็กที่สุด",
    f"{DETERMINISTIC_NOTE}: ตัวตรวจจับที่ไม่ขึ้นกับ seed ให้ผลเหมือนกันทุก seed "
    "ช่วงความเชื่อมั่นจึงมาจากการ bootstrap ข้ามเหตุการณ์เท่านั้น",
    "ref_supervised เป็น supervised reference (uses labels) ใช้เทียบเท่านั้น",
    f"{AGNOSTIC_NOTE}: ชุดตัวตรวจจับที่ไม่มี rule หรือ stat.cusum ระบุเทคนิคไม่ได้ (ได้ MM-T000 "
    "เสมอ) จึงไม่แสดง recall_correct",
    "bootstrap unit: event = สุ่มซ้ำทีละเหตุการณ์จริง; symbol-day = สุ่มซ้ำทีละวันของสัญลักษณ์ "
    "เพราะเหตุการณ์ SEMI-SYN ทุก seed และทุก family ถูกฉีดลงในวันเดียวกัน; "
    f"{UNIT_LEGACY} = ผล SEMI-SYN รุ่นเก่าที่ไม่มีวันกำกับ ช่วงความเชื่อมั่นจึงแคบเกินจริง",
    "recall และช่วงความเชื่อมั่นใช้ตัวประมาณแบบอัตราส่วนเดียวกัน: รวมจำนวนที่ตรวจพบ / จำนวนทั้งหมด "
    "ของทุก (seed, เหตุการณ์) และสุ่มซ้ำทีละกลุ่ม (cluster)",
]


def _budget_key(b: float) -> str:
    return f"{float(b):g}"


def _finite_mean(values: list[Any]) -> float | None:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else None


def _bootstrap_unit(records: list[dict]) -> str:
    if not any(r["provenance"] == "SEMI-SYN" for r in records):
        return UNIT_EVENT
    if all("day" in pe for r in records for pe in r["per_event"].values()):
        return UNIT_SYMBOL_DAY
    # older SEMI-SYN records without `day`: events can only be resampled one by one
    return UNIT_LEGACY


def cluster_counts(records: list[dict], key: str, unit: str) -> tuple[np.ndarray, np.ndarray]:
    """`(hits, n)` per cluster, summed over every `(seed, event)` value of a cell.

    `unit == "symbol-day"` clusters by `(symbol, day)` of the event; otherwise by label_id.
    Clusters are returned in sorted key order.
    """
    hits: dict[tuple, float] = defaultdict(float)
    n: dict[tuple, int] = defaultdict(int)
    for r in records:
        for lid, pe in r["per_event"].items():
            k = (pe["symbol"], pe["day"]) if unit == UNIT_SYMBOL_DAY else (lid,)
            hits[k] += float(pe[key])
            n[k] += 1
    keys = sorted(n)
    return (
        np.array([hits[k] for k in keys], dtype=np.float64),
        np.array([n[k] for k in keys], dtype=np.float64),
    )


def _cell(records: list[dict], n_boot: int) -> dict[str, Any]:
    ok = [r for r in records if r.get("metrics") is not None]
    out: dict[str, Any] = {
        "provenance": "/".join(sorted({r["provenance"] for r in records})),
        "n_seeds": len({r["seed"] for r in records}),
        "seeds": sorted({r["seed"] for r in records}),
        "n_runs": len(records),
        "tau_saturated_any": any(bool(r.get("tau_saturated")) for r in ok),
        "deterministic": all(bool(r.get("deterministic")) for r in records),
        "errors": sorted({r["error"] for r in records if r.get("error")}),
        "note": next((r["note"] for r in records if r.get("note")), None),
        "n_events": max((r.get("n_events") or 0 for r in records), default=0),
        "n_symbol_days": max((r.get("n_symbol_days") or 0 for r in records), default=0),
        "classifies": all(r.get("classifies", True) for r in records),
        "ci_unit": _bootstrap_unit(records),
        "mean": None,
        "ci": None,
    }
    if ok:
        out["mean"] = {m: _finite_mean([r["metrics"].get(m) for r in ok]) for m in MEAN_METRICS}
        ci = {}
        for metric, key in CI_METRICS.items():
            # point estimate and CI from the same cluster ratio estimator
            hits, n = cluster_counts(ok, key, out["ci_unit"])
            mean, (lo, hi) = ratio_bootstrap_ci(hits, n, n_boot, CI_SEED)
            out["mean"][metric] = mean if math.isfinite(mean) else None
            ci[metric] = [lo, hi] if math.isfinite(lo) else None
        out["ci"] = ci
        if not out["classifies"]:  # always MM-T000: recall_correct (and F1) carry no information
            out["mean"]["recall_correct"] = None
            out["mean"]["f1"] = None
            ci["recall_correct"] = None
    return out


def _labels_summary(runs: list[dict]) -> list[dict[str, Any]]:
    """Per distinct split: labels considered / eligible as events / dropped by reason, plus the
    symbol-days lost at window loading (`n_skipped`, max per set and reason over its runs, so a
    set loaded by many variants and budgets is counted once)."""
    seen: dict[str, dict[str, Any]] = {}
    for r in runs:
        sd = r.get("split_def") or {}
        if "n_labels" not in sd:
            continue
        sha = sd.get("sha12")
        if sha not in seen:
            by_period = sd.get("labels_dropped") or {}
            reasons = {k: sum(v.get(k, 0) for v in by_period.values()) for k in DROP_REASONS}
            n = int(sd["n_labels"])
            seen[sha] = {
                "split_sha12": sha,
                "n_labels": n,
                "eligible": n - sum(reasons.values()),
                "dropped": reasons,
                "dropped_by_period": by_period,
                "skipped_symbol_days": {},
            }
        skipped = seen[sha]["skipped_symbol_days"]
        for set_name, by_reason in (r.get("n_skipped") or {}).items():
            for reason, k in by_reason.items():
                key = f"{set_name}:{reason}"
                skipped[key] = max(skipped.get(key, 0), int(k))
    for v in seen.values():
        v["skipped_symbol_days"] = {k: n for k, n in sorted(v["skipped_symbol_days"].items()) if n}
    return [seen[k] for k in sorted(seen)]


def _paired(runs: list[dict], n_boot: int, min_effect: float) -> dict[str, Any]:
    """AE vs each baseline on the real_t001 events (baselines experiment, budget 2)."""
    arms: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in runs:
        if (
            r["experiment"] == "baselines"
            and r.get("metrics") is not None
            and float(r["budget"]) == PAIRED_BUDGET
        ):
            arms[r["variant"]][r["seed"]] = r["per_event"]
    rows, notes = [], []
    if AE not in arms:
        notes.append("ข้าม: ไม่มีผล baselines ของ ai.autoencoder_trade ที่งบ 2")
    for base in (b for b in BASELINES if b != AE):
        if AE not in arms:
            break
        seeds = sorted(set(arms[AE]) & set(arms.get(base, {})))
        if not seeds:
            notes.append(f"ข้าม {base}: ไม่มี seed ที่มีผลทั้งสองฝั่ง")
            continue
        events = sorted(set.intersection(*[set(arms[k][s]) for k in (AE, base) for s in seeds]))
        if not events:
            notes.append(f"ข้าม {base}: ไม่มีเหตุการณ์ร่วมกัน")
            continue

        def mat(k: str, seeds: list[int] = seeds, events: list[str] = events) -> np.ndarray:
            return np.array([[float(arms[k][s][e]["detected_any"]) for e in events] for s in seeds])

        diff, (lo, hi) = paired_bootstrap(mat(AE), mat(base), n_boot, CI_SEED)
        rows.append(
            {
                "baseline": base,
                "n_seeds": len(seeds),
                "n_events": len(events),
                "diff_mean": diff,
                "ci": [lo, hi],
                "significant": significant(diff, (lo, hi), min_effect),
                "provenance": "REAL",
            }
        )
    return {"budget": PAIRED_BUDGET, "metric": "recall_any", "rows": rows, "notes": notes}


def _fmt(x: float | None, nd: int = 3) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def _fmt_ci(mean: float | None, ci: list[float] | None) -> str:
    s = _fmt(mean)
    return f"{s} [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else s


def _row_md(variant: str, budget: str, c: dict[str, Any]) -> str:
    flags = []
    if c["tau_saturated_any"]:
        flags.append(SATURATED_MARK)
    if c["deterministic"]:
        flags.append(DETERMINISTIC_NOTE)
    if c["note"]:
        flags.append(c["note"])
    if not c["classifies"]:
        flags.append(AGNOSTIC_NOTE)
    flags += [f"error: {e.splitlines()[0][:160]}" for e in c["errors"]]
    m, ci = c["mean"], c["ci"] or {}
    if m is None:
        cells = ["–"] * 8
    else:
        cells = [
            _fmt_ci(m["recall_any"], ci.get("recall_any")),
            _fmt_ci(m["recall_correct"], ci.get("recall_correct")),
            c["ci_unit"],
            f"{_fmt(m['precision'])} (prevalence {_fmt(m['prevalence'], 4)})",
            _fmt(m["fa_per_symbol_day"]),
            _fmt(m["fa_per_symbol_day_hard"]),
            f"{_fmt(m['ttd_median_s'], 0)} / {_fmt(m['ttd_p90_s'], 0)}",
            _fmt(m["pr_auc"]),
        ]
    return (
        f"| {variant} | {budget} | {c['provenance']} | {c['n_seeds']} | {c['n_events']} | "
        + " | ".join(cells)
        + f" | {'; '.join(flags)} |"
    )


def _uniq(runs: list[dict], key: str) -> list[str]:
    return sorted({str(r.get(key)) for r in runs})


def build_report(
    research_db: str | Path | Store,
    out_md: str | Path,
    out_json: str | Path,
    *,
    cfg: Settings | None = None,
) -> dict[str, Any]:
    """Write the Markdown and JSON report from every run record; returns the JSON content.

    `research_db` is the research DuckDB path (opened read-only, no side effects) or an open
    `Store`. `cfg` supplies `n_boot` and `min_effect` (default: `load_settings()`).
    """
    cfg = cfg or load_settings()
    if isinstance(research_db, Store):
        runs = research_db.list_runs()
    else:
        con = duckdb.connect(str(Path(research_db)), read_only=True)
        try:  # same rows and order as Store.list_runs
            sql = "SELECT record_json FROM runs ORDER BY created_us, run_id"
            rows = con.execute(sql).fetchall()
        finally:
            con.close()
        runs = [json.loads(r[0]) for r in rows]
    n_boot = cfg.eval.n_boot

    grouped: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in runs:
        grouped[r["experiment"]][r["variant"]][_budget_key(r["budget"])].append(r)
    order = [e for e in EXPERIMENTS if e in grouped] + sorted(set(grouped) - set(EXPERIMENTS))

    sig = _uniq(runs, "signatures_sha")
    header = f"runs: {len(runs)} | signatures_sha: {', '.join(sig) or '–'}"
    experiments: dict[str, Any] = {}
    for exp in order:
        experiments[exp] = {
            v: {
                b: _cell(recs, n_boot)
                for b, recs in sorted(by_b.items(), key=lambda kv: float(kv[0]))
            }
            for v, by_b in sorted(grouped[exp].items())
        }
    paired = _paired(runs, n_boot, cfg.eval.min_effect)
    data = {
        "header": header,
        "runs": len(runs),
        "signatures_sha": sig,
        "git_sha": _uniq(runs, "git_sha"),
        "config_hash": _uniq(runs, "config_hash"),
        "experiments": experiments,
        "paired_bootstrap": paired,
        "labels": _labels_summary(runs),
        "notes": NOTES_TH,
    }

    lines = [
        "# รายงานผลการประเมิน Market IDS (เพื่อการศึกษา)",
        "",
        f"runs: {len(runs)}",
        "",
        f"signatures_sha: {', '.join(sig) or '–'}",
        "",
        f"git_sha: {', '.join(data['git_sha']) or '–'} · "
        f"config_hash: {', '.join(data['config_hash']) or '–'}",
        "",
        *[
            f"labels (split {lab['split_sha12']}): eligible {lab['eligible']} of "
            f"{lab['n_labels']}; dropped: "
            + ", ".join(f"{k} {v}" for k, v in lab["dropped"].items())
            + "; symbol-days skipped at window loading: "
            + (", ".join(f"{k} {v}" for k, v in lab["skipped_symbol_days"].items()) or "none")
            for lab in data["labels"]
        ],
        "",
        "ทุกตัวเลขมาจากตาราง runs ทั้งหมด; ค่าเฉลี่ยข้าม seed และช่วงความเชื่อมั่น 95 % "
        "จากการ bootstrap (หน่วยตามคอลัมน์ bootstrap unit)",
    ]
    cols = (
        "| variant | budget | provenance | seeds | events | recall_any [95% CI] | "
        "recall_correct [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day | "
        "FA/symbol-day hard | TTD median / p90 (s) | PR-AUC | notes |"
    )
    for exp in order:
        lines += ["", f"## {exp}", "", cols, "|" + "---|" * 14]
        for v, by_b in experiments[exp].items():
            lines += [_row_md(v, b, c) for b, c in by_b.items()]
    lines += [
        "",
        f"## AE vs baselines (paired bootstrap, budget {PAIRED_BUDGET:g}, real_t001 events)",
        "",
        "| baseline | provenance | seeds | events | diff recall_any (AE − baseline) | 95% CI | "
        "ผล |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in paired["rows"]:
        verdict = "AE ดีกว่าอย่างมีนัย" if row["significant"] else "ไม่มีนัยสำคัญ"
        lines.append(
            f"| {row['baseline']} | {row['provenance']} | {row['n_seeds']} | {row['n_events']} | "
            f"{_fmt(row['diff_mean'])} | [{_fmt(row['ci'][0])}, {_fmt(row['ci'][1])}] | "
            f"{verdict} |"
        )
    if paired["notes"]:
        lines += ["", *[f"- {n}" for n in paired["notes"]]]
    lines += [
        "",
        f"เกณฑ์: CI ล่าง > 0 และผลต่าง ≥ {cfg.eval.min_effect:g}",
        "",
        "## หมายเหตุ",
        "",
    ]
    lines += [f"- {n}" for n in NOTES_TH]

    out_md, out_json = Path(out_md), Path(out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps(safe := _json_safe(data), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return safe


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_json_safe(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x
