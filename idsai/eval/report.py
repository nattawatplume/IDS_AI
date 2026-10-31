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

Seed spread: `std` of a cell is the sample standard deviation (ddof 1) over seeds of the recall
of each seed, None with fewer than two seeds. A cell where recall is 0 or 1 has a bootstrap CI
of one point, so `rule_of_three` adds the bound `3 / n` over the same independent clusters.

Retrain decision: `retrain_drift` records at budget 2 on the accept split give the
champion/challenger verdict (`idsai.ai.learning.compare`, a paired event bootstrap): full stack
`v1` vs `v2` (what would be deployed), and the AE alone `v1:ai` vs `v2:ai` as a diagnostic.
The retrain and AE-vs-baselines blocks each use only the records of the most recent
`config_hash` feeding them, with a note counting the older-config records ignored.
Its `series` block (recall_any per version, same ratio estimator as the tables) feeds the
dashboard's before/after chart.
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from idsai.ai.learning import compare
from idsai.config import Settings, load_settings
from idsai.data.store import Store
from idsai.eval.experiments import AE, BASELINES, EXPERIMENTS
from idsai.eval.metrics import (
    MIN_EVENTS,
    few_events_th,
    paired_bootstrap,
    ratio_bootstrap_ci,
    significant,
)

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
MIXED_CONFIG_NOTE = (
    "ตาราง runs รวมผลจากหลาย config_hash แต่การตัดสิน champion/challenger ใช้เฉพาะ "
    "config_hash ล่าสุด ค่าในตาราง retrain_drift จึงอาจไม่ตรงกับกราฟและบล็อกการตัดสิน"
)
NOTES_TH = [
    "หน้าต่างเวลาที่ทับซ้อนกันมี autocorrelation สูง จึงทำให้การรับประกันของ conformal p-value "
    "อ่อนลง: อัตราการแจ้งเตือนจริงอาจสูงกว่างบที่ตั้งไว้",
    "เวลาตรวจจับ (TTD, latency) ไม่รวมเวลารอปิดแท่ง 1 วินาทีและเวลารอ stride 10 วินาที",
    "precision แสดงคู่กับ prevalence (สัดส่วนหน้าต่างที่เป็นบวก) เสมอ; F1 เก็บไว้ใน JSON แต่ไม่ใช่ตัวชี้วัดหลัก",
    "คะแนนรวม (fused) คือ anomaly p-value หรือ “ความหายากเทียบตลาดปกติ” ไม่ใช่ความน่าจะเป็นว่ามีการปั่นราคา",
    "ทุกการแจ้งเตือนคือความผิดปกติทางสถิติ ไม่ได้กล่าวหาบุคคลใด",
    f"{SATURATED_MARK}: งบการแจ้งเตือนสูงกว่าอัตราสูงสุดที่ข้อมูลสอบเทียบทำได้ τ จึงถูกตั้งที่ค่า F บวกที่เล็กที่สุด "
    "หรือเหนือค่า F ทั้งหมดเมื่อไม่มีค่า F บวกเลย (เช่น กฎที่ไม่เคยทำงานในช่วงสอบเทียบ) "
    "งบจึงไม่ได้ถูกทดสอบกับข้อมูลจริง",
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
    "rule of three: เมื่อตรวจพบ 0 ครั้งหรือครบทุกครั้ง bootstrap CI จะแคบเหลือจุดเดียวทั้งที่ข้อมูลมีน้อย "
    "จึงแสดงขอบเขตโดยประมาณเพิ่ม: ขอบบน 3/n เมื่อตรวจไม่พบเลย หรือขอบล่าง 1 − 3/n เมื่อตรวจพบครบ "
    "โดย n คือจำนวนกลุ่มอิสระตามคอลัมน์ bootstrap unit (ไม่นับซ้ำข้าม seed)",
    "± std คือส่วนเบี่ยงเบนมาตรฐานของ recall ข้าม seed (– เมื่อมี seed เดียว)",
    f"การเปรียบเทียบสองฝั่งต้องมีอย่างน้อย {MIN_EVENTS} เหตุการณ์ ถ้าน้อยกว่านั้นจะไม่สรุปว่าดีกว่า",
    "TTD ติดลบหมายความว่าการแจ้งเตือนเริ่มก่อนเวลาเริ่มของป้ายกำกับ (label)",
    "งบการแจ้งเตือนเป็นเพดาน ไม่ใช่เป้าหมาย: อัตรา FA ที่วัดได้จึงอาจต่ำกว่างบที่ตั้งไว้",
    "recall_correct นับเหตุการณ์ที่มีการแจ้งเตือนทับช่วงเหตุการณ์และระบุเทคนิคตรงกับป้ายกำกับ: "
    "เทคนิคของการแจ้งเตือนมาจากแถวที่จัดประเภทได้และมีค่า F สูงสุดของ episode "
    "ส่วนหลักฐานมาจากแถวจุดสูงสุดของ episode ซึ่งอาจเป็นคนละแถวกัน",
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


def _rule_of_three(hits: np.ndarray, n: np.ndarray, unit: str) -> dict[str, Any] | None:
    """Bound on a recall of 0 or 1 from `n` independent clusters; None for anything between."""
    k = int(n.size)
    if k == 0 or 0 < hits.sum() < n.sum():
        return None
    margin = min(3.0 / k, 1.0)
    zero = hits.sum() == 0
    return {
        "side": "upper" if zero else "lower",
        "bound": margin if zero else 1.0 - margin,
        "hits": 0 if zero else k,
        "n": k,
        "unit": UNIT_SYMBOL_DAY if unit == UNIT_SYMBOL_DAY else UNIT_EVENT,
    }


def _seed_std(records: list[dict], key: str) -> float | None:
    """Sample std over seeds of each seed's recall (its `(seed, event)` values pooled)."""
    by_seed: dict[int, list[float]] = defaultdict(list)
    for r in records:
        by_seed[r["seed"]] += [float(pe[key]) for pe in r["per_event"].values()]
    recalls = [float(np.mean(v)) for v in by_seed.values() if v]
    return float(np.std(recalls, ddof=1)) if len(recalls) > 1 else None


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
        "std": None,
        "rule_of_three": None,
    }
    if ok:
        out["mean"] = {m: _finite_mean([r["metrics"].get(m) for r in ok]) for m in MEAN_METRICS}
        ci, std, r3 = {}, {}, {}
        for metric, key in CI_METRICS.items():
            # point estimate and CI from the same cluster ratio estimator
            hits, n = cluster_counts(ok, key, out["ci_unit"])
            mean, (lo, hi) = ratio_bootstrap_ci(hits, n, n_boot, CI_SEED)
            out["mean"][metric] = mean if math.isfinite(mean) else None
            ci[metric] = [lo, hi] if math.isfinite(lo) else None
            std[metric] = _seed_std(ok, key)
            r3[metric] = _rule_of_three(hits, n, out["ci_unit"])
        out["ci"], out["std"], out["rule_of_three"] = ci, std, r3
        if not out["classifies"]:  # always MM-T000: recall_correct (and F1) carry no information
            out["mean"]["recall_correct"] = None
            out["mean"]["f1"] = None
            ci["recall_correct"] = None
            std["recall_correct"] = None
            r3["recall_correct"] = None
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


def _latest_config(recs: list[dict]) -> tuple[list[dict], str | None]:
    """Keep only the records of the most recent `config_hash` among `recs`.

    `recs` arrive in created order (`Store.list_runs`: `ORDER BY created_us, run_id`), so the
    last record's config is the most recent. Returns the kept records and a Thai note naming
    how many older-config records were ignored (None when nothing was ignored).
    """
    if not recs:
        return recs, None
    latest = recs[-1].get("config_hash")
    kept = [r for r in recs if r.get("config_hash") == latest]
    n_old = len(recs) - len(kept)
    if not n_old:
        return kept, None
    return kept, f"ไม่นับ {n_old} run ของ config_hash เก่า (ใช้เฉพาะ config_hash ล่าสุด {latest})"


def _paired(runs: list[dict], n_boot: int, min_effect: float) -> dict[str, Any]:
    """AE vs each baseline on the real_t001 events (baselines experiment, budget 2).

    Only records of the most recent `config_hash` feeding this block are used.
    """
    recs, config_note = _latest_config(
        [r for r in runs if r["experiment"] == "baselines" and float(r["budget"]) == PAIRED_BUDGET]
    )
    arms: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in recs:
        if r.get("metrics") is not None:
            arms[r["variant"]][r["seed"]] = r["per_event"]
    rows: list[dict[str, Any]] = []
    notes = [config_note] if config_note else []
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
        enough = len(events) >= MIN_EVENTS
        rows.append(
            {
                "baseline": base,
                "n_seeds": len(seeds),
                "n_events": len(events),
                "diff_mean": diff,
                "ci": [lo, hi],
                "significant": enough and significant(diff, (lo, hi), min_effect),
                "reason_th": None if enough else few_events_th(len(events)),
                "provenance": "REAL",
            }
        )
    return {"budget": PAIRED_BUDGET, "metric": "recall_any", "rows": rows, "notes": notes}


RETRAIN = "retrain_drift"
RETRAIN_PAIRS = {"primary": ("v1", "v2"), "ai_only": ("v1:ai", "v2:ai")}
AI_ONLY_LABEL = "AE alone (diagnostic)"
NO_RETRAIN_TH = "ยังไม่มีผล retrain_drift"


def _retrain_shell(champion: str, challenger: str, provenance: str | None) -> dict[str, Any]:
    """A retrain decision block with nothing decided yet (filled by `_retrain_block`)."""
    return {
        "champion": champion,
        "challenger": challenger,
        "budget": PAIRED_BUDGET,
        "metric": "recall_any",
        "split": "accept",
        "n_seeds": 0,
        "seeds": [],
        "n_events": 0,
        "series": None,
        "diff_mean": None,
        "ci": None,
        "accept": None,
        "reason_th": "",
        "provenance": provenance,
        "config_hash": None,
        "notes": [],
    }


def _retrain_block(
    runs: list[dict], champion: str, challenger: str, n_boot: int, min_effect: float
) -> dict[str, Any] | None:
    """Champion vs challenger on the accept split (recall_any, budget 2); None without records.

    Arrays `(n_seeds, n_events)` of per-event `detected_any` over the seeds where both arms have
    an OK record (`metrics` not null) and the events every such record shares. The decision is
    `compare` (paired event bootstrap); the series uses the cluster ratio estimator of the
    report tables (unit = event) on the same records and events. Only records of the most
    recent `config_hash` feeding this block are used, while the tables pool every config_hash,
    so the series equals the table cell only when no seed or event is dropped and the runs
    share one config_hash (the report says so when they do not: `MIXED_CONFIG_NOTE`).
    """
    recs, config_note = _latest_config(
        [
            r
            for r in runs
            if r["experiment"] == RETRAIN
            and r["variant"] in (champion, challenger)
            and float(r["budget"]) == PAIRED_BUDGET
        ]
    )
    if not recs:
        return None
    block = _retrain_shell(champion, challenger, "/".join(sorted({r["provenance"] for r in recs})))
    block["config_hash"] = recs[-1].get("config_hash")
    notes = block["notes"]
    if config_note:
        notes.append(config_note)
    arms: dict[str, dict[int, dict]] = {champion: {}, challenger: {}}
    for r in recs:
        if r.get("metrics") is not None:
            arms[r["variant"]][r["seed"]] = r
    n_null = sum(r.get("metrics") is None for r in recs)
    seeds = sorted(set(arms[champion]) & set(arms[challenger]))
    dropped = sorted((set(arms[champion]) | set(arms[challenger])) - set(seeds))
    if n_null:
        notes.append(f"ไม่นับ {n_null} run ที่ metrics เป็น null (เช่น CalibrationError)")
    if dropped:
        notes.append(
            f"ตัด seed {', '.join(map(str, dropped))}: ไม่มีผลครบทั้ง {champion} และ {challenger}"
        )
    if not seeds:
        block["reason_th"] = f"ยังตัดสินไม่ได้: ไม่มี seed ที่มีผล {champion} และ {challenger} ครบทั้งสองฝั่ง"
        return block
    ok = [arms[k][s] for k in (champion, challenger) for s in seeds]
    events = sorted(set.intersection(*[set(r["per_event"]) for r in ok]))
    n_union = len(set().union(*[set(r["per_event"]) for r in ok]))
    if n_union > len(events):
        notes.append(f"ใช้เฉพาะ {len(events)} จาก {n_union} เหตุการณ์ที่ทุก run มีร่วมกัน")

    def mat(k: str) -> np.ndarray:
        return np.array(
            [[arms[k][s]["per_event"][e]["detected_any"] for e in events] for s in seeds],
            dtype=bool,
        ).reshape(len(seeds), len(events))

    d = compare(mat(champion), mat(challenger), n_boot, CI_SEED, min_effect)
    series = {}
    for k in (champion, challenger):
        aligned = [{"per_event": {e: arms[k][s]["per_event"][e] for e in events}} for s in seeds]
        hits, n = cluster_counts(aligned, "detected_any", UNIT_EVENT)
        mean, (lo, hi) = ratio_bootstrap_ci(hits, n, n_boot, CI_SEED)
        finite = math.isfinite(mean)
        series[k] = {"mean": mean if finite else None, "ci": [lo, hi] if finite else None}
    block.update(
        seeds=seeds,
        n_seeds=len(seeds),
        n_events=len(events),
        series=series,
        diff_mean=d.diff_mean,
        ci=list(d.ci),
        # `compare` rejects below MIN_EVENTS; the report says that nothing was decided
        accept=None if len(events) < MIN_EVENTS else d.accept,
        reason_th=d.reason_th,
    )
    return block


def _retrain_decision(runs: list[dict], n_boot: int, min_effect: float) -> dict[str, Any] | None:
    """Primary decision (full stack v1 vs v2) plus the AE-only diagnostic under `ai_only`.

    None without any retrain_drift record; a missing pair is None (`ai_only`) or an undecided
    block with a Thai reason (primary, so the JSON shape stays stable).
    """
    if not any(r["experiment"] == RETRAIN for r in runs):
        return None
    pairs = RETRAIN_PAIRS
    primary = _retrain_block(runs, *pairs["primary"], n_boot, min_effect)
    ai_only = _retrain_block(runs, *pairs["ai_only"], n_boot, min_effect)
    if primary is None:
        primary = _retrain_shell(*pairs["primary"], None)
        primary["reason_th"] = f"{NO_RETRAIN_TH} ของ full stack ที่งบ {PAIRED_BUDGET:g}"
    elif ai_only is not None and ai_only["config_hash"] != primary["config_hash"]:
        note = (
            f"config_hash ไม่ตรงกัน: full stack ใช้ {primary['config_hash']} แต่ "
            f"{AI_ONLY_LABEL} ใช้ {ai_only['config_hash']} จึงเทียบสองบล็อกนี้กันตรง ๆ ไม่ได้"
        )
        primary["notes"].append(note)
        ai_only["notes"].append(note)
    return {**primary, "ai_only": ai_only}


def _retrain_row_md(label: str, b: dict[str, Any]) -> str:
    series = b["series"] or {}

    def recall(k: str) -> str:
        s = series.get(k)
        return _fmt_ci(s["mean"], s["ci"]) if s else "–"

    ci = f"[{_fmt(b['ci'][0])}, {_fmt(b['ci'][1])}]" if b["ci"] else "–"
    verdict = {True: "ยอมรับ", False: "ไม่ยอมรับ", None: "ยังตัดสินไม่ได้"}[b["accept"]]
    return (
        f"| {label} | {b['champion']} → {b['challenger']} | {b['provenance'] or '–'} | "
        f"{b['n_seeds']} | {b['n_events']} | {recall(b['champion'])} | "
        f"{recall(b['challenger'])} | {_fmt(b['diff_mean'])} | {ci} | {verdict} |"
    )


def _retrain_md(rd: dict[str, Any] | None, min_effect: float) -> list[str]:
    lines = [
        "",
        f"## retrain_drift: champion v1 vs challenger v2 (paired bootstrap, budget "
        f"{PAIRED_BUDGET:g}, accept split)",
        "",
    ]
    if rd is None:
        return [*lines, NO_RETRAIN_TH]
    lines += [
        "| comparison | champion → challenger | provenance | seeds | events | "
        "recall_any champion [95% CI] | recall_any challenger [95% CI] | "
        "diff (challenger − champion) | 95% CI | ผล |",
        "|" + "---|" * 10,
    ]
    blocks = [("full stack (deployed)", rd)]
    if rd["ai_only"] is not None:
        blocks.append((AI_ONLY_LABEL, rd["ai_only"]))
    lines += [_retrain_row_md(label, b) for label, b in blocks]
    lines.append("")
    for label, b in blocks:
        lines.append(f"- {label}: {b['reason_th']}")
        lines += [f"  - {n}" for n in b["notes"]]
    if rd["ai_only"] is None:
        lines.append(f"- {AI_ONLY_LABEL}: {NO_RETRAIN_TH}")
    lines += [
        "",
        f"challenger แทน champion เมื่อ CI ล่างของผลต่าง > 0 และผลต่าง ≥ {min_effect:g} "
        "(bootstrap แบบจับคู่ข้ามเหตุการณ์ ใช้ดัชนีสุ่มซ้ำเดียวกันทั้งสองฝั่ง)",
    ]
    return lines


def _error_th(error: str, limit: int = 300) -> str:
    """The Thai sentence of a bilingual error (before its English part), whole when it fits;
    a cut is marked with "…" (the JSON keeps the full text)."""
    text = error.splitlines()[0] if error else ""
    thai = re.split(r"(?<=\.) (?=[A-Z])", text, maxsplit=1)[0]
    return thai if len(thai) <= limit else thai[: limit - 1] + "…"


def _fmt(x: float | None, nd: int = 3) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def _fmt_ci(mean: float | None, ci: list[float] | None) -> str:
    s = _fmt(mean)
    return f"{s} [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else s


def _fmt_recall(mean: float | None, std: float | None, ci: list[float] | None) -> str:
    if mean is None:
        return "–"
    s = f"{_fmt(mean)} ± {_fmt(std)}"
    return f"{s} [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else s


def _rule_of_three_th(metric: str, r: dict[str, Any]) -> str:
    unit = "เหตุการณ์" if r["unit"] == UNIT_EVENT else r["unit"]
    reach = "สูงถึง" if r["side"] == "upper" else "ต่ำถึง"
    return (
        f"{metric} {r['hits']} จาก {r['n']} {unit}: ช่วงจริงอาจ{reach}ราว "
        f"{r['bound'] * 100:.0f}% (rule of three)"
    )


def _mixed_notes(runs: list[dict]) -> list[str]:
    """Thai warnings naming what differs across the runs of one report."""
    configs = {r.get("config_hash") for r in runs} - {None}
    notes = [MIXED_CONFIG_NOTE] if len(configs) > 1 else []
    gits = sorted({r["git_sha"] for r in runs if r.get("git_sha")})
    if len(gits) > 1:
        notes.append(
            f"ตาราง runs รวมผลจากหลาย git_sha ({', '.join(gits)}) ผลอาจมาจากโค้ดคนละรุ่น "
            "(รุ่น -dirty นับเป็นคนละรุ่น)"
        )
    splits = sorted(
        {(r.get("split_def") or {}).get("sha12") for r in runs} - {None},
    )
    if len(splits) > 1:
        notes.append(f"ตาราง runs รวมผลจากหลาย split ({', '.join(splits)}) ผลอาจมาจากชุดข้อมูลคนละชุด")
    return notes


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
    flags += [f"error: {_error_th(e)}" for e in c["errors"]]
    flags += [_rule_of_three_th(metric, r) for metric, r in (c["rule_of_three"] or {}).items() if r]
    m, ci, std = c["mean"], c["ci"] or {}, c["std"] or {}
    if m is None:
        cells = ["–"] * 8
    else:
        cells = [
            _fmt_recall(m["recall_any"], std.get("recall_any"), ci.get("recall_any")),
            _fmt_recall(m["recall_correct"], std.get("recall_correct"), ci.get("recall_correct")),
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
    retrain = _retrain_decision(runs, n_boot, cfg.eval.min_effect)
    data = {
        "header": header,
        "runs": len(runs),
        "signatures_sha": sig,
        "git_sha": _uniq(runs, "git_sha"),
        "config_hash": _uniq(runs, "config_hash"),
        "experiments": experiments,
        "paired_bootstrap": paired,
        "retrain_decision": retrain,
        "labels": _labels_summary(runs),
        "notes": NOTES_TH,
    }
    mixed = _mixed_notes(runs)
    data["notes"] = [*mixed, *NOTES_TH]

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
        *[line for n in mixed for line in ("", f"**คำเตือน:** {n}")],
    ]
    cols = (
        "| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | "
        "recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day | "
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
        verdict = row["reason_th"] or ("AE ดีกว่าอย่างมีนัย" if row["significant"] else "ไม่มีนัยสำคัญ")
        lines.append(
            f"| {row['baseline']} | {row['provenance']} | {row['n_seeds']} | {row['n_events']} | "
            f"{_fmt(row['diff_mean'])} | [{_fmt(row['ci'][0])}, {_fmt(row['ci'][1])}] | "
            f"{verdict} |"
        )
    if paired["notes"]:
        lines += ["", *[f"- {n}" for n in paired["notes"]]]
    lines += ["", f"เกณฑ์: CI ล่าง > 0 และผลต่าง ≥ {cfg.eval.min_effect:g}"]
    lines += _retrain_md(retrain, cfg.eval.min_effect)
    lines += [
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
