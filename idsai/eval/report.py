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
Its `series` block (recall_any per version, same ratio estimator as the tables) feeds the
dashboard's before/after chart.

Ablation: `paired_ablation` (budget 2) compares `all` with each subset on REAL (paired event
bootstrap) and SEMI-SYN (per-seed pairing, symbol-day clusters resampled with the same indices
for both arms, `paired_ratio_bootstrap`).

The retrain, AE-vs-baselines and ablation blocks each use only the records of the most recent
`config_hash` feeding them, with a note counting the older-config records ignored.

False alerts: when every record of a cell carries `fa_by_day` (per symbol-day unmatched-episode
counts, zero days included), FA and FA-hard get a 95 % CI: per symbol-day the mean over the
cell's records (seeds), then symbol-days resampled with replacement (`bootstrap_ci`). Older
records without it show the mean alone. When a record also carries `<which>_mult` (how many
parts loaded each day, e.g. the three SEMI-SYN families of one ablation arm; missing keys = 1),
the CI is the symbol-day ratio bootstrap of `sum n / sum m` (`ratio_bootstrap_ci`); without it a
day summed over three parts read as three times its rate.

M1 (`m1_fusion`, budget 2): one table per split (accept, test, semisyn) of every AI fusion
variant, with the paired recall_any difference against V0 `all`. The variant is chosen on the
accept split only, by a rule fixed in advance (`m1_select`): among the variants whose accept
REAL recall_any is not significantly below V0 (paired event bootstrap, `significant` with
`min_effect`), the lowest accept-hard FA; ties go to the simpler one, `all` < `no_ai` <
`ai_shift1` < `ai_shift2` < `ai_mondrian` (V0 < V1 < V3 < V2). Test numbers are only reported.

M4 (`m4_sell`, budget 2): one table of the `base` (no sell-side rules) and `sell` variants on
the REAL test split: recall per phase with a 95 % CI (pump phase = recall_any on the label's onset
window; sell phase = an episode of any technique with an exceedance row in
`(t_peak + 60 s, t_peak + dump_tau]`; "correct" = that episode names the sell phase), FA and
FA-hard with their CI, and the `bear_raid` SEMI-SYN recall of the same variant. The paired
differences `sell - base` of pump-phase, sell-phase and sell-phase-correct recall use the
ablation's paired bootstrap (`_ablation_pair`); the sell rules must not lower pump recall.
Sell-phase CIs use only the events with a sell window (one cluster per REAL event).

Technique confusion (`confusion`, budget 2, `real_t001 full` and every `semisyn` family): per
label family, the share of events whose matched episodes carry each parent technique (or none),
averaged over seeds; an event with several techniques counts once in each column. Needs
`per_event[*]["techniques"]` (records written after Part C).
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
    UNCLASSIFIED,
    _parent,
    bootstrap_ci,
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
    "FA/symbol-day [95% CI]: เมื่อผลมีจำนวนการแจ้งเตือนเท็จรายวัน (fa_by_day) จะเฉลี่ยข้าม seed "
    "ในแต่ละ symbol-day แล้ว bootstrap สุ่มซ้ำทีละ symbol-day (นับวันที่ไม่มีการแจ้งเตือนด้วย); "
    "ผลรุ่นเก่าที่ไม่มีข้อมูลนี้แสดงเฉพาะค่าเฉลี่ย; (n วัน) คือจำนวน symbol-day ที่ใช้นับ",
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
    _, hits, n = _cluster_counts_keyed(records, key, unit)
    return hits, n


def _cluster_counts_keyed(
    records: list[dict], key: str, unit: str
) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """`cluster_counts` plus the sorted cluster keys (`(symbol, day)` or `(label_id,)`)."""
    hits: dict[tuple, float] = defaultdict(float)
    n: dict[tuple, int] = defaultdict(int)
    for r in records:
        for lid, pe in r["per_event"].items():
            k = (pe["symbol"], pe["day"]) if unit == UNIT_SYMBOL_DAY else (lid,)
            hits[k] += float(pe[key])
            n[k] += 1
    keys = sorted(n)
    return (
        keys,
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
        "fa_ci": None,
        "n_fa_days": None,
        "n_hard_days": None,
    }
    if ok:
        out["fa_ci"] = {m: _fa_ci(ok, which, n_boot) for m, which in FA_DAYS.items()}
        for k in ("n_fa_days", "n_hard_days"):
            vals = [r["metrics"].get(k) for r in ok if r["metrics"].get(k) is not None]
            out[k] = int(max(vals)) if vals else None
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


FA_DAYS = {"fa_per_symbol_day": "normal", "fa_per_symbol_day_hard": "hard"}


def _fa_ci(records: list[dict], which: str, n_boot: int) -> list[float] | None:
    """95 % CI of a cell's FA rate from `fa_by_day[which]` (None unless every record has it).

    Per symbol-day the mean over the records that list it (seeds), then a bootstrap over days.
    """
    days = [(r.get("fa_by_day") or {}).get(which) for r in records]
    if not days or not all(days):
        return None
    mults = [(r.get("fa_by_day") or {}).get(f"{which}_mult") for r in records]
    if any(m is not None for m in mults):
        # several parts per day: the rate is sum(count) / sum(parts) over days (ratio bootstrap)
        n: dict[str, float] = defaultdict(float)
        m: dict[str, float] = defaultdict(float)
        for d, mult in zip(days, mults, strict=True):
            for k, v in d.items():
                n[k] += float(v)
                m[k] += float((mult or {}).get(k, 1))
        keys = sorted(n)
        _, (lo, hi) = ratio_bootstrap_ci(
            np.array([n[k] for k in keys]), np.array([m[k] for k in keys]), n_boot, CI_SEED
        )
        return [lo, hi] if math.isfinite(lo) else None
    per_day: dict[str, list[float]] = defaultdict(list)
    for d in days:
        for k, v in d.items():
            per_day[k].append(float(v))
    means = np.array([np.mean(per_day[k]) for k in sorted(per_day)], dtype=np.float64)
    _, (lo, hi) = bootstrap_ci(means, n_boot, CI_SEED)
    return [lo, hi] if math.isfinite(lo) else None


CONFUSION_SETS = (("real_t001", "full"), ("semisyn", None))  # None: every variant (family)
NONE_COL = "none"
NO_TECHNIQUES_TH = (
    "ยังไม่มีข้อมูลเทคนิคที่ระบุต่อเหตุการณ์ (techniques) ในผลชุดนี้ "
    "ต้องรัน `idsai eval` ใหม่จึงจะสร้างตาราง confusion ได้"
)
CONFUSION_NOTE_TH = (
    "ค่าในตาราง = สัดส่วนเหตุการณ์ของ family นั้น (เฉลี่ยข้าม seed) ที่ episode ที่จับคู่ได้ระบุเทคนิคหลัก "
    f"(parent) ตามคอลัมน์; {UNCLASSIFIED} = ระบุเทคนิคไม่ได้; {NONE_COL} = ตรวจไม่พบ; "
    "เหตุการณ์ที่ถูกระบุหลายเทคนิคนับในทุกคอลัมน์ของเทคนิคเหล่านั้น ผลรวมของแถวจึงอาจเกิน 1"
)


def _confusion(runs: list[dict]) -> dict[str, Any]:
    """Label family x predicted parent technique (budget 2), see the module docstring."""
    out: dict[str, Any] = {}
    for exp, variant in CONFUSION_SETS:
        name = exp if variant is None else f"{exp} {variant}"
        recs = [
            r
            for r in runs
            if r["experiment"] == exp
            and (variant is None or r["variant"] == variant)
            and float(r["budget"]) == PAIRED_BUDGET
            and r.get("metrics") is not None
            and r.get("per_event")
        ]
        usable = [r for r in recs if all("techniques" in pe for pe in r["per_event"].values())]
        block: dict[str, Any] = {"budget": PAIRED_BUDGET, "columns": [], "rows": [], "notes": []}
        if usable and len(usable) < len(recs):
            block["notes"].append(f"ไม่นับ {len(recs) - len(usable)} run ที่ไม่มีข้อมูล techniques")
        # per seed and label family: events, and events per predicted column
        n: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        cnt: dict[int, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        cols: set[str] = set()
        for r in usable:
            for pe in r["per_event"].values():
                fam = pe.get("family") or "?"
                n[r["seed"]][fam] += 1
                parents = sorted({_parent(t) for t in pe["techniques"]}) or [NONE_COL]
                for c in parents:
                    cnt[r["seed"]][fam][c] += 1
                cols.update(parents)
        cols.discard(NONE_COL)
        columns = [*sorted(cols), NONE_COL]
        for fam in sorted({f for by_fam in n.values() for f in by_fam}):
            seeds = sorted(s for s in n if n[s].get(fam))
            block["rows"].append(
                {
                    "family": fam,
                    "n_seeds": len(seeds),
                    "n_events": max(n[s][fam] for s in seeds),
                    "share": {
                        c: float(np.mean([cnt[s][fam].get(c, 0) / n[s][fam] for s in seeds]))
                        for c in columns
                    },
                }
            )
        if block["rows"]:
            block["columns"] = columns
        out[name] = block
    return out


def _confusion_md(conf: dict[str, Any]) -> list[str]:
    lines = ["", f"## confusion: family ของป้ายกำกับ × เทคนิคที่ระบุ (budget {PAIRED_BUDGET:g})"]
    for name, block in conf.items():
        lines += ["", f"### {name}", ""]
        if not block["rows"]:
            lines.append(f"- {NO_TECHNIQUES_TH}")
            continue
        cols = block["columns"]
        lines += [
            "| family | seeds | events | " + " | ".join(cols) + " |",
            "|" + "---|" * (3 + len(cols)),
        ]
        lines += [
            f"| {row['family']} | {row['n_seeds']} | {row['n_events']} | "
            + " | ".join(_fmt(row["share"][c]) for c in cols)
            + " |"
            for row in block["rows"]
        ]
        lines += ["", f"- {CONFUSION_NOTE_TH}", *[f"- {n}" for n in block["notes"]]]
    return lines


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


ABLATION = "ablation"
ABLATION_FULL = "all"
ABLATION_ARMS = ("rules+stats", "rules", "stats", "ai")
ABLATION_SETS = (("real_t001", "REAL"), ("semisyn", "SEMI-SYN"))
NO_PAIRED_ABLATION_TH = (
    f"ยังไม่มี paired bootstrap ของ ablation ที่งบ {PAIRED_BUDGET:g} "
    "จึงไม่สรุปว่าความต่างใดมีนัย (ห้ามใช้ CI ไม่ทับกันแทน)"
)


def paired_ratio_bootstrap(
    hits_a: np.ndarray, hits_b: np.ndarray, n: np.ndarray, n_boot: int, seed: int
) -> tuple[float, tuple[float, float]]:
    """`((sum(hits_a) - sum(hits_b)) / sum(n), 95 % CI)` resampling clusters with replacement.

    Both arms share the clusters and `n` (aligned `(seed, event)` values), and each resample
    uses the same cluster indices for both, so the difference is that of two cluster ratio
    estimators (`ratio_bootstrap_ci`). With one cluster per event and equal `n` this equals
    `paired_bootstrap` draw for draw. No clusters gives NaNs.
    """
    d = np.asarray(hits_a, dtype=np.float64) - np.asarray(hits_b, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    if d.shape != n.shape or d.ndim != 1:
        raise ValueError(f"hits and n must be equal-length 1-D arrays: {d.shape} {n.shape}")
    if n.size == 0 or n.sum() <= 0:
        return math.nan, (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n.size, size=(n_boot, n.size))
    lo, hi = np.percentile(d[idx].sum(axis=1) / n[idx].sum(axis=1), [2.5, 97.5])
    return float(d.sum() / n.sum()), (float(lo), float(hi))


def _ablation_pair(
    full: dict[int, dict],
    other: dict[int, dict],
    n_boot: int,
    min_effect: float,
    key: str = "detected_any",
) -> tuple[dict[str, Any] | None, str | None]:
    """One `all` vs subset row (or None and a Thai reason it was skipped), on per-event `key`
    (default `detected_any`; events whose `key` is None in any record are left out, M4).

    REAL events (label_id) are shared by every seed, so like `_paired` only the events every
    record of both arms has are kept. SEMI-SYN scenarios are drawn per seed (their ids differ
    across seeds), so the events are intersected per seed; the `(seed, event)` values are then
    clustered by `(symbol, day)` as in `_cell`.
    """
    seeds = sorted(set(full) & set(other))
    if not seeds:
        return None, "ไม่มี seed ที่มีผลทั้งสองฝั่ง"
    recs = [arm[s] for arm in (full, other) for s in seeds]
    unit = _bootstrap_unit(recs)
    if unit == UNIT_EVENT:
        common = set.intersection(*[set(r["per_event"]) for r in recs])
        events = {s: sorted(common) for s in seeds}
    else:
        events = {s: sorted(set(full[s]["per_event"]) & set(other[s]["per_event"])) for s in seeds}
    if key != "detected_any":
        events = {
            s: [
                e
                for e in ev
                if all(arm[s]["per_event"][e].get(key) is not None for arm in (full, other))
            ]
            for s, ev in events.items()
        }
    if not any(events.values()):
        return None, "ไม่มีเหตุการณ์ร่วมกัน"

    def aligned(arm: dict[int, dict]) -> list[dict]:
        return [{"per_event": {e: arm[s]["per_event"][e] for e in events[s]}} for s in seeds]

    keys_a, hits_a, n = _cluster_counts_keyed(aligned(full), key, unit)
    keys_b, hits_b, n_b = _cluster_counts_keyed(aligned(other), key, unit)
    # same events, different (symbol, day): not comparable (equal sizes alone are not enough)
    if keys_a != keys_b or not np.array_equal(n, n_b):
        return None, "เหตุการณ์เดียวกันอยู่คนละ symbol-day ในสองฝั่ง"
    diff, (lo, hi) = paired_ratio_bootstrap(hits_a, hits_b, n, n_boot, CI_SEED)
    k = int(n.size)  # independent clusters: events (REAL) or symbol-days (SEMI-SYN)
    enough = k >= MIN_EVENTS
    if enough:
        reason = None
    elif unit == UNIT_EVENT:
        reason = few_events_th(k)
    else:
        reason = f"ยังตัดสินไม่ได้ มีเพียง {k} กลุ่ม ({unit}) ต้องมีอย่างน้อย {MIN_EVENTS} กลุ่มจึงจะตัดสินได้"
    row = {
        "n_seeds": len(seeds),
        "n_events": max(len(v) for v in events.values()),
        "n_values": int(n.sum()),
        "n_clusters": k,
        "ci_unit": unit,
        "diff_mean": diff,
        "ci": [lo, hi],
        "significant": enough and significant(diff, (lo, hi), min_effect),
        "significant_reverse": enough and significant(-diff, (-hi, -lo), min_effect),
        "reason_th": reason,
    }
    return row, None


def _paired_ablation(runs: list[dict], n_boot: int, min_effect: float) -> dict[str, Any]:
    """`all` vs each of rules+stats, rules, stats, ai (ablation experiment, budget 2).

    REAL (`real_t001:*`) pairs events like `_paired`; SEMI-SYN (`semisyn:*`) pairs the
    `(seed, event)` values of each seed and resamples symbol-day clusters with the same
    indices for both arms (`paired_ratio_bootstrap`). `diff_mean` is recall_any of `all` minus
    that of the subset; `significant` means `all` is better, `significant_reverse` the subset.
    Only records of the most recent `config_hash` feeding this block are used.
    """
    recs, config_note = _latest_config(
        [r for r in runs if r["experiment"] == ABLATION and float(r["budget"]) == PAIRED_BUDGET]
    )
    arms: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in recs:
        if r.get("metrics") is not None:
            arms[r["variant"]][r["seed"]] = r
    rows: list[dict[str, Any]] = []
    notes = [config_note] if config_note else []
    for set_name, provenance in ABLATION_SETS:
        full = f"{set_name}:{ABLATION_FULL}"
        if full not in arms:
            if any(v.startswith(f"{set_name}:") for v in arms):
                notes.append(f"ข้าม {set_name}: ไม่มีผล {full} ที่งบ {PAIRED_BUDGET:g}")
            continue
        for sub in ABLATION_ARMS:
            variant = f"{set_name}:{sub}"
            row, why = _ablation_pair(arms[full], arms.get(variant, {}), n_boot, min_effect)
            if row is None:
                notes.append(f"ข้าม {variant}: {why}")
                continue
            if row["ci_unit"] == UNIT_LEGACY:
                notes.append(f"{variant}: ผล SEMI-SYN รุ่นเก่าไม่มีวันกำกับ CI จึงแคบเกินจริง")
            rows.append({"set": set_name, "variant": variant, "provenance": provenance, **row})
    return {
        "budget": PAIRED_BUDGET,
        "metric": "recall_any",
        "full": ABLATION_FULL,
        "rows": rows,
        "notes": notes,
    }


def _paired_ablation_md(pa: dict[str, Any], min_effect: float) -> list[str]:
    lines = [
        "",
        f"### ablation: all เทียบชุดย่อย (paired bootstrap, budget {PAIRED_BUDGET:g})",
        "",
    ]
    if not pa["rows"]:
        return [*lines, f"- {NO_PAIRED_ABLATION_TH}", *[f"- {n}" for n in pa["notes"]]]
    lines += [
        "| เทียบ | provenance | seeds | events | bootstrap unit (กลุ่ม) | "
        "ผลต่าง recall_any (all − X) [95 % CI] | อ่านว่า |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in pa["rows"]:
        sub = row["variant"].split(":", 1)[1]
        if row["reason_th"]:
            verdict = row["reason_th"]
        elif row["significant"]:
            verdict = f"all ดีกว่า {sub} อย่างมีนัย"
        elif row["significant_reverse"]:
            verdict = f"{sub} ดีกว่า all อย่างมีนัย"
        else:
            verdict = "ไม่มีนัยสำคัญ"
        lines.append(
            f"| all vs {row['variant']} | {row['provenance']} | {row['n_seeds']} | "
            f"{row['n_events']} | {row['ci_unit']} ({row['n_clusters']}) | "
            f"{_fmt_ci(row['diff_mean'], row['ci'])} | {verdict} |"
        )
    lines += [
        "",
        "- ผลต่าง > 0 แปลว่าชุดเต็ม (all) ตรวจพบมากกว่า; ผลต่าง < 0 แปลว่าชุดย่อยตรวจพบมากกว่า",
        f"- เกณฑ์ (ทั้งสองทิศ): CI ล่างของผลต่าง > 0 และผลต่าง ≥ {min_effect:g} (ห้ามใช้ CI ไม่ทับกันแทน)",
        "- REAL: สุ่มซ้ำทีละเหตุการณ์ด้วยดัชนีเดียวกันทั้งสองฝั่ง; SEMI-SYN: จับคู่ (seed, เหตุการณ์) "
        "ของแต่ละ seed แล้วสุ่มซ้ำทีละ symbol-day ด้วยดัชนีเดียวกันทั้งสองฝั่ง "
        "(ตัวประมาณแบบอัตราส่วนเดียวกับตาราง)",
        *[f"- {n}" for n in pa["notes"]],
    ]
    return lines


M1 = "m1_fusion"
M1_V0 = "all"
# tie order of the selection rule: simpler first (V0 < V1 < V3 < V2)
M1_SIMPLICITY = ("all", "no_ai", "ai_shift1", "ai_shift2", "ai_mondrian")
M1_LABEL = {
    "all": "V0 all",
    "no_ai": "V1 no_ai",
    "ai_shift1": "V3 ai_shift1",
    "ai_shift2": "V3 ai_shift2",
    "ai_mondrian": "V2 ai_mondrian",
}
M1_SPLITS = (("accept", "REAL"), ("test", "REAL"), ("semisyn", "SEMI-SYN"))
M1_TIE_TOL = 1e-12
M1_RULE_TH = (
    "กฎเลือก (กำหนดไว้ก่อนดูผล test และใช้ split accept เท่านั้น): ตัดตัวแปรที่ recall_any บน accept "
    "ต่ำกว่า V0 all อย่างมีนัย (paired bootstrap ข้ามเหตุการณ์, CI ล่างของ (V0 − X) > 0 และผลต่าง ≥ "
    "min_effect) และตัวแปรที่ตัดสิน recall เทียบ V0 ไม่ได้ (เหตุการณ์น้อยเกินไป) "
    "แล้วเลือกตัวที่ FA ของวันผันผวนสูงของ accept (accept-hard) ต่ำสุดที่งบ 2; "
    "ถ้าเท่ากันเลือกตัวที่ง่ายกว่า: all < no_ai < ai_shift1 < ai_shift2 < ai_mondrian"
)
M1_NOTES_TH = [
    "ผล test และ semisyn รายงานไว้ดูเท่านั้น ไม่ได้ใช้เลือกตัวแปร",
    "ทุกตัวแปรใช้คะแนนของตัวตรวจจับชุดเดียวกัน (fit และ score ครั้งเดียวต่อ seed) ต่างกันเฉพาะการรวมคะแนน "
    "และ τ ที่ fit ใหม่บนข้อมูลสอบเทียบของแต่ละตัวแปร",
    "no_ai: AE ยังถูกคำนวณและแสดง p แต่ไม่เข้า F, agreement และเทคนิค; ai_shiftδ: AE ต้องหายากขึ้น "
    "(ส่วนของ AE ใน F = max(0, −log10 p − δ)); ai_mondrian: p ของ AE เทียบกับคะแนนสอบเทียบ"
    "ในช่วงเดียวกันของช่วงราคาขาขึ้นในหน้าต่าง 1800 วินาที (jump_1800) (3 ช่วงจาก tercile บนข้อมูลสอบเทียบ); "
    "jump_1800 ไม่ใช่ความผันผวนสัมบูรณ์ และแถวที่มีการปั่นราคาขึ้นมีค่า jump สูงโดยธรรมชาติ "
    "ai_mondrian จึงอาจลด recall ของ AE ซึ่งเกณฑ์ recall ของกฎเลือกจะคัดออกเอง",
    "F คือ −log10 ของ anomaly p-value (ความหายากเทียบตลาดปกติ) ไม่ใช่ความน่าจะเป็นว่ามีการปั่นราคา",
    "accept-hard = วัน accept-normal ที่ผันผวนสูงสุดตามเกณฑ์เดียวกับ hard negative ของ test "
    "(realized volatility 1 นาที) และยังนับรวมใน FA ของ accept ด้วย",
]
NO_M1_TH = "ยังไม่มีผล m1_fusion"


def _m1_arms(recs: list[dict], split: str) -> dict[str, dict[int, dict]]:
    arms: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in recs:
        sp, _, v = r["variant"].partition(":")
        if sp == split and r.get("metrics") is not None:
            arms[v][r["seed"]] = r
    return arms


def m1_select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The M1 selection rule on accept rows (`variant`, `fa_hard`, `below_v0`, `undecided`); see
    the module docstring. A variant whose recall comparison with V0 could not be decided
    (`undecided`: no paired result, or too few clusters) is excluded: it cannot win on FA alone.
    V0 itself is always eligible. Returns the chosen variant (None if none qualifies) and the
    excluded ones with a Thai reason."""
    rank = {v: i for i, v in enumerate(M1_SIMPLICITY)}
    excluded = {r["variant"]: "recall ต่ำกว่า V0 อย่างมีนัย" for r in rows if r["below_v0"]}
    for r in rows:
        if r["variant"] != M1_V0 and r["variant"] not in excluded and r.get("undecided"):
            why = r.get("reason_th") or "ไม่มีผลเทียบแบบจับคู่กับ V0"
            excluded[r["variant"]] = f"ตัดสิน recall เทียบ V0 ไม่ได้ ({why}) จึงชนะด้วย FA อย่างเดียวไม่ได้"
    for r in rows:
        if r["variant"] not in excluded and r["fa_hard"] is None:
            excluded[r["variant"]] = "ไม่มีค่า FA-hard"
    ok = [r for r in rows if r["variant"] not in excluded]
    if not ok:
        return {"chosen": None, "excluded": excluded, "candidates": []}
    best = min(r["fa_hard"] for r in ok)
    tied = [r["variant"] for r in ok if r["fa_hard"] <= best + M1_TIE_TOL]
    chosen = min(tied, key=lambda v: (rank.get(v, len(rank)), v))
    return {"chosen": chosen, "excluded": excluded, "candidates": sorted(r["variant"] for r in ok)}


def _m1(runs: list[dict], n_boot: int, min_effect: float) -> dict[str, Any] | None:
    """M1 tables per split, paired recall_any vs V0, and the accept-only selection (budget 2)."""
    recs, config_note = _latest_config(
        [r for r in runs if r["experiment"] == M1 and float(r["budget"]) == PAIRED_BUDGET]
    )
    if not recs:
        return None
    notes = [config_note] if config_note else []
    splits: dict[str, Any] = {}
    for split, provenance in M1_SPLITS:
        arms = _m1_arms(recs, split)
        if not arms:
            continue
        rows = []
        order = [v for v in M1_SIMPLICITY if v in arms] + sorted(set(arms) - set(M1_SIMPLICITY))
        for v in order:
            c = _cell(list(arms[v].values()), n_boot)
            m, ci, fa_ci = c["mean"] or {}, c["ci"] or {}, c["fa_ci"] or {}
            row: dict[str, Any] = {
                "variant": v,
                "label": M1_LABEL.get(v, v),
                "n_seeds": c["n_seeds"],
                "n_events": c["n_events"],
                "recall_any": m.get("recall_any"),
                "recall_any_ci": ci.get("recall_any"),
                "recall_correct": m.get("recall_correct"),
                "fa": m.get("fa_per_symbol_day"),
                "fa_ci": fa_ci.get("fa_per_symbol_day"),
                "fa_hard": m.get("fa_per_symbol_day_hard"),
                "fa_hard_ci": fa_ci.get("fa_per_symbol_day_hard"),
                "vs_v0": None,
                "below_v0": False,
                "reason_th": None,
            }
            if v != M1_V0 and M1_V0 in arms:
                pair, why = _ablation_pair(arms[M1_V0], arms[v], n_boot, min_effect)
                if pair is None:
                    row["reason_th"] = why
                else:
                    # diff_mean = V0 - X; `significant` = V0 detects significantly more
                    row["vs_v0"] = {k: pair[k] for k in ("diff_mean", "ci", "ci_unit")}
                    row["below_v0"] = bool(pair["significant"])
                    row["above_v0"] = bool(pair["significant_reverse"])
                    row["reason_th"] = pair["reason_th"]
            # recall vs V0 not decided (no paired result, or too few clusters)
            row["undecided"] = v != M1_V0 and (row["vs_v0"] is None or row["reason_th"] is not None)
            rows.append(row)
        splits[split] = {"provenance": provenance, "rows": rows}
    selection = None
    if "accept" in splits:
        if M1_V0 not in _m1_arms(recs, "accept"):
            notes.append("เลือกตัวแปรไม่ได้: ไม่มีผล accept:all (V0) ที่งบ 2")
        else:
            selection = m1_select(splits["accept"]["rows"])
    return {
        "budget": PAIRED_BUDGET,
        "splits": splits,
        "selection": selection,
        "rule_th": M1_RULE_TH,
        "notes": notes,
    }


def _m1_md(m1: dict[str, Any] | None, min_effect: float) -> list[str]:
    lines = ["", f"### m1_fusion: ตัวแปรการรวมคะแนนของ AE (budget {PAIRED_BUDGET:g})"]
    if m1 is None:
        return [*lines, "", f"- {NO_M1_TH}"]
    for split, block in m1["splits"].items():
        lines += [
            "",
            f"#### {split} ({block['provenance']})",
            "",
            "| ตัวแปร | seeds | events | recall_any [95% CI] | recall_correct | "
            "FA/symbol-day [95% CI] | FA hard [95% CI] | "
            "ผลต่าง recall_any (V0 − X) [95% CI] | อ่านว่า |",
            "|" + "---|" * 9,
        ]
        for r in block["rows"]:
            if r["variant"] == M1_V0:
                diff, verdict = "–", "ตัวอ้างอิง"
            elif r["vs_v0"] is None:
                diff, verdict = "–", r["reason_th"] or "–"
            else:
                diff = _fmt_ci(r["vs_v0"]["diff_mean"], r["vs_v0"]["ci"])
                if r["reason_th"]:
                    verdict = r["reason_th"]
                elif r["below_v0"]:
                    verdict = "ต่ำกว่า V0 อย่างมีนัย"
                elif r.get("above_v0"):
                    verdict = "สูงกว่า V0 อย่างมีนัย"
                else:
                    verdict = "ไม่ต่างจาก V0 อย่างมีนัย"
            lines.append(
                f"| {r['label']} | {r['n_seeds']} | {r['n_events']} | "
                f"{_fmt_ci(r['recall_any'], r['recall_any_ci'])} | {_fmt(r['recall_correct'])} | "
                f"{_fmt_ci(r['fa'], r['fa_ci'])} | {_fmt_ci(r['fa_hard'], r['fa_hard_ci'])} | "
                f"{diff} | {verdict} |"
            )
    lines += ["", f"- {m1['rule_th']} (min_effect = {min_effect:g})"]
    sel = m1["selection"]
    if sel is not None:
        for v, why in sorted(sel["excluded"].items()):
            lines.append(f"- ตัดออก {M1_LABEL.get(v, v)}: {why}")
        if sel["chosen"] is None:
            lines.append("- **ผลการเลือก (accept):** ไม่มีตัวแปรที่ผ่านเกณฑ์")
        else:
            lines.append(f"- **ผลการเลือก (accept):** {M1_LABEL.get(sel['chosen'], sel['chosen'])}")
    lines += [f"- {n}" for n in [*M1_NOTES_TH, *m1["notes"]]]
    return lines


M4 = "m4_sell"
M4_VARIANTS = ("base", "sell")
M4_LABEL = {"base": "base (ไม่มีกฎฝั่งขาย)", "sell": "sell (มีกฎฝั่งขาย)"}
M4_SELL_KEYS = {
    "recall_sell_phase": "detected_sell_phase",
    "recall_sell_phase_correct": "detected_sell_phase_correct",
}
NO_M4_TH = "ยังไม่มีผล m4_sell"
# paired sell - base differences: metric -> per-event key
M4_PAIRED = {"recall_pump_phase": "detected_any", **M4_SELL_KEYS}
M4_PAIRED_TH = {
    "recall_pump_phase": "recall ช่วงพุ่ง",
    "recall_sell_phase": "recall ช่วงเทขาย",
    "recall_sell_phase_correct": "ระบุช่วงเทขายถูก",
}
M4_NOTES_TH = [
    "ช่วงพุ่ง (pump phase) = recall_any บนหน้าต่างรอบเวลาประกาศของเหตุการณ์ (เหมือน real_t001)",
    "ช่วงเทขาย (sell phase) = มีตอนแจ้งเตือนใดก็ได้ที่มีแถวเกินเกณฑ์ในช่วง (จุดสูงสุด + 60 วินาที, "
    "จุดสูงสุด + dump_tau] ที่วัดจากราคาจริง (dump_tau อย่างน้อย 60 วินาที ไม่เกิน 1 ชั่วโมง; "
    "ถ้าราคาไม่กลับลงภายใน 1 ชั่วโมงใช้ 1 ชั่วโมงเต็ม) ส่วนหางของตอนแจ้งเตือนช่วงพุ่งที่เลยจุดสูงสุด"
    "ไม่ถึง 60 วินาทีจึงไม่นับ และ 'ระบุช่วงถูก' = ตอนแจ้งเตือนนั้นระบุว่าเป็นช่วงกำลังเทขาย "
    "(แถวล่าสุดที่กฎจัดประเภทเป็น MM-T001.003 หรือ MM-T006)",
    "เหตุการณ์ที่วัดช่วงเทขายไม่ได้ไม่นับใน recall ช่วงเทขาย (นับจำนวนตามสถานะไว้ในคอลัมน์สุดท้าย)",
    "bear raid เป็นข้อมูลกึ่งสังเคราะห์ (SEMI-SYN) ทั้งหมด: ฉีดแรงขายเข้าไปในวัน test-normal จริง",
    "ทุกตัวเลขเป็นความผิดปกติทางสถิติ ไม่ใช่ความน่าจะเป็นว่ามีการปั่นหรือทุบราคา และไม่ได้บอกว่าใครทำอะไร",
    "เกณฑ์ผลต่าง sell − base: ลดลงอย่างมีนัย = CI บนของผลต่าง < 0 และผลต่าง ≤ −min_effect",
]


def _m4_arms(recs: list[dict], split: str) -> dict[str, dict[int, dict]]:
    return _m1_arms(recs, split)


def _sell_ci(records: list[dict], key: str, n_boot: int) -> tuple[float | None, list | None, int]:
    """Cluster-ratio recall and 95 % CI of a sell-phase key over the events that have it (REAL:
    one cluster per label_id; None values are events without a sell window)."""
    hits: dict[str, float] = defaultdict(float)
    n: dict[str, int] = defaultdict(int)
    for r in records:
        for lid, pe in r["per_event"].items():
            if pe.get(key) is None:
                continue
            hits[lid] += float(pe[key])
            n[lid] += 1
    keys = sorted(n)
    if not keys:
        return None, None, 0
    h = np.array([hits[k] for k in keys], dtype=np.float64)
    m = np.array([n[k] for k in keys], dtype=np.float64)
    mean, (lo, hi) = ratio_bootstrap_ci(h, m, n_boot, CI_SEED)
    return (
        mean if math.isfinite(mean) else None,
        [lo, hi] if math.isfinite(lo) else None,
        len(keys),
    )


def _m4(runs: list[dict], n_boot: int, min_effect: float) -> dict[str, Any] | None:
    """M4 table (budget 2): per variant the per-phase recall, FA, FA-hard, bear-raid recall."""
    recs, config_note = _latest_config(
        [r for r in runs if r["experiment"] == M4 and float(r["budget"]) == PAIRED_BUDGET]
    )
    if not recs:
        return None
    notes = [config_note] if config_note else []
    test, raid = _m4_arms(recs, "test"), _m4_arms(recs, "bear_raid")
    rows = []
    for v in [x for x in M4_VARIANTS if x in test or x in raid] + sorted(
        (set(test) | set(raid)) - set(M4_VARIANTS)
    ):
        row: dict[str, Any] = {"variant": v, "label": M4_LABEL.get(v, v)}
        if v in test:
            tr = list(test[v].values())
            c = _cell(tr, n_boot)
            m, ci, fa_ci = c["mean"] or {}, c["ci"] or {}, c["fa_ci"] or {}
            row |= {
                "n_seeds": c["n_seeds"],
                "n_events": c["n_events"],
                "recall_pump_phase": m.get("recall_any"),
                "recall_pump_phase_ci": ci.get("recall_any"),
                "fa": m.get("fa_per_symbol_day"),
                "fa_ci": fa_ci.get("fa_per_symbol_day"),
                "fa_hard": m.get("fa_per_symbol_day_hard"),
                "fa_hard_ci": fa_ci.get("fa_per_symbol_day_hard"),
            }
            for metric, key in M4_SELL_KEYS.items():
                mean, sci, k = _sell_ci(tr, key, n_boot)
                row[metric], row[f"{metric}_ci"] = mean, sci
                row["n_sell_events"] = k
            status: dict[str, int] = {}
            for r in tr[:1]:  # the same labels in every seed
                status = dict((r["metrics"] or {}).get("sell_status_counts") or {})
            row["sell_status_counts"] = status
            # dump over within 60 s: empty credit interval, out of the sell-phase denominators
            row["n_sell_short"] = len(
                {
                    lid
                    for r in tr
                    for lid, pe in r["per_event"].items()
                    if pe.get("sell_window_short")
                }
            )
        if v in raid:
            c = _cell(list(raid[v].values()), n_boot)
            m, ci = c["mean"] or {}, c["ci"] or {}
            row |= {
                "bear_raid_recall_any": m.get("recall_any"),
                "bear_raid_recall_any_ci": ci.get("recall_any"),
                "bear_raid_recall_correct": m.get("recall_correct"),
                "bear_raid_recall_correct_ci": ci.get("recall_correct"),
                "bear_raid_provenance": "SEMI-SYN",
            }
        rows.append(row)
    pairs: dict[str, Any] = dict.fromkeys(M4_PAIRED)
    if "base" in test and "sell" in test:
        for metric, key in M4_PAIRED.items():
            pair, why = _ablation_pair(test["sell"], test["base"], n_boot, min_effect, key)
            if pair is None:
                pairs[metric] = {"reason_th": why}
                continue
            # diff_mean = sell - base; `significant_reverse` = base detects significantly more
            p = {
                k: pair[k]
                for k in ("diff_mean", "ci", "ci_unit", "n_clusters", "n_seeds", "reason_th")
            }
            p["drop"] = bool(pair["significant_reverse"])
            p["rise"] = bool(pair["significant"])
            pairs[metric] = p
    return {
        "budget": PAIRED_BUDGET,
        "rows": rows,
        "paired_pump": pairs["recall_pump_phase"],
        "paired": pairs,
        "notes": notes,
    }


def _m4_md(m4: dict[str, Any] | None, min_effect: float) -> list[str]:
    lines = ["", f"### m4_sell: ตรวจช่วงกำลังพุ่งและช่วงกำลังเทขาย (budget {PAIRED_BUDGET:g})"]
    if m4 is None:
        return [*lines, "", f"- {NO_M4_TH}"]
    lines += [
        "",
        "| ตัวแปร | seeds | events (REAL) | recall ช่วงพุ่ง [95% CI] | recall ช่วงเทขาย [95% CI] | "
        "ระบุช่วงเทขายถูก [95% CI] | FA/symbol-day [95% CI] | FA hard [95% CI] | "
        "bear raid recall_any [95% CI] (SEMI-SYN) | bear raid recall_correct [95% CI] (SEMI-SYN) | "
        "สถานะการวัดช่วงเทขาย |",
        "|" + "---|" * 11,
    ]
    for r in m4["rows"]:
        g = r.get
        status = ", ".join(f"{k} {v}" for k, v in (r.get("sell_status_counts") or {}).items())
        lines.append(
            f"| {r['label']} | {r.get('n_seeds', '–')} | {r.get('n_events', '–')} | "
            f"{_fmt_ci(r.get('recall_pump_phase'), r.get('recall_pump_phase_ci'))} | "
            f"{_fmt_ci(r.get('recall_sell_phase'), r.get('recall_sell_phase_ci'))} | "
            f"{_fmt_ci(g('recall_sell_phase_correct'), g('recall_sell_phase_correct_ci'))} | "
            f"{_fmt_ci(r.get('fa'), r.get('fa_ci'))} | "
            f"{_fmt_ci(r.get('fa_hard'), r.get('fa_hard_ci'))} | "
            f"{_fmt_ci(r.get('bear_raid_recall_any'), r.get('bear_raid_recall_any_ci'))} | "
            f"{_fmt_ci(r.get('bear_raid_recall_correct'), r.get('bear_raid_recall_correct_ci'))} | "
            f"{status or '–'} |"
        )
    lines.append("")
    for metric, p in m4["paired"].items():
        name = M4_PAIRED_TH[metric]
        if p is None:
            lines.append(f"- ผลต่าง {name} (sell − base): ต้องมีผลทั้งสองตัวแปร")
            continue
        if "diff_mean" not in p:
            lines.append(f"- ผลต่าง {name} (sell − base): {p['reason_th']}")
            continue
        if p["reason_th"]:
            verdict = p["reason_th"]
        elif p["drop"]:
            verdict = (
                "กฎฝั่งขายทำให้ recall ช่วงพุ่งลดลงอย่างมีนัย"
                if metric == "recall_pump_phase"
                else f"{name}ลดลงอย่างมีนัย"
            )
        elif p["rise"]:
            verdict = f"{name}สูงขึ้นอย่างมีนัย"
        else:
            verdict = (
                "recall ช่วงพุ่งไม่ลดลงอย่างมีนัย"
                if metric == "recall_pump_phase"
                else f"{name}ไม่ต่างกันอย่างมีนัย"
            )
        lines.append(
            f"- ผลต่าง {name} (sell − base): {_fmt_ci(p['diff_mean'], p['ci'])} "
            f"({p['n_clusters']} {p['ci_unit']}, paired bootstrap) → {verdict} "
            f"(min_effect = {min_effect:g})"
        )
    for r in m4["rows"]:
        if r.get("n_sell_short"):
            lines.append(
                f"- {r['label']}: เหตุการณ์ที่เทขายจบภายใน 60 วินาที {r['n_sell_short']} รายการ "
                "ไม่นับในตัวหารของ recall ช่วงเทขาย (และผลต่าง sell − base)"
            )
    lines += [f"- {n}" for n in [*M4_NOTES_TH, *m4["notes"]]]
    return lines


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


def _fmt_fa(mean: float | None, ci: list[float] | None, n_days: int | None) -> str:
    """`x [lo, hi] (n วัน)`; the CI and the day count only when known."""
    s = _fmt_ci(mean, ci)
    return f"{s} ({n_days} วัน)" if mean is not None and n_days else s


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
    fa_ci = c.get("fa_ci") or {}
    if m is None:
        cells = ["–"] * 8
    else:
        cells = [
            _fmt_recall(m["recall_any"], std.get("recall_any"), ci.get("recall_any")),
            _fmt_recall(m["recall_correct"], std.get("recall_correct"), ci.get("recall_correct")),
            c["ci_unit"],
            f"{_fmt(m['precision'])} (prevalence {_fmt(m['prevalence'], 4)})",
            _fmt_fa(m["fa_per_symbol_day"], fa_ci.get("fa_per_symbol_day"), c.get("n_fa_days")),
            _fmt_fa(
                m["fa_per_symbol_day_hard"],
                fa_ci.get("fa_per_symbol_day_hard"),
                c.get("n_hard_days"),
            ),
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
    paired_ablation = _paired_ablation(runs, n_boot, cfg.eval.min_effect)
    retrain = _retrain_decision(runs, n_boot, cfg.eval.min_effect)
    m1 = _m1(runs, n_boot, cfg.eval.min_effect)
    m4 = _m4(runs, n_boot, cfg.eval.min_effect)
    confusion = _confusion(runs)
    data = {
        "header": header,
        "runs": len(runs),
        "signatures_sha": sig,
        "git_sha": _uniq(runs, "git_sha"),
        "config_hash": _uniq(runs, "config_hash"),
        "experiments": experiments,
        "paired_bootstrap": paired,
        "paired_ablation": paired_ablation,
        "retrain_decision": retrain,
        "m1_fusion": m1,
        "m4_sell": m4,
        "confusion": confusion,
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
        "recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | "
        "FA/symbol-day [95% CI] | FA/symbol-day hard [95% CI] | "
        "TTD median / p90 (s) | PR-AUC | notes |"
    )
    for exp in order:
        lines += ["", f"## {exp}", "", cols, "|" + "---|" * 14]
        for v, by_b in experiments[exp].items():
            lines += [_row_md(v, b, c) for b, c in by_b.items()]
        if exp == ABLATION:
            lines += _paired_ablation_md(paired_ablation, cfg.eval.min_effect)
        if exp == M1:
            lines += _m1_md(m1, cfg.eval.min_effect)
        if exp == M4:
            lines += _m4_md(m4, cfg.eval.min_effect)
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
    lines += _confusion_md(confusion)
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
