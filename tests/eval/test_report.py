import json

import numpy as np
import pytest

from idsai.config import load_settings
from idsai.data.store import Store
from idsai.eval.metrics import bootstrap_ci
from idsai.eval.report import (
    AGNOSTIC_NOTE,
    DETERMINISTIC_NOTE,
    SATURATED_MARK,
    build_report,
    cluster_values,
)

EVENTS = [f"e{i}" for i in range(20)]
SPLIT_DEF = {
    "sha12": "split0000001",
    "n_labels": 30,
    "labels_dropped": {"none": {"out_of_period": 4}, "test": {"missing_day": 3, "no_baseline": 1}},
}


def rec(experiment, variant, budget, seed, detected, *, provenance="REAL", days=None, **kw):
    """`detected` per event; `days[i]` is the (symbol, day) an event was injected into."""
    per_event = {}
    for i, (lid, d) in enumerate(zip(EVENTS[: len(detected)], detected, strict=True)):
        sym, day = days[i] if days else ("ABCBTC", f"2020-07-{1 + i % 28:02d}")
        per_event[f"{lid}-s{seed}" if days else lid] = {
            "symbol": sym,
            "day": day,
            "detected_any": bool(d),
            "detected_correct": bool(d),
            "ttd_s": 1.0 if d else None,
        }
    metrics = {
        "recall_any": sum(detected) / len(detected),
        "recall_correct": sum(detected) / len(detected),
        "precision": 0.5,
        "prevalence": 0.001,
        "fa_per_symbol_day": 1.5,
        "fa_per_symbol_day_hard": 3.0,
        "ttd_median_s": 1.0,
        "ttd_p90_s": 2.0,
        "pr_auc": None,
        "f1": 0.4,
        "n_alerts": 4,
    }
    base = {
        "run_id": f"{experiment}|{variant}|b{budget:g}|s{seed}|cfg",
        "experiment": experiment,
        "variant": variant,
        "budget": float(budget),
        "seed": seed,
        "provenance": provenance,
        "git_sha": "abc",
        "config_hash": "cfg",
        "signatures_sha": "sig123456789",
        "split_def": SPLIT_DEF,
        "metrics": metrics,
        "per_event": per_event,
        "tau": 2.0,
        "tau_saturated": False,
        "deterministic": False,
        "classifies": True,
        "error": None,
        "n_events": len(detected),
        "n_symbol_days": 7,
    }
    base.update(kw)
    return base


@pytest.fixture
def research(tmp_path):
    with Store(tmp_path / "data", tmp_path / "research.duckdb") as s:
        yield s


def write(store, records):
    for r in records:
        store.insert_run(r)


def test_report_tables_markers_and_paired_bootstrap(research, tmp_path):
    ae = [1] * 15 + [0] * 5
    kk = [1] * 5 + [0] * 15
    agnostic = {"classifies": False}
    records = []
    for seed in (0, 1):
        records.append(rec("baselines", "ai.autoencoder_trade", 2, seed, ae, **agnostic))
        records.append(rec("baselines", "baseline.kk", 2, seed, kk, deterministic=True, **agnostic))
        records.append(
            rec("baselines", "baseline.pca", 2, seed, ae, deterministic=True, **agnostic)
        )
    records.append(rec("real_t001", "full", 10, 0, ae, tau_saturated=True))
    records.append(
        rec("real_t001", "full", 0.1, 0, ae, metrics=None, per_event={}, error="too small\nmore")
    )
    records.append(rec("semisyn", "pump_burst", 2, 0, kk, provenance="SEMI-SYN"))
    write(research, records)

    cfg = load_settings(overrides={"eval": {"n_boot": 300}})
    md, js = tmp_path / "out" / "report.md", tmp_path / "out" / "report.json"
    data = build_report(research, md, js, cfg=cfg)
    text = md.read_text(encoding="utf-8")
    raw = js.read_text(encoding="utf-8")
    assert "runs: 9" in text and "signatures_sha: sig123456789" in text
    assert "accuracy" not in text.lower() and "accuracy" not in raw.lower()
    assert SATURATED_MARK in text and DETERMINISTIC_NOTE in text and "error: too small" in text
    assert "ความผิดปกติทางสถิติ" in text and "autocorrelation" in text and "prevalence" in text
    # labels used vs dropped, by reason
    assert (
        "labels (split split0000001): used 22 of 30; dropped: out_of_period 4, cal_period 0, "
        "missing_day 3, no_baseline 1"
    ) in text
    # every table row carries a provenance
    rows = [ln for ln in text.splitlines() if ln.startswith("| ") and "---" not in ln]
    body = [
        ln for ln in rows if not ln.startswith("| variant") and not ln.startswith("| baseline |")
    ]
    assert body and all("REAL" in ln or "SEMI-SYN" in ln for ln in body)
    # technique-agnostic detectors: recall_correct shown as "–" with a note, not 0.000
    (kk_row,) = [ln for ln in body if ln.startswith("| baseline.kk | 2 |")]
    cells = [c.strip() for c in kk_row.split("|")]
    assert cells[7] == "–" and AGNOSTIC_NOTE in kk_row and cells[8] == "event"

    loaded = json.loads(raw)
    assert loaded == data  # the returned dict is exactly what was written
    assert loaded["runs"] == 9 and "runs: 9" in loaded["header"]
    assert loaded["labels"][0]["used"] == 22
    cell = loaded["experiments"]["baselines"]["ai.autoencoder_trade"]["2"]
    assert cell["n_seeds"] == 2 and cell["provenance"] == "REAL" and cell["ci_unit"] == "event"
    assert cell["mean"]["recall_any"] == pytest.approx(0.75)
    assert cell["mean"]["recall_correct"] is None and cell["ci"]["recall_correct"] is None
    lo, hi = cell["ci"]["recall_any"]
    assert lo < 0.75 < hi
    assert loaded["experiments"]["baselines"]["baseline.kk"]["2"]["deterministic"] is True
    full = loaded["experiments"]["real_t001"]["full"]
    assert full["10"]["tau_saturated_any"] is True and full["10"]["mean"]["recall_correct"] == 0.75
    assert full["0.1"]["mean"] is None and full["0.1"]["errors"] == ["too small\nmore"]
    semi = loaded["experiments"]["semisyn"]["pump_burst"]["2"]
    assert semi["provenance"] == "SEMI-SYN" and semi["ci_unit"] == "symbol-day"

    paired = {r["baseline"]: r for r in loaded["paired_bootstrap"]["rows"]}
    assert paired["baseline.kk"]["diff_mean"] == pytest.approx(0.5)
    assert paired["baseline.kk"]["significant"] is True
    assert paired["baseline.pca"]["diff_mean"] == 0.0
    assert paired["baseline.pca"]["ci"] == [0.0, 0.0] and not paired["baseline.pca"]["significant"]
    notes = loaded["paired_bootstrap"]["notes"]
    assert any("baseline.random" in n for n in notes)  # missing arm is skipped with a note


def test_semisyn_ci_clusters_by_symbol_day(research, tmp_path):
    """5 seeds inject into the same 4 days; detection depends only on the day."""
    days = [("ABCBTC", f"2020-07-0{1 + i % 4}") for i in range(4)]
    hit = [1, 0, 1, 0]  # days 1 and 3 are easy, 2 and 4 hard
    recs = [
        rec("semisyn", "pump_burst", 2, seed, hit, provenance="SEMI-SYN", days=days)
        for seed in range(5)
    ]
    write(research, recs)
    clustered = cluster_values(recs, "detected_any", "symbol-day")
    assert clustered.tolist() == [1.0, 0.0, 1.0, 0.0]
    pooled = np.array([float(pe["detected_any"]) for r in recs for pe in r["per_event"].values()])
    assert pooled.size == 20
    _, (plo, phi) = bootstrap_ci(pooled, 1000, 0)

    cfg = load_settings(overrides={"eval": {"n_boot": 1000}})
    data = build_report(research, tmp_path / "r.md", tmp_path / "r.json", cfg=cfg)
    cell = data["experiments"]["semisyn"]["pump_burst"]["2"]
    lo, hi = cell["ci"]["recall_any"]
    assert cell["ci_unit"] == "symbol-day" and cell["mean"]["recall_any"] == 0.5
    assert (hi - lo) > 1.5 * (phi - plo)  # 4 independent days, not 20 independent events
    assert "symbol-day" in (tmp_path / "r.md").read_text(encoding="utf-8")


def test_report_reads_db_path_read_only(tmp_path):
    db = tmp_path / "db" / "research.duckdb"
    with Store(tmp_path / "data", db) as s:
        s.insert_run(rec("real_t001", "full", 2, 0, [1, 0]))
    before = sorted(p.name for p in db.parent.iterdir())
    data = build_report(db, tmp_path / "r.md", tmp_path / "r.json", cfg=load_settings())
    assert data["runs"] == 1
    assert sorted(p.name for p in db.parent.iterdir()) == before  # no trades/ dir, no new files


def test_report_on_empty_db(tmp_path):
    db = tmp_path / "research.duckdb"
    Store(tmp_path, db).close()
    data = build_report(db, tmp_path / "r.md", tmp_path / "r.json", cfg=load_settings())
    assert data["runs"] == 0 and data["experiments"] == {} and data["labels"] == []
    assert "runs: 0" in (tmp_path / "r.md").read_text(encoding="utf-8")
    assert data["paired_bootstrap"]["rows"] == []
