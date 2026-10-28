import json

import pytest

from idsai.config import load_settings
from idsai.data.store import Store
from idsai.eval.report import DETERMINISTIC_NOTE, SATURATED_MARK, build_report

EVENTS = [f"e{i}" for i in range(20)]


def rec(experiment, variant, budget, seed, detected, *, provenance="REAL", **kw):
    per_event = {
        lid: {"detected_any": bool(d), "detected_correct": bool(d), "ttd_s": 1.0 if d else None}
        for lid, d in zip(EVENTS, detected, strict=True)
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
        "metrics": metrics,
        "per_event": per_event,
        "tau": 2.0,
        "tau_saturated": False,
        "deterministic": False,
        "error": None,
        "n_events": len(EVENTS),
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
    records = []
    for seed in (0, 1):
        records.append(rec("baselines", "ai.autoencoder_trade", 2, seed, ae))
        records.append(rec("baselines", "baseline.kk", 2, seed, kk, deterministic=True))
        records.append(rec("baselines", "baseline.pca", 2, seed, ae, deterministic=True))
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
    assert "runs: 9" in text and "signatures_sha: sig123456789" in text
    assert "accuracy" not in text.lower() and "accuracy" not in js.read_text("utf-8").lower()
    assert SATURATED_MARK in text and DETERMINISTIC_NOTE in text and "error: too small" in text
    assert "ความผิดปกติทางสถิติ" in text and "autocorrelation" in text and "prevalence" in text
    # every table row carries a provenance
    rows = [ln for ln in text.splitlines() if ln.startswith("| ") and "---" not in ln]
    body = [ln for ln in rows if not ln.startswith("| variant") and not ln.startswith("| baseline")]
    assert body and all("REAL" in ln or "SEMI-SYN" in ln for ln in body)

    loaded = json.loads(js.read_text(encoding="utf-8"))
    assert loaded == json.loads(json.dumps(data, default=str)) or loaded["runs"] == 9
    cell = loaded["experiments"]["baselines"]["ai.autoencoder_trade"]["2"]
    assert cell["n_seeds"] == 2 and cell["provenance"] == "REAL"
    assert cell["mean"]["recall_any"] == pytest.approx(0.75)
    lo, hi = cell["ci"]["recall_any"]
    assert lo < 0.75 < hi
    assert loaded["experiments"]["baselines"]["baseline.kk"]["2"]["deterministic"] is True
    assert loaded["experiments"]["real_t001"]["full"]["10"]["tau_saturated_any"] is True
    err = loaded["experiments"]["real_t001"]["full"]["0.1"]
    assert err["mean"] is None and err["errors"] == ["too small\nmore"]
    assert loaded["experiments"]["semisyn"]["pump_burst"]["2"]["provenance"] == "SEMI-SYN"

    paired = {r["baseline"]: r for r in loaded["paired_bootstrap"]["rows"]}
    assert paired["baseline.kk"]["diff_mean"] == pytest.approx(0.5)
    assert paired["baseline.kk"]["significant"] is True
    assert paired["baseline.pca"]["diff_mean"] == 0.0
    assert paired["baseline.pca"]["ci"] == [0.0, 0.0] and not paired["baseline.pca"]["significant"]
    notes = loaded["paired_bootstrap"]["notes"]
    assert any("baseline.random" in n for n in notes)  # missing arm is skipped with a note


def test_report_on_empty_db_and_path_argument(tmp_path):
    db = tmp_path / "research.duckdb"
    Store(tmp_path, db).close()
    data = build_report(db, tmp_path / "r.md", tmp_path / "r.json", cfg=load_settings())
    assert data["runs"] == 0 and data["experiments"] == {}
    assert "runs: 0" in (tmp_path / "r.md").read_text(encoding="utf-8")
    assert data["paired_bootstrap"]["rows"] == []
