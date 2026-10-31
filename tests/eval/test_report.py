import json

import numpy as np
import polars as pl
import pytest

from idsai.ai.chief import Chief
from idsai.config import load_settings
from idsai.data.store import Store
from idsai.detection.rules import Condition, Signature, SignatureDetector
from idsai.eval.metrics import MIN_EVENTS, bootstrap_ci
from idsai.eval.report import (
    AGNOSTIC_NOTE,
    DETERMINISTIC_NOTE,
    MIXED_CONFIG_NOTE,
    NOTES_TH,
    SATURATED_MARK,
    UNIT_LEGACY,
    build_report,
    cluster_counts,
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
    skips = {"cal": {}, "eval": {"no_baseline": 2}}  # same set seen by two budgets: count once
    records.append(rec("real_t001", "full", 10, 0, ae, tau_saturated=True, n_skipped=skips))
    records.append(
        rec(
            "real_t001",
            "full",
            0.1,
            0,
            ae,
            metrics=None,
            per_event={},
            error="too small\nmore",
            n_skipped=skips,
        )
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
    # labels eligible vs dropped, by reason, next to the window-loading losses
    assert (
        "labels (split split0000001): eligible 22 of 30; dropped: out_of_period 4, cal_period 0, "
        "missing_day 3, no_baseline 1; symbol-days skipped at window loading: eval:no_baseline 2"
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
    assert loaded["labels"][0]["eligible"] == 22
    assert loaded["labels"][0]["skipped_symbol_days"] == {"eval:no_baseline": 2}
    cell = loaded["experiments"]["baselines"]["ai.autoencoder_trade"]["2"]
    assert cell["n_seeds"] == 2 and cell["provenance"] == "REAL" and cell["ci_unit"] == "event"
    assert cell["mean"]["recall_any"] == pytest.approx(0.75)
    assert cell["mean"]["recall_correct"] is None and cell["ci"]["recall_correct"] is None
    lo, hi = cell["ci"]["recall_any"]
    assert lo < 0.75 < hi
    # REAL, equal seed counts per event: the ratio estimator equals the event bootstrap
    _, old = bootstrap_ci(np.array([ae, ae], dtype=float), 300, 0)
    assert [lo, hi] == pytest.approx(list(old))
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
    hits, n = cluster_counts(recs, "detected_any", "symbol-day")
    assert hits.tolist() == [5.0, 0.0, 5.0, 0.0] and n.tolist() == [5.0] * 4
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


def test_shown_recall_and_ci_share_the_ratio_estimator(research, tmp_path):
    """Unequal cluster sizes (e.g. tick-skipped scenarios): mean = sum(hits) / sum(n)."""
    a, b = ("ABCBTC", "2020-07-01"), ("ABCBTC", "2020-07-02")
    recs = [
        rec(
            "semisyn",
            "pump_burst",
            2,
            0,
            [1, 1, 1, 1, 0],
            provenance="SEMI-SYN",
            days=[a, a, a, a, b],
        ),
        rec("semisyn", "pump_burst", 2, 1, [1, 0, 0], provenance="SEMI-SYN", days=[a, b, b]),
    ]
    write(research, recs)
    cfg = load_settings(overrides={"eval": {"n_boot": 1000}})
    data = build_report(research, tmp_path / "r.md", tmp_path / "r.json", cfg=cfg)
    cell = data["experiments"]["semisyn"]["pump_burst"]["2"]
    mean = cell["mean"]["recall_any"]
    assert mean == pytest.approx(5 / 8)  # not (0.8 + 1/3) / 2 per seed, not 0.5 per cluster
    lo, hi = cell["ci"]["recall_any"]
    assert lo <= mean <= hi


def test_semisyn_records_without_day_are_labelled_legacy(research, tmp_path):
    r = rec("semisyn", "pump_burst", 2, 0, [1, 0, 1], provenance="SEMI-SYN")
    for pe in r["per_event"].values():
        del pe["day"]
    write(research, [r])
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    build_report(research, md, js, cfg=load_settings(overrides={"eval": {"n_boot": 100}}))
    cell = json.loads(js.read_text(encoding="utf-8"))["experiments"]["semisyn"]["pump_burst"]["2"]
    assert cell["ci_unit"] == UNIT_LEGACY == "event (legacy, optimistic)"
    (row,) = [ln for ln in md.read_text(encoding="utf-8").splitlines() if ln.startswith("| pump")]
    assert "| event (legacy, optimistic) |" in row


def test_paired_block_uses_only_the_latest_config_hash(research, tmp_path):
    ae = [1] * 15 + [0] * 5
    kk = [1] * 5 + [0] * 15
    t = 1_000
    records = []
    for seed in (0, 1):  # an older config where kk looked better and pca existed
        for variant, det in (
            ("ai.autoencoder_trade", kk),
            ("baseline.kk", ae),
            ("baseline.pca", ae),
        ):
            t += 1
            records.append(
                rec(
                    "baselines",
                    variant,
                    2,
                    seed,
                    det,
                    config_hash="old",
                    created_us=t,
                    run_id=f"baselines|{variant}|b2|s{seed}|old",
                )
            )
    for seed in (0, 1):
        for variant, det in (("ai.autoencoder_trade", ae), ("baseline.kk", kk)):
            t += 1
            records.append(rec("baselines", variant, 2, seed, det, created_us=t))
    write(research, records)
    cfg = load_settings(overrides={"eval": {"n_boot": 300}})
    data = build_report(research, tmp_path / "r.md", tmp_path / "r.json", cfg=cfg)
    paired = data["paired_bootstrap"]
    rows = {r["baseline"]: r for r in paired["rows"]}
    assert set(rows) == {"baseline.kk"}  # pca only exists under the old config
    assert rows["baseline.kk"]["diff_mean"] == pytest.approx(0.5)
    assert rows["baseline.kk"]["n_seeds"] == 2
    ignored = "ไม่นับ 6 run ของ config_hash เก่า (ใช้เฉพาะ config_hash ล่าสุด cfg)"
    assert ignored in paired["notes"]
    assert any("baseline.pca" in n for n in paired["notes"])
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert f"- {ignored}" in text.splitlines()


def test_paired_block_without_older_configs_has_no_config_note(research, tmp_path):
    write(research, [rec("baselines", "ai.autoencoder_trade", 2, 0, [1, 0])])
    data = build_report(research, tmp_path / "r.md", tmp_path / "r.json", cfg=load_settings())
    assert not any("config_hash" in n for n in data["paired_bootstrap"]["notes"])


def test_error_text_keeps_the_whole_thai_sentence():
    from idsai.eval.report import _error_th

    th = "ชุดข้อมูลสอบเทียบเล็กเกินไป (6/(n_cal+1) = 0.000231, n_cal = 25920) — ต้องเพิ่มข้อมูล."
    assert _error_th(th + " Calibration set too small; calibrate on more rows.") == th
    assert _error_th("x" * 400).endswith("…") and len(_error_th("x" * 400)) == 300
    assert _error_th("only english error") == "only english error"


def _report(research, tmp_path, n_boot=200):
    cfg = load_settings(overrides={"eval": {"n_boot": n_boot}})
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    data = build_report(research, md, js, cfg=cfg)
    return data, md.read_text(encoding="utf-8")


def _row(text, prefix):
    (row,) = [ln for ln in text.splitlines() if ln.startswith(prefix)]
    return row


# ---------------------------------------------------------------- saturation marker


def test_rule_that_never_fires_in_cal_shows_the_saturation_marker(research, tmp_path):
    sig = Signature(
        sid=1,
        technique="MM-T001.001",
        layer="rule",
        rev=1,
        name="t",
        conditions=[Condition(feature="a", op=">=", thr=1.0)],
    )
    det = SignatureDetector(sig)
    n = 2000
    cal = pl.DataFrame(
        {"symbol": ["A"] * n, "t_end_us": list(range(n)), f"score__{det.name}": [-1.0] * n}
    )
    chief = Chief([det], load_settings().chief, "cfg", {})
    chief.calibrate(cal)
    with pytest.warns(RuntimeWarning, match="tau_saturated"):
        chief.fit_threshold(chief.fuse(cal), 2.0)
    write(research, [rec("ablation", "rules", 2, 0, [1, 0], tau_saturated=chief.tau_saturated)])
    _, text = _report(research, tmp_path)
    assert SATURATED_MARK in _row(text, "| rules |")


# ---------------------------------------------------------------- rule of three


def test_zero_hits_add_the_rule_of_three_upper_bound(research, tmp_path):
    write(research, [rec("real_t001", "full", 2, s, [0] * 6) for s in (0, 1)])  # 2 seeds, 6 events
    data, text = _report(research, tmp_path)
    cell = data["experiments"]["real_t001"]["full"]["2"]
    assert cell["ci"]["recall_any"] == [0.0, 0.0]  # the bootstrap CI stays, for reference
    r3 = cell["rule_of_three"]["recall_any"]
    assert r3 == {"side": "upper", "bound": 0.5, "hits": 0, "n": 6, "unit": "event"}
    assert "0 จาก 6 เหตุการณ์: ช่วงจริงอาจสูงถึงราว 50% (rule of three)" in _row(text, "| full |")


def test_all_hits_add_the_rule_of_three_lower_bound(research, tmp_path):
    write(research, [rec("real_t001", "full", 2, 0, [1] * 7)])
    data, text = _report(research, tmp_path)
    r3 = data["experiments"]["real_t001"]["full"]["2"]["rule_of_three"]["recall_any"]
    assert r3["side"] == "lower" and r3["bound"] == pytest.approx(1 - 3 / 7)
    assert (r3["hits"], r3["n"]) == (7, 7)
    assert "7 จาก 7 เหตุการณ์: ช่วงจริงอาจต่ำถึงราว 57% (rule of three)" in _row(text, "| full |")


def test_rule_of_three_is_clipped_and_absent_otherwise(research, tmp_path):
    write(
        research, [rec("real_t001", "none", 2, 0, [0, 0]), rec("real_t001", "both", 2, 0, [1, 1])]
    )
    write(research, [rec("real_t001", "mixed", 2, 0, [1, 0, 1])])
    data, text = _report(research, tmp_path)
    exp = data["experiments"]["real_t001"]
    assert exp["none"]["2"]["rule_of_three"]["recall_any"]["bound"] == 1.0
    assert exp["both"]["2"]["rule_of_three"]["recall_any"]["bound"] == 0.0
    assert exp["mixed"]["2"]["rule_of_three"] == {"recall_any": None, "recall_correct": None}
    assert "rule of three" not in _row(text, "| mixed |")


def test_rule_of_three_counts_independent_clusters_not_seed_event_pairs(research, tmp_path):
    days = [("ABCBTC", f"2020-07-0{1 + i % 4}") for i in range(8)]  # 8 events on 4 days
    write(
        research,
        [
            rec("semisyn", "pump_burst", 2, s, [0] * 8, provenance="SEMI-SYN", days=days)
            for s in range(3)
        ],
    )
    data, text = _report(research, tmp_path)
    r3 = data["experiments"]["semisyn"]["pump_burst"]["2"]["rule_of_three"]["recall_any"]
    assert r3 == {"side": "upper", "bound": 0.75, "hits": 0, "n": 4, "unit": "symbol-day"}
    assert "0 จาก 4 symbol-day: ช่วงจริงอาจสูงถึงราว 75% (rule of three)" in text


def test_technique_agnostic_cell_has_no_recall_correct_rule_of_three(research, tmp_path):
    write(research, [rec("baselines", "baseline.kk", 2, 0, [0] * 6, classifies=False)])
    data, _ = _report(research, tmp_path)
    r3 = data["experiments"]["baselines"]["baseline.kk"]["2"]["rule_of_three"]
    assert r3["recall_any"]["side"] == "upper" and r3["recall_correct"] is None


def test_error_cell_has_no_rule_of_three_or_std(research, tmp_path):
    write(research, [rec("real_t001", "full", 2, 0, [1], metrics=None, per_event={}, error="x")])
    cell = _report(research, tmp_path)[0]["experiments"]["real_t001"]["full"]["2"]
    assert cell["rule_of_three"] is None and cell["std"] is None


def test_rule_of_three_note_explains_without_hardcoded_values():
    (note,) = [n for n in NOTES_TH if "rule of three" in n]
    assert "3/n" in note and "%" not in note


# ---------------------------------------------------------------- seed spread


def test_std_over_seeds_in_json_and_md(research, tmp_path):
    write(research, [rec("real_t001", "full", 2, 0, [1, 1, 0, 0])])
    write(research, [rec("real_t001", "full", 2, 1, [1, 1, 1, 0])])
    data, text = _report(research, tmp_path)
    cell = data["experiments"]["real_t001"]["full"]["2"]
    want = float(np.std([0.5, 0.75], ddof=1))
    assert cell["std"]["recall_any"] == pytest.approx(want)
    assert cell["std"]["recall_correct"] == pytest.approx(want)
    cells = [c.strip() for c in _row(text, "| full |").split("|")]
    assert cells[6].startswith(f"{cell['mean']['recall_any']:.3f} ± {want:.3f} [")
    assert "recall_any ± std [95% CI]" in text and "recall_correct ± std [95% CI]" in text


def test_std_with_one_seed_is_a_dash(research, tmp_path):
    write(research, [rec("real_t001", "full", 2, 0, [1, 0])])
    data, text = _report(research, tmp_path)
    cell = data["experiments"]["real_t001"]["full"]["2"]
    assert cell["std"] == {"recall_any": None, "recall_correct": None}
    cells = [c.strip() for c in _row(text, "| full |").split("|")]
    assert cells[6].startswith("0.500 ± – [")


def test_std_pools_records_of_one_seed_and_skips_agnostic_recall_correct(research, tmp_path):
    # two configs of seed 0 pool into one seed value: (2 + 0) / 4 = 0.5; seed 1 gives 1.0
    write(
        research,
        [
            rec("baselines", "b", 2, 0, [1, 1], classifies=False, run_id="a"),
            rec("baselines", "b", 2, 0, [0, 0], classifies=False, run_id="b"),
            rec("baselines", "b", 2, 1, [1, 1], classifies=False, run_id="c"),
        ],
    )
    cell = _report(research, tmp_path)[0]["experiments"]["baselines"]["b"]["2"]
    assert cell["std"]["recall_any"] == pytest.approx(float(np.std([0.5, 1.0], ddof=1)))
    assert cell["std"]["recall_correct"] is None


# ---------------------------------------------------------------- minimum events


def test_paired_verdict_needs_the_minimum_event_count(research, tmp_path):
    n = MIN_EVENTS - 1
    write(
        research,
        [
            rec("baselines", "ai.autoencoder_trade", 2, 0, [1] * n),
            rec("baselines", "baseline.kk", 2, 0, [0] * n),
        ],
    )
    data, text = _report(research, tmp_path)
    (row,) = data["paired_bootstrap"]["rows"]
    assert row["n_events"] == n and row["diff_mean"] == 1.0
    assert row["significant"] is False
    assert f"เพียง {n} ครั้ง" in row["reason_th"] and f"อย่างน้อย {MIN_EVENTS}" in row["reason_th"]
    md_row = _row(text, "| baseline.kk | REAL |")
    assert row["reason_th"] in md_row and "AE ดีกว่า" not in md_row


def test_paired_verdict_at_the_minimum_event_count(research, tmp_path):
    write(
        research,
        [
            rec("baselines", "ai.autoencoder_trade", 2, 0, [1] * MIN_EVENTS),
            rec("baselines", "baseline.kk", 2, 0, [0] * MIN_EVENTS),
        ],
    )
    data, text = _report(research, tmp_path)
    (row,) = data["paired_bootstrap"]["rows"]
    assert row["significant"] is True and row["reason_th"] is None
    assert "AE ดีกว่าอย่างมีนัย" in _row(text, "| baseline.kk | REAL |")


# ---------------------------------------------------------------- mixed runs warning


def test_no_mixed_warning_when_runs_share_everything(research, tmp_path):
    write(research, [rec("real_t001", "full", 2, s, [1, 0]) for s in (0, 1)])
    data, text = _report(research, tmp_path)
    assert data["notes"] == NOTES_TH and "**คำเตือน:**" not in text


def test_mixed_git_sha_warns_and_dirty_counts_as_different(research, tmp_path):
    write(
        research,
        [
            rec("real_t001", "full", 2, 0, [1, 0], git_sha="abc"),
            rec("real_t001", "full", 2, 1, [1, 0], git_sha="abc-dirty"),
        ],
    )
    data, text = _report(research, tmp_path)
    (note,) = [n for n in data["notes"] if "git_sha" in n]
    assert "abc, abc-dirty" in note
    assert f"**คำเตือน:** {note}" in text.splitlines()
    assert MIXED_CONFIG_NOTE not in data["notes"]


def test_mixed_split_warns_with_the_shas(research, tmp_path):
    other = {**SPLIT_DEF, "sha12": "split0000002"}
    write(
        research,
        [
            rec("real_t001", "full", 2, 0, [1, 0]),
            rec("real_t001", "full", 2, 1, [1, 0], split_def=other),
        ],
    )
    data, text = _report(research, tmp_path)
    (note,) = [n for n in data["notes"] if "split0000001" in n]
    assert "split0000002" in note and f"**คำเตือน:** {note}" in text.splitlines()


def test_mixed_config_git_and_split_all_warn(research, tmp_path):
    other = {**SPLIT_DEF, "sha12": "split0000002"}
    write(
        research,
        [
            rec("real_t001", "full", 2, 0, [1, 0]),
            rec("real_t001", "full", 2, 1, [1, 0], config_hash="c2", git_sha="d", split_def=other),
        ],
    )
    data, text = _report(research, tmp_path)
    assert data["notes"][0] == MIXED_CONFIG_NOTE
    assert text.count("**คำเตือน:**") == 3
    assert "abc, d" in data["notes"][1] and "split0000001, split0000002" in data["notes"][2]


# ---------------------------------------------------------------- notes


def test_notes_explain_negative_ttd_and_the_budget_cap():
    text = " ".join(NOTES_TH)
    assert "TTD ติดลบ" in text and "ก่อน" in text
    assert "งบการแจ้งเตือนเป็นเพดาน" in text and "ต่ำกว่า" in text


def test_note_defines_recall_correct():
    (note,) = [n for n in NOTES_TH if n.startswith("recall_correct")]
    assert "ทับ" in note and "F สูงสุด" in note and "หลักฐาน" in note and "จุดสูงสุด" in note
    assert "สังเคราะห์" not in note and "ข้อมูลจริง" not in note
