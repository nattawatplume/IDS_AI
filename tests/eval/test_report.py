import json

import numpy as np
import polars as pl
import pytest

from idsai.ai.chief import Chief
from idsai.config import load_settings
from idsai.data.store import Store
from idsai.detection.rules import Condition, Signature, SignatureDetector
from idsai.eval.metrics import (
    MIN_EVENTS,
    bootstrap_ci,
    few_events_th,
    paired_bootstrap,
    ratio_bootstrap_ci,
)
from idsai.eval.report import (
    AGNOSTIC_NOTE,
    DETERMINISTIC_NOTE,
    MIXED_CONFIG_NOTE,
    NO_PAIRED_ABLATION_TH,
    NOTES_TH,
    SATURATED_MARK,
    UNIT_LEGACY,
    build_report,
    cluster_counts,
    m1_select,
    paired_ratio_bootstrap,
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


# ---------------------------------------------------------------- ablation paired bootstrap


def _ablation(prefix, arms, seeds=(0, 1), **kw):
    """Ablation records at budget 2: `arms` maps the subset name to its per-event detections."""
    prov = "SEMI-SYN" if prefix == "semisyn" else "REAL"
    return [
        rec("ablation", f"{prefix}:{sub}", 2, s, det, provenance=prov, **kw)
        for sub, det in arms.items()
        for s in seeds
    ]


def _section(text, title):
    return text.split(f"## {title}\n", 1)[1].split("\n## ", 1)[0]


def test_paired_ratio_bootstrap_matches_the_event_bootstrap_with_one_cluster_per_event():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, size=(3, 25)).astype(float)
    b = rng.integers(0, 2, size=(3, 25)).astype(float)
    n = np.full(25, 3.0)
    got = paired_ratio_bootstrap(a.sum(axis=0), b.sum(axis=0), n, 500, 0)
    want = paired_bootstrap(a, b, 500, 0)
    assert got[0] == pytest.approx(want[0]) and got[1] == pytest.approx(want[1])
    assert paired_ratio_bootstrap(a.sum(axis=0), b.sum(axis=0), n, 500, 0) == got  # seeded
    assert paired_ratio_bootstrap(a.sum(axis=0), a.sum(axis=0), n, 500, 0) == (0.0, (0.0, 0.0))
    nan, (lo, hi) = paired_ratio_bootstrap(np.array([]), np.array([]), np.array([]), 10, 0)
    assert np.isnan(nan) and np.isnan(lo) and np.isnan(hi)
    with pytest.raises(ValueError):
        paired_ratio_bootstrap(np.ones(2), np.ones(2), np.ones(3), 10, 0)


def test_paired_ablation_real_rows_json_and_md(research, tmp_path):
    full = [1] * 15 + [0] * 5
    arms = {
        "all": full,
        "rules+stats": [1] * 5 + [0] * 15,  # all clearly better
        "rules": full,  # identical: diff 0
        "ai": [1] * 20,  # the subset clearly better
    }  # stats missing: skipped with a note
    write(research, _ablation("real_t001", arms))
    data, text = _report(research, tmp_path, n_boot=300)
    pa = data["paired_ablation"]
    assert pa["budget"] == 2.0 and pa["metric"] == "recall_any"
    rows = {r["variant"]: r for r in pa["rows"]}
    assert set(rows) == {"real_t001:rules+stats", "real_t001:rules", "real_t001:ai"}
    rs = rows["real_t001:rules+stats"]
    assert rs["provenance"] == "REAL" and rs["ci_unit"] == "event"
    assert rs["n_seeds"] == 2 and rs["n_events"] == 20 and rs["n_clusters"] == 20
    assert rs["diff_mean"] == pytest.approx(0.5) and rs["significant"]
    assert not rs["significant_reverse"]
    # REAL mirrors `_paired`: the event-level paired bootstrap over sorted label ids, draw for draw
    order = sorted(range(20), key=lambda i: EVENTS[i])
    a = np.array([[full[i] for i in order]] * 2, dtype=float)
    b = np.array([[arms["rules+stats"][i] for i in order]] * 2, dtype=float)
    _, ci = paired_bootstrap(a, b, 300, 0)
    assert rs["ci"] == pytest.approx(list(ci))
    assert rows["real_t001:rules"]["ci"] == [0.0, 0.0]
    assert not rows["real_t001:rules"]["significant"]
    ai = rows["real_t001:ai"]
    assert ai["diff_mean"] == pytest.approx(-0.25) and ai["significant_reverse"]
    assert not ai["significant"]
    assert any("real_t001:stats" in n for n in pa["notes"])
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["paired_ablation"] == pa

    section = _section(text, "ablation")
    assert "### ablation: all เทียบชุดย่อย (paired bootstrap, budget 2)" in section
    assert "ผลต่าง recall_any (all − X) [95 % CI]" in section and "อ่านว่า" in section
    row = _row(text, "| all vs real_t001:rules+stats |")
    assert "| REAL | 2 | 20 | event (20) | 0.500 [" in row
    assert "all ดีกว่า rules+stats อย่างมีนัย" in row
    assert "ai ดีกว่า all อย่างมีนัย" in _row(text, "| all vs real_t001:ai |")
    assert "ไม่มีนัยสำคัญ" in _row(text, "| all vs real_t001:rules |")
    assert NO_PAIRED_ABLATION_TH not in text


def test_paired_ablation_semisyn_pairs_per_seed_and_clusters_by_symbol_day(research, tmp_path):
    """Scenario ids differ per seed; 4 days, and both arms' detection depends only on the day."""
    days = [("ABCBTC", f"2020-07-0{1 + i % 4}") for i in range(12)]
    full = [1, 0, 1, 0] * 3
    sub = [0, 0, 1, 0] * 3  # misses day 1 only
    write(research, _ablation("semisyn", {"all": full, "rules": sub}, seeds=range(5), days=days))
    data, text = _report(research, tmp_path, n_boot=1000)
    (row,) = data["paired_ablation"]["rows"]
    assert row["variant"] == "semisyn:rules" and row["provenance"] == "SEMI-SYN"
    assert row["ci_unit"] == "symbol-day" and row["n_clusters"] == 4
    assert row["n_seeds"] == 5 and row["n_events"] == 12 and row["n_values"] == 60
    # the difference of the two table recalls (same cluster ratio estimator)
    cells = data["experiments"]["ablation"]
    diff = cells["semisyn:all"]["2"]["mean"]["recall_any"]
    diff -= cells["semisyn:rules"]["2"]["mean"]["recall_any"]
    assert row["diff_mean"] == pytest.approx(diff) and diff == pytest.approx(0.25)
    # 4 symbol-days, not 60 independent values: below MIN_EVENTS clusters, not decided
    assert not row["significant"] and not row["significant_reverse"]
    assert "4 กลุ่ม (symbol-day)" in row["reason_th"]
    lo, hi = row["ci"]
    _, (elo, ehi) = paired_bootstrap(
        np.array([full], dtype=float), np.array([sub], dtype=float), 1000, 0
    )
    assert (hi - lo) > (ehi - elo)  # wider than resampling 12 events as independent
    assert "symbol-day (4)" in _row(text, "| all vs semisyn:rules |")
    again, _ = _report(research, tmp_path / "again", n_boot=1000)
    assert again["paired_ablation"] == data["paired_ablation"]  # deterministic (CI_SEED)


def test_paired_ablation_semisyn_with_enough_days_decides(research, tmp_path):
    days = [("ABCBTC", f"2020-07-{1 + i:02d}") for i in range(MIN_EVENTS)]
    arms = {"all": [1] * MIN_EVENTS, "rules+stats": [0] * MIN_EVENTS}
    write(research, _ablation("semisyn", arms, days=days))
    data, _ = _report(research, tmp_path)
    (row,) = data["paired_ablation"]["rows"]
    assert row["n_clusters"] == MIN_EVENTS and row["reason_th"] is None
    assert row["diff_mean"] == 1.0 and row["ci"] == [1.0, 1.0] and row["significant"]


def test_paired_ablation_uses_the_latest_config_and_min_events(research, tmp_path):
    t = 0
    records = []
    for cfg_hash, det in (("old", [1] * 9), ("cfg", [0] * 9)):
        for r in _ablation("real_t001", {"all": [1] * 9, "rules": det}):
            t += 1
            r.update(config_hash=cfg_hash, created_us=t, run_id=r["run_id"] + cfg_hash)
            records.append(r)
    write(research, records)
    data, text = _report(research, tmp_path)
    pa = data["paired_ablation"]
    (row,) = pa["rows"]
    assert "ไม่นับ 4 run ของ config_hash เก่า (ใช้เฉพาะ config_hash ล่าสุด cfg)" in pa["notes"]
    assert row["diff_mean"] == 1.0  # the old config (diff 0) is not pooled in
    assert row["n_events"] == 9 and not row["significant"]
    assert row["reason_th"] == few_events_th(9)
    assert row["reason_th"] in _row(text, "| all vs real_t001:rules |")


def test_paired_ablation_without_the_full_arm_says_no_paired_bootstrap(research, tmp_path):
    write(research, _ablation("real_t001", {"rules": [1, 0]}))
    data, text = _report(research, tmp_path)
    pa = data["paired_ablation"]
    assert pa["rows"] == [] and any("real_t001:all" in n for n in pa["notes"])
    section = _section(text, "ablation")
    assert NO_PAIRED_ABLATION_TH in section and "| all vs" not in section


def test_no_ablation_records_no_ablation_section(research, tmp_path):
    write(research, [rec("real_t001", "full", 2, 0, [1, 0])])
    data, text = _report(research, tmp_path)
    assert data["paired_ablation"]["rows"] == [] and data["paired_ablation"]["notes"] == []
    assert "### ablation" not in text


def test_paired_ablation_skips_when_cluster_sizes_match_but_keys_differ(research, tmp_path):
    """Same event ids, one cluster each, but arm `rules` puts e1 on another day."""
    days_a = [("ABCBTC", "2020-07-01"), ("ABCBTC", "2020-07-02")]
    days_b = [("ABCBTC", "2020-07-01"), ("ABCBTC", "2020-07-03")]
    records = [
        rec("ablation", "semisyn:all", 2, 0, [1, 0], provenance="SEMI-SYN", days=days_a),
        rec("ablation", "semisyn:rules", 2, 0, [1, 0], provenance="SEMI-SYN", days=days_b),
    ]
    write(research, records)
    data, _ = _report(research, tmp_path)
    pa = data["paired_ablation"]
    assert pa["rows"] == []
    assert any("semisyn:rules" in n and "คนละ symbol-day" in n for n in pa["notes"])


# ---------------------------------------------------------------- FA CI and day counts (Part C)
def _fa_days(counts, prefix="ABCBTC|2020-08-"):
    return {f"{prefix}{i + 1:02d}": c for i, c in enumerate(counts)}


def test_fa_ci_bootstraps_seed_mean_per_symbol_day(research, tmp_path):
    normal = {0: [0, 2, 1, 0, 3], 1: [1, 0, 1, 2, 1]}
    hard = {0: [4, 0, 2], 1: [2, 2, 0]}
    records = []
    for seed in (0, 1):
        r = rec("real_t001", "full", 2, seed, [1, 0])
        r["fa_by_day"] = {"normal": _fa_days(normal[seed]), "hard": _fa_days(hard[seed], "X|")}
        r["metrics"].update(
            fa_per_symbol_day=sum(normal[seed]) / 5,
            fa_per_symbol_day_hard=sum(hard[seed]) / 3,
            n_fa_days=5,
            n_hard_days=3,
        )
        records.append(r)
    write(research, records)
    data, text = _report(research, tmp_path, n_boot=400)
    c = data["experiments"]["real_t001"]["full"]["2"]
    assert c["n_fa_days"] == 5 and c["n_hard_days"] == 3
    day_means = np.mean([normal[0], normal[1]], axis=0)
    mean, ci = bootstrap_ci(day_means, 400, 0)
    assert c["mean"]["fa_per_symbol_day"] == pytest.approx(mean)  # seed mean = day mean
    assert c["fa_ci"]["fa_per_symbol_day"] == pytest.approx(list(ci))
    _, ci_h = bootstrap_ci(np.mean([hard[0], hard[1]], axis=0), 400, 0)
    assert c["fa_ci"]["fa_per_symbol_day_hard"] == pytest.approx(list(ci_h))
    row = _row(text, "| full | 2 |")
    assert f"{mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (5 วัน)" in row
    assert f"[{ci_h[0]:.3f}, {ci_h[1]:.3f}] (3 วัน)" in row
    assert "FA/symbol-day [95% CI]" in text


def test_fa_without_fa_by_day_shows_the_mean_alone(research, tmp_path):
    new = rec("real_t001", "full", 2, 0, [1, 0], fa_by_day={"normal": _fa_days([1, 2]), "hard": {}})
    old = rec("real_t001", "full", 2, 1, [1, 0])  # older record: no fa_by_day, no n_*_days
    write(research, [new, old])
    data, text = _report(research, tmp_path)
    c = data["experiments"]["real_t001"]["full"]["2"]
    assert c["fa_ci"] == {"fa_per_symbol_day": None, "fa_per_symbol_day_hard": None}
    assert c["n_fa_days"] is None and c["n_hard_days"] is None
    assert "| 1.500 | 3.000 |" in _row(text, "| full | 2 |")


# ---------------------------------------------------------------- technique confusion (Part C)
def _tech_rec(experiment, variant, seed, events, provenance="REAL"):
    """`events`: list of (family, techniques)."""
    r = rec(experiment, variant, 2, seed, [bool(t) for _, t in events], provenance=provenance)
    for (fam, techs), pe in zip(events, r["per_event"].values(), strict=True):
        pe.update(family=fam, techniques=techs)
    return r


def test_confusion_table_shares_over_seeds_and_multi_technique_events(research, tmp_path):
    real = {
        0: [
            ("telegram_pump", ["MM-T001.2"]),
            ("telegram_pump", ["MM-T001", "MM-T000"]),  # counts in both columns
            ("telegram_pump", []),
            ("telegram_pump", ["MM-T003"]),
        ],
        1: [
            ("telegram_pump", ["MM-T001"]),
            ("telegram_pump", []),
            ("telegram_pump", []),
            ("telegram_pump", []),
        ],
    }
    records = [_tech_rec("real_t001", "full", s, ev) for s, ev in real.items()]
    records.append(
        _tech_rec("real_t001", "full", 0, real[0]) | {"budget": 10.0, "run_id": "b10"}
    )  # other budget: ignored
    for seed in (0, 1):
        records.append(
            _tech_rec("semisyn", "pump_burst", seed, [("pump_burst", ["MM-T001"])], "SEMI-SYN")
        )
        records.append(_tech_rec("semisyn", "wash_ring", seed, [("wash_ring", [])] * 2, "SEMI-SYN"))
    write(research, records)
    data, text = _report(research, tmp_path)
    cf = data["confusion"]
    rt = cf["real_t001 full"]
    assert rt["columns"] == ["MM-T000", "MM-T001", "MM-T003", "none"]
    (row,) = rt["rows"]
    assert row["family"] == "telegram_pump" and row["n_seeds"] == 2 and row["n_events"] == 4
    assert row["share"] == pytest.approx(
        {
            "MM-T001": (2 / 4 + 1 / 4) / 2,
            "MM-T000": 1 / 8,
            "MM-T003": 1 / 8,
            "none": (1 / 4 + 3 / 4) / 2,
        }
    )
    ss = cf["semisyn"]
    assert ss["columns"] == ["MM-T001", "none"]
    rows = {r["family"]: r["share"] for r in ss["rows"]}
    assert rows == {
        "pump_burst": {"MM-T001": 1.0, "none": 0.0},
        "wash_ring": {"MM-T001": 0.0, "none": 1.0},
    }
    section = _section(text, "confusion: family ของป้ายกำกับ × เทคนิคที่ระบุ (budget 2)")
    assert "| family | seeds | events | MM-T000 | MM-T001 | MM-T003 | none |" in section
    assert "| telegram_pump | 2 | 4 | 0.125 | 0.375 | 0.125 | 0.500 |" in section
    assert "นับในทุกคอลัมน์" in section


def test_confusion_needs_a_rerun_for_records_without_techniques(research, tmp_path):
    write(research, [rec("real_t001", "full", 2, 0, [1, 0])])
    data, text = _report(research, tmp_path)
    assert data["confusion"]["real_t001 full"]["rows"] == []
    assert data["confusion"]["semisyn"]["rows"] == []
    section = _section(text, "confusion: family ของป้ายกำกับ × เทคนิคที่ระบุ (budget 2)")
    assert section.count("ต้องรัน `idsai eval` ใหม่") == 2


# ---------------------------------------------------------------- FA multiplicity (M1 fix)
def test_fa_ci_divides_by_the_parts_that_loaded_each_day(research, tmp_path):
    """An ablation SEMI-SYN arm sums three family parts per symbol-day: without `normal_mult`
    its FA CI reads three times the rate; with it the CI is the ratio sum n / sum m."""
    rng = np.random.default_rng(0)
    n_days, n_parts = 12, 3
    days = [("ABCBTC", f"2020-07-{1 + i:02d}") for i in range(4)]
    counts = {s: rng.integers(0, 3, size=(n_parts, n_days)) for s in (0, 1)}
    records = {"legacy": [], "fixed": []}
    for kind, recs in records.items():
        for s in (0, 1):
            r = rec(
                "ablation", f"semisyn:{kind}", 2, s, [1, 0, 1, 0], provenance="SEMI-SYN", days=days
            )
            summed = counts[s].sum(axis=0)
            r["fa_by_day"] = {"normal": _fa_days(summed.tolist()), "hard": {}}
            if kind == "fixed":
                r["fa_by_day"]["normal_mult"] = _fa_days([n_parts] * n_days)
            r["metrics"]["fa_per_symbol_day"] = summed.sum() / (n_parts * n_days)
            recs.append(r)
    write(research, [*records["legacy"], *records["fixed"]])
    data, _ = _report(research, tmp_path, n_boot=400)
    cells = data["experiments"]["ablation"]
    legacy, fixed = cells["semisyn:legacy"]["2"], cells["semisyn:fixed"]["2"]
    rate = np.mean([counts[s].sum() / (n_parts * n_days) for s in (0, 1)])
    assert legacy["mean"]["fa_per_symbol_day"] == pytest.approx(rate)
    lo, hi = legacy["fa_ci"]["fa_per_symbol_day"]
    assert lo > 2 * rate  # the bug: the CI sits around 3x the rate it brackets
    n = counts[0].sum(axis=0) + counts[1].sum(axis=0)
    m = np.full(n_days, 2.0 * n_parts)
    want = ratio_bootstrap_ci(n.astype(float), m, 400, 0)
    assert want[0] == pytest.approx(rate)
    assert fixed["fa_ci"]["fa_per_symbol_day"] == pytest.approx(list(want[1]))
    lo, hi = fixed["fa_ci"]["fa_per_symbol_day"]
    assert lo <= rate <= hi


def test_fa_ci_multiplicity_defaults_to_one_for_missing_keys(research, tmp_path):
    r = rec("ablation", "semisyn:all", 2, 0, [1, 0], provenance="SEMI-SYN")
    r["fa_by_day"] = {"normal": _fa_days([3, 1, 2]), "hard": {}, "normal_mult": _fa_days([3])}
    write(research, [r])
    data, _ = _report(research, tmp_path, n_boot=200)
    got = data["experiments"]["ablation"]["semisyn:all"]["2"]["fa_ci"]["fa_per_symbol_day"]
    want = ratio_bootstrap_ci(np.array([3.0, 1.0, 2.0]), np.array([3.0, 1.0, 1.0]), 200, 0)
    assert got == pytest.approx(list(want[1]))


# ---------------------------------------------------------------- M1 selection
def _m1_row(variant, fa_hard, below=False):
    return {"variant": variant, "fa_hard": fa_hard, "below_v0": below}


def test_m1_select_lowest_fa_among_not_worse_with_simplicity_ties():
    rows = [
        _m1_row("all", 11.0),
        _m1_row("no_ai", 5.0),
        _m1_row("ai_shift1", 5.0),
        _m1_row("ai_shift2", 6.0),
        _m1_row("ai_mondrian", 5.0),
    ]
    assert m1_select(rows)["chosen"] == "no_ai"  # tie: V1 < V3 < V2
    assert m1_select([r for r in rows if r["variant"] != "no_ai"])["chosen"] == "ai_shift1"
    assert m1_select([rows[0], rows[4], _m1_row("ai_shift2", 5.0)])["chosen"] == "ai_shift2"
    assert m1_select([rows[0], rows[4]])["chosen"] == "ai_mondrian"
    # significantly lower recall: excluded even with the lowest FA
    sel = m1_select([*rows[:1], _m1_row("no_ai", 1.0, below=True), *rows[2:]])
    assert sel["chosen"] == "ai_shift1"
    assert "no_ai" in sel["excluded"] and "no_ai" not in sel["candidates"]
    # V0 wins a tie with any variant (nothing gained), and when all others are worse
    assert m1_select([_m1_row("all", 5.0), _m1_row("no_ai", 5.0)])["chosen"] == "all"
    only_v0 = m1_select([_m1_row("all", 9.0), _m1_row("no_ai", 1.0, below=True)])
    assert only_v0["chosen"] == "all"
    assert m1_select([_m1_row("no_ai", None)])["chosen"] is None


def _m1_recs(split, variant, detected, fa_hard, seeds=(0, 1)):
    out = []
    for s in seeds:
        r = rec("m1_fusion", f"{split}:{variant}", 2, s, detected)
        r["metrics"]["fa_per_symbol_day_hard"] = fa_hard
        out.append(r)
    return out


def test_m1_report_selects_on_accept_only(research, tmp_path):
    v0 = [1] * 16 + [0] * 4
    worse = [1] * 6 + [0] * 14  # significantly below V0
    records = [
        *_m1_recs("accept", "all", v0, 11.0),
        *_m1_recs("accept", "no_ai", v0, 6.0),
        *_m1_recs("accept", "ai_shift1", v0[:-1] + [1], 6.0),  # tie with no_ai on FA-hard
        *_m1_recs("accept", "ai_shift2", worse, 1.0),  # lowest FA but recall drops
        *_m1_recs("accept", "ai_mondrian", v0, 7.0),
        # test numbers would favour ai_mondrian: they must not change the choice
        *_m1_recs("test", "all", v0, 11.0),
        *_m1_recs("test", "no_ai", v0, 9.0),
        *_m1_recs("test", "ai_mondrian", v0, 0.5),
    ]
    write(research, records)
    data, text = _report(research, tmp_path)
    m1 = data["m1_fusion"]
    assert m1["selection"]["chosen"] == "no_ai"
    assert set(m1["selection"]["excluded"]) == {"ai_shift2"}
    acc = {r["variant"]: r for r in m1["splits"]["accept"]["rows"]}
    assert [r["variant"] for r in m1["splits"]["accept"]["rows"]] == [
        "all",
        "no_ai",
        "ai_shift1",
        "ai_shift2",
        "ai_mondrian",
    ]
    assert acc["ai_shift2"]["below_v0"] is True and acc["ai_shift2"]["vs_v0"]["diff_mean"] > 0
    assert acc["no_ai"]["vs_v0"]["diff_mean"] == 0.0 and acc["all"]["vs_v0"] is None
    assert acc["no_ai"]["fa_hard"] == 6.0
    assert set(m1["splits"]) == {"accept", "test"}
    section = _section(text, "m1_fusion")
    assert "### m1_fusion" in section and "#### accept (REAL)" in section
    assert "**ผลการเลือก (accept):** V1 no_ai" in section
    assert "ตัดออก V3 ai_shift2" in section
    assert "ไม่ใช่ความน่าจะเป็น" in section and "accuracy" not in section.lower()


def test_m1_report_without_v0_does_not_select(research, tmp_path):
    write(research, _m1_recs("accept", "no_ai", [1] * 12, 3.0))
    data, text = _report(research, tmp_path)
    assert data["m1_fusion"]["selection"] is None
    assert any("V0" in n for n in data["m1_fusion"]["notes"])


def test_m1_select_excludes_variants_whose_recall_vs_v0_is_undecided():
    rows = [
        _m1_row("all", 11.0),
        {**_m1_row("no_ai", 1.0), "undecided": True, "reason_th": "few"},
        _m1_row("ai_shift1", 6.0),
    ]
    sel = m1_select(rows)
    assert sel["chosen"] == "ai_shift1"  # no_ai cannot win on FA alone
    assert "no_ai" in sel["excluded"] and "few" in sel["excluded"]["no_ai"]
    # V0 always stays eligible, even if flagged
    only = m1_select([{**_m1_row("all", 9.0), "undecided": True}, rows[1]])
    assert only["chosen"] == "all" and "all" not in only["excluded"]


def test_m1_report_excludes_undecided_variants_with_a_thai_reason(research, tmp_path):
    few = [1, 0, 1, 0, 1]  # fewer than MIN_EVENTS events: the paired comparison is undecided
    write(
        research,
        [*_m1_recs("accept", "all", few, 11.0), *_m1_recs("accept", "no_ai", few, 1.0)],
    )
    data, text = _report(research, tmp_path)
    m1 = data["m1_fusion"]
    row = {r["variant"]: r for r in m1["splits"]["accept"]["rows"]}["no_ai"]
    assert row["undecided"] is True and row["below_v0"] is False
    assert m1["selection"]["chosen"] == "all"
    assert "no_ai" in m1["selection"]["excluded"]
    section = _section(text, "m1_fusion")
    assert "ตัดออก V1 no_ai: ตัดสิน recall เทียบ V0 ไม่ได้" in section
    assert "**ผลการเลือก (accept):** V0 all" in section


# ---------------------------------------------------------------- M4 sell side


def _m4_recs(variant, pump, sell, sell_ok, seeds=(0, 1), fa=1.0):
    """test:<variant> records; `sell[i]` None = event i has no sell window."""
    out = []
    for s in seeds:
        r = rec("m4_sell", f"test:{variant}", 2, s, pump, sell_side=variant == "sell")
        r["metrics"]["fa_per_symbol_day"] = fa
        for i, pe in enumerate(r["per_event"].values()):
            pe["sell_status"] = "measured" if sell[i] is not None else "no_rise"
            pe["detected_sell_phase"] = sell[i]
            pe["detected_sell_phase_correct"] = None if sell[i] is None else sell_ok[i]
        n_none = sum(x is None for x in sell)
        r["metrics"]["sell_status_counts"] = {"measured": len(sell) - n_none, "no_rise": n_none}
        out.append(r)
    return out


def _raid_recs(variant, detected, seeds=(0, 1)):
    days = [("ABCBTC", f"2021-01-{1 + i:02d}") for i in range(len(detected))]
    return [
        rec("m4_sell", f"bear_raid:{variant}", 2, s, detected, provenance="SEMI-SYN", days=days)
        for s in seeds
    ]


def test_m4_report_section_per_phase_and_paired_pump_recall(research, tmp_path):
    pump = [1] * 15 + [0] * 5
    sell = [True] * 6 + [False] * 10 + [None] * 4
    records = [
        *_m4_recs("base", pump, sell, [False] * 20),
        *_m4_recs(
            "sell", pump, [True] * 12 + [False] * 4 + [None] * 4, [True] * 10 + [False] * 10, fa=1.2
        ),  # fmt: skip
        *_raid_recs("base", [0] * 12),
        *_raid_recs("sell", [1] * 9 + [0] * 3),
    ]
    write(research, records)
    data, text = _report(research, tmp_path)
    m4 = data["m4_sell"]
    rows = {r["variant"]: r for r in m4["rows"]}
    assert [r["variant"] for r in m4["rows"]] == ["base", "sell"]
    b, s = rows["base"], rows["sell"]
    assert b["recall_pump_phase"] == pytest.approx(0.75) and b["recall_pump_phase_ci"]
    assert b["n_sell_events"] == 16  # events without a sell window are left out
    assert b["recall_sell_phase"] == pytest.approx(6 / 16)
    assert s["recall_sell_phase"] == pytest.approx(12 / 16)
    assert b["recall_sell_phase_correct"] == 0.0
    assert s["recall_sell_phase_correct"] == pytest.approx(10 / 16)
    assert s["fa"] == pytest.approx(1.2) and b["fa_hard"] == pytest.approx(3.0)
    assert b["sell_status_counts"] == {"measured": 16, "no_rise": 4}
    assert s["bear_raid_recall_any"] == pytest.approx(0.75) and s["bear_raid_recall_any_ci"]
    assert b["bear_raid_recall_any"] == 0.0 and s["bear_raid_provenance"] == "SEMI-SYN"
    p = m4["paired_pump"]
    assert p["diff_mean"] == 0.0 and p["drop"] is False and p["n_clusters"] == 20
    section = _section(text, "m4_sell")
    assert "### m4_sell" in section and "recall ช่วงเทขาย" in section
    assert "recall ช่วงพุ่งไม่ลดลงอย่างมีนัย" in section
    assert "ความผิดปกติทางสถิติ" in section and "ไม่ใช่ความน่าจะเป็น" in section
    assert "SEMI-SYN" in section and "accuracy" not in section.lower()


def test_m4_report_flags_a_significant_pump_recall_drop(research, tmp_path):
    sell = [True] * 20
    write(
        research,
        [
            *_m4_recs("base", [1] * 18 + [0] * 2, sell, sell),
            *_m4_recs("sell", [1] * 8 + [0] * 12, sell, sell),
        ],
    )
    data, text = _report(research, tmp_path)
    p = data["m4_sell"]["paired_pump"]
    assert p["diff_mean"] == pytest.approx(-0.5) and p["drop"] is True
    assert "กฎฝั่งขายทำให้ recall ช่วงพุ่งลดลงอย่างมีนัย" in _section(text, "m4_sell")


def test_m4_report_without_records(research, tmp_path):
    write(research, [rec("real_t001", "full", 2, 0, [1, 0])])
    data, _ = _report(research, tmp_path)
    assert data["m4_sell"] is None


def test_m4_report_paired_sell_phase_differences(research, tmp_path):
    pump = [1] * 20
    sell_base = [False] * 18 + [None] * 2
    sell_sell = [True] * 15 + [False] * 3 + [None] * 2
    write(
        research,
        [
            *_m4_recs("base", pump, sell_base, [False] * 20),
            *_m4_recs("sell", pump, sell_sell, [True] * 15 + [False] * 5),
        ],
    )
    data, text = _report(research, tmp_path)
    paired = data["m4_sell"]["paired"]
    assert set(paired) == {"recall_pump_phase", "recall_sell_phase", "recall_sell_phase_correct"}
    sp = paired["recall_sell_phase"]
    assert sp["n_clusters"] == 18  # events without a sell window are left out
    assert sp["diff_mean"] == pytest.approx(15 / 18) and sp["rise"] is True
    assert paired["recall_sell_phase_correct"]["diff_mean"] == pytest.approx(15 / 18)
    assert paired["recall_pump_phase"]["diff_mean"] == 0.0
    section = _section(text, "m4_sell")
    assert "ผลต่าง recall ช่วงเทขาย (sell − base)" in section
    assert "recall ช่วงเทขายสูงขึ้นอย่างมีนัย" in section
    assert "ผลต่าง ระบุช่วงเทขายถูก (sell − base)" in section
