# Phase 0 + M0: alert-to-peak timing, post-alert price, report gaps

Source: `docs/ROADMAP.md` (Phase 0, M0). Offline and read-only toward `data/app.duckdb`. No live-system change. No Hu data. Every number is labelled with its provenance (REAL / SEMI-SYN), and the wording follows the usual rules: a statistical anomaly, not an accusation; F is not a probability; no accuracy metric.

## What already exists (no rerun needed)
- `data/research.duckdb` table `runs`: 5 seeds × 4 budgets per variant.
  - `record_json.per_event[label_id] = {symbol, day, family, detected_any, detected_correct, ttd_s}`.
  - `ttd_s = (first episode row t_end_us − label t_start_us)/1e6`, where `first_us` is the alert onset.
- `scripts/measure_pumps.py::measure_label(data_dir, symbol, t0)` returns `t_peak_us`, `p_ref`, `impact`, `dump_tau_s` and `status` from the stored trades in `data/trades/…`. Today it is only called for train-period labels, but the function works for any label.
- `scripts/measure_pumps.py::read_trades(data_dir, symbol, start_us, end_us)`.
- `idsai/pipeline/stream.py` already records `latency_ms` per processed bar.

## Part A: `scripts/post_alert_outcomes.py` (Phase 0.1 + 0.2)

**Input**
- Runs of experiment `real_t001`, variant `full`, budget `--budget` (default 2), all seeds, read-only from `data/research.duckdb`. If several config_hash values exist, use only the most recent one, as `report._latest_config` does.
- Label `t_start_us` comes from the labels in the research DB (or from the label files, as in `measure_pumps.load_labels`).

**Per (label, seed) where `detected_any`**
- `alert_us = t0 + ttd_s·1e6`.

**Per label, independent of seed**
- Take `measure_label(...)`.
- Keep `status == "measured"` for the peak statistics. Count every other status by its reason.

**Fields**
- `lead_to_peak_s = (t_peak_us − alert_us)/1e6`. A positive value means the alert came before the peak.
- `p_alert`: the price of the last trade at or before `alert_us`.
- `p_peak`: the price at the peak.
- `ret_h = p(alert_us + h)/p_alert − 1` for h ∈ {15 min, 1 h, 24 h}. `p(t)` is the last trade price at or before t. If there is no trade in the hour before t, the value is missing and is counted.
- `ret_peak = p_peak/p_alert − 1`: the most a buyer at alert time could still have gained.
- Report the following two values next to each other, not one alone:
  - `drawdown_h = min price in (alert_us, alert_us+h] / p_alert − 1`
  - `max_up_h`, defined the same way with the max

**Summary (JSON + Markdown, deterministic, same inputs give the same bytes)**
- `n_labels`, `n_detected`, `n_measured`, and the exclusion counts.
- `frac_alert_before_peak`: the mean over (label, seed) pairs, plus an event-bootstrap 95 % CI using `idsai.eval.metrics.bootstrap_ci` on an (n_seeds, n_events) matrix. An event that a seed did not detect is left out of that seed's row. The rule for that must be written down. If it is simpler, use a per-label seed-mean and bootstrap over labels; whichever is used, write it in the output.
- Quantiles (10/50/90) of `lead_to_peak_s`, `ret_peak`, `ret_15m`, `ret_1h`, `ret_24h`, `drawdown_1h`, `max_up_1h`, and `frac(ret_1h < 0)` with a CI.
- The per-seed median of `lead_to_peak_s` (stability).

**Output**
- `data/post_alert_outcomes.json` and `.md` (both gitignored under `data/`), printed as summary lines.

**Caveats to print in the output**
- The labels are announcement times (BTC pairs, 2018–2021).
- Alert time ignores the 1 s bar close and the 10 s stride, the same note as for TTD.
- These are historical statistics, not advice.

**Tests**
- `tests/test_post_alert_outcomes.py` with synthetic trades, a tmp dir, and a tmp DuckDB. It must never touch real `data/`.
- Cover: lead sign, `p(t)` lookup edges, the missing-trade path, determinism, and skipping undetected events.

## Part B: ablation paired bootstrap (M0, report only)

In `idsai/eval/report.py`, add `_paired_ablation(runs, n_boot, min_effect)` at budget 2:
- **REAL:** `real_t001:all` vs each of `rules+stats`, `rules`, `stats`, `ai`.
- **SEMI-SYN:** the same for `semisyn:*`. Clusters follow the existing `_bootstrap_unit` and `cluster_counts` rules for injected events, so reuse what `_cell` does for SEMI-SYN. If cluster-aware pairing is not available, restrict this block to REAL and say so in its notes.

Other requirements:
- Mirror `_paired` (latest config_hash only, intersect seeds and events, `MIN_EVENTS`, `significant`).
- Add it to the JSON as `paired_ablation` and as a Markdown table under the ablation section, written in Thai.
- Tests: extend `tests/` for the report, using synthetic run records.

## Part C: richer run records (M0, needs a rerun)

In `idsai/eval/experiments.py::_evaluate`:
- `per_event[lid]["techniques"]`: the sorted unique `technique_id` values of the matched alert episodes. This needs `match_events` to also return them, through a new optional field on `MatchResult` with an empty-dict default, so other callers stay unaffected.
- `metrics["n_hard_days"] = n_h`, and `metrics["n_fa_days"] = fa_days`.
- `fa_by_day`: per symbol-day unmatched-alert counts, both normal and hard, as `{"SYM|YYYY-MM-DD": n}`. These go at the record level next to `per_event`, not inside `metrics`.

In `idsai/eval/report.py`:
- FA and FA-hard 95 % CI: bootstrap over symbol-days, pooled across seeds as a seed-mean per day, then resample days. Shown in the table cell as `x [lo, hi]` when `fa_by_day` is present, otherwise as before.
- `n_hard_days` is shown in the summary.
- **Confusion table per family** (REAL `real_t001 full` and `semisyn` families, budget 2):
  - Rows are label families. Columns are the predicted parent technique among the matched episodes, plus "none".
  - Count share = the mean over seeds.
  - An event with several techniques counts once for each technique, and this must be stated.

**Rerun**
- `idsai eval --experiment real_t001` and `--experiment semisyn` (about 4 h and 10 h historically).
- Ablation and baselines can stay as they are.
- Copy `data/research.duckdb` to a backup first.
- After the rerun, `idsai report` rebuilds `docs/REPORT.md`, keeping the mixed-config note.

**Tests**
- `techniques` on `match_events`, `fa_by_day` counts, the FA CI on synthetic records, and the confusion table.

## Part D: latency p99 (M0)

`scripts/bench_latency.py`:
- Replay one stored test symbol-day through the stream pipeline at full speed (`speed 0`).
- Collect `latency_ms`, then print the median, p99, max and the number of bars. Output goes to `data/latency.json`.
- If the replay at speed 0 cannot reach `latency_ms` without starting the API, document it and measure through the same code path the stream uses.

## Out of scope
- ROADMAP 0.3 (market-wide hard-day FA). It needs hard-day alert episodes, which Part C does not store. That moves to M1.
- Any change to the detectors or thresholds.
