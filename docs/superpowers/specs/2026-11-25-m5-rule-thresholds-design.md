# M5: reachable pump-rule thresholds (100101 rev 2, 100102 rev 2)

Sources: the M3 head diagnosis (train range only), REPORT §12, `configs/signatures.yaml`. Branch `feat/m5`, cut from `main` (eeb3e7e). The live stack is rules + stat.robust_z + stat.cusum + ai.iforest (AI shift 1), with `sell_side` and `attr_rules` on.

## Why
- After M3, naming is nearly saturated: on test, REAL MM-T000 is 0.001. A cusum-attribution change (`attribution_min`) would only name more false alarms, so **it is dropped from this milestone**. Record that in the report.
- The remaining lever is detection. Rule 100101 (burst) cannot fire on these pairs: `z_quote_vol_60 >= 6` is out of reach (per-event window max median 3.2 on 183 train pumps, 0/183 fire). Rule 100102 fails mostly on `cusum_imb >= 8` (window max median 6.0).
- Diagnosis numbers (train range, mini-Chief without AI and without attr):
  - 100101 at 3: recall_any 0.814 → 0.858, FA +1.4 %.
  - 100101 at 2.5: FA +9 %.
  - With 100102 `cusum_imb` at 5 added: FA about +10 %.
  - These must be re-measured with the live stack.

## Goal
Raise recall_any (REAL and SEMI-SYN) at budget 2 with FA and FA-hard at most +2 % over base. Rules enter F, so FA can move. τ is refit per variant on cal as usual, which keeps the per-budget FA control.

## Changes
1. `signatures.yaml` gains rev-2 alternatives for 100101 and 100102, loaded by a flag. Do not edit rev 1 in place.
   - Suggested: `configs/signatures_r2.yaml` or a variant mechanism matching how M3 did `signatures_wash2.yaml`. Reuse that mechanism.
   - Rev history goes in comments, with the train fire rates.
2. Thresholds are chosen on **train only** before the eval and frozen in the spec's "Deviations/Real run" section, with their train fire rates. Measure:
   - the row fire rate on train-normal days;
   - the per-event fire count on train REAL pumps.
   - Candidates:
     - 100101 `z_quote_vol_60` ∈ {3, 2.5}
     - 100102 `cusum_imb` ∈ {6, 5}
   - Keep the other conditions unchanged.
   - Pick at most 2 values per rule for the eval. Prefer a row fire rate on train-normal ≤ 1e-3.
3. Off by default: `chief.rule_rev2` or an equivalent. With the flag off, output is bit-identical and the flag is absent from config dumps, using the same hash trick as `sell_side` and `attr_rules`. Pin:
   - the golden;
   - `eval_prev_live.yaml` hash 53e3b662ab96;
   - the current default hash 9aff5e3482f2.

## Experiment `m5_rules`
- Per seed, fit and score once on the live stack. Re-score the rule layer per variant if that is cheap; otherwise refit per variant and state the cost.
- Variants (names illustrative):
  - `base`
  - `r101` (100101 rev 2)
  - `r102` (100102 rev 2)
  - `r101+r102`
  - optionally one more 100101 value
- Splits: accept, test, semisyn (test only). Records as in M2 and M3: recall_any, recall_correct [CI], FA [CI], FA hard, the named-FA share, confusion.
- **Selection rule (accept only, fixed in advance):**
  - Eligible: FA and FA-hard ≤ 1.02 × base at budget 2.
  - Pick the largest significant paired gain in recall_any (CI low > 0 and ≥ min_effect 0.02).
  - Ties go to the simpler variant: base < r101 < r102 < r101+r102.
  - No significant gain → base.
- Report: a Thai table at budget 2 (accept/test/semisyn), the paired deltas, and confusion. Test and SEMI-SYN are reported once.

## Tests (synthetic only)
- The flag off is bit-identical (golden).
- Rev-2 rules fire at the new thresholds and not below.
- Parity and state round-trip.
- `m5_rules` smoke test: fit counted once per seed; selection rule on synthetic records.

## Run
`idsai --config configs/eval_prev_live.yaml eval --experiment m5_rules`, after review, by the head, overnight. Expected 6–9 h.

## Out of scope
- `attribution_min` and the cusum cap: dropped (naming is saturated).
- wash rev 2: needs independent validation.
- stealth naming.

## Real run / deviations

### Train-only threshold check (2026-11-25, before any eval)
Data: train range 2018-01-01..2019-06-30 only, feature cache read only (rule columns are the same with and without `sell_side`); test, accept and `data/research.duckdb` untouched. Script: scratchpad `m5/m5_train_fire.py`.
- Train-normal rows: 3,075,840 rows on 356 train-range symbol-days at least `min_gap_days` (2) from every label of the symbol. "Fire episodes" = runs of firing rows split by gaps > 300 s.
- Train REAL pumps: 183 of 188 train labels (Sapienza + Hu) with cached rows, rows in [t_start − 120 s, t_start + 300 s].

| rule, threshold | rows fired | row fire rate | fire episodes / symbol-day | train REAL pumps fired |
|---|---|---|---|---|
| 100101 z_quote_vol_60 ≥ 6 (rev 1) | 0 | 0 | 0 | 0/183 |
| 100101 ≥ 3 | 58 | 1.89e-5 | 0.034 | 105/183 |
| 100101 ≥ 2.5 | 202 | 6.57e-5 | 0.149 | 122/183 |
| 100102 cusum_imb ≥ 8 (rev 1) | 5,862 | 1.91e-3 | 0.848 | 42/183 |
| 100102 ≥ 6 | 7,013 | 2.28e-3 | 0.963 | 59/183 |
| 100102 ≥ 5 | 7,670 | 2.49e-3 | 1.048 | 72/183 |

### Frozen choices
- **100101 rev 2: z_quote_vol_60 ≥ 3** (`configs/signatures_r2.yaml`, variant `r101`), plus the alternative **2.5** (`configs/signatures_r2_alt.yaml`, eval only, variant `r101_25`). Both are far below the 1e-3 row-rate preference.
- **100102 rev 2: cusum_imb ≥ 6** (variant `r102`; with 100101 at 3: `r101+r102`). Deviation: no 100102 candidate meets the ≤ 1e-3 preference, because rev 1 itself already fires on 1.91e-3 of train-normal rows. 6 adds fewer rows (+1,151, +3.7e-4) than 5 (+1,808); 5 is not evaluated.
- Tie order for the selection: base < r101 < r101_25 < r102 < r101+r102.

### Implementation notes
- Flag: `chief.rule_rev2: list[int]` (e.g. `[100101, 100102]`). Each listed sid's rev 2 replaces its rev 1 under the same detector name `rule.<sid>`. Empty (default) is absent from dumps: default hash 9aff5e3482f2 and `eval_prev_live.yaml` 53e3b662ab96 unchanged, golden digests unchanged. A non-empty flag changes `config_hash`, so bundles must be retrained.
- `m5_rules` base = today's live stack: `features.sell_side` on (sell rules 100104/100601 in F), naming rules on in every variant, `ai.iforest` shift 1, all rules at rev 1. The eval config keeps its `config_hash`.
- One `ai.iforest` fit per seed and one scoring pass per set (the three candidates are extra rule columns). Variants only re-fuse, with their own calibration and τ (5 variants × 3 splits = 15 Chief fits per seed).
- Expected run cost: like `m3_attr` per seed. The first seed also computes the sell-side feature cache for the accept days (456 symbol-days not yet cached under that feature hash; train-normal, cal and test already are). Seeds 1–4 reuse it.

### Real run / results (2026-11-25, `m5_rules`, 5 seeds, budget 2)
Command as in "Run". Full tables with CIs: `data/report.md` section `m5_rules` (REPORT §13 and its appendix).

| split | variant | recall_any | Δ recall_any vs base [95% CI] | FA/symbol-day | FA hard |
|---|---|---|---|---|---|
| accept (94 events) | base | 0.806 | – | 2.517 | 7.700 |
| | r101 | 0.806 | +0.000 [0.000, 0.000] | 2.534 | 7.750 |
| | r101_25 | 0.817 | +0.011 [0.000, 0.032] | 2.592 | 7.800 |
| | r102 | 0.817 | +0.011 [0.000, 0.032] | 2.529 | 8.050 |
| | r101+r102 | 0.817 | +0.011 [0.000, 0.032] | 2.546 | 8.100 |
| test (200 events, report only) | base | 0.789 | – | 1.806 | 5.850 |
| | r101 | 0.798 | +0.009 [0.000, 0.023] | 1.818 | 5.850 |
| semisyn (report only) | base | 0.730 | – | 1.581 | – |
| | r101_25 | 0.737 | +0.007 [0.005, 0.010] | 1.661 | – |
| | r102 | 0.714 | −0.016 [−0.021, −0.012] | 1.582 | – |
| | r101+r102 | 0.715 | −0.015 [−0.020, −0.011] | 1.594 | – |

- **Selection (accept): base.** Excluded: r101_25 (FA 2.59 > 1.02 × 2.52), r102 (FA-hard 8.05 > 1.02 × 7.7), r101+r102 (FA-hard 8.1). r101 is eligible but its gain is +0.000, so no significant gain → base. The live config is unchanged (no `chief.rule_rev2` in `configs/default.yaml`; default hash 9aff5e3482f2, `eval_prev_live.yaml` 53e3b662ab96).
- Test: r101 +0.009 [0.000, 0.023] at FA 1.818 vs 1.806, not significant. SEMI-SYN: r102 and r101+r102 are significantly worse, by less than min_effect (pump_burst missed 0.348 → 0.397); r101_25 is significantly better by +0.007, below min_effect, at FA +5 %.
- Reading: the unreachable thresholds were not the binding constraint once the live stack (stat + iforest + attr) is in place; the pumps rev 2 catches are already caught by other detectors. Remaining misses (~20 % of Telegram pumps) are not reached by loosening these two rules within the FA budget.
- `attribution_min` was not changed (naming is saturated; noted in the report).
- **Promotion would need retraining.** A non-empty `chief.rule_rev2` changes `config_hash`, so every bundle must be retrained, and τ is calibrated with the rule set in F. `bundle.json` holds `config_hash` but not the sha of `configs/signatures_r2.yaml`, so that sha should be added to the bundle meta before any promotion (an edit to the rev 2 file would otherwise go undetected). Not done, since base was kept.
- Review follow-ups: `chief.rule_rev2` is sorted on load so the dump and hash do not depend on the written order (repeats are still rejected); the report reads a paired CI wholly below / above 0 that misses `min_effect` as a significant decrease / increase smaller than `min_effect`, not as "no difference" (display only; selection unchanged).
