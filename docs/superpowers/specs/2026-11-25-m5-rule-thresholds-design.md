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
