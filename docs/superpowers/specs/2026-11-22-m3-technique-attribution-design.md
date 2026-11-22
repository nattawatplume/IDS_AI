# M3: name the technique of detected events

Source: `docs/ROADMAP.md` M3, the confusion table in `docs/REPORT.md`, and the head's train-only diagnosis (summarised below). Branch `feat/m3`, cut from `feat/live-ai`, so the live stack is rules + stat.robust_z + stat.cusum + ai.iforest with AI shift 1.

**Diagnosis (train range only; no test or accept numbers were used).**
- Only rules and `stat.cusum` can name a technique (`idsai/ai/chief.py` ~418-428). An episode is named from its highest-F classified row (~594-599).
- Rule 100101 (pump burst) fires on 0 of 183 train REAL pumps: `z_quote_vol_60 >= 6` is out of reach on these pairs (per-event max median 3.2).
- `stat.cusum` can never name: its score is capped at 100, so the best -log10 p is about 2.84, below `attribution_min` 3.0.
- Wash rule 100501 fires on 0 of 24 train-range wash injections: `z_quote_vol_300 >= 3` never holds (max ≈ 1–1.6). `z_pingpong_300 >= 5` alone fires on 0 of 1.38M train-normal rows.
- Without AI: recall_correct 0.208, and 0.607 of events are detected only as MM-T000.

## Goal
Raise recall_correct (REAL and SEMI-SYN) with **no change in FA and no change in recall_any**, by changing only how an already-detected episode is named. Tuning uses train data only. The choice of variant is made on accept by a rule fixed in advance. Test and SEMI-SYN are reported once.

## Changes
1. **Naming-only signatures** (`configs/signatures_attr.yaml`, new; loaded by the Chief).
   - These rules never enter F, p-values, agreement or τ, so the set of rows over τ is identical by construction.
   - They are consulted only for an episode whose exceedance rows include no row named by an F-rule or by cusum. Scan that episode's rows (the same rows the episode spans) and take the first naming rule that matches, in a fixed priority order.
   - **Initial rules** (thresholds from the diagnosis; re-check on train before freezing, and write the train fire rates in comments):
     - pump burst (MM-T001.x): `ret_60 >= 0.02 and z_rush_60 >= 4` (≈0.019 % of train-normal rows)
     - wash (MM-T005.x): `z_pingpong_300 >= 5 and abs(ret_300) <= 0.002` (0 train-normal rows)
     - Do **not** add the stealth variant: it fires on 0.8 % of rows and only names more false alarms.
   - Off by flag `chief.attr_rules: bool = False`. When off, every output is bit-identical and the flag is absent from config dumps (the same hash trick as `features.sell_side`). Add a golden test.
   - Batch/stream parity must hold. Extend the parity and state tests.
   - Evidence and the UI say the name came from a "naming rule" (กฎระบุเทคนิค), separate from the detection evidence.
2. **Wash signature 100501 rev 2:** drop the `z_quote_vol_300` condition. Verify on train that it still fires on ~0 train-normal rows and on no train REAL-pump day. Keep the rev history in a comment. Caveat: our own injector defines wash, so tuning against it is partly circular. The real-data FA check is the guard.
3. **No change to 100101 or to `attribution_min`** in this milestone (both change F or naming of FAs). List them as follow-ups.

## Experiment `m3_attr`
- Per seed: fit and score once (live stack). Variants:
  - `base`
  - `attr` (naming rules on)
  - `wash2` (100501 rev 2)
  - `attr+wash2`
- Note: `wash2` changes an F-rule, so its FA can move. Report it.
- Splits: accept, test, semisyn (test only). Records as in M1 and M2.
- Report: a Thai table at budget 2 with:
  - recall_any, recall_correct [CI], FA [CI], FA hard;
  - **the share of FA episodes that got a name** (naming a false alarm is a cost);
  - the confusion table per family.
- **Selection rule (accept only, fixed in advance):**
  - Eligible: FA and FA-hard ≤ 1.02 × base (the naming-only variants are equal by construction), and recall_any not significantly lower.
  - Pick the variant with the largest significant paired gain in recall_correct. Ties go to the simpler variant: base < attr < wash2 < attr+wash2.
  - Accept has only REAL pumps, so wash naming is judged on SEMI-SYN test as report-only.
- Guard against trivial naming: the report also shows recall_correct for a "name every detected episode MM-T001" strawman on REAL, so readers can see that REAL alone cannot tell good naming from blanket naming. The SEMI-SYN confusion table (wash must not be called pump, and pump must not be called wash) is the real check.

## Tests (synthetic only)
- The flag off is bit-identical (golden).
- A naming rule names only episodes left MM-T000 and never changes F, τ, alerts or agreement.
- Priority order. Parity. State round-trip.
- `m3_attr` smoke test (fit counted once per seed). Selection rule on synthetic records.

## Run
`idsai --config configs/eval_prev_live.yaml eval --experiment m3_attr`, after review, by the head. This eval config keeps its hash; the live stack is built inside the experiment.

## Out of scope
- Changing detection thresholds that enter F (100101, cusum cap, `attribution_min`). Those are follow-ups.
- Live deployment: turning `attr_rules` on in `configs/default.yaml` is a separate step after the results.
