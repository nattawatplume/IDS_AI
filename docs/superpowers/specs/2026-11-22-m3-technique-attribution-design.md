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

## Results / deviations (implementation, 2026-11-23)

**Train re-check** (train range 2018-01..2019-06 only; feature cache and raw trades; no cal, accept or test; research.duckdb never opened). Sample as the head's diagnosis: 160 train-normal days at least 2 days from any label (1,382,400 rows, seed 1), the 183 train REAL pumps, wash_pingpong injected into 24 train-normal days.
- Burst naming rule 100191 (`ret_60 >= 0.02`, `z_rush_60 >= 4`): fires on 231 train-normal rows (0.017 %; the diagnosis said 0.019 % on its sample) and in the match window of 161 of 183 REAL pumps.
- Wash naming rule 100591 and 100501 rev 2 (same two conditions: `z_pingpong_300 >= 5`, `|ret_300| <= 0.002`): 0 train-normal rows, 0 rows on the REAL pump days, 24 of 24 injections (train and test scenario ranges alike). 100501 rev 1 fires in 0 of 24.
- The rates are written in the YAML comments.

**Choices.**
- Naming rules live in `configs/signatures_attr.yaml` with a new signature layer `attr` (sids 100191, 100591; priority = file order, burst first). `naming_detectors(cfg)` returns them only with `chief.attr_rules`.
- The Chief takes them as a separate `naming=` list, not as detectors: no calibration, no `p__` column, not in `F`/`best`/`agreement`/`tau` or the state (`state_dict` is byte-identical with or without them). `Chief.score_detectors` lists what a caller must score; the rules are read from their `score__attr.<sid>` columns, because eval frames keep only scores.
- Span = every row of the symbol from the episode's first to its last exceedance (as the diagnosis measured, V7), not only exceedance rows. The stream keeps the naming hits up to the last exceedance plus "pending" hits after it; pending hits count only if a later exceedance extends the episode, so the closed alert equals the batch one. A name that arrives emits `updated`.
- `Alert.technique_source = "naming_rule"` (absent from dumps when None, like `phase`); `signature_id`/`rev` are the naming rule's; `peak_technique_id`, `fused_p`, detectors, agreement and evidence stay the detection's. Explainer: "ชื่อเทคนิคมาจากกฎระบุเทคนิค ซึ่งไม่ใช่หลักฐานการตรวจจับ …" in place of the other-row clause; UI badge "ชื่อเทคนิคจากกฎระบุเทคนิค (ไม่ใช่หลักฐานการตรวจจับ)".
- Live wiring (flag still off in `configs/default.yaml`): `StreamPipeline` scores `chief.naming` too; `load_bundle` rebuilds the naming rules from config (they hold no state; the flag changes `config_hash`, so a bundle must be retrained when it is switched on).
- **Deviation:** 100501 rev 2 is a separate file `configs/signatures_wash2.yaml` read only by `m3_attr` (scored as `rule.100501.rev2`), so `signatures.yaml`, its recorded sha and the live wash rule stay rev 1 until the results. Promoting it is the live step.
- **Selection reading:** "largest significant paired gain" = among eligible variants whose paired recall_correct gain over base is significant, the largest wins (ties base < attr < wash2 < attr+wash2); none → base. Eligibility: FA and FA-hard each ≤ 1.02 × base, recall_any not significantly below base (paired).
- Report (`### m3_attr`, budget 2, per split): recall_any / recall_correct with CI, FA / FA-hard with CI, share of FA episodes named (and named by a naming rule), the "every detected episode = MM-T001" strawman, paired recall_correct and recall_any vs base, and a confusion table per variant for test and semisyn.

**Proof that F/τ/alerts are unchanged with the flag on** (tests): golden hash of flag-off outputs pinned from the pre-M3 Chief; with naming on, `fuse` output equal, alert ids/windows/fused_p/detectors/agreement/evidence/severity equal and only MM-T000 episodes renamed (`tests/ai/test_chief_naming.py`); full stream pipeline replay with naming equal to batch and to the flag-off run except names (`tests/pipeline/test_stream.py`); `m3_attr` smoke: `attr` has base's tau, n_cal, fa_by_day, FA, FA-hard, recall_any, n_alerts and per-event detected_any exactly.

**Expected run cost** of `idsai --config configs/eval_prev_live.yaml eval --experiment m3_attr` (5 seeds): per seed one `ai.iforest` fit and one scoring pass over cal, accept, test, test-hard and the three SEMI-SYN families (the live stack plus three cheap rule columns; no autoencoder training), then 12 Chief fits (4 variants × 3 splits) with 4 budgets each. Clearly cheaper than `m2_ai` (three AE trainings and 16 variants per seed); about m1_fusion's loading time without its AE fit.

**Follow-ups (unchanged here):** 100101 `z_quote_vol_60 >= 6` (unreachable on these pairs), the cusum cap vs `attribution_min`, turning `chief.attr_rules` on in `configs/default.yaml`, promoting 100501 rev 2.
