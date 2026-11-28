# M8: rule 100102 (stealth): naming only, or with a trading-activity gate

Source: M7 (REPORT §15, ROADMAP item 8). On train, 100102 drives F at the peak of 0.49 of the FA and 0.78 of the hard-day FA. Dropping it from F gave:
- FA: 385 → 216;
- FA_hard: 63 → 18;
- DET: 160 → 160;
- correctly named: 108 → 89.

M7 pre-registered two variants. Branch `feat/m8-stealth`, cut from `feat/m7-diag` (21b6302; PR #33 is open). The pattern follows M5 (`rule_rev2`, `configs/signatures_r2.yaml`, the `m5_rules` experiment, `docs/superpowers/specs/2026-11-25-m5-rule-thresholds-design.md`). Read those first.

## Variants (fixed in advance; nothing else is tried)
- **base:** today's live stack: rules + robust_z + cusum + iforest shift 1, `sell_side` and `attr_rules` on, budget 2.
- **v1 `stealth_attr`:** 100102 leaves the F rule layer and becomes a naming-only rule in the M3 `attr` layer. It can name a technique but never adds to F.
- **v2 `stealth_act`:** 100102 rev 3 = rev 1 conditions plus `z_n_trades_60 >= 3`.
  - The threshold 3 is frozen now. It came from train only: the M7 FA onset median is 2.40, and the DET Q1 is 3.53.
  - Record the row fire rate on train and the per-event fire on train pumps in the YAML header, as M5 did for rev 2. The train range only, read from the feature cache.

## Config (hash-safe, like `sell_side`, `attr_rules` and `rule_rev2`)
- One new optional chief setting, e.g. `chief.stealth: "" | "attr" | "act"`, or an equivalent clean design. Rev 3 lives in `configs/signatures_r3.yaml`, with its sha in the bundle meta when it is used.
- It is absent from dumps when off. Output is bit-identical when off and the golden test does not change. Pin both hashes: default 9aff5e3482f2 and `eval_prev_live.yaml` 53e3b662ab96.
- Validate the conflicts: `act` together with `rule_rev2` containing 102 is an error. `attr` needs `attr_rules` on, or say clearly what happens.
- Live and replay use the flag through the registry, like `rule_rev2`. Turning it on in `configs/default.yaml` is a later, separate step and needs the user's OK. Do not do it.

## Experiment `m8_stealth`
- Variants: base, v1, v2. Splits: accept (with accept-hard as in M5), test and semisyn. Records match M5:
  - recall_any, recall_correct;
  - FA, FA hard;
  - the named-FA share;
  - confusion.
- 5 seeds, budget 2. τ is refit per variant on cal, as the eval does.
- **Selection, on accept only, the M5 rule:**
  - eligible if recall_any ≥ base − 0.01, with the paired CI low of (X − base) > −0.03;
  - **and** recall_correct with paired CI low > −0.03;
  - pick the largest significant FA-hard reduction (paired CI high < 0, relative cut ≥ 10 %);
  - if nothing qualifies, keep base.
- Test and SEMI-SYN are report-only.
- Report generation in `idsai/eval/report.py` must handle the new experiment like `m5_rules`.

## Tests
Synthetic, temp stores only:
- flag on/off dumps and both hash pins;
- v1 never adds to F but can name;
- v2 rev 3 conditions;
- the conflict validation;
- experiment wiring on a tiny synthetic context, like the existing m5 tests.

Ruff clean. Full suite `-m "not network and not slow"` green.

## Run (head)
`idsai --config configs/eval_prev_live.yaml eval --experiment m8_stealth` in the background, overnight-scale. Then write REPORT §16 and ROADMAP, in Thai, with the usual wording rules: "ความผิดปกติทางสถิติ"; p and F are never a probability; no accuracy metric.

## Rules
- No order-sending code, no API keys.
- Never commit `data/`, `models/`, `.claude/` or `.superpowers/`. Never use `git add -f`.
- Tune on train, select on accept, look at test once.

## Implementation / deviations

### Flag
- `chief.stealth: null | "attr" | "act"` (default null, absent from dumps; default hash 9aff5e3482f2 and `eval_prev_live.yaml` 53e3b662ab96 unchanged).
- `attr`: `rule_signatures` drops 100102; `naming_detectors` appends `attr.100102` (100102 rev 1 of `signatures.yaml` on the naming layer: same sid, technique, rev and conditions) after the M3 naming rules, so it can only name an episode still MM-T000. Needs `chief.attr_rules: true` (else a config error).
- `act`: rev 3 from `configs/signatures_r3.yaml` replaces rev 1 under the name `rule.100102`. `train` writes `signatures_r3_sha` into `bundle.json`.
- Either mode with 100102 in `chief.rule_rev2` is a config error.

### Train-only check (2026-11-27, before any eval)
Train range 2018-01-01..2019-06-30, feature cache read only, the same day and event selection as the M5 check. Script: scratchpad `m8/m8_train_fire.py`.

| rule | rows fired | row fire rate | fire episodes / symbol-day | train REAL pumps fired |
|---|---|---|---|---|
| 100102 rev 1 | 5,862 | 1.91e-3 | 0.848 | 42/183 |
| 100102 rev 3 (+ z_n_trades_60 ≥ 3) | 691 | 2.25e-4 | 0.258 | 38/183 |

### Experiment
- `m8_stealth`: the variants `base`, `stealth_attr` and `stealth_act`, with one fit and one scoring pass per seed. The variants only re-fuse those scores, and each one gets its own calibration and τ.
- Paired FA-hard: per accept-hard symbol-day, the mean over the common seeds, then a day bootstrap with the same indices for both arms.
- `m8_select` also excludes a variant whose paired recall comparison cannot decide (too few events).

## ผล / Results (28 Nov 2026, `m8_stealth`, 5 seeds, budget 2)
Command as in "Run". Full tables with CIs and the confusion tables: `data/report.md` section `m8_stealth` (REPORT §16).

| split | variant | recall_any | Δ recall_any [95% CI] | Δ recall_correct [95% CI] | FA/symbol-day | FA hard | Δ FA hard [95% CI] (rel) | named FA (by naming rule) |
|---|---|---|---|---|---|---|---|---|
| accept (94 events) | base | 0.806 | – | – (0.787) | 2.517 | 7.700 | – | 0.347 (0.016) |
| | stealth_attr | 0.749 | −0.057 [−0.111, −0.019] | −0.057 [−0.111, −0.019] | 2.037 | 6.150 | −1.550 [−2.550, −0.649] (−20 %) | 0.181 (0.086) |
| | stealth_act | 0.774 | −0.032 [−0.074, 0.000] | −0.043 [−0.096, −0.011] | 2.190 | 6.450 | −1.250 [−2.250, −0.400] (−16 %) | 0.217 (0.019) |
| test (200 events, report only) | base | 0.789 | – | – (0.783) | 1.806 | 5.850 | – | 0.582 (0.026) |
| | stealth_attr | 0.756 | −0.033 [−0.058, −0.010] | −0.033 [−0.058, −0.010] | 1.136 | 4.050 | −1.800 [−3.250, −0.700] (−31 %) | 0.327 (0.167) |
| | stealth_act | 0.779 | −0.010 [−0.025, 0.000] | −0.008 [−0.025, 0.010] | 1.233 | 4.250 | −1.600 [−3.100, −0.499] (−27 %) | 0.348 (0.043) |
| semisyn (report only) | base | 0.730 | – | – (0.535) | 1.581 | – | – | 0.625 (0.027) |
| | stealth_attr | 0.590 | −0.140 [−0.150, −0.131] | −0.139 [−0.148, −0.130] | 0.936 | – | – | 0.359 (0.177) |
| | stealth_act | 0.732 | +0.002 [−0.005, 0.008] | −0.132 [−0.145, −0.120] | 1.020 | – | – | 0.379 (0.048) |

- **Selection (accept): base.** Excluded: stealth_attr (recall_any −0.057, beyond −0.01) and stealth_act (recall_any −0.032, beyond −0.01). Both also fail the CI-low gates (recall_any CI low −0.111 / −0.074, recall_correct CI low −0.111 / −0.096, all ≤ −0.03). Both cut FA hard significantly and by more than 10 %, but neither is eligible. The live config is unchanged (no `chief.stealth` in `configs/default.yaml`; default hash 9aff5e3482f2, `eval_prev_live.yaml` 53e3b662ab96).
- Test (looked at once, report only): stealth_act would look eligible on test alone (recall_any −0.010, recall_correct −0.008, FA hard −27 %). It does not change the choice: selecting on test would make test a selection split. The accept and test differences overlap within their CIs.
- SEMI-SYN (report only): stealth_attr loses pump_burst detection (missed 0.348 → 0.759), so 100102 in F also helps fast pumps reach τ on the simulator. stealth_act keeps recall_any, but pump_stealth naming drops (MM-T000 0.524 → 0.925, MM-T001 0.480 → 0.060): the simulator's stealth pumps have no trade-count burst, so rev 3 never fires on them. pump_burst is about unchanged (missed 0.348 → 0.331).
- Reading: both variants cut accept FA hard by 16–20 %, at a cost of about 3–6 points of recall_any (about 3 and 5 of 94 events). So 100102 also lets some real pumps reach τ. This differs from the M7 train counterfactual (160 → 160), which kept τ, used one seed and had no naming rules. That explanation is not tested.
- Review follow-ups: `load_bundle` now checks `signatures_r3_sha` against `configs/signatures_r3.yaml` when `chief.stealth` is "act" (a changed file, a changed sha or a missing sha is a `BundleError`; nothing changes with stealth off). `M8_RULE_TH` now states that a variant whose recall comparison cannot decide (fewer than 10 events) is excluded, as `m8_select` already did.
- Next (hypothesis, not implemented): on train only, look at the real pumps that are lost when 100102 is dropped with τ refit at the same budget, and at how they differ from this rule's FA. Accept stays untouched as the selection split. Only then pre-register a single new condition and test it with the same discipline.
