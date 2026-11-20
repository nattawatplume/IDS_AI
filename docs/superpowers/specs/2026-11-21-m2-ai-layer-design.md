# M2: make the AI layer earn its place

Source: `docs/ROADMAP.md` M2 and the M1 results (`docs/REPORT.md` §9). Branch `feat/m2` (worktree `IdsAI-m2`), cut from `feat/m4`.

**Facts going in.**
- M1 picked `no_ai` (the AE adds hard-day false alarms and no REAL recall).
- Yet `ai_shift2` (the AE must be 2 decades rarer than its p-value says) kept REAL hard FA near `no_ai` (5.62 vs 5.55) and lifted SEMI-SYN recall from 0.710 to 0.874.
- On its own, iforest beats the AE on REAL recall by about 0.16 (paired, significant).

**Goal.** Find an AI layer and fusion that adds recall over `no_ai` without paying in hard-day FA, chosen by a rule fixed in advance on the accept split. If nothing qualifies, the answer is "keep `no_ai`", and that is a valid result.

## Rule against fooling ourselves
- Select on the **accept** split only (94 REAL events, accept-hard days computed as in M1). Test and SEMI-SYN are reported once, never used to choose.
- **Reference:** `no_ai` (rules + stats; the AI is scored but not fused).
- **Eligible:** a variant whose accept-hard FA at budget 2 is ≤ 1.10 × the reference's accept-hard FA, and whose accept FA (all days) is ≤ 1.10 × the reference's.
- **Pick:** among the eligible variants, the one with the largest paired accept recall_any gain over the reference (paired bootstrap; reuse M1's `significant` with `min_effect`). It wins only if that gain is significant. Otherwise the pick is `no_ai`.
- Ties go to the simpler variant: fewer AI models first, then AE before iforest, then fuse < shift1 < shift2.
- The rule is written into the report code and applied mechanically, the same way M1's `m1_select` works.

## Variants (detectors fit and scored once per seed and shared across fusions)
- **AI models:**
  - `ae` (today's config)
  - `ae_wide` (hidden 128, bottleneck 16)
  - `ae_narrow` (hidden 64, bottleneck 8)
  - `iforest` (`IsolationForestDetector` used as the AI layer)
  - `ae+iforest` (both AI models enter the fusion)
- **Fusions:** `fuse`, `shift1`, `shift2`. Reuse M1's `AiFusion`. If needed, generalize it so the AI set can hold several detectors, the shift applies to each AI detector, and `exclude` drops them all. Keep `"fuse"` with today's AE bit-identical.
- **Plus the reference `no_ai`.** That is 5 × 3 + 1 = 16 fusion variants per seed, but only 4 model fits (ae, ae_wide, ae_narrow, iforest).
- Keep the AE grid small on purpose (CPU budget on a laptop). Epochs, lr and the rest stay as in config. Write the grid in the spec, and report every grid point, not only the winner.

## Code
- `idsai/ai/chief.py`:
  - Support more than one AI detector in the fusion modes.
  - Raw `iforest` scores must go through the same split-conformal p-value path as the AE, which they should already do because the Chief is score-agnostic. Verify that and test it.
  - Default behaviour stays bit-identical. Extend the existing golden or parity tests rather than replacing them.
- `idsai/eval/experiments.py`: new experiment `m2_ai`.
  - Fit the 4 models once per seed and score cal / accept / accept-hard / test / test-hard once.
  - For each variant, build a Chief, calibrate, fit τ per budget and evaluate with `_evaluate`.
  - Records: experiment `m2_ai`, variant `"<split>:<model>:<fusion>"` and `"<split>:no_ai"`, split ∈ {accept, test}, provenance REAL.
  - SEMI-SYN on test only if it fits without a second scoring pass (as in M1). Otherwise skip it and say so.
  - Add it to `EXPERIMENTS` and the CLI choices.
- `idsai/eval/report.py`: a Thai `m2_ai` section.
  - A table per split at budget 2: recall_any [CI], recall_correct, FA [CI], FA-hard [CI], and the paired recall difference vs `no_ai`.
  - The selection rule, printed and applied on accept, with the chosen variant.
  - A paired line "AE alone vs iforest alone" on test, answering the ROADMAP M2 goal.
  - F is never called a probability.
- Do **not** change the live default (`configs/default.yaml`). Switching the live AI layer is a separate step after the results.

## Tests (synthetic only; never the real `data/`)
- Multi-AI fusion:
  - `fuse`, `shift`, `exclude` with two AI detectors (F formula, agreement, technique).
  - Batch/stream parity.
  - State round-trip; v2 states still load.
- iforest through the conformal path gives p-values in (0, 1] and the parity holds.
- `m2_ai` smoke test on the existing synthetic eval fixtures:
  - each model is fit once per seed (count the calls);
  - the `no_ai` record equals M1's `no_ai` on the same fixture.
- Selection rule on synthetic records:
  - ineligible by hard FA;
  - ineligible by FA;
  - a non-significant gain leaves the pick at `no_ai`;
  - the tie order.

## Run (by the head, after review, when the CPU is free)
- `idsai --config configs/eval_prev_live.yaml eval --experiment m2_ai` (5 seeds).

## Out of scope
- A large hyperparameter sweep, and new AE features (the sell-side features stay out of the model inputs).
- Changing the live deployment.

## Results / deviations (implementation, before the real run)

**AE grid (every point is reported).** All other `ae` settings (epochs, lr, weight_decay, batch_size, patience, clip, train_stride_s) stay as configured.

| model | detector name | hidden | bottleneck |
|---|---|---|---|
| `ae` | `ai.autoencoder_trade` | config (64) | config (16) |
| `ae_wide` | `ai.autoencoder_wide` | 128 | 16 |
| `ae_narrow` | `ai.autoencoder_narrow` | 64 | 8 |
| `iforest` | `ai.iforest` | IsolationForest, 200 trees, same features and seed as `baseline.iforest` | |

**Tests.** Full suite: 1608 passed, 2 skipped, 4 deselected (slow); `ruff check .` clean. Existing golden/parity tests (including `tests/detection/test_sell_side_off.py`) pass unchanged.

**Code as built.**
- `idsai/ai/chief.py`: no logic change was needed. The Chief already treats every `ai`-layer detector separately: each gets its own conformal p-value, `shift` applies to each, `exclude` drops them all, and Mondrian bins each one. The `ai` layer still counts once in `agreement`. This is documented in the module docstring and covered by new tests (F formula, agreement, technique, batch/stream parity and state round-trip in every mode with two AI detectors, plus a v1 state and a one-AI v2 state loading). The state layout did not change, so `STATE_VERSION` stays 2.
- `ai.iforest`: new `AiIsolationForestDetector` (a subclass of `IsolationForestDetector`, layer `ai`), registered in the detector registry. Its raw scores are bit-identical to `baseline.iforest`. A test shows they go through the conformal path: p in (0, 1], NaN gives p = 1, and batch/stream parity holds.
- `idsai/eval/experiments.py`: `m2_ai`, `m2_ai_models`, `m2_variants`. It is in `EXPERIMENTS`, `INJECTING` and the CLI choices. `configs/default.yaml` is unchanged.
- `idsai/eval/report.py`: `_m2`, `_m2_md`, `m2_select`, and the JSON key `m2_ai`.

**Deviations and choices the spec left open.**
1. **Score columns of the grid points.** The shared scoring pass needs a distinct column per model, so `ae_wide` and `ae_narrow` carry their own detector names. Only the instance name changes; the class name is untouched. `ae` keeps `ai.autoencoder_trade`.
2. **SEMI-SYN is included** (`semisyn:<variant>`). The injected test-normal days are scored in the same single pass as everything else, as in M1. They are reported only and never used to choose.
3. **Diagnostic records `test:ae:alone` and `test:iforest:alone`.** Each is that AI model alone through its own Chief (default fusion), using the same scores. They feed the report's "AE เดี่ยว เทียบ iforest เดี่ยว" paired line, because the in-stack variants are not "alone". They cost 2 extra fusions and no extra fit or scoring, and they never enter the selection.
4. **`no_ai`** = the configured rules and stats plus today's AE with `AiFusion("exclude")`, which is exactly M1's `no_ai` detector set. The smoke test checks equality with `m1_fusion`'s `no_ai` on metrics, per_event, fa_by_day, tau, n_cal, detectors and model_versions, for every split and budget. Caveat: if a config listed the AE before other detectors, the detector order would differ from M1's (this only matters for `best_detector` ties). The default config lists it last.
5. **Tie order inside the AE grid** was not specified. I chose `ae` < `ae_narrow` < `ae_wide` (today's config first, then the smaller net). The full key is (number of AI models, `ae` < `ae_narrow` < `ae_wide` < `iforest`, `fuse` < `shift1` < `shift2`).
6. **Selection details.**
   - The FA caps compare seed means: ≤ 1.10 × no_ai, with a 1e-12 tolerance.
   - A variant without a paired gain vs `no_ai` cannot be ranked, so it is not eligible.
   - A variant with too few clusters is ranked, but it can never be significant. If it has the top gain, the pick stays `no_ai`.
   - Following the spec literally, only the top-gain eligible variant can win. A smaller significant gain elsewhere does not.

**Expected cost of the real run vs `m1_fusion` (per seed).**
- Model fits: 4 vs 1. The fits are 3 AE plus 1 IsolationForest. The AE fits are single-threaded: `ae_narrow` costs about the same as `ae`, and `ae_wide` about 2× (twice the hidden width), so AE training is about 4× M1's. The IsolationForest fit is small (256-row subsamples per tree).
- Scoring passes: 1 vs 1. The same symbol-days and injected days are loaded once, so feature computation is unchanged. Scoring the AI models in that pass costs about 4× M1's AE scoring, plus IsolationForest `score_samples` on every row (the cost the `baselines` experiment already pays).
- Fusion/threshold/evaluation: 16 × 3 + 2 = 50 `_variant` calls vs 15 (about 3.3×). Each call re-calibrates, fuses and evaluates; there is no window recomputation.
- Overall, I expect about 1.5–2.5× M1's wall time if window loading dominates, as it usually does, and up to about 3–4× if model training and scoring dominate. Memory: 4 extra Float64 score columns on every loaded set.

**Open risks.**
- Selection runs over 15 candidates on the accept split, so the largest gain carries a winner's-curse bias. The single test report is the check.
- The FA eligibility uses point means, not CIs. With ~94 accept events, a small FA difference can flip eligibility.
- `ae+iforest` fuses one more detector, which raises the `fit_threshold` resolution floor. A tight budget can then give a `CalibrationError` record for that variant only.
- Each AE fit is single-threaded, so the 3 AE fits are the slowest part of each seed on a busy laptop.
