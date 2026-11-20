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
