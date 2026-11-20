# M1: cut false alarms that come from the AE on volatile days

Source: `docs/ROADMAP.md` M1 and `docs/REPORT.md` §3.

**The problem.** At budget 2, `all` (rules+stats+ai) has REAL recall 0.739 and hard FA 11.17. `rules+stats` has recall 0.745 and hard FA 5.55. The paired difference all − rules+stats is −0.006 [−0.039, 0.028] on REAL and −0.157 on SEMI-SYN. In other words, the AE adds false alarms on volatile days and adds no recall.

**Goal.** Get hard FA ≤ 5.55 with REAL recall_any not significantly below `all` (paired, budget 2).

## Rule against fooling ourselves (selection ≠ test)
- **Choose the variant on the accept split only.**
  - Events: `accept_events` (94 REAL).
  - Normal days: `accept_normal`.
  - "accept-hard": the `hard_negative_k`, capped at ⌊len(accept_normal)/3⌋, most volatile accept-normal days, picked with the same `realized_vol_1m` rule as the test hard negatives. Compute this list inside the experiment. Do **not** change the `Split` dataclass or `split_def`, because that would change the split sha of the stored runs.
- **The selection rule is fixed before seeing any test number.** Among the variants whose accept REAL recall_any is not significantly below V0 (paired bootstrap, `significant` with `min_effect`, reverse direction), pick the one with the lowest accept-hard FA at budget 2. A variant whose recall comparison with V0 cannot be decided (no paired result, or fewer than MIN_EVENTS clusters) is excluded, so it cannot win on FA alone; V0 itself always stays eligible. On a tie, pick the simpler one, in the order V1 < V3 < V2.
- **Then report every variant on test, once.** Test numbers are reported, never used to choose.

## Variants (fusion only; detectors are fit and scored once per seed and shared)
- **V0 `all`:** today's fusion, `F = max(rule, stat, ai)`.
- **V1 `no_ai`:** `F = max(rule, stat)`. The AE is still scored and kept in `p__` and the evidence, but it does not enter `F`, `agreement` or the technique.
- **V2 `ai_mondrian`:** Mondrian (regime-conditional) conformal for the AE only.
  - The regime variable is an existing causal feature over the longest window. **Chosen: `jump_1800`** (raw), the upward price range in the 1800 s window (highest price in the window against the close before it, minus 1; "ช่วงราคาขาขึ้นในหน้าต่าง 1800 วินาที (jump_1800)"). The registry has no return-dispersion feature, and this is its only range feature.
  - Caveat: `jump_1800` is an upward range, not absolute volatility. Pump rows have a high jump by construction, so Mondrian may lower AE recall on pumps. The selection's recall gate against V0 handles that.
  - Make 3 bins from tercile edges of that feature on the cal rows.
  - The AE p-value of a row is computed against the cal AE scores in the same bin. Rules and stats are unchanged.
- **V3 `ai_shift`:** the AE needs extra rarity: its contribution is `max(0, −log10 p_ai − δ)` with δ ∈ {1, 2}, which gives two sub-variants, `ai_shift1` and `ai_shift2`.

For every variant, τ is re-fit on the cal rows of that variant's own `F` with the usual `fit_threshold` and budgets. Each variant must keep the batch/stream parity property: the lookup-table approach stays valid per bin and per shift.

## Code
- **`idsai/ai/chief.py`:**
  - Add a fusion option to `Chief` / `ChiefCfg`: `ai_mode: Literal["fuse", "exclude", "mondrian", "shift"] = "fuse"`, plus `ai_shift: float = 0.0` and `ai_regime_feature: str | None`.
  - The default `"fuse"` must be bit-identical to today: existing tests stay green, and stored models/state still load.
  - Add the state_dict and load_state_dict fields with a version bump that still loads older states.
  - ChiefStream works for every mode.
- **`idsai/eval/experiments.py`:**
  - New experiment `m1_fusion`. Per seed: fit the detectors once, score cal / accept / accept-hard / test / test-hard once, then for each variant build a Chief, calibrate, fit τ per budget and evaluate with the existing `_evaluate`.
  - Write records as experiment `m1_fusion`, variant `"<split>:<variant>"` (split ∈ {accept, test}), with provenance REAL.
  - Add it to `EXPERIMENTS` and the CLI choices.
  - SEMI-SYN is optional. Include `semisyn:<variant>` on test only if it fits without a second scoring pass of the injected days. Otherwise skip it and say so.
- **Fix the latent bug from the M0 review:** the ablation `semisyn:*` `fa_by_day` currently sums several family parts under the same `SYM|DAY` key.
  - Store per-day counts *and* per-day multiplicity (how many parts loaded that day): `fa_by_day = {"normal": {k: n}, "hard": {...}, "normal_mult": {k: m}}`, with `m` defaulting to 1.
  - The report CI then uses the ratio Σn/Σm through a symbol-day ratio bootstrap (`ratio_bootstrap_ci` exists in metrics).
  - Records without `*_mult` behave as before. Add a test that reproduces the 3× inflation and shows the fix.
- **`idsai/eval/report.py`:** a `m1_fusion` section.
  - Per split, a table of the variants at budget 2: recall_any [CI], recall_correct, FA [CI], FA-hard [CI].
  - Paired recall difference vs V0.
  - The selection rule, applied mechanically on accept, printing the chosen variant.
  - Everything in Thai. Never call F a probability.

## Tests (synthetic only, never the real data/)
- Chief: `"fuse"` is identical to before. `exclude` drops the AE from F, agreement and technique. In `mondrian`, p uses only same-bin cal scores, with edge bins and NaN regime treated as their own case (documented). For `shift`, the F formula is correct. Parity holds between `fuse()` and `ChiefStream.step` for every mode. State round-trips, and an old state still loads.
- Experiment: a smoke test on the existing synthetic eval fixtures (see `tests/eval/test_experiments_smoke.py`) for `m1_fusion`, showing that the detectors are fit and scored once per seed (count the calls).
- Report: the selection rule on synthetic records, covering ties, a variant with significantly lower recall being excluded, and the order V1 < V3 < V2.
- The `fa_by_day` multiplicity fix.

## Run (after review, launched by the head, not the implementer)
- `idsai --config configs/eval_prev_live.yaml eval --experiment m1_fusion` (5 seeds).
- After `m1_fusion`, also re-run `ablation` so that its FA CIs use the fixed multiplicity (about 12 h).

## Out of scope
- Retraining or re-tuning the AE (that is M2).
- Changing the live deployment default. If the chosen variant is not V0, switching `ai_mode` in `configs/default.yaml` for live is a separate, explicit step after the results.
