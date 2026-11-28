# M7: what the remaining false alarms and missed pumps look like (diagnosis only)

Source: ROADMAP item 8: "next, look at what the false alarms on hard days look like before proposing a new feature". The follow-up list in M5 adds "the common traits of the real pumps still missed (about 20 %)".

Branch `feat/m7-diag`, cut from `main` (508f29b). This step is **read-only diagnosis on the TRAIN range only**. Accept, test and SEMI-SYN are never read. Detection, configs and the config hash (9aff5e3482f2) do not change, and no flag is added.

## Script `scripts/diagnose_fa.py`
Copy the setup of `scripts/diagnose_market.py`:
- the same split and live stack (rules, stat.robust_z, stat.cusum, ai.iforest shift 1);
- iforest fit as in the eval;
- Chief calibrated on `cal_normal`, τ at budget 2;
- the DuckDB copied to a temp dir; JSON written to `--out` outside `data/`.

Share code rather than duplicating it where that is clean, for example by moving the common setup into a small helper module under `scripts/` or by importing from `diagnose_market`. Do not change `diagnose_market`'s output.

**Sets**
- **FA:** unmatched episodes on `train_normal` days.
- **Hard train days:** the `m1_hard_k(cfg, n)` most volatile train-normal days, picked with the same `_most_volatile` rule the eval uses. FA_hard is the FA episodes on those days.
- **DET:** train REAL pumps that are detected.
- **MISS:** train REAL pumps with no matched episode.

**Profile each set.** Report counts and medians with IQR:
1. **Which layer drives the alert:** at the episode's peak-F row, the detector with the smallest p (rules / robust_z / cusum / iforest), as shares. Count how often more than one layer is under τ-equivalent.
2. **Episode length** (minutes) and peak F/τ.
3. **Feature values at onset**, for the main features the detectors use: z_quote_vol_60, ret_60, jump_1800, imbalance or cusum_imb, trade count z. Find the real column names in the feature code.
4. **Context:**
   - UTC hour of day;
   - day volatility rank;
   - symbol concentration: the share of FA in the top 5 symbols, and the Gini over symbols;
   - FA per day, to show whether they are spread out or clustered.
5. **Direction:** the share with the price up vs down over the 30 min after onset (descriptive only).
6. **FA vs DET separability:** for each feature in (3), the AUC of FA-onset vs DET-onset values, computed with rank statistics in numpy. List the features with AUC ≥ 0.65 or ≤ 0.35. This is descriptive, not a selection. Use the same for MISS vs DET.

**Output**
- A compact printed table, and JSON with all numbers.
- Runtime should be similar to `diagnose_market` (minutes, not hours). Use the stored archive only, with no downloads.

## Write-up
- `docs/REPORT.md` §15, "M7 ลักษณะการเตือนผิดและ pump ที่หลุด (train)", in Thai:
  - the tables and the 3–5 main observations;
  - the limitations: train only, 2018–19 BTC pairs vs live USDT 2026;
  - descriptive only, with no p-hacking claims.
- `docs/ROADMAP.md`: update item 8 and the "งานต่อ" line with one concrete next step, which is a hypothesis to test later with the usual discipline (fixed rule, chosen on accept, test once). Do not implement it.
- Spec file `docs/superpowers/specs/2026-11-26-m7-fa-diagnosis-design.md`: copy this text and add the results section.

## Rules
- Wording: "ความผิดปกติทางสถิติ". Never call p or F a probability. No accuracy metric. Accuse no one.
- Never commit `data/`, `models/`, `.claude/` or `.superpowers/`. Never use `git add -f`.
- Tests: a small unit test for the pure helpers (the AUC, Gini and share functions) on synthetic arrays. Never touch the real `data/app.duckdb`, other than reading the temp copy inside the script.
- Ruff must be clean, and the full suite `-m "not network and not slow"` green.

## Results (26 Nov 2026)

**Run:** `scripts/diagnose_fa.py` on branch feat/m7-diag (wall time 418 s). config_hash 9aff5e3482f2, split 7fb702a841c5, seed 0, budget 2, τ = 2.904 (the same stack and τ as M6). FA 385 on 356 train-normal days; hard k = 20 days with 63 FA; DET 160 and MISS 23 of 183 train pumps. Full tables in `docs/REPORT.md` §15.

**Layer at the peak-F row** (shares; "F" = the layer whose shifted −log10 p is F; a row with F = 0 has no driver):

| set | n | smallest p: rules / robust_z / cusum / iforest | F: rules / robust_z / cusum / iforest | ≥ 2 layers at τ |
|---|---|---|---|---|
| FA | 385 | 0.49 / 0.17 / 0.12 / 0.22 | 0.57 / 0.21 / 0.12 / 0.10 | 0.08 |
| FA_hard | 63 | 0.76 / 0.08 / 0.08 / 0.08 | 0.81 / 0.08 / 0.08 / 0.03 | 0.06 |
| DET | 160 | 0.31 / 0.54 / 0.00 / 0.14 | 0.33 / 0.62 / 0.00 / 0.06 | 0.76 |
| MISS | 23 | 0.04 / 0.13 / 0.13 / 0.70 | 0.13 / 0.52 / 0.22 / 0.13 | 0.00 |

When rules drive F, the rule with the smallest p is 100102 in 188 FA (0.49 of all FA) and 49 FA_hard (0.78), and never in DET (DET: 100104 51, 100601 1).

**Counterfactual without rule 100102 in F** (train only; the other detectors' calibration and τ unchanged, τ not refit; no naming rules in this stack, as in M6):
- FA 385 → 216 (172 of the original episodes vanish, 0.45); 1.08 → 0.61 per symbol-day.
- FA_hard 63 → 18 (46 vanish, 0.73).
- Detected pumps 160 → 160 (none lost). 12 change their first matched episode (7 change onset).
- Correctly named pumps 108 → 89: 19 first episodes change technique, because 100102 also names stealth pumps.

**Separable features (AUC ≥ 0.65 or ≤ 0.35; labels sharing a first episode count once, 2 DET onset rows dropped):**
- FA vs DET at the episode onset: z_quote_vol_60 0.162, z_n_trades_60 0.113, z_rush_60 0.113, ret_60 0.102, cusum_ret 0.149 (0.10–0.16; FA_hard 0.04–0.10), jump_1800 0.182, ret_1800 0.210, and cusum_imb 0.774 (FA_hard 0.870).
- MISS vs DET at the label start (like for like): z_rush_60 0.305, z_buy_imb_60 0.284, ret_60 0.345, cusum_ret 0.350, buy_imb_1800 0.302 (z_quote_vol_60 0.378 and z_n_trades_60 0.351 not flagged).
- MISS vs DET at the window peak: z_quote_vol_60 0.154, z_n_trades_60 0.051, z_rush_60 0.050, jump_1800 0.175, cusum_ret 0.250, ret_1800 0.111, buy_imb_1800 0.251.

**Concentration:**
- Symbols: 79/89 have FA. The top 5 hold 0.203 and the top 10 % (8 symbols) hold 0.270; Gini 0.449. Every symbol has 4 normal days (1.08 FA per day overall). Top 5: PPTBTC 6.25 per day, GVTBTC 4.00, ARNBTC 3.25, BRDBTC 3.00, EVXBTC 3.00.
- Overlap: 5 of the 20 hard days belong to the top 5 symbols and hold 36 of the 63 FA_hard (PPTBTC and GVTBTC together 34).
- Days: 0.489 of days have any FA (0.511 have none, which drives part of the concentration). The top 10 % of days hold 0.468, max 13 per day, Gini 0.696. Hard days have 3.15 FA per day, the rest 0.96.

**Observations (descriptive, train):**
1. Rule 100102 (stealth) drives F at the peak in 0.49 of FA and 0.78 of FA_hard, and never at the peak of a detected pump. Without it in F (same τ), FA drops 385 → 216 and FA_hard 63 → 18 with no detected pump lost. Correct naming drops 108 → 89, though. On train, this rule is behind about half of all FA and about 3/4 of hard-day FA.
2. FA show less of a trading burst than real pumps. Activity is still elevated at FA onset (z_n_trades_60 median 2.40, z_rush_60 2.33), but clearly lower than DET (AUCs above). cusum_imb is higher in FA. FA mostly reach τ in one layer only (0.92), just above it (F/τ 1.11). Real pumps reach τ in at least two layers (0.76).
3. FA cluster by day more than by symbol, but the two overlap: over half of FA_hard come from the top 5 symbols' hard days, and the zeros drive part of the day concentration. The cause is likely the coin's own volatility that day (consistent with M6, not tested).
4. Compared at the same anchor (the label start), the 23 missed pumps are quieter than the detected ones: lower rush, buy imbalance, 60 s return and cusum_ret. Their closest point reaches F/τ 0.70. iforest has their smallest raw p (0.70), but after the shift it drives F in only 0.13. n is small, so this is descriptive only.
5. Hour and 30-min direction are not usable separators. Train pumps all start 12–23 UTC (the pump groups' schedule in the labels), and price falls after onset in both FA (0.60) and DET (0.72).

**Next step (hypothesis, not implemented):** at most 2 pre-registered variants of 100102 against base:
- V1: 100102 as a naming-only rule (M3 `attr` layer). It is out of F like the counterfactual, but can still name stealth pumps.
- V2: 100102 rev 3 = rev 1 + `z_n_trades_60 >= 3`. The threshold is from train, between the FA median 2.40 and the DET Q1 3.53.

Both are chosen on accept with the usual rule (FA hard down, recall_any and recall_correct not significantly lower), with τ refit as the eval does, and tested once. The main risk is recall_correct.

**Limits:**
- Train only, seed 0, many comparisons: no AUC here is evidence of separability, and none is a selection.
- The counterfactual keeps τ. A refit τ at the same budget would be lower and bring some FA back, and live also has naming rules that may recover part of the recall_correct loss.
- F is capped at the p-value resolution (F = 5.134, F/τ = 1.77). Most DET peaks sit at that cap, so the FA vs DET F/τ gap is understated.
- The data are 2018–19 BTC-quoted pairs; live trades USDT pairs in 2026.
- Event-day volatility includes the pump itself.
- MISS has no episode, so it is compared with DET only at the label start and at the window peak.
- iforest scores its own fit rows.
