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

**Run:** `scripts/diagnose_fa.py` on branch feat/m7-diag (wall time 656 s). config_hash 9aff5e3482f2, split 7fb702a841c5, seed 0, budget 2, τ = 2.904 (the same stack and τ as M6). FA 385 on 356 train-normal days; hard k = 20 days with 63 FA; DET 160 and MISS 23 of 183 train pumps. Full tables in `docs/REPORT.md` §15.

**Layer at the peak-F row** (shares; "F" = the layer whose shifted −log10 p is F):

| set | n | smallest p: rules / robust_z / cusum / iforest | F: rules / robust_z / cusum / iforest | ≥ 2 layers at τ |
|---|---|---|---|---|
| FA | 385 | 0.49 / 0.17 / 0.12 / 0.22 | 0.57 / 0.21 / 0.12 / 0.10 | 0.08 |
| FA_hard | 63 | 0.76 / 0.08 / 0.08 / 0.08 | 0.81 / 0.08 / 0.08 / 0.03 | 0.06 |
| DET | 160 | 0.31 / 0.54 / 0.00 / 0.14 | 0.33 / 0.62 / 0.00 / 0.06 | 0.76 |
| MISS | 23 | 0.04 / 0.13 / 0.13 / 0.70 | 0.13 / 0.52 / 0.22 / 0.13 | 0.00 |

When rules drive F, the rule with the smallest p is 100102 in 188 FA (0.49 of all FA) and 49 FA_hard (0.78), and never in DET (DET: 100104 51, 100601 1).

**Separable features (AUC ≥ 0.65 or ≤ 0.35), FA vs DET at onset:** z_quote_vol_60 0.163, z_n_trades_60 0.112, z_rush_60 0.114, ret_60 0.101, jump_1800 0.180, cusum_imb 0.774, cusum_ret 0.149, ret_1800 0.207 (FA_hard vs DET: same set, cusum_imb 0.871). MISS vs DET at the window peak: z_quote_vol_60 0.154, z_n_trades_60 0.051, z_rush_60 0.050, jump_1800 0.175, cusum_ret 0.250, ret_1800 0.111, buy_imb_1800 0.251.

**Concentration:** 79/89 symbols have FA; top 5 symbols hold 0.203, Gini 0.449. 0.489 of days have any FA; the top 10 % of days hold 0.468, max 13 per day, Gini over days 0.696. Hard days 3.15 FA per day vs 0.96 on the rest.

**Observations (descriptive):**
1. Rule 100102 (stealth) drives about half of all FA and 0.78 of hard-day FA, and never drives the peak of a detected pump.
2. FA are slow, single-layer exceedances (0.92 reach τ in one layer, peak F/τ 1.11) with high cusum_imb and low volume, trade count, rush and 60 s return; real pumps light up several layers (0.76).
3. FA cluster by day, not by symbol.
4. The 23 missed pumps are quiet at the label start (no rush, ~0 return, negative imbalance) and peak at F/τ 0.70; iforest has their smallest raw p (0.70) but after the shift drives F in only 0.13. n is small.
5. Hour and 30-min direction are not usable separators: train pumps all start 12–23 UTC (the pump groups' schedule in the labels), and price falls after onset in both FA (0.60) and DET (0.72).

**Next step (hypothesis, not implemented):** a fixed 100102 rev 3 that also needs trading activity, or 100102 as a naming-only rule (M3 `attr` layer). Thresholds from train only, chosen on accept with the usual rule (FA hard down, recall_any and recall_correct not significantly lower), test once. Risk: 100102 may name stealth pumps, so recall_correct must be watched.

**Limits:** train only, seed 0, many comparisons (no AUC here is evidence of separability or a selection); 2018–19 BTC-quoted pairs vs live USDT 2026; event-day volatility includes the pump; MISS uses the label start, not an episode onset; iforest scores its own fit rows.
