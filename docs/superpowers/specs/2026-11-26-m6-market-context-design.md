# M6: market context (phase 3.1), BTC as the market reference

Sources: `docs/ROADMAP.md` phase 3.1 and REPORT §8–13. Branch `feat/m6`, cut from `main` (c263287). Live stack: rules + stat.robust_z + stat.cusum + ai.iforest (shift 1), with `sell_side` and `attr_rules` on. Main budget 2.

## Why BTC, not a market median
The research archive has a median of 3 symbols per day: 1,331 days, of which only 346 have at least 5 symbols. A cross-symbol median would be noise. Research pairs are XXX/BTC, and the live coins are USDT pairs that move with BTC, so the market reference is **BTCUSDT 1-minute klines** from data.binance.vision. Those files are small. Verify each one with the CHECKSUM, as the trades downloader does. Never download full BTCUSDT trades, because one day ingests about 1.5 GB.

## Step A — diagnosis, train range only (gate; report before step B)
1. Add `idsai download --market` (or an equivalent). It fetches BTCUSDT 1m klines for every date in the research split, plus D−1, and stores them under `data/market/` (not committed). Add a CHECKSUM test with mocked HTTP.
2. Market move feature: `mkt_absret_z` per row. Take |BTC log-return| over the same window as the row's main return feature (find the jump/return window in the feature code, e.g. 60 s or 1800 s). Express it as a robust z: median/MAD over the trailing 24 h of BTC minutes. Use the kline close nearest at or before the row time; never look ahead.
3. Diagnosis on **train** only (train normal days and train REAL pumps), with the live stack and its frozen τ at budget 2:
   - Of the false-alarm episodes on train-normal days, what share starts within ±10 min of a BTC move with `mkt_absret_z ≥ 3`? Compare that with the base rate (the share of all train minutes with `mkt_absret_z ≥ 3`).
   - Of the train REAL pump detections, what share has `mkt_absret_z ≥ 3` near its start?
4. **Gate:** go to step B only if the FA share near BTC moves is at least 2× the base rate **and** at least 0.10 in absolute terms, and the pump share is below half the FA share. Otherwise stop. Write the numbers into the spec's results section and REPORT §14 as "M6 หยุดที่ขั้นวินิจฉัย", then report.

## Step B — only if the gate passes
- A flag `chief.market_context` (default off) using the same hash-safe trick as `sell_side`, `attr_rules` and `rule_rev2`:
  - It is absent from dumps when off.
  - With it off, output is bit-identical and the golden test is unchanged.
  - Pin: the default hash 9aff5e3482f2 and `eval_prev_live.yaml` 53e3b662ab96.
- Two variants, both chosen from train only:
  - **`mkt_damp`:** for rows where `mkt_absret_z ≥ k`, add log10(d) to the threshold the Chief compares F against. In practice, require F ≥ τ + log10(d), with d ∈ {2, 5} picked on train. τ is still fitted on cal as usual. Implement it so that the fit and the per-budget FA control stay valid. Document how.
  - **`mkt_tag`:** detection is unchanged; alerts gain `market_move: true` (a new optional Alert field, left out of dumps when None, like `phase`). It is a display-only control and can never be selected for detection gains.
- Experiment `m6_market`: variants base, mkt_damp, mkt_tag. Splits are accept, test and semisyn (semisyn is test-only and has no BTC context, so mark it as not applicable and skip it if the kline data is missing). Records match M5: recall_any, recall_correct, FA, FA hard, named-FA share, and confusion.
- **Selection (accept only, fixed in advance):**
  - Eligible variants have recall_any ≥ base − 0.01, with the paired CI low of (X − base) > −0.03.
  - Pick the one with the largest significant FA-hard reduction: the paired CI high < 0 and a relative cut of at least 10 %.
  - If nothing qualifies, keep base.
  - mkt_tag is never selected; it is reported only.
- Live: if mkt_damp is picked, the live pipeline needs BTCUSDT klines streamed or polled. Live symbols already include BTC (BTCUSDT bars), so reuse those bars for the live reference. Write a note on how live gets `mkt_absret_z`. Enabling it in `configs/default.yaml` is a later, separate step and needs the user's OK.

## Rules
- No order-sending code and no API keys.
- Never commit `data/`, `models/` or `.claude/`.
- Tests are synthetic or use temp stores, never the real `data/app.duckdb`.
- Wording: "ความผิดปกติทางสถิติ"; never call p or F a probability; no accuracy metric; Thai report text.
- Tune on train, select on accept, and look at test once.

## Run
Step A runs locally: downloads plus a short diagnosis script. Step B's eval runs overnight on the head's call: `idsai --config configs/eval_prev_live.yaml eval --experiment m6_market`.

## Results: step A (2026-11-25), gate result STOP

**Run:** `scripts/diagnose_market.py` on branch feat/m6 (wall time 1,140 s).
- Config: `configs/default.yaml` (config_hash 9aff5e3482f2), the live stack. It runs rule.100101, 100102, 100501, 100104 and 100601, plus stat.robust_z, stat.cusum and ai.iforest with `AiFusion("shift", 1.0)`.
- Split sha 7fb702a841c5, seed 0.
- ai.iforest was fit on the masked, grid-thinned train-normal rows (512,640 rows).
- The Chief was calibrated on cal_normal (347 days, 2,998,080 rows). τ = 2.904 at budget 2 (not saturated).
- The train range was scored: 356 train-normal days and 174 train event days, 4,579,200 rows in all, with 0 days skipped.

**Market data:** BTCUSDT 1m klines from data.binance.vision, every file CHECKSUM verified.
- 1,358 days were downloaded, none missing. That is every date of `research_symdays`, D−1 included. They take 69 MB under `data/market/BTCUSDT-1m/`.
- The train diagnosis read 693,822 klines, of which 682,260 have a finite z.

**Feature:** `mkt_absret_z` is |BTC 1m log-return| as a robust z.
- Window: 60 s. This is the window of `ret_60`, the return condition of the pump-burst rule 100101 and of the sell rules 100104 and 100601.
- Scale: median and 1.4826·MAD over the 1,440 minutes strictly before, with at least 360 samples required.
- Each value is known at its kline's close.

**Definitions:**
- *Near:* a kline closing within ±10 min of the episode onset has z ≥ k.
- *Base minute:* the share of train-normal symbol-day minutes with z ≥ k. This is the spec's base rate.
- *Base near:* the share of those minutes that are themselves within ±10 min of a z ≥ k minute. This is the like-for-like chance level for "near".
- *Pump:* the onset of the first matched episode of a detected train REAL pump.

Train results: 385 FA episodes on 356 normal days (1.08 per symbol-day), and 160 of 183 train REAL pumps detected.

| k | FA near / FA | FA share | base minute | base near | pumps near / detected | pump share |
|---|---|---|---|---|---|---|
| 2 | 257 / 385 | 0.668 | 0.1153 (58,823 / 510,231) | 0.700 | 122 / 160 | 0.762 |
| **3** | **168 / 385** | **0.436** | **0.0601 (30,666 / 510,231)** | **0.453** | **76 / 160** | **0.475** |
| 4 | 110 / 385 | 0.286 | 0.0340 (17,351 / 510,231) | 0.289 | 44 / 160 | 0.275 |

Using the pump label's `t_start` instead of the detection onset gives 78 / 160 (0.488) at k = 3.

**Gate (k = 3, as written):**
- FA share ≥ 2× base rate: 0.436 ≥ 0.120 holds.
- FA share ≥ 0.10: holds.
- Pump share < half the FA share: 0.475 is not below 0.218, so this condition **fails**.

**Decision: STOP.** Step B is not built.

**Reading:**
- FA onsets are no closer to BTC moves than a random minute of the same days. At every k, the FA share (0.668 / 0.436 / 0.286) is at or below the like-for-like chance level "base near" (0.700 / 0.453 / 0.289).
- The first gate condition passes only because "base minute" compares a ±10 min window (21 klines) with a single minute.
- Real pump detections sit near BTC moves at the same rate as false alarms, or a slightly higher one. Damping on BTC moves would cost recall about as much as it cuts FA.
- On these pairs a BTC market-move context does not separate false alarms from pumps.

**Method notes:**
- Seed 0 only.
- ai.iforest scores its own train rows (in-sample). Only the AI layer is affected; the rules and stats are not fitted.
- The research pairs are XXX/BTC, so their price is quoted in BTC, and a BTC/USDT move shows up in every pair's ret. That is exactly the effect tested here.
- A z ≥ 3 on |r| holds for 6 % of minutes. |r| is skewed, so the MAD scale is small and "≥ 3" is not rare. That is why k = 4 was also reported, and the conclusion is the same at every k.
