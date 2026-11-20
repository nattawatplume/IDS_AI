# M4: detect the sell side (dump phase + bear raid)

Source: `docs/ROADMAP.md` M4. Branch `feat/m4` (worktree), cut from `feat/m1`.

**Goal.** An alert says whether the market is "กำลังพุ่ง" (pump phase) or "กำลังเทขาย" (sell phase), and the report gives a recall per phase. Bear raid (a sell-side push without a pump before it) becomes a technique in the taxonomy.

## Hard constraints
- **Off by default.** A new config flag `detection.sell_side: bool = False` (put it in the existing detection or feature config class that fits; read `idsai/config.py`).
  - With the flag off, every existing output must be bit-identical: features used by the models, rule hits, alerts, Chief state, the config hash of `configs/default.yaml` and `configs/eval_prev_live.yaml`, and the signatures sha256 that runs record.
  - Prove this with a test: the default-config feature frame, `model_features(cfg)` and the rule outputs on a synthetic day are unchanged.
  - If the signatures sha is computed over the whole `configs/signatures.yaml`, put the new sell-side signatures in a separate file `configs/signatures_sell.yaml` that is loaded only when the flag is on.
- **AE inputs never change.** `model_features(cfg)` must return exactly today's list. New window bases are excluded from it explicitly (keep an explicit list of model bases instead of "all z_*"). M4 does not retrain anything.
- **Batch/stream parity** holds for the new features, the same way it holds today (`kernel.py`, `streaming.py`, `bars.py` streaming path). There is an existing parity test pattern; extend it.
- No order sending, no API keys, no accusation wording ("ความผิดปกติทางสถิติ"), F is never a probability, no accuracy metric. Tests never touch the real `data/`.

## Features (only computed when the flag is on, or computed always but unused; pick whichever keeps the flag-off outputs bit-identical with less code)
- Bar column `sell_rush_orders`: distinct `ts_us // 1000` in the second with ≥ 2 sell-taker fills (mirror of `rush_orders`).
- Window bases (raw + z, same windows):
  - `sell_rush`: sum of `sell_rush_orders`, transform log1p, like `rush`.
  - `drop`: lowest low in the window against the close before the window, minus 1 (mirror of `jump`). Thai desc: "ราคาต่ำสุดในหน้าต่างเทียบราคาปิดก่อนหน้าต่าง (−1)".
- `sell_quote` already exists in the bars, and `buy_imb` already goes negative on selling. Use them; do not add a separate `sell_imb` unless needed.
- State feature `cusum_sell`: one-sided CUSUM of the negative standardized 10 s imbalance (mirror of `cusum_imb`), only if it fits cleanly; otherwise skip and say so.
- Registry `desc_th` for each new feature, in Thai.

## Taxonomy and signatures
- Taxonomy (`idsai/detection/taxonomy.yaml`):
  - `MM-T001.003` "ช่วงเทขายหลังปั่น (dump phase)" under MM-T001.
  - A new parent `MM-T006` "ทุบราคา (bear raid)" with `MM-T006.001` "เทขายแบบพุ่งเร็ว (burst)". Follow the file's existing fields.
- Sell-side rule signatures (new sids, `rev: 1`), thresholds mirror the pump rules and are written in the spec of the file as "set from train-period data only":
  - dump phase: price falling fast after a recent rise (e.g. `ret_60 <= -0.02`, `z_sell_rush_60 >= 4` or `z_quote_vol_60 >= 6`, and `jump_1800 >= 0.03`, meaning there was a rise in the last 30 min).
  - bear raid burst: the same falling conditions but `jump_1800 < 0.03` (no rise before).
  - The implementer may tune these thresholds only on the train split (≤ 2019-06-30), with a small script, and must write the chosen values and how in the spec's results section.
- Alert phase: `Alert` gets an optional field `phase: Literal["pump", "sell"] | None` (default None, so old alerts and serialized data still load). It is derived from the technique: MM-T001.001/.002 → pump, MM-T001.003 and MM-T006.* → sell, else None. The explainer template (`idsai/product/explainer/template.py`) adds one Thai line for the phase when set: "ช่วง: กำลังพุ่ง" / "ช่วง: กำลังเทขาย". The web alert card shows it if the field exists (small badge, match the HUD style). The Telegram message gets it for free through the explainer.

## Evaluation: new experiment `m4_sell`
- Per seed, same split and fitting path as `real_t001` (reuse it; read `idsai/eval/experiments.py`). Run two variants: `base` (flag off) and `sell` (flag on). Records: experiment `m4_sell`, variant `"<split>:<variant>"` for split ∈ {test}, provenance labelled.
- **REAL sell-phase labels.** For each REAL label, `scripts/measure_pumps.measure_label` gives `t_peak_us` and `dump_tau_s` when status is measured. The sell window is `[t_peak_us, t_peak_us + max(dump_tau_s, 60 s)]`, capped at 1 h. Move the needed pieces into `idsai/eval/` (do not import from `scripts/` in library code) or call through a thin helper; the label derivation must be deterministic and cached in the experiment so it is computed once.
  - Metrics per phase: `recall_pump_phase` (existing match on the label's onset window), `recall_sell_phase` (an alert episode, any technique, that overlaps the sell window), and `recall_sell_phase_correct` (episode technique is in the sell set). Exclusion counts per status.
- **SEMI-SYN bear raid.** Add an injection family `bear_raid` to `idsai/detection/simulator/inject.py`: synthetic taker-sell fills, in rush clusters, pushing the price down by a sampled impact, mirroring `pump_burst`. Provenance SEMI-SYN on every number. Recall_any and recall_correct for `bear_raid` on the `sell` variant.
- FA and FA-hard per variant with the existing `fa_by_day` (+ multiplicity) machinery, so the cost of the sell rules is visible.
- Report (`idsai/eval/report.py`): a Thai `m4_sell` section: table per variant at budget 2 with recall per phase [CI], FA [CI], FA-hard [CI], bear-raid recall (SEMI-SYN), and a paired difference `sell − base` for pump-phase recall (it must not drop significantly).
- Add `m4_sell` to `EXPERIMENTS` and the CLI choices.

## Tests (synthetic only)
- `sell_rush_orders` and `drop` values on hand-built trades; batch/stream parity for both.
- Flag off: bit-identical outputs (see constraints), `model_features` unchanged.
- Signatures: the dump rule fires on a synthetic rise-then-crash and not on a flat day; bear raid fires on a crash without a prior rise; MM-T001.001 behaviour unchanged.
- Alert `phase` derivation, old alert JSON without `phase` still loads, explainer line.
- `bear_raid` injection: price moves down, fills are sell-taker, deterministic for a seed.
- Sell-window derivation from a synthetic measured label, and the `m4_sell` smoke test on the existing synthetic eval fixtures.
- Report section on synthetic records.

## Run (by the head, after review, when the CPU is free)
- `idsai --config configs/eval_prev_live.yaml eval --experiment m4_sell`.

## Out of scope
- Turning `sell_side` on in live. That is a separate step after the results.
- Retraining the AE with sell features (later, M2 territory).
- Phase timeline UI beyond the badge.
