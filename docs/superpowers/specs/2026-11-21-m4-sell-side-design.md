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

## Results / deviations (implementer, 2026-11-21)

### What was built
- `features.sell_side: bool = False` (in `FeatureCfg`). Bars always carry `sell_rush_orders`
  (cheap, never persisted, not read with the flag off); the window bases `sell_rush`, `drop`
  (raw + z) and the state `cusum_sell` (one-sided CUSUM of the negative 10 s imbalance z) exist
  only with the flag on and are appended after every existing column, so no existing column moves
  and their values are bit-identical with the flag on or off (tested). Batch == stream with the
  flag on (tested on a 30 h stream).
- `model_features` now lists its bases explicitly (`MODEL_BASES`, `MODEL_STATE`); the list is
  unchanged with the flag on or off.
- Taxonomy `MM-T001.003` and `MM-T006` / `MM-T006.001`; sell rules in
  `configs/signatures_sell.yaml`, loaded only with the flag (`signatures.yaml` and its sha are
  untouched). `Alert.phase` (pump / sell); explainer line "ช่วง: กำลังพุ่ง/กำลังเทขาย"; web
  alert card and alert detail show a small phase badge.
- `bear_raid` injection family (SEMI-SYN), `m4_sell` experiment (+ CLI choice), Thai report
  section, `idsai/eval/sell_labels.py`, `scripts/tune_sell_rules.py`.
- Flag-off proof: `tests/detection/test_sell_side_off.py` pins digests (computed on the pre-M4
  code, commit fdf7eb4) of the default feature frame, bars, `model_features`, rule / stat scores,
  Chief state, alerts JSON, the config hashes of `default.yaml` and `eval_prev_live.yaml`, and
  the signatures sha.

### Thresholds (train split only, `scripts/tune_sell_rules.py`)
- Data: the first 30 REAL train labels (<= 2019-06-30) measured as pumps
  (`pump_measure`, statuses on the way: 30 measured, 1 late_peak, 2 no_rise, 1 missing_day) and
  20 train normal days (one stored day per symbol, >= 3 days from any label). Read-only from the
  main checkout's Parquet partitions; no DuckDB opened; output JSON in a scratch dir.
- Grid of the shared falling conditions: `ret_60 <= {-0.01, -0.02, -0.03}` x
  `z_sell_rush_60 >= {2, 3, 4}` x `z_quote_vol_60 >= {2, 3, 4, 6}`. Score: share of measured
  pumps where the conditions fire inside the sell window, and fired episodes per normal day.
- Result: the mirrored `z_quote_vol_60 >= 6` never fired in a sell window (0/30). `>= 3` fired in
  9/30 (at ret -0.02, sell-rush 4) with 0 episodes on the 20 normal days; `>= 2` reaches 19/30 but
  costs 0.25-1.3 episodes per normal day. `ret_60` and `z_sell_rush_60` barely mattered (one event
  of 30 between -0.01 and -0.02).
- Chosen (rule: change as few mirrored thresholds as possible; keep 0 normal-day episodes): only
  `z_quote_vol_60` 6 -> 3. Final, both rules: `z_quote_vol_60 >= 3`, `ret_60 <= -0.02`,
  `z_sell_rush_60 >= 4`, plus `jump_1800 >= 0.03` (dump, sid 100104) or `jump_1800 <= 0.03`
  (bear raid, sid 100601). `rev: 1`.
- Caveat: 30 events / 20 normal days is a small sample; the FA cost on the full test split is what
  `m4_sell` will show. No train bear raids exist (the real labels are pumps), so the bear-raid
  rule is only checked on SEMI-SYN.

### Deviations
1. Flag location: `features.sell_side`, not `detection.sell_side` (there is no detection config
   class; the flag changes the feature frame, so `FeatureCfg` fits). A wrap serializer drops it
   from every dump when False, so `config_hash`, the feature-cache key (`_features_hash`) and any
   stored config are byte-identical to pre-M4; with it on they differ (separate feature cache).
2. Alert phase is derived from the episode's **signature** (its full sub-technique), because
   `Alert.technique_id` holds the parent (`MM-T001`), which cannot tell .001/.002 from .003. It is
   set only when the Chief has at least one sell-side rule; otherwise every pump alert would gain
   `phase: "pump"` with the flag off and the alerts would no longer be bit-identical. `phase` None
   is also left out of the JSON dump, so old and flag-off alerts serialize exactly as before.
   Episodes attributed by `stat.cusum` (no signature) have phase None.
3. Signatures are conjunctions only, so the spec's "`z_sell_rush_60 >= 4` or `z_quote_vol_60 >= 6`"
   became both conditions (the pump-burst rule's shape). `jump_1800 < 0.03` is written `<= 0.03`
   (the only ops are >=, <=, abs<=); at exactly 0.03 both rules can fire and the Chief takes the
   first (dump).
4. Dump-phase sid is 100104: 100103 (the family's next number) is the Kamps-Kleinberg baseline.
5. Sell window: `[t_peak, t_peak + clip(dump_tau_s, 60 s, 1 h)]`; a right-censored dump
   (`dump_tau_s` None) gets the full hour and is flagged `dump_censored`. Labels with any other
   status are excluded from the sell-phase recall and counted per status. `recall_pump_phase` is
   `recall_any` (no extra per-event key). The measurement code moved from
   `scripts/measure_pumps.py` to `idsai/eval/pump_measure.py` (the script re-exports it; its tests
   pass unchanged).
6. `m4_sell` computes every feature with the flag on and scores once; `base` = the configured
   detectors minus the sell rules, `sell` = all. Because existing columns are bit-identical,
   `test:base` reproduces `real_t001 full` exactly (asserted in the smoke test). All records keep
   the invoking config's `config_hash` (so the report keeps both variants in one block) and add
   `sell_side` and `signatures_sell_sha`. `bear_raid:<v>` records exist for both variants (base
   as reference).
7. The bear-raid scenario ranges reuse the `pump_burst` test ranges for both regions (there are no
   measured bear raids to fit).

### Open risks
- A real dump usually follows its pump within the Chief's 300 s suppression, so the pump and dump
  rows merge into one episode named by its highest-F classified row (often a pump rule): such an
  episode overlaps the sell window (counts for `recall_sell_phase`) but its phase is "pump", so
  `recall_sell_phase_correct` can stay low by construction. The synthetic test shows exactly this
  (rise-then-crash -> one "pump" episode).
- The head's `m4_sell` run writes a second feature cache (`data/features/fs1-<hash with
  sell_side>`), about the size of the existing one, and adds one SEMI-SYN family's worth of loads.
- Not run here: the real `m4_sell` eval (only the synthetic smoke test).
