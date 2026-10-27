# Market IDS MVP (M0–M4 + LIVE trade mode) Implementation Plan — rev 2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the end-to-end Track T (trade-flow) Market IDS: Binance archive ingest → store → features → Red-Team injection → L1/L2/L3 detectors + baselines → Chief → evaluation on real Telegram pump labels and semi-synthetic attacks → FastAPI + SOC dashboard with REPLAY (Time Machine) and LIVE trade modes.

**Architecture:** One Python package `idsai` with contract-first modules. Feature math lives in **one shared NumPy kernel** used by both the batch path (eval) and the streaming path (replay/live), so batch/stream parity is exact by construction. Detectors are stateless functions of window feature rows; sequence state (baselines, CUSUM) lives in the feature layer. Chief converts detector scores to conformal p-values, fuses L1/L2/L3, and thresholds to an alert budget.

**Tech Stack:** Python 3.13, uv, pydantic v2, PyYAML, NumPy, Polars, DuckDB, scikit-learn (+SciPy), PyTorch CPU, httpx, websockets, FastAPI, uvicorn, pytest, hypothesis, ruff; static HTML/JS + TradingView lightweight-charts 5.2.1 (vendored).

**Spec:** `docs/superpowers/specs/2026-10-27-market-ids-design.md` (sections §N). Out of scope here (separate plan, M5–M7): Track B / BookBuilder / T002–T004, Ollama explainer, wash statistical tests, co-evolution, VIDEO pack.

**Revision 2 (2026-10-27):** incorporates independent plan review (window off-by-one, BTC-pair scaling, budget-by-construction, acceptance split, baseline leakage into fusion, label timezone check, shared kernel, paired event bootstrap, disk/caching, interface gaps).

## Global Constraints

- `requires-python = ">=3.13,<3.14"`; uv with `uv.lock` committed; torch from `https://download.pytorch.org/whl/cpu`.
- All timestamps `int64` µs UTC (`*_us`). Spot archive: ms before 2025-01-01, µs from 2025-01-01; normalize by magnitude (`> 1e14` ⇒ µs).
- `taker_side = "sell"` when `isBuyerMaker` is true, else `"buy"`.
- Allowed network hosts: `data.binance.vision`, `data-api.binance.vision`, `data-stream.binance.vision`, `raw.githubusercontent.com` (labels script). No order-sending code, no API keys.
- Red-Team injection takes the real current `mode` and raises `EthicsError` when it is `"live"`.
- 1 s bars indexed by bar start `t_us`. A window row at `t_end_us` covers bars `t_us ∈ [t_end − W, t_end)`; `close_t` = close of bar `t_end − 1 s`; `close_{t−W}` = close of bar `t_end − W − 1 s`. Nothing at or after `t_end` is used.
- Windows `[10, 60, 300, 1800]` s, stride `10` s. Baseline: prior 24 h, sampled every `60` s, refreshed every `300` s, valid when ≥ `360` non-NaN samples (= 6 h).
- Robust z scale = `max(IQR/1.349, tail, floor)` where for log-transformed features the scale is split: `tail = (q99 − med)/2.326` when `x ≥ med`, `(med − q01)/2.326` when `x < med`; and `(q99.5 − q0.5)/(2·2.576)` for ret/jump/ratio features (zero-inflated sparse pairs; rev 2.1 ruling). KK ratios are NaN when the 12 h mean is 0.
- Quote-volume transform `log1p(x / quote_unit)`, `quote_unit = {"BTC": 1e-4, "USDT": 1.0}` by quote asset suffix.
- No hour-of-day / minute-of-day features.
- Fusion uses only layers `{rule, stat, ai}`; `agreement` counts only those layers (max 3). Baselines run only alone (baselines experiment).
- Chief research budget is per symbol-day: `budget_per_symbol_day = 2.0`; eval sweeps `[0.1, 0.5, 2, 10]`. LIVE converts `live.budget_per_hour = 2.0` over `live.universe_size = 5` to per symbol-day. `suppress_for_s = 300`, `agreement_p = 0.01`, `attribution_min = 3.0`, unclassified `"MM-T000"`.
- Eval: an alert episode `[first_us, last_us]` matches a label if it overlaps `[t_start − 120 s, t_end + 300 s]`. Primary FA metric `fa_per_symbol_day` on clean test-normal days, and separately on hard negatives. No accuracy, no point-adjusted F1, F1 is not a headline. Precision is always printed with prevalence.
- Significance = paired bootstrap over events (same resample indices for both arms), `n_boot = 1000`, accept iff 95 % CI lower bound > 0 **and** mean diff ≥ `0.02`. Never "non-overlapping CIs".
- `split_seed = 0` fixed; run seeds `[0, 1, 2, 3, 4]` vary only AE init, IsolationForest, and scenario sampling. Deterministic detectors report bootstrap CIs, never seed std.
- `configs/signatures.yaml` thresholds may be tuned only on train-period data; its sha256 is recorded in every run.
- Every metric row carries `provenance ∈ {"REAL", "SEMI-SYN", "SYN"}`; every run records `git_sha, config_hash, signatures_sha, seed, model_versions`.
- Fused score is an "anomaly p-value" / "ความหายากเทียบตลาดปกติ", never a probability. User-facing alert text contains "ความผิดปกติทางสถิติ" and never accuses a person.
- Code/identifiers/comments in English; docs and UI copy in Thai.
- Hu et al. labels: download by script, never commit. Sapienza labels (MIT): vendored with LICENSE.
- UI shows "Binance Vision (CC BY-NC-SA 4.0)" and TradingView NOTICE + link. Default serve port `8800`.
- Two DuckDB files: `data/app.duckdb` (alerts, feedback, labels; written by `serve`) and `data/research.duckdb` (runs; written by `eval`). `serve` reads experiments only from `data/report.json`.

## Review Focus

1. Sparse BTC pairs on a satoshi price grid (most seconds empty, one tick = 0.5–3 %) — z stays finite and not pinned at the clip; ret floor follows tick size. Tests in Task 5.
2. Label timestamps in the wrong timezone or offset — alignment check against the real volume peak before any REAL metric is trusted. Tasks 4 and 20.
3. Missing archive day or missing D−1 — `skipped` reason recorded, no crash; NaN features ⇒ score NaN ⇒ p = 1. Tasks 5, 9, 10.
4. Duplicate / out-of-order trades (overlapping files, live reconnect, ms/µs boundary) — dedup `(symbol, trade_id)`, sort `(ts_us, trade_id)`, single monotonic µs series. Tasks 2, 3, 16.
5. Corrupt or partial download — retry once then `ChecksumError`; unverified files never ingested. Task 2.

---

## File Structure

```
IdsAI/
  pyproject.toml, uv.lock, README.md, NOTICE.md
  configs/default.yaml, splits.yaml, signatures.yaml, scenarios.yaml
  idsai/
    config.py                 # Settings + load_settings + config_hash + file_sha
    contracts/frames.py       # Polars schemas: TRADES, LABELS
    contracts/events.py       # Trade (pydantic)
    contracts/alert.py        # Alert, DetectorRef
    contracts/detector.py     # Detector ABC, Evidence
    bus.py
    data/archive.py, store.py, replay.py, live.py
    detection/taxonomy.yaml
    detection/features/registry.py   # FeatureSpec, feature_columns, model_features
    detection/features/bars.py       # trades → 1 s bars (batch) + BarBuilder (stream)
    detection/features/kernel.py     # shared NumPy math: prefix sums, baseline z, cusum, kk
    detection/features/windows.py    # batch driver: compute_windows
    detection/features/streaming.py  # stream driver: StreamingWindows
    detection/simulator/scenarios.py, inject.py, guard.py
    detection/rules.py, stats.py, registry.py
    ai/autoencoder.py, baselines.py, chief.py, learning.py, auditor.py
    eval/labels.py, eval/data/pump_telegram.csv, eval/data/LICENSE-sapienza
    eval/splits.py, metrics.py, experiments.py, report.py
    pipeline/batch.py, stream.py
    product/explainer/template.py
    product/api/state.py, app.py
    product/web/index.html, app.js, styles.css, vendor/…
    cli.py
  scripts/fetch_labels.py
  tests/…  (mirrors package; markers: slow, network)
  docs/ETHICS.md, DATA_LIMITATIONS.md, TEAM_GUIDE.md, REPORT.md
  .github/workflows/ci.yml
```

Team ownership (TEAM_GUIDE.md): Role 1 `data/`, `bus.py`, CI · Role 2 `detection/` · Role 3 `ai/`, `eval/` · Role 4 `product/`, docs.

---

### Task 1: Scaffold, Settings, contracts

**Files:**
- Create: `pyproject.toml`, `configs/default.yaml`, `configs/splits.yaml`, `idsai/__init__.py`, `idsai/config.py`, `idsai/contracts/{__init__,frames,events,alert}.py`
- Test: `tests/test_config.py`, `tests/contracts/test_contracts.py`

**Interfaces:**
- Produces:
  - `Settings` with sub-models:
    - `PathsCfg(data_dir: Path, models_dir: Path, app_db: Path, research_db: Path, report_json: Path)`
    - `FeatureCfg(windows_s=[10,60,300,1800], stride_s=10, baseline_s=86400, baseline_sample_s=60, baseline_refresh_s=300, min_baseline_samples=360, quote_unit={"BTC":1e-4,"USDT":1.0})`
    - `ChiefCfg(budget_per_symbol_day=2.0, suppress_for_s=300, agreement_p=0.01, attribution_min=3.0)`
    - `EvalCfg(before_s=120, after_s=300, seeds=[0,1,2,3,4], split_seed=0, n_boot=1000, min_effect=0.02, budgets_per_symbol_day=[0.1,0.5,2,10], hard_negative_k=20)`
    - `SplitsCfg(train, accept, cal, test: tuple[date, date], normal_days_per_symbol=4, min_gap_days=2, label_offset_s={"sapienza":0,"hu":0})` loaded from `configs/splits.yaml`
    - `AECfg(hidden=64, bottleneck=16, epochs=50, lr=1e-3, weight_decay=1e-5, batch_size=512, patience=5, clip=10.0, train_stride_s=60)`
    - `LiveCfg(symbols=["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT","PEPEUSDT"], ws_url="wss://data-stream.binance.vision/stream", budget_per_hour=2.0, universe_size=5, train_days=7, bar_grace_s=2)`
    - `detectors: list[str] = ["rules","stat.robust_z","stat.cusum","ai.autoencoder_trade"]`, `mode: Literal["replay","live"]`, `port: int = 8800`
  - `load_settings(path=None, overrides=None) -> Settings`; `config_hash(s) -> str` (sha256 canonical JSON, 12 hex); `file_sha(path) -> str` (sha256, 12 hex); `quote_unit_for(symbol: str, cfg: FeatureCfg) -> float` (suffix match, `KeyError` if none).
  - `configs/splits.yaml`: `train: [2018-01-01, 2019-06-30]`, `cal: [2019-07-01, 2019-12-31]`, `accept: [2020-01-01, 2020-06-30]`, `test: [2020-07-01, 2022-01-31]`.
  - `frames.TRADES` = `ts_us Int64, recv_us Int64, symbol String, price Float64, qty Float64, quote_qty Float64, taker_side String, trade_id Int64, n_fills Int32, source String`; `frames.LABELS` = `label_id String, source String, symbol String, t_start_us Int64, t_end_us Int64, technique_id String, family String, scenario_seed Int64 (nullable), provenance String`; `frames.empty(schema)`, `frames.conform(df, schema)` (raises `SchemaError` on missing column).
  - `events.Trade` pydantic mirroring TRADES (`taker_side: Literal["buy","sell"]`, `source: Literal["archive","live","sim"]`).
  - `alert.DetectorRef(name, layer, p, score)`; `alert.Alert`: `alert_id, ts_event_us, symbol, technique_id, signature_id: int | None, rev: int | None, severity (1–3), status ("open"|"updated"|"closed"), fused_p, detectors: list[DetectorRef], agreement (0–3), evidence: list[tuple[str,float,float]], window: tuple[int,int], dedup_key, suppress_for_s, config_hash, model_versions: dict[str,str], provenance ("live"|"replay"|"sim"), schema_v=1`.

- [ ] **Step 1: Init.** `uv init --package --name idsai --python 3.13`; `uv add pydantic pyyaml numpy polars duckdb scikit-learn httpx websockets fastapi "uvicorn[standard]" torch`; `uv add --dev pytest hypothesis ruff`. Torch via `[tool.uv.sources] torch = { index = "pytorch-cpu" }` + `[[tool.uv.index]] name="pytorch-cpu" url="https://download.pytorch.org/whl/cpu" explicit=true`. Pytest markers `slow`, `network`. Ruff line-length 100.
- [ ] **Step 2: Failing tests**

```python
def test_defaults_match_spec():
    s = load_settings()
    assert s.features.windows_s == [10, 60, 300, 1800] and s.features.stride_s == 10
    assert s.features.min_baseline_samples == 360 and s.features.baseline_sample_s == 60
    assert s.chief.budget_per_symbol_day == 2.0 and s.chief.attribution_min == 3.0
    assert s.eval.split_seed == 0 and s.port == 8800

def test_splits_strictly_ordered():
    sp = load_settings().splits
    assert sp.train[1] < sp.cal[0] <= sp.cal[1] < sp.accept[0] <= sp.accept[1] < sp.test[0]

def test_config_hash_stable_and_sensitive(): ...
def test_stride_must_divide_windows_and_sample(): ...   # stride 7 → ValidationError
def test_quote_unit_for(): assert quote_unit_for("BRDBTC", cfg) == 1e-4 and quote_unit_for("PEPEUSDT", cfg) == 1.0
def test_conform_rejects_missing_column(): ...
def test_alert_bounds(): ...  # severity 4 or agreement 4 → ValidationError
```
- [ ] **Step 3:** `uv run pytest tests/test_config.py tests/contracts -v` → FAIL.
- [ ] **Step 4:** Implement (YAML → deep-merge overrides → `model_validate`; validators: splits ordered, stride divides windows and `baseline_sample_s`, sample divides refresh).
- [ ] **Step 5:** tests PASS; `uv run ruff check .` clean.
- [ ] **Step 6: Commit** `feat: scaffold idsai package, settings and contracts`.

### Task 2: Archive download + CHECKSUM + zip reader

**Files:** Create `idsai/data/archive.py`; Test `tests/data/test_archive.py`

**Interfaces:**
- Produces:
  - `archive_url(symbol, day: date, kind: Literal["trades","aggTrades"]="trades") -> str` = `https://data.binance.vision/data/spot/daily/{kind}/{SYM}/{SYM}-{kind}-{YYYY-MM-DD}.zip`.
  - `download_day(symbol, day, dest_dir: Path, client: httpx.Client, kind="trades") -> Path | None` — None on 404; verifies `<zip>.CHECKSUM` sha256; skips an existing verified file; 429/418 → sleep `Retry-After` and retry (≤ 3); mismatch → delete, retry once, then `ChecksumError`.
  - `read_trades_zip(path, symbol) -> pl.DataFrame` (TRADES; `source="archive"`, `recv_us=ts_us`, `n_fills=1`; sorted `(ts_us, trade_id)`). Columns `id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch`; tolerate a header row; booleans `True/true/False/false`.

- [ ] **Step 1: Failing tests** (`httpx.MockTransport`, zips built in `tmp_path`): URL pattern for `BRDBTC 2020-06-01`; 404 → None; checksum mismatch → 2 zip requests then `ChecksumError` and no zip left; 429 honours `Retry-After` (monkeypatched sleep); 2024-12-31 (ms) + 2025-01-01 (µs) concat is sorted and all `ts_us > 1.7e15`; `isBuyerMaker` True/False → `sell`/`buy`.
- [ ] **Step 2–4:** FAIL → implement (`zipfile` + `pl.read_csv(has_header=False)`; header = first field non-numeric) → PASS.
- [ ] **Step 5: Commit** `feat(data): archive download with checksum and trade normalization`.

### Task 3: Store

**Files:** Create `idsai/data/store.py`; Test `tests/data/test_store.py`

**Interfaces:**
- Produces `class Store(data_dir: Path, db_path: Path)`:
  - `write_trades(df)` → `trades/symbol={S}/date={YYYY-MM-DD}/part.parquet`, replaces the partition, dedups `(symbol, trade_id)`; records partition sha in `trades/_index.parquet`.
  - `read_trades(symbol, start_us, end_us) -> pl.DataFrame` ([start, end), sorted; empty TRADES frame if absent); `has_day(symbol, day) -> bool`; `partition_sha(symbol, day) -> str | None`; `last_trade_before(symbol, ts_us) -> dict | None`.
  - `write_labels(df)`, `read_labels(symbol=None)`.
  - `insert_alert(a)`, `upsert_alert(a)` (by `alert_id`), `list_alerts(limit=200, symbol=None)`, `get_alert(id)`, `insert_feedback(alert_id, verdict: Literal["tp","fp","unsure"], analyst)`, `list_feedback(alert_id)`.
  - `insert_run(record: dict)`, `list_runs(experiment=None) -> list[dict]` (`metrics_json` TEXT).

- [ ] **Step 1: Failing tests:** roundtrip; same day written twice ⇒ same row count; duplicate ids deduped; out-of-order input read back sorted; `[start, end)` boundaries; alerts upsert/feedback/runs roundtrip; empty read has TRADES schema; partition sha changes when content changes.
- [ ] **Step 2–4:** FAIL → implement (`pl.scan_parquet` + filters; DuckDB tables created on open) → PASS.
- [ ] **Step 5: Commit** `feat(data): parquet trade store and duckdb tables`.

### Task 4: Real pump labels + alignment check

**Files:** Create `idsai/eval/labels.py`, `idsai/eval/data/pump_telegram.csv` (copy of SystemsLab-Sapienza `pump_telegram.csv`, 1110 lines incl. header), `idsai/eval/data/LICENSE-sapienza` (MIT text fetched from that repo), `scripts/fetch_labels.py`; Test `tests/eval/test_labels.py`

**Interfaces:**
- Produces:
  - `load_sapienza(path, offset_s=0) -> LABELS` — `exchange == "binance"`; `symbol = f"{symbol}BTC"`; `t_start_us = t_end_us = date + hour UTC + offset`; `source="sapienza"`, `technique_id="MM-T001"`, `family="telegram_pump"`, `provenance="REAL"`.
  - `load_hu(path, offset_s=0) -> LABELS` — TSV `channel_id, session_id, coin, exchange, pair, timestamp`; keep `exchange == "binance" and pair == "BTC"`; `symbol = coin + "BTC"`; `source="hu"`.
  - `merge_dedup(frames, tol_s=300) -> LABELS` — per symbol, sort by time, single-linkage clusters (consecutive gap ≤ `tol_s`); one label per cluster at earliest time; `source` = sorted unique joined by `+`; `label_id` = sha1(`symbol|t_start_us`)[:12].
  - `label_alignment(store, labels, half_window_s=10800) -> pl.DataFrame` — per label: `lag_s` = start of the 1-min bucket with max quote volume within `t0 ± half_window_s`, minus `t0`; `has_data`.
  - `scripts/fetch_labels.py` → first verifies the Hu repo path by listing `https://api.github.com/repos/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency/contents/Data/Telegram/Labeled`, then downloads `PD_logs_cleaned.txt` to `data/labels/hu_pd_logs.tsv` (gitignored) and checks the header equals the 6 columns above.

- [ ] **Step 1: Failing tests** (inline fixtures): binance-only + symbol suffix + exact µs for `2019-06-01 16:00`; offset applied; Hu keeps only binance BTC; dedup merges 16:00/16:02/16:06 chain into one (single-linkage) and keeps 16:00 vs 16:10 separate; `label_alignment` on a synthetic store with a volume spike 60 s after t0 returns `lag_s == 60`.
- [ ] **Step 2–4:** FAIL → implement → PASS; vendored file sanity: `load_sapienza(...).height == 520`.
- [ ] **Step 5: Commit** `feat(eval): telegram pump labels with dedup and alignment check`.

### Task 5: Feature registry, bars, shared kernel, batch windows

**Files:** Create `idsai/detection/features/{__init__,registry,bars,kernel,windows}.py`; Test `tests/detection/test_bars.py`, `tests/detection/test_windows.py`

**Interfaces:**
- Consumes: `TRADES`, `FeatureCfg`, `quote_unit_for`.
- Produces:
  - `FeatureSpec(name, unit, desc_th, transform: Literal["none","log1p","log1p_quote"], floor: Literal["log","ret","ratio"])`; `REGISTRY`; `feature_columns(cfg) -> list[str]`; `model_features(cfg) -> list[str]` = all `z_*` + `buy_imb_*` + `round_frac_*` + `cusum_*` (deterministic order).
  - `trades_to_bars(trades, symbol, start_us, end_us) -> pl.DataFrame` — one row per second `t_us` in `[start, end)`: `symbol, t_us, n_trades, buy_quote, sell_quote, quote_vol, open, high, low, close, rush_orders, max_trade_quote, pingpong, n_round`. Empty seconds: counts 0, OHLC = previous close; before the first trade OHLC = NaN.
    - `rush_orders`: distinct `ts_us // 1000` in the second with ≥ 2 buy-taker trades.
    - `pingpong`: consecutive trade pairs, equal qty and price, opposite side, `Δts ≤ 50_000 µs`.
    - `n_round`: `qty / 10**floor(log10(qty))` within `1e-9` of an integer.
  - `kernel.py` (pure NumPy, used by batch and stream):
    - `prefix(x: np.ndarray) -> np.ndarray` cumulative sums (float64; counts int64); `window_sum(P, i, w) = P[i] − P[i − w]`.
    - `raw_row(bar_arrays, i_end: int, cfg, quote_unit) -> dict[str, float]` — for each W: `n_trades_W, quote_vol_W, buy_quote_W, rush_W, pingpong_W, round_frac_W, buy_imb_W, ret_W, jump_W` from bars `[i_end − W, i_end)`. `buy_imb = 0` when `buy + sell < 1e-12`, else `(b − s)/(b + s)`. `ret_W = ln(close[i_end−1] / close[i_end−W−1])`, `jump_W = max(high[i_end−W:i_end]) / close[i_end−W−1] − 1` (NaN if either close NaN). `round_frac_W = n_round/n_trades` (0 if no trades).
    - `transform(name, x, quote_unit)` — `log1p(x)` or `log1p(x / quote_unit)` per spec.
    - `Baseline` — holds samples taken every `baseline_sample_s` of transformed raw rows plus 1 s closes; `refresh(t_us)` (every `baseline_refresh_s`) computes per feature `med, q25, q75` with `np.nanquantile(..., method="linear")` over samples in `[t − 86400 s, t)`, and `tick_rel` = median of positive `|Δclose|/close` over 1 s closes in the same range; `z(row) -> dict` with `scale = max(iqr/1.349, floor)`, floor `0.25` (log), `max(1e-4, tick_rel)` (ret/jump), `0.05` (ratio); NaN if valid samples < `min_baseline_samples`; `z = clip((x − med)/scale, −50, 50)`.
    - `cusum_step(s_prev, z) = min(max(0, s_prev + clip(z, −10, 10) − 0.5), 100)` with NaN z ⇒ 0; applied to `z_ret_10` → `cusum_ret` and `z_buy_imb_10` → `cusum_imb`.
    - `kk_row(history, i_end)` — `kk_vol_ratio = quote_vol_1800 / mean(quote_vol_1800 at i_end − k·1800, k=1..24)`, `kk_price_ratio = close / mean(close at same points)`; NaN if any missing.
  - `compute_windows(bars, cfg, quote_unit) -> pl.DataFrame` — rows at every stride boundary `t_end_us` inside the bars range; columns `symbol, t_end_us` + raw + `z_<raw>` + `cusum_ret, cusum_imb, kk_vol_ratio, kk_price_ratio`. Iterates the kernel in time order (vectorized prefix sums; Baseline refresh only every 300 s).

- [ ] **Step 1: Failing bar tests:** empty seconds filled; leading NaN OHLC; two buys in one ms ⇒ `rush_orders == 1`, sells don't count; pingpong at 10 ms = 1 and 60 ms = 0; `n_round` counts `0.5, 20, 300` not `0.37`.
- [ ] **Step 2: Failing window tests:**

```python
def test_window_sums_exact():  # 1 trade/s, quote 2.0
    w = compute_windows(bars_const(hours=8), cfg, 1.0)
    assert w["n_trades_60"][-1] == 60 and w["quote_vol_300"][-1] == 600.0

def test_trade_at_t_end_excluded():
    w = compute_windows(bars_with_trade_at(t_end), cfg, 1.0)
    assert row(w, t_end)["n_trades_10"] == 0 and row(w, t_end + 10e6)["n_trades_10"] == 1

def test_z_nan_until_6h_of_samples(): ...  # 5 h of data ⇒ all z NaN; 7 h ⇒ finite

def test_satoshi_grid_sparse_pair():  # price ∈ {100, 101} sat, 1 trade / 5 min, quote ~0.001 BTC
    w = compute_windows(bars_satoshi_sparse(hours=30), cfg, 1e-4)
    z = w.select(pl.col("^z_ret_.*$")).to_numpy()
    assert np.nanmax(np.abs(z)) < 50  # not pinned at clip by tick bounce

def test_btc_pump_z_reaches_rule_threshold():  # 2 BTC/min burst on a 0.01 BTC/min pair
    assert row(compute_windows(bars_btc_burst(), cfg, 1e-4), t_peak)["z_quote_vol_60"] >= 6

def test_cusum_on_10s_rows(): ...  # single +3σ 10 s move adds ≤ 2.5, not ×6

@settings(deadline=None, max_examples=10)
@given(st.integers(min_value=1, max_value=3600))
def test_causal(extra_s): ...  # appending future bars never changes existing rows

def test_no_time_of_day_columns(): ...

@pytest.mark.slow
def test_one_symbol_day_under_5s(): ...  # 2 days of bars, rows for D only
```
- [ ] **Step 3–5:** FAIL → implement → PASS.
- [ ] **Step 6: Commit** `feat(detection): 1s bars and shared causal feature kernel`.

### Task 6: Red-Team simulator (T001 burst/stealth, T005 pingpong)

**Files:** Create `idsai/detection/simulator/{__init__,scenarios,inject,guard}.py`, `configs/scenarios.yaml`; Test `tests/detection/test_simulator.py`

**Interfaces:**
- Produces:
  - `EthicsError(RuntimeError)`; `assert_injection_allowed(mode: str)` raises for `"live"`.
  - `Scenario`: `family: Literal["pump_burst","pump_stealth","wash_pingpong"], technique_id, sub_technique, t_start_us, duration_s, impact, vol_mult, rush_size, dump_tau_s, seed, region: Literal["train","test"]`.
  - `sample_scenario(family, region, t_start_us, rng, ranges: dict) -> Scenario`.
  - `base_quote_rate(prev_day_trades: pl.DataFrame) -> float` = `max(total_quote / 86400, median_trade_quote / 60)`; `0.0` raises `ValueError` (no trades ⇒ caller skips).
  - `inject(trades, sc, base_quote_per_s: float, mode: str) -> tuple[pl.DataFrame, pl.DataFrame]` — calls `assert_injection_allowed(mode)` first; returns new trades (sorted) and a one-row LABELS frame (`provenance="SEMI-SYN"`, `t_end_us = t_start_us + duration_s·1e6`).
- `configs/scenarios.yaml` (train | test region; train ranges are re-fit from train-period real pumps in Task 20 Step 3):
  - `pump_burst`: duration 30–180 s; impact 0.05–0.30 | 0.03–0.05; vol_mult 5–30; rush_size 3–8; dump_tau 120–600 s.
  - `pump_stealth`: duration 1800–5400 s | 5400–7200 s; impact 0.02–0.08; vol_mult 1.2–2.0; rush_size 1.
  - `wash_pingpong`: duration 300–1800 s; vol_mult 2–5 | 1.5–2; impact 0.
- Model: synthetic taker-buy trades at Poisson rate giving `vol_mult × base` extra quote/s; bursts arrive in clusters of `rush_size` fills sharing one ms; price multiplier `m(t)` linear 1 → `1+impact` over the duration then `1 + impact·exp(−(t − t_peak)/dump_tau)`; every trade with `ts ≥ t_start` gets `price × m(t)` (rounded to the symbol's observed price tick). Pingpong: buy then sell 1–20 ms later, equal round qty, current price. Synthetic trades `source="sim"`, negative unique `trade_id`.

- [ ] **Step 1: Failing tests:** determinism by seed; pre-`t_start` trades unchanged; peak price ≥ `(1 + 0.9·impact) × pre-price`; burst ms-clusters of `rush_size`; pingpong equal qty/opposite sides/price unchanged; **sparse pair (one trade per 5 min) injection adds quote volume ≥ 0.5 × expected**; `mode="live"` raises `EthicsError`; label technique ids and provenance; test-region samples inside test ranges.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(redteam): semi-synthetic pump and wash-like injection`.

### Task 7: Detector contract, rules, stats, baselines, registry, taxonomy

**Files:** Create `idsai/contracts/detector.py`, `idsai/detection/{rules,stats,registry}.py`, `idsai/ai/baselines.py`, `configs/signatures.yaml`, `idsai/detection/taxonomy.yaml`; Test `tests/detection/test_detectors.py`

**Interfaces:**
- Produces:
  - `Evidence(top: list[tuple[str, float, float]])` ≤ 5, sorted by contribution desc.
  - `class Detector(ABC)`: `name, layer: Literal["rule","stat","ai","baseline"], track: Literal["T","B"], technique_ids: list[str]` (`["ANY"]` = agnostic), `requires: list[str]`, `signature_id: int | None = None`, `rev: int | None = None`; `fit(normal: pl.DataFrame) -> None` (default no-op); `score(windows: pl.DataFrame) -> np.ndarray` (float64, NaN if a required feature is NaN); `fired(scores) -> np.ndarray[bool]` (default all True for non-rules); `evidence(row: dict) -> Evidence`; `state_dict()`, `load_state_dict(d)`.
  - `SignatureDetector(sig)` — name `rule.{sid}` (layer `rule`) or `baseline.kk` (sid 100103); `score = min_i dir_i·(x_i − thr_i)/max(|thr_i|, 1e-9)` (`>=` → +1, `<=` → −1, `abs<=` → `(thr − |x|)/thr`); `fired = score ≥ 0`.
  - `RobustZDetector(features)` `stat.robust_z` (`["ANY"]`, max of listed `z_*`); `CusumDetector()` `stat.cusum` (`["MM-T001"]`, `max(cusum_ret, cusum_imb)`).
  - `IsolationForestDetector(features, seed)` `baseline.iforest` (`−score_samples`), `PCADetector(features, n_components=0.95)` `baseline.pca`, `RandomDetector(seed)` `baseline.random` (uniform from hash of `(symbol, t_end_us, seed)`); all `["ANY"]`, features default `model_features(cfg)`.
  - `build_detectors(names, cfg) -> list[Detector]` — `"rules"` = every signature with `layer: rule` (excludes `baseline.kk`); unknown name → `KeyError` listing valid names.
- `configs/signatures.yaml` (initial, frozen before test runs; tune only on train period):
  - `100101` MM-T001.001 burst (rule): `z_quote_vol_60 >= 6`, `ret_60 >= 0.02`, `z_rush_60 >= 4`.
  - `100102` MM-T001.002 stealth (rule): `cusum_imb >= 8`, `ret_1800 >= 0.03`, `buy_imb_1800 >= 0.2`.
  - `100103` Kamps–Kleinberg adapted to 30-min candles (baseline): `kk_vol_ratio >= 4.0`, `kk_price_ratio >= 1.05`.
  - `100501` MM-T005.001 wash-like (rule): `z_pingpong_300 >= 5`, `ret_300 abs<= 0.002`, `z_quote_vol_300 >= 3`.
- `taxonomy.yaml`: MM-T000…T005 with `name_th, definition_th, reference, observability_th, sub_techniques`.

- [ ] **Step 1: Failing tests:** margin math at threshold = 0 and `abs<=`; NaN required ⇒ NaN; `fired`; robust_z max; random reproducible per seed; iforest/pca outlier > p99 of normal; `build_detectors(["rules"])` → sids `{100101, 100102, 100501}` only; `build_detectors(["baseline.kk"])` → layer `baseline`; unknown name `KeyError`; evidence ≤ 5 sorted.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(detection): detector contract, signatures, stats and baselines`.

### Task 8: Autoencoder, model registry, training mask

**Files:** Create `idsai/ai/autoencoder.py`, `idsai/ai/learning.py`; Test `tests/ai/test_autoencoder.py`, `tests/ai/test_training_mask.py`

**Interfaces:**
- Produces:
  - `AutoencoderDetector(features: list[str], cfg: AECfg, seed: int)` — `ai.autoencoder_trade`, layer `ai`, `["ANY"]`. `fit`: keep rows at `train_stride_s` spacing, drop NaN rows, `RobustScaler`, clip ±`clip`, last 10 % (time order) for early stopping; `torch.manual_seed(seed)`, `torch.use_deterministic_algorithms(True)`, `num_threads` fixed. `score` = per-row MSE; `evidence` = top-5 per-feature squared error.
  - `ModelRegistry(models_dir)`: `save(det, meta) -> str` (`models/{name}/v{n}/` with `state.*` + `meta.json`: `data_hash, config_hash, seed, created_at, train_range`), `load(name, version=None) -> tuple[dict, dict]`, `versions(name)`.
  - `training_mask(windows, labels, margin_s=3600) -> tuple[pl.Series, float]` — keep-mask False only within a label ± margin of the same symbol; returns excluded fraction. No alert parameter.

- [ ] **Step 1: Failing tests:** ROC-AUC > 0.9 on 20 k correlated-Gaussian normal + 200 shifted rows; same seed ⇒ `np.array_equal` scores; save → load → identical scores; NaN row ⇒ NaN; `@slow` fit 100 k × 40 < 120 s; mask excludes only the labeled symbol ± margin; `"alerts" not in inspect.signature(training_mask).parameters`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(ai): autoencoder detector, model registry and label-only training mask`.

### Task 9: Chief

**Files:** Create `idsai/ai/chief.py`; Test `tests/ai/test_chief.py`

**Interfaces:**
- Consumes: detectors, `ChiefCfg`, `Alert`.
- Produces:
  - `class CalibrationError(ValueError)`.
  - `Chief(detectors, cfg: ChiefCfg, config_hash: str, model_versions: dict[str, str])`.
  - `calibrate(cal: pl.DataFrame)` — `cal` has `symbol, t_end_us, score__<name>`; stores sorted finite scores per detector; records `n_cal`, `cal_symbol_days`.
  - `pvalues(name, scores) -> np.ndarray` — `(1 + #{s_i ≥ s}) / (n + 1)`; NaN ⇒ 1.0; for rule detectors, `fired == False` ⇒ 1.0.
  - `fuse(scored) -> pl.DataFrame` adds `p__<name>`, `F = max_d (−log10 p_d)` over fused detectors (layers rule/stat/ai; if the Chief holds only baselines, over those), `best_detector`, `agreement` (layers in {rule, stat, ai} with any p < `agreement_p`), `technique`: highest −log10 p among **fired rules** → its technique; else `stat.cusum` with −log10 p ≥ `attribution_min` → `MM-T001`; else `MM-T000`.
  - `fit_threshold(cal_fused, budget_per_symbol_day: float) -> float` — candidate τ = unique F values descending; returns the smallest τ whose `episodes(...)` rate per symbol-day on `cal_fused` ≤ budget (scan, then verify); raises `CalibrationError` if `budget_per_symbol_day / rows_per_symbol_day < n_detectors / (n_cal + 1)`.
  - `episodes(fused, tau=None, provenance="replay") -> list[Alert]` — rows with `F ≥ τ` grouped **per symbol** (gap ≤ `suppress_for_s` merges); `window = (first, last)`, `ts_event_us = first`, technique/evidence/signature from the peak row, `fused_p = 10**−F_peak`, `severity = 1 + (agreement ≥ 2) + (agreement ≥ 3)`, `dedup_key = f"{symbol}|{first // 300_000_000}"`, `config_hash`, `model_versions`.
  - `ChiefStream(chief)`: `step(row) -> Alert | None` — open on first exceedance, `updated` when F peak rises, `closed` after `suppress_for_s` quiet; alert ids equal those `episodes()` would produce.
  - `state_dict()/load_state_dict()`.

- [ ] **Step 1: Failing tests:**

```python
def test_pvalue_formula():
    ch.calibrate(cal_with("stat.cusum", [1, 2, 3, 4]))
    assert ch.pvalues("stat.cusum", np.array([5.0, 2.0, np.nan])).tolist() == [0.2, 0.8, 1.0]

def test_unfired_rule_gets_p1(): ...  # rule score −0.1 above every cal score ⇒ p == 1.0

def test_threshold_respects_budget_out_of_sample():  # 50 symbols × 10 days iid
    tau = ch.fit_threshold(ch.fuse(iid(seed=0)), 2.0)
    assert len(ch.episodes(ch.fuse(iid(seed=1)), tau)) / 500 <= 2.0 * 1.5

def test_threshold_guard_raises_on_tiny_calibration(): ...
def test_episode_merge_split_per_symbol(): ...  # t=0,200 s → 1; t=0,400 s → 2; technique flip inside → still 1
def test_technique_from_fired_rule_else_cusum_else_T000(): ...
def test_agreement_ignores_baseline_layer_and_caps_at_3(): ...
def test_severity_from_agreement_only(): ...
def test_stream_matches_batch_episode_ids(): ...
```
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(ai): chief with conformal calibration and alert budget`.

### Task 10: Batch pipeline per symbol-day

**Files:** Create `idsai/pipeline/{__init__,batch}.py`; Test `tests/pipeline/test_batch.py`

**Interfaces:**
- Produces:
  - `DayResult(symbol, day, windows: pl.DataFrame, labels: pl.DataFrame, skipped: Literal["missing_day","no_baseline","no_trades"] | None)`.
  - `load_day_windows(store, symbol, day, cfg, scenario: Scenario | None = None, include_next_minutes: int = 0) -> DayResult` — reads `[D−1, D+1 + include_next_minutes)`; `skipped="missing_day"` if `not has_day(D)`, `"no_baseline"` if `not has_day(D−1)`; with a scenario: `inject(..., base_quote_rate(D−1 trades), mode="replay")` (`"no_trades"` if D−1 empty). Keeps rows with `t_end_us ∈ [D, D+1 + include_next_minutes)`. Clean (no-scenario) windows cached as float32 Parquet at `data/features/{FEATURE_SET_V}-{config_hash(features)}/{symbol}/{day}-{partition_sha(D−1)}{partition_sha(D)}.parquet`; scenario windows are never cached.
  - `score_windows(windows, detectors) -> pl.DataFrame` adds `score__<name>`.

- [ ] **Step 1: Failing tests** (synthetic store `TESTBTC`): 8640 rows for D; skips for missing D / missing D−1; injected burst ⇒ `score__rule.100101` > 0 in label window; second clean call hits cache (patched `compute_windows` not called); scenario call never writes cache; `include_next_minutes=10` adds 60 rows.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(pipeline): batch symbol-day processing with cache`.

### Task 11: Splits, metrics, experiments, report

**Files:** Create `idsai/eval/{splits,metrics,experiments,report}.py`; Test `tests/eval/test_splits.py`, `tests/eval/test_metrics.py`, `tests/eval/test_experiments_smoke.py`

**Interfaces:**
- Produces:
  - `SymDay(symbol, day, include_next_minutes=0)`; `Split(train_normal, train_events, cal_normal, accept_events, accept_normal, test_events, test_normal, test_hard_negative: list[SymDay])`; `make_split(labels, store, cfg, seed=cfg.eval.split_seed) -> Split` — event days = days containing a label (plus `include_next_minutes=10` when `t0` is within 10 min of midnight); normal days = days of labeled symbols ≥ `min_gap_days` from any label of that symbol, `normal_days_per_symbol` sampled per period with `seed`, only days with D and D−1 in the store; `test_hard_negative` = top `hard_negative_k` test-normal days by realized 1-min volatility (moved out of `test_normal`); asserts period order and no label within `min_gap_days` of a normal day.
  - `match_events(alerts, labels, before_s, after_s) -> MatchResult` (`n_events, detected_any: dict[label_id, bool], detected_correct: dict[label_id, bool], n_alerts, n_alerts_matched, ttd_s: dict[label_id, float]`) — overlap rule from Global Constraints; TTD = `first_us − t_start` of the earliest matching episode (may be negative).
  - `summarize(mr, symbol_days: float, n_windows_pos: int, n_windows: int) -> dict` → `recall_any, recall_correct, precision, prevalence, fa_per_symbol_day, ttd_median_s, ttd_p90_s, f1` (f1 stored, not a headline).
  - `window_pr_auc(F: np.ndarray, y: np.ndarray) -> float` (`average_precision_score`).
  - `bootstrap_ci(per_event, n_boot, seed)`; `paired_bootstrap(a, b, n_boot, seed) -> tuple[float, tuple[float, float]]` (a, b shape `(n_seeds, n_events)`, mean over seeds, same event resample indices).
  - `run_experiment(name, cfg, store, seed) -> list[dict]`; every record: `run_id, experiment, variant, budget, seed, provenance, git_sha, config_hash, signatures_sha, model_versions, split_def, metrics, per_event` and is written with `store.insert_run` (research db). Protocol for all: AE fit on `train_normal` windows with `training_mask`; Chief calibrated and τ fit on `cal_normal`; metrics at every budget in `budgets_per_symbol_day`.
    - `real_t001` — full stack on `test_events ∪ test_normal`, `REAL`; FA also reported on `test_hard_negative`.
    - `semisyn` — per test-normal day and family one test-region scenario (seeded), `SEMI-SYN`, per family.
    - `ablation` — `rules`, `stats` (`stat.robust_z`+`stat.cusum`), `ai`, `rules+stats`, `all` on the `real_t001` and `semisyn` sets.
    - `baselines` — each of `baseline.random`, `baseline.kk`, `baseline.iforest`, `baseline.pca`, `stat.robust_z`, `ai.autoencoder_trade` alone through Chief.
    - `ref_supervised` — RandomForest(200, max_depth=5, seed) on `train_events` windows (positive = `[t0, t0+300 s]`) + `train_normal`, using `model_features`; RF probability treated as a detector score through the same Chief; labeled "supervised reference".
    - `retrain_drift` — AE v1 trained on train-normal days in `[2018-01-01, 2018-12-31]`, v2 on `[2018-07-01, 2019-06-30]`; both evaluated on `accept_events ∪ accept_normal` (never test); records per-event detection for the Task 12 decision.
  - `build_report(research_db, out_md, out_json)` — tables per experiment × variant × budget: mean over seeds + bootstrap CI (deterministic detectors marked "deterministic: CI from event bootstrap"); header shows total run count and signatures sha; every row has provenance; AE-vs-each-baseline paired bootstrap at budget 2; notes: autocorrelation weakens the conformal guarantee; latency excludes bar close + stride wait.

- [ ] **Step 1: Failing metric tests:** overlap edges (episode ending at `t0 − 121 s` misses, ending at `t0 − 119 s` hits; episode starting 10 min early but lasting into the window hits and is not counted as FA); correct vs any (MM-T000 counts only for any); `fa_per_symbol_day` units; negative TTD; paired bootstrap uses identical indices (equal arrays ⇒ CI == (0, 0)); PR-AUC on toy data.
- [ ] **Step 2: Failing split tests:** period order; no normal day within 2 days of a label; hard negatives disjoint from test_normal; same seed ⇒ same split; midnight labels include next minutes.
- [ ] **Step 3: Failing smoke test:** synthetic research store (3 symbols × 16 days spanning overridden splits) → `semisyn` and `retrain_drift` records have the right provenance and finite metrics; `build_report` writes md + json containing `"runs: "`.
- [ ] **Step 4–6:** FAIL → implement → PASS.
- [ ] **Step 7: Commit** `feat(eval): time splits, event metrics, experiment runner and report`.

### Task 12: Champion/challenger decision

**Files:** Modify `idsai/ai/learning.py`, `idsai/eval/report.py`; Test `tests/ai/test_learning.py`

**Interfaces:**
- Produces: `Decision(accept: bool, diff_mean: float, ci: tuple[float, float], reason_th: str)`; `compare(champion: np.ndarray, challenger: np.ndarray, n_boot: int, seed: int, min_effect: float = 0.02) -> Decision` — arrays `(n_seeds, n_events)` of 0/1 detection; uses `paired_bootstrap`; accept iff CI low > 0 and diff ≥ `min_effect`. Report adds the `retrain_drift` decision (v2 vs v1, `recall_any` at budget 2, accept split) to md + json, plus the before/after series for the dashboard chart.

- [ ] **Step 1: Failing tests:** clear improvement (challenger detects 30 more of 100 events in every seed) ⇒ accept; identical ⇒ reject with CI (0, 0); +0.01 consistent ⇒ reject by `min_effect`; noisy mixed ⇒ reject.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(ai): champion-challenger decision with paired event bootstrap`.

### Task 13: Auditor + leakage guard

**Files:** Create `idsai/ai/auditor.py`; Test `tests/ai/test_auditor.py`, `tests/test_leakage_guard.py`

**Interfaces:**
- Produces: `Check(name, status: Literal["ok","warn","fail"], value, detail_th)`; `HealthReport(checks, ts_us)` with `overall` = worst; `Auditor(cfg)`:
  - `alert_rate(n_alerts, symbol_days, budget_per_symbol_day)` (Poisson upper tail p < 0.01 ⇒ warn)
  - `calibration(pvals)` (KS vs uniform p < 0.01 ⇒ warn; < 200 values ⇒ ok with note)
  - `calibration_resolution(n_cal, n_detectors, rows_per_symbol_day, budget)` (rate < 10·D/(n+1) ⇒ warn)
  - `latency(samples_ms)` (p99 > 2000 ⇒ warn)
  - `trade_gaps(trades)` (non-sim `trade_id` jumps > 1)
  - `exclusion(frac)` (> 0.02 ⇒ warn)
  - `staleness(meta, now_us, max_age_days=30)`
  - `warmup(valid_samples, min_samples)` (LIVE/replay: warn with hours remaining)
  - `report(**inputs) -> HealthReport`
- Leakage guard: AST-scan modules under `idsai/detection/` (except `detection/simulator/`) and `idsai/ai/` (except `ai/learning.py`); none may import `idsai.eval` or reference `LABELS`.

- [ ] **Step 1–4:** failing tests for every ok/warn boundary + AST guard (and a planted violating module in `tmp_path` is caught) → implement → PASS.
- [ ] **Step 5: Commit** `feat(ai): auditor health checks and label leakage guard`.

### Task 14: Template explainer

**Files:** Create `idsai/product/explainer/{__init__,template}.py`; Test `tests/product/test_explainer.py`

**Interfaces:**
- Produces: `Explanation(text_th, facts: list[tuple[str, str]])`; `explain(alert) -> Explanation`; `numbers_in(text) -> list[float]`; `validate_numbers(text, allowed, rel_tol=0.01) -> bool`; `BANNED_TH = ["โกง", "มิจฉาชีพ", "ผู้ปั่น", "คนร้าย", "ฉ้อโกง"]`.
- Exact skeleton: `"พบความผิดปกติทางสถิติ ({technique_id} {name_th}) ที่ {symbol} เวลา {HH:MM:SS} UTC: {facts}; {agreement} จาก 3 ชั้นตรวจจับเห็นพ้อง; ความหายากเทียบตลาดปกติ p ≈ {fused_p:.1e}"`; facts from evidence using `desc_th` (z 1 decimal, returns % 2 decimals). MM-T000 name: `"ความผิดปกติที่ยังไม่จัดประเภท"`.

- [ ] **Step 1–4:** failing tests (phrase present; `validate_numbers` True against evidence-derived allowed list incl. time digits, agreement, 3, fused_p; no banned word; MM-T000 wording) → implement → PASS.
- [ ] **Step 5: Commit** `feat(product): thai template explainer with number validator`.

### Task 15: Bus, StreamingWindows, Time Machine, StreamPipeline

**Files:** Create `idsai/bus.py`, `idsai/detection/features/streaming.py`, `idsai/data/replay.py`, `idsai/pipeline/stream.py`; Modify `idsai/detection/features/bars.py` (add `BarBuilder`); Test `tests/test_bus.py`, `tests/detection/test_streaming_parity.py`, `tests/data/test_replay.py`, `tests/pipeline/test_stream.py`

**Interfaces:**
- Produces:
  - `Bus(maxsize=1000)`: `subscribe(topic) -> asyncio.Queue`, `publish(topic, msg)` (drop-oldest, `dropped[topic]`); topics `bars, scores, alerts, health`.
  - `BarBuilder(symbol, grace_s: int)`: `on_trade(t)`, `on_tick(t_us) -> list[dict]` — closes second `s` when `t_us ≥ s + 1 s + grace_s`; rows identical to `trades_to_bars`; late trades for closed seconds counted in `late`.
  - `StreamingWindows(symbol, cfg, quote_unit)`: `push_bar(bar) -> dict | None` — uses `kernel.py` exactly as `compute_windows` (same prefix-sum order, same Baseline refresh schedule) so output equals batch rows bit-for-bit.
  - `Replayer(trades, speed=1.0, start_us=None, preroll_from_us=None)`: async iterator of `("trade", Trade)` / `("tick", t_us)` (one tick per event-time second); pre-roll from `preroll_from_us` to `start_us` runs at max speed with a `("preroll_done", t_us)` marker; `pause()`, `resume()`, `set_speed(x)` (`0` = max), `seek(t_us)` (clamped; re-pre-rolls 24 h), `inject(sc, base_quote_per_s, mode)` (only trades after cursor), `cursor_us`.
  - `StreamPipeline(cfg, store, detectors, chief, bus, mode)`: `async run(source)` — source → BarBuilder → StreamingWindows → `score_windows` (1-row) → `ChiefStream.step` → `store.upsert_alert` + bus; nothing published before `preroll_done`; `latency_ms` deque(10_000) measured from tick receipt to Chief decision; keeps a ring buffer of the last 24 h of `{t_end_us, close, F, lane_rule, lane_stat, lane_ai}` for `/api/series`.

- [ ] **Step 1: Failing parity test:** 30 h random sparse trades → batch `compute_windows(trades_to_bars(...))` vs BarBuilder(grace 0) → StreamingWindows: `np.array_equal(..., equal_nan=True)` on every column.
- [ ] **Step 2: Failing replay tests:** order; empty-second ticks; seek clamps; pause/resume; inject after cursor only; `inject(mode="live")` raises; preroll marker emitted once.
- [ ] **Step 3: Failing pipeline test:** 2 days fixture + injected burst at speed 0 → ≥ 1 alert with `MM-T001` overlapping the label window; no alert before preroll_done; p99 latency < 2000 ms; bus received the alert.
- [ ] **Step 4–6:** FAIL → implement → PASS.
- [ ] **Step 7: Commit** `feat(stream): bus, streaming features with exact parity, time machine and stream pipeline`.

### Task 16: LIVE trade sensor

**Files:** Create `idsai/data/live.py`; Test `tests/data/test_live.py`

**Interfaces:**
- Produces: `LiveTradeSensor(symbols, url, rotate_s=23*3600)`: async iterator of the same tuples as `Replayer` (wall-clock ticks); URL `{url}?streams={s}@trade/...&timeUnit=MICROSECOND` (lowercase); `parse_trade(msg) -> Trade` (`e,E,s,t,p,q,T,m`; `m` true ⇒ `sell`; `recv_us` = local clock); dedup `(symbol, trade_id)` over the last 10 000; reconnect backoff 1→60 s; proactive reconnect after `rotate_s`; counters `reconnects, dupes`.
- Warm-up: `serve --mode live` pre-rolls yesterday's archive day (if published) through the same pipeline before connecting; Auditor `warmup` check shows hours remaining when baseline samples < 360.

- [ ] **Step 1: Failing tests:** `parse_trade` on a recorded payload; dedup across reconnect; fake `websockets.serve` server closing after 3 messages ⇒ sensor reconnects and continues; URL format. Network smoke test marked `network`: `BTCUSDT` yields ≥ 10 trades in 10 s.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(data): live trade sensor with reconnect and dedup`.

### Task 17: API

**Files:** Create `idsai/product/api/{__init__,state,app}.py`; Test `tests/product/test_api.py`

**Interfaces:**
- Produces: `AppState(cfg, store, mode, pipeline, replayer, auditor)`; `create_app(state) -> FastAPI`:
  - static `/` from `product/web`
  - `GET /api/alerts?symbol=&limit=` (Alert JSON + `explanation_th`); `GET /api/alerts/{id}` (+ Explanation + feedback; 404 unknown); `POST /api/alerts/{id}/feedback {verdict, analyst}` → 201
  - `GET /api/series?symbol&start_us&end_us` → `{t_end_us[], close[], F[], lane_rule[], lane_stat[], lane_ai[], tau}` from the pipeline ring buffer
  - `GET /api/symbols`, `GET /api/health`, `GET /api/experiments` (report JSON or `{"runs": 0}`), `GET /api/taxonomy`
  - `POST /api/replay/control {action: play|pause|speed|seek, value?}` → 409 in live
  - `POST /api/redteam/inject {family, region, offset_s}` → 403 with Thai message in live, else 202 + Scenario (calls `replayer.inject(..., mode=state.mode)`)
  - `WS /ws/stream` messages `{type: bar|score|alert|health, data}`, per-client 4 Hz throttle for bar/score; alerts never dropped.

- [ ] **Step 1–4:** failing TestClient tests for every endpoint (shapes, 404, 201, 403, 409, series lengths equal, ws receives a published alert) → implement → PASS.
- [ ] **Step 5: Commit** `feat(api): rest and websocket api`.

### Task 18: CLI, demo pack, docs, CI

**Files:** Create `idsai/cli.py` (`[project.scripts] idsai = "idsai.cli:main"`), `README.md`, `NOTICE.md`, `docs/{ETHICS,DATA_LIMITATIONS,TEAM_GUIDE}.md`, `.github/workflows/ci.yml`; Modify spec §3.3 to the frame-based Detector interface; Test `tests/test_cli.py`

**Interfaces:**
- Subcommands: `labels` (Sapienza + Hu if present, offsets from splits.yaml → app db); `labels --align` (runs `label_alignment`, prints lag histogram, exit 1 if median |lag| ≥ 120 s); `download --split research | --symbols S --days N` (includes D−1); `train [--seed N] [--profile research|live]` (live: last `train_days` of `live.symbols`, calibrate Chief on the last 2 days, save via ModelRegistry); `eval --experiment NAME|all --seeds 0,1,2,3,4`; `report`; `demo-pack` (3 labeled pump days + D−1, 20 normal days of the same small symbols, then trains and calibrates demo models into `data/demo/`); `serve --mode replay|live [--symbol S --day D] [--port 8800]`.
- Docs: ETHICS.md (§15 expanded); DATA_LIMITATIONS.md (§1 + label alignment result placeholder filled in Task 20 + "claims we do not make"); TEAM_GUIDE.md (roles, add-a-detector in 3 steps, tests, glossary); README quickstart (Thai): `uv sync` → `uv run idsai demo-pack` → `uv run idsai serve --mode replay`.
- CI: ubuntu + windows, Python 3.13, `uv sync --frozen`, `uv run ruff check .`, `uv run pytest -q -m "not network and not slow"`.

- [ ] **Step 1–4:** failing tests (`--help` lists all subcommands; serve default port 8800; `train --profile live` with empty symbols → error; `labels --align` exit code logic with a stubbed store) → implement → full `uv run pytest -q -m "not network"` PASS.
- [ ] **Step 5: Commit** `feat: cli, demo pack, docs and ci`.

### Task 19: SOC dashboard

**Files:** Create `idsai/product/web/{index.html,app.js,styles.css}`, `vendor/lightweight-charts.standalone.production.js` (from `https://unpkg.com/lightweight-charts@5.2.1/dist/lightweight-charts.standalone.production.js`; version in `vendor/VERSION`), `vendor/NOTICE-lightweight-charts`; Test `tests/product/test_web_static.py` + browser verification

**Interfaces:**
- Consumes only Task 17 endpoints and WS types.
- Produces hash routes `#/monitor`, `#/alerts`, `#/alerts/{id}`, `#/redteam`, `#/eval`, `#/taxonomy`, `#/about`:
  - Top bar: mode badge, event clock, Auditor status dot (+ warm-up text), alert-budget meter.
  - Monitor: price chart with alert markers; F lane with τ line; L1/L2/L3 lanes (from `/api/series` on load, then WS); symbol picker; replay controls (hidden in LIVE).
  - Alerts: triage table → detail (Thai explanation, evidence table, detector p-values + agreement, alert window shaded on chart, technique card, TP/FP/unsure).
  - Red-Team: family/region pickers, inject, time-to-detect timer; disabled with notice in LIVE.
  - Eval: tables from `/api/experiments` with REAL/SEMI-SYN/SYN badges, budget selector, run count, prevalence next to precision, retrain before/after chart with CI whiskers and the Decision text.
  - Taxonomy cards; About: limitations, ethics, attribution (Binance Vision CC BY-NC-SA 4.0, TradingView link via `attributionLogo: true`).
  - Clean Apple-like visual language: system font stack, CSS custom properties, light/dark via `prefers-color-scheme`, works at 360 px without horizontal scroll, no build step.

- [ ] **Step 1:** failing test: `GET /` contains `"Market IDS"`, `"Binance Vision"`, `"tradingview"`; vendor JS served with JS content type.
- [ ] **Step 2–3:** implement → PASS.
- [ ] **Step 4: Browser verification** (preview tools): `idsai serve --mode replay` on the demo pack; no console errors; Monitor shows chart and lanes; injecting a burst produces an alert in Alerts; dark mode and 375 px layout OK; screenshots saved.
- [ ] **Step 5: Commit** `feat(web): soc dashboard`.

### Task 20: Real run and honest report

**Files:** Create `docs/REPORT.md`; Modify `docs/DATA_LIMITATIONS.md`, possibly `configs/splits.yaml` (label offsets), `configs/scenarios.yaml` (train ranges), `configs/signatures.yaml` (train-only tuning)

- [ ] **Step 1:** `uv run idsai labels && uv run idsai download --split research` — record symbol-days, 404 count, disk.
- [ ] **Step 2:** `uv run idsai labels --align` — record the lag histogram per source in DATA_LIMITATIONS.md; if median |lag| ≥ 120 s for a source, set its `label_offset_s` from the train period only, rerun, and document.
- [ ] **Step 3:** From **train-period** real pumps only: measure vol multiple, price impact, rush cluster sizes and duration; set `scenarios.yaml` train ranges to their 10–90 % range; optionally adjust signature thresholds on train-period data; then freeze (`signatures_sha` recorded) and commit before Step 4.
- [ ] **Step 4:** `uv run idsai eval --experiment all --seeds 0,1,2,3,4` in the background; record wall time.
- [ ] **Step 5:** `uv run idsai report` → `docs/REPORT.md` + `data/report.json`.
- [ ] **Step 6:** Write the Thai discussion from the generated numbers only: targets met/missed (recall ≥ 0.8 per technique at budget 2, FA per symbol-day out-of-sample vs budget, hard-negative FA, p99 latency < 2 s), AE vs baselines (paired bootstrap), ablation, `retrain_drift` Decision, supervised reference gap, and limitations.
- [ ] **Step 7:** `uv run idsai serve --mode replay` and verify the Eval page matches REPORT.md.
- [ ] **Step 8: Commit** `docs: experiment report with real and semi-synthetic results`.
