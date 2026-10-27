# Market IDS MVP (M0–M4 + LIVE trade mode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the end-to-end Track T (trade-flow) Market IDS: Binance archive ingest → store → features → Red-Team injection → L1/L2/L3 detectors + baselines → Chief → evaluation on real Telegram pump labels and semi-synthetic attacks → FastAPI + SOC dashboard with REPLAY (Time Machine) and LIVE trade modes.

**Architecture:** One Python package `idsai` with contract-first modules. Batch path (eval) is vectorized Polars; streaming path (replay/live) is incremental NumPy and is pinned to the batch path by a parity test. Detectors are stateless functions of window feature rows; all sequence state (rolling baselines, CUSUM) lives in the feature layer. Chief converts detector scores to conformal p-values, fuses, and thresholds to an alert budget.

**Tech Stack:** Python 3.13, uv, pydantic v2, PyYAML, NumPy, Polars, DuckDB, scikit-learn (+SciPy), PyTorch CPU, httpx, websockets, FastAPI, uvicorn, pytest, hypothesis, ruff; static HTML/JS + TradingView lightweight-charts 5.x (vendored).

**Spec:** `docs/superpowers/specs/2026-10-27-market-ids-design.md` (sections referenced as §N). Out of scope for this plan: §4.2 Track B, §5 T002–T004, BookBuilder, Ollama explainer, wash statistical tests, co-evolution (M5–M7, separate plan).

## Global Constraints

- `requires-python = ">=3.13,<3.14"`; dependencies managed by uv with `uv.lock` committed; torch from `https://download.pytorch.org/whl/cpu`.
- All timestamps are `int64` microseconds UTC (`*_us`). Spot archive before 2025-01-01 is ms, from 2025-01-01 is µs; normalize by magnitude (`> 1e14` ⇒ µs).
- `taker_side = "sell"` when `isBuyerMaker` is true, else `"buy"`.
- Network hosts allowed: `data.binance.vision`, `data-api.binance.vision`, `data-stream.binance.vision`. No order-sending code, no API keys, no authenticated endpoints.
- Red-Team injection must raise `EthicsError` when mode is `"live"`.
- Windows `[10, 60, 300, 1800]` s, stride `10` s, baseline lookback `86400` s, minimum baseline `21600` s; features use only data ≤ `t_end_us`.
- No hour-of-day / minute-of-day features.
- Chief defaults: `budget_per_hour = 2.0` for `universe_size = 5`, `suppress_for_s = 300`, `agreement_p = 0.01`, `attribution_min = 2.0` (−log10 p), unclassified technique id `"MM-T000"`.
- Eval defaults: match window `[t_start − 120 s, t_end + 300 s]`, seeds `[0, 1, 2, 3, 4]`, `n_boot = 1000`. No accuracy, no point-adjusted F1. Every metric row carries `provenance ∈ {"REAL", "SEMI-SYN", "SYN"}`.
- Fused score is named "anomaly p-value" / "ความหายากเทียบตลาดปกติ" — never "probability of manipulation". User-facing alert text contains "ความผิดปกติทางสถิติ" and never accuses a person.
- Every run records `git_sha`, `config_hash`, `seed`, `model_versions`.
- Code, identifiers and code comments in English; docs and UI copy in Thai (technical terms may stay English).
- Hu et al. labels are downloaded by script, never committed. Sapienza labels (MIT) are vendored with their LICENSE.
- UI shows attribution "Binance Vision (CC BY-NC-SA 4.0)" and TradingView lightweight-charts NOTICE + link.

## Review Focus

1. Dead small-cap symbols (minutes with zero trades, IQR = 0) — robust z must stay finite via per-feature scale floors; windows still emitted. Test in Task 5.
2. Missing archive day (404) or missing D−1 — baseline < 6 h ⇒ NaN features ⇒ detector score NaN ⇒ Chief p = 1; eval counts the day as `skipped`, never crashes. Tests in Tasks 5, 9, 11.
3. Mixed timestamp units across the 2025-01-01 boundary — a 2024-12-31 + 2025-01-01 pair yields one monotonic µs series. Test in Task 2.
4. Duplicate / out-of-order trades (overlapping files, live reconnect) — dedup on `(symbol, trade_id)` and sort by `(ts_us, trade_id)` before bars. Tests in Tasks 3 and 16.
5. Corrupt or partial download (CHECKSUM mismatch) — retry once, then raise `ChecksumError`; unverified files are never ingested. Test in Task 2.

---

## File Structure

```
IdsAI/
  pyproject.toml, uv.lock, README.md, NOTICE.md
  configs/default.yaml        # all tunables (§14)
  configs/splits.yaml         # research time splits (§11.1)
  configs/signatures.yaml     # L1 rules (§6)
  configs/scenarios.yaml      # Red-Team param ranges, train/test regions (§5)
  idsai/
    config.py                 # Settings (pydantic) + load_settings + config_hash
    contracts/frames.py       # Polars schemas: TRADES, BARS, LABELS
    contracts/events.py       # Trade (pydantic) for streaming/API
    contracts/alert.py        # Alert, DetectorRef
    contracts/detector.py     # Detector ABC, Evidence
    bus.py                    # in-process pub/sub
    data/archive.py           # URLs, download+CHECKSUM, read zip → TRADES frame
    data/store.py             # Parquet trades + DuckDB tables
    data/replay.py            # Time Machine Replayer
    data/live.py              # LiveTradeSensor
    detection/taxonomy.yaml   # MM-T000..T005 (§13)
    detection/features/registry.py   # FeatureSpec table
    detection/features/bars.py       # trades → 1 s bars
    detection/features/windows.py    # bars → window features (batch)
    detection/features/streaming.py  # StreamingWindows (incremental)
    detection/simulator/scenarios.py # Scenario, sample_scenario
    detection/simulator/inject.py    # inject()
    detection/simulator/guard.py     # EthicsError, assert_injection_allowed
    detection/rules.py        # SignatureDetector
    detection/stats.py        # RobustZDetector, CusumDetector
    detection/registry.py     # build_detectors(cfg)
    ai/autoencoder.py         # AutoencoderDetector
    ai/baselines.py           # IsolationForest, PCA, Random
    ai/chief.py               # Chief (batch) + ChiefStream
    ai/learning.py            # ModelRegistry, champion/challenger, training mask
    ai/auditor.py             # Auditor checks → HealthReport
    eval/labels.py            # Sapienza/Hu loaders, dedup
    eval/data/pump_telegram.csv, eval/data/LICENSE-sapienza
    eval/splits.py, eval/metrics.py, eval/experiments.py, eval/report.py
    pipeline/batch.py         # process_day
    pipeline/stream.py        # StreamPipeline
    product/explainer/template.py
    product/api/app.py, product/api/state.py
    product/web/index.html, app.js, styles.css, vendor/…
    cli.py
  scripts/fetch_labels.py
  tests/…                     # mirrors package layout
  docs/ETHICS.md, DATA_LIMITATIONS.md, TEAM_GUIDE.md, REPORT.md
  .github/workflows/ci.yml
```

Team ownership (for TEAM_GUIDE.md): Role 1 `data/`, `bus.py`, CI · Role 2 `detection/` · Role 3 `ai/`, `eval/` · Role 4 `product/`, docs.

---

### Task 1: Scaffold, Settings, contracts

**Files:**
- Create: `pyproject.toml`, `configs/default.yaml`, `idsai/__init__.py`, `idsai/config.py`, `idsai/contracts/{__init__,frames,events,alert}.py`
- Test: `tests/test_config.py`, `tests/contracts/test_contracts.py`

**Interfaces:**
- Produces:
  - `Settings` (pydantic) with sub-models `PathsCfg(data_dir: Path, models_dir: Path, db_path: Path)`, `FeatureCfg(windows_s: list[int], stride_s: int, baseline_s: int, min_baseline_s: int)`, `ChiefCfg(budget_per_hour: float, universe_size: int, suppress_for_s: int, agreement_p: float, attribution_min: float)`, `EvalCfg(before_s: int, after_s: int, seeds: list[int], n_boot: int)`, `LiveCfg(symbols: list[str], ws_url: str)`, `detectors: list[str]`, `mode: Literal["replay","live"]`.
  - `load_settings(path: Path | None = None, overrides: dict | None = None) -> Settings`; `config_hash(s: Settings) -> str` (sha256 of canonical JSON, first 12 hex).
  - `frames.TRADES: dict[str, pl.DataType]` = `ts_us Int64, recv_us Int64, symbol String, price Float64, qty Float64, quote_qty Float64, taker_side String, trade_id Int64, n_fills Int32, source String`; `frames.LABELS` = `label_id String, source String, symbol String, t_start_us Int64, t_end_us Int64, technique_id String, family String, scenario_seed Int64 (nullable), provenance String`; `frames.empty(schema) -> pl.DataFrame`; `frames.conform(df, schema) -> pl.DataFrame` (casts, raises `SchemaError` on missing column).
  - `events.Trade` pydantic mirroring TRADES with `taker_side: Literal["buy","sell"]`, `source: Literal["archive","live","sim"]`.
  - `alert.DetectorRef(name: str, layer: str, p: float, score: float)`; `alert.Alert` with fields exactly as §3.4 plus `schema_v: int = 1`; `severity: int` constrained 1–3; `status: Literal["open","updated","closed"]`; `provenance: Literal["live","replay","sim"]`.

- [ ] **Step 1: Init project with uv.** `uv init --package --name idsai --python 3.13` in repo root, then add deps: `uv add pydantic pyyaml numpy polars duckdb scikit-learn httpx websockets fastapi "uvicorn[standard]" torch` and `uv add --dev pytest hypothesis ruff`. Configure `[tool.uv.sources] torch = { index = "pytorch-cpu" }` with `[[tool.uv.index]] name = "pytorch-cpu"`, `url = "https://download.pytorch.org/whl/cpu"`, `explicit = true`. Ruff: line-length 100.
- [ ] **Step 2: Write failing tests**

```python
def test_defaults_match_spec():
    s = load_settings()
    assert s.features.windows_s == [10, 60, 300, 1800]
    assert s.features.stride_s == 10 and s.features.baseline_s == 86400
    assert s.features.min_baseline_s == 21600
    assert s.chief.budget_per_hour == 2.0 and s.chief.universe_size == 5
    assert s.chief.suppress_for_s == 300 and s.eval.seeds == [0, 1, 2, 3, 4]

def test_config_hash_stable_and_sensitive():
    a, b = load_settings(), load_settings()
    assert config_hash(a) == config_hash(b) and len(config_hash(a)) == 12
    assert config_hash(load_settings(overrides={"chief": {"budget_per_hour": 3.0}})) != config_hash(a)

def test_stride_must_divide_windows():
    with pytest.raises(ValidationError):
        load_settings(overrides={"features": {"stride_s": 7}})

def test_conform_rejects_missing_column():
    with pytest.raises(SchemaError):
        conform(pl.DataFrame({"ts_us": [1]}), TRADES)

def test_alert_severity_bounds():
    with pytest.raises(ValidationError):
        Alert(**{**valid_alert_kwargs(), "severity": 4})
```
- [ ] **Step 3: Run** `uv run pytest tests/test_config.py tests/contracts -v` → FAIL (import errors).
- [ ] **Step 4: Implement** `config.py` (YAML → deep-merge overrides → `Settings.model_validate`), `configs/default.yaml` with the Global Constraints values, `contracts/*`.
- [ ] **Step 5: Run tests** → PASS. `uv run ruff check .` → clean.
- [ ] **Step 6: Commit** `feat: scaffold idsai package, settings and contracts`.

### Task 2: Archive download + CHECKSUM + zip reader

**Files:**
- Create: `idsai/data/archive.py`
- Test: `tests/data/test_archive.py`

**Interfaces:**
- Consumes: `frames.TRADES`, `frames.conform`.
- Produces:
  - `archive_url(symbol: str, day: date, kind: Literal["trades","aggTrades"] = "trades") -> str` → `https://data.binance.vision/data/spot/daily/{kind}/{SYM}/{SYM}-{kind}-{YYYY-MM-DD}.zip`.
  - `download_day(symbol, day, dest_dir: Path, client: httpx.Client, kind="trades") -> Path | None` (None on 404; verifies `.CHECKSUM` sha256; skips if a verified file exists; on 429/418 sleeps `Retry-After` then retries up to 3 times; checksum mismatch → delete, retry once, then `ChecksumError`).
  - `read_trades_zip(path: Path, symbol: str) -> pl.DataFrame` (TRADES schema, `source="archive"`, `recv_us = ts_us`, `n_fills = 1`, sorted by `(ts_us, trade_id)`). Columns without header: `id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch`; tolerate a header row; booleans `True/true/False/false`.

- [ ] **Step 1: Failing tests** (use `httpx.MockTransport` and zips built in `tmp_path`):

```python
def test_url_pattern():
    assert archive_url("BRDBTC", date(2020, 6, 1)) == \
        "https://data.binance.vision/data/spot/daily/trades/BRDBTC/BRDBTC-trades-2020-06-01.zip"

def test_404_returns_none(tmp_path): assert download_day("X", D, tmp_path, client_404) is None

def test_checksum_mismatch_retries_once_then_raises(tmp_path):
    with pytest.raises(ChecksumError): download_day("X", D, tmp_path, client_bad_sum)
    assert calls["zip"] == 2 and not list(tmp_path.glob("*.zip"))

def test_429_honours_retry_after(tmp_path, monkeypatch): ...  # sleep called with 1, then success

def test_ms_and_us_normalized(tmp_path):
    df = pl.concat([read_trades_zip(zip_ms_2024_12_31, "XBTC"), read_trades_zip(zip_us_2025_01_01, "XBTC")])
    assert df["ts_us"].is_sorted() and df["ts_us"].min() > 1_700_000_000_000_000

def test_taker_side_from_is_buyer_maker(tmp_path):
    df = read_trades_zip(zip_with_rows([(1, "True"), (2, "False")]), "XBTC")
    assert df["taker_side"].to_list() == ["sell", "buy"]
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** with `zipfile` + `pl.read_csv(has_header=False, new_columns=…)`; header detection = first field not numeric.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(data): archive download with checksum and trade normalization`.

### Task 3: Store (Parquet trades + DuckDB tables)

**Files:**
- Create: `idsai/data/store.py`
- Test: `tests/data/test_store.py`

**Interfaces:**
- Consumes: `TRADES`, `LABELS`, `Alert`.
- Produces: `class Store(root: Path, db_path: Path)` with
  - `write_trades(df) -> None` — partitions `trades/symbol={S}/date={YYYY-MM-DD}/part.parquet`; rewriting a partition replaces it; dedups `(symbol, trade_id)` keeping first.
  - `read_trades(symbol, start_us, end_us) -> pl.DataFrame` — inclusive start, exclusive end, sorted `(ts_us, trade_id)`; empty frame with TRADES schema when absent.
  - `has_day(symbol, day) -> bool`.
  - `write_labels(df)`, `read_labels(symbol: str | None = None) -> pl.DataFrame`.
  - `insert_alert(a: Alert)`, `list_alerts(limit=200, symbol=None) -> list[Alert]`, `get_alert(alert_id) -> Alert | None`, `insert_feedback(alert_id, verdict: Literal["tp","fp","unsure"], analyst: str)`.
  - `insert_run(record: dict)`, `list_runs(experiment: str | None = None) -> list[dict]` (DuckDB table `runs`, `metrics_json` as JSON text).

- [ ] **Step 1: Failing tests:** roundtrip equality; rewrite partition idempotent (row count unchanged after writing the same day twice); overlapping duplicate trade ids deduped; out-of-order input read back sorted; time filter boundaries; alerts/feedback/runs roundtrip; empty read returns TRADES schema.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (`pl.scan_parquet` with filters; DuckDB tables created on open). **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(data): parquet trade store and duckdb tables`.

### Task 4: Real pump labels (Sapienza + Hu)

**Files:**
- Create: `idsai/eval/labels.py`, `idsai/eval/data/pump_telegram.csv` (copy of SystemsLab-Sapienza `pump_telegram.csv`), `idsai/eval/data/LICENSE-sapienza` (MIT text from that repo), `scripts/fetch_labels.py`
- Test: `tests/eval/test_labels.py`

**Interfaces:**
- Produces:
  - `load_sapienza(path: Path) -> pl.DataFrame[LABELS]` — keep `exchange == "binance"`; `symbol = f"{symbol}BTC"`; `t_start_us = t_end_us = date + hour` (UTC); `source="sapienza"`, `technique_id="MM-T001"`, `family="telegram_pump"`, `provenance="REAL"`.
  - `load_hu(path: Path) -> pl.DataFrame[LABELS]` — TSV `channel_id, session_id, coin, exchange, pair, timestamp`; keep `exchange == "binance"`; `symbol = coin + pair`; `source="hu"`.
  - `merge_dedup(frames: list[pl.DataFrame], tol_s: int = 300) -> pl.DataFrame` — same symbol within `tol_s` ⇒ one label (earliest time, `source` joined like `"hu+sapienza"`); `label_id` = sha1 of `symbol|t_start_us` first 12 hex.
  - `scripts/fetch_labels.py` downloads Hu `Data/Telegram/Labeled/PD_logs_cleaned.txt` from `https://raw.githubusercontent.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency/master/...` into `data/labels/hu_pd_logs.tsv` (gitignored).

- [ ] **Step 1: Failing tests** with inline fixture text:

```python
def test_sapienza_binance_only_and_symbol():
    df = load_sapienza(fx("symbol,group,date,hour,exchange\nBRD,G1,2019-06-01,16:00,binance\nR,G2,2018-01-01,16:00,yobit\n"))
    assert df["symbol"].to_list() == ["BRDBTC"]
    assert df["t_start_us"][0] == 1559404800_000000

def test_dedup_merges_groups_within_5min():
    out = merge_dedup([labels("BRDBTC", "2019-06-01T16:00"), labels("BRDBTC", "2019-06-01T16:02")])
    assert out.height == 1 and out["t_start_us"][0] == us("2019-06-01T16:00")

def test_dedup_keeps_distinct_symbols_and_far_times(): ...
```
- [ ] **Step 2–4:** Run → FAIL, implement, run → PASS. Then run against the real vendored file: `uv run python -c "from idsai.eval.labels import *; print(load_sapienza(...).height)"` → `520`.
- [ ] **Step 5: Commit** `feat(eval): telegram pump label loaders with dedup`.

### Task 5: Feature registry, 1 s bars, batch window features

**Files:**
- Create: `idsai/detection/features/{__init__,registry,bars,windows}.py`
- Test: `tests/detection/test_bars.py`, `tests/detection/test_windows.py`

**Interfaces:**
- Consumes: `TRADES`, `FeatureCfg`.
- Produces:
  - `FeatureSpec(name: str, unit: str, desc_th: str, log1p: bool, z_floor: float)`; `REGISTRY: dict[str, FeatureSpec]`; `feature_columns(cfg: FeatureCfg) -> list[str]` (deterministic order).
  - `trades_to_bars(trades: pl.DataFrame, symbol: str, start_us: int, end_us: int) -> pl.DataFrame` — one row per second in `[start_us, end_us)`; columns `symbol, t_us, n_trades, buy_quote, sell_quote, quote_vol, open, high, low, close, rush_orders, max_trade_quote, pingpong, n_round`. Empty seconds: counts 0, OHLC = previous close (forward fill; leading empties null).
    - `rush_orders`: number of distinct millisecond timestamps (`ts_us // 1000`) in the second with ≥ 2 buy-taker trades.
    - `pingpong`: consecutive trade pairs with equal `qty`, equal `price`, opposite `taker_side`, `Δts ≤ 50_000 µs`.
    - `n_round`: trades whose `qty / 10**floor(log10(qty))` is within `1e-9` of an integer.
  - `compute_windows(bars: pl.DataFrame, cfg: FeatureCfg) -> pl.DataFrame` — rows at `t_end_us` multiples of `stride_s` (window covers bars with `t_us ∈ (t_end − W, t_end]`, i.e. ending at the bar starting `t_end − 1 s`); columns `symbol, t_end_us` + for each W: `n_trades_W, quote_vol_W, buy_quote_W, rush_W, pingpong_W, round_frac_W, buy_imb_W, ret_W, jump_W`; for each of those a `z_<col>`; plus `cusum_ret, cusum_imb, kk_vol_ratio, kk_price_ratio`.
- Algorithms the tests don't pin:
  - `buy_imb = (buy − sell) / (buy + sell + 1e-12)`; `ret_W = ln(close_t / close_{t−W})`; `jump_W = max(high in window) / close_{t−W} − 1`.
  - Robust z: transform `x' = log1p(x)` when `spec.log1p`; baseline over **prior** stride rows within `baseline_s` (exclude current row): `med = rolling_quantile(0.5)`, `iqr = q75 − q25` (linear interpolation), `scale = max(iqr / 1.349, spec.z_floor)`; `z = clip((x' − med) / scale, −50, 50)`; NaN when fewer than `min_baseline_s / stride_s` prior rows.
  - `cusum_ret`: `s_t = min(max(0, s_{t−1} + clip(z_ret_60, −10, 10) − 0.5), 100)`, NaN z ⇒ treated as 0; `cusum_imb` same on `z_buy_imb_60`.
  - `kk_vol_ratio = quote_vol_1800_t / mean(quote_vol_1800 at t − k·1800 s, k = 1..24)`; `kk_price_ratio = close_t / mean(close at same 24 points)`; NaN if any point missing.
  - `z_floor` defaults: `0.25` for `log1p` features, `1e-4` for `ret_*`/`jump_*`, `0.05` for `buy_imb_*`, `round_frac_*`.

- [ ] **Step 1: Failing bar tests:** empty seconds filled; `rush_orders == 1` for two buys sharing one ms; sells sharing a ms not counted; `pingpong == 1` for buy/sell equal qty 10 ms apart, `0` at 60 ms; `n_round` counts `0.5, 20, 300` but not `0.37`; taker volumes split correctly.
- [ ] **Step 2: Failing window tests:**

```python
def test_window_sums_exact(): # constant 1 trade/s, quote 2.0
    w = compute_windows(bars_const(hours=8), cfg)
    assert w["n_trades_60"].tail(1).item() == 60 and w["quote_vol_300"].tail(1).item() == 600.0

def test_z_nan_before_min_baseline():
    w = compute_windows(bars_const(hours=5), cfg)
    assert w["z_quote_vol_60"].is_nan().all()

def test_dead_market_z_finite():  # 23 h of zero trades then one burst of 50 trades
    w = compute_windows(bars_dead_then_burst(), cfg)
    z = w["z_n_trades_10"].drop_nans()
    assert z.is_finite().all() and z.max() > 6

@given(st.integers(min_value=1, max_value=3600))
def test_causal(extra_s):  # appending future bars never changes existing rows
    base = compute_windows(bars_random(hours=7, seed=1), cfg)
    more = compute_windows(append_random(bars_random(hours=7, seed=1), extra_s), cfg)
    assert_frame_equal(base, more.head(base.height))

def test_no_time_of_day_columns(): assert not any("hour" in c or "minute" in c for c in w.columns)

def test_one_day_under_10s():  # performance guard, 86400 bars
    t = time.perf_counter(); compute_windows(bars_random(hours=24, seed=2), cfg); assert time.perf_counter() - t < 10
```
- [ ] **Step 3: Run** → FAIL. **Step 4: Implement.** **Step 5: Run** → PASS.
- [ ] **Step 6: Commit** `feat(detection): 1s bars and causal window features`.

### Task 6: Red-Team simulator for Track T (T001 burst/stealth, T005 pingpong)

**Files:**
- Create: `idsai/detection/simulator/{__init__,scenarios,inject,guard}.py`, `configs/scenarios.yaml`
- Test: `tests/detection/test_simulator.py`

**Interfaces:**
- Consumes: `TRADES`, `LABELS`.
- Produces:
  - `class EthicsError(RuntimeError)`; `assert_injection_allowed(mode: str) -> None` (raises for `"live"`).
  - `Scenario` (pydantic): `family: Literal["pump_burst","pump_stealth","wash_pingpong"]`, `technique_id`, `sub_technique`, `t_start_us`, `duration_s`, `impact`, `vol_mult`, `rush_size`, `dump_tau_s`, `seed`, `region: Literal["train","test"]`.
  - `sample_scenario(family, region, t_start_us, rng: np.random.Generator, ranges: dict) -> Scenario`.
  - `inject(trades: pl.DataFrame, sc: Scenario, base_quote_per_s: float, mode: str = "replay") -> tuple[pl.DataFrame, pl.DataFrame]` — returns (new trades sorted, one-row LABELS frame with `provenance="SEMI-SYN"`, `t_end_us = t_start_us + duration_s·1e6`). Calls `assert_injection_allowed(mode)` first.
- `configs/scenarios.yaml` ranges (train | test region):
  - `pump_burst`: duration 30–180 s, impact 0.05–0.30 | duration 30–180 s, impact 0.03–0.05 (smaller, unseen); vol_mult 5–30; rush_size 3–8; dump_tau 120–600 s.
  - `pump_stealth`: duration 1800–5400 s, impact 0.02–0.08 | duration 5400–7200 s; vol_mult 1.2–2.0; rush_size 1.
  - `wash_pingpong`: duration 300–1800 s, vol_mult 2–5 | 1.5–2; impact 0.
- Injection model: synthetic taker-buy trades at Poisson rate giving `vol_mult × base_quote_per_s` extra quote volume; for bursts, trades arrive in clusters of `rush_size` fills sharing one ms; price multiplier `m(t)` rises linearly from 1 to `1+impact` over the duration, then decays as `1 + impact·exp(−(t−t_peak)/dump_tau)`; all real and synthetic trades with `ts ≥ t_start` get `price × m(t)`. Pingpong: pairs (buy then sell 1–20 ms later) of equal round qty at the current price; `m(t) = 1`. Synthetic trades: `source="sim"`, `trade_id` negative and unique.

- [ ] **Step 1: Failing tests:** same seed ⇒ identical frames; trades before `t_start` unchanged; max price within duration ≥ `(1 + 0.9·impact) × pre-price`; burst has ms-clusters of size `rush_size`; pingpong pairs have equal qty and opposite sides and price unchanged; `inject(..., mode="live")` raises `EthicsError`; label row has `technique_id` `MM-T001`/`MM-T005` and `provenance == "SEMI-SYN"`; test-region samples fall inside test ranges.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(redteam): semi-synthetic pump and wash-like injection`.

### Task 7: Detector contract, rules, stats, baselines, registry

**Files:**
- Create: `idsai/contracts/detector.py`, `idsai/detection/{rules,stats,registry}.py`, `idsai/ai/baselines.py`, `configs/signatures.yaml`, `idsai/detection/taxonomy.yaml`
- Test: `tests/detection/test_detectors.py`

**Interfaces:**
- Produces:
  - `Evidence(top: list[tuple[str, float, float]])` — `(feature, value, contribution)`, ≤ 5 items, sorted by contribution desc.
  - `class Detector(ABC)`: attrs `name: str`, `layer: Literal["rule","stat","ai","baseline"]`, `track: Literal["T","B"]`, `technique_ids: list[str]` (`["ANY"]` = technique-agnostic), `requires: list[str]`; methods `fit(normal: pl.DataFrame) -> None` (default no-op), `score(windows: pl.DataFrame) -> np.ndarray` (float64, len = rows, higher = more anomalous, NaN if any required feature NaN), `evidence(row: dict[str, float]) -> Evidence`, `state_dict() -> dict`, `load_state_dict(d: dict) -> None`.
  - `SignatureDetector(sig: dict)` — name `f"rule.{sid}"`; `score = min_i dir_i·(x_i − thr_i)/max(|thr_i|, 1e-9)` with `dir=+1` for `">="`, `−1` for `"<="`, and `"abs<="` scored as `(thr − |x|)/thr`; fires when score ≥ 0.
  - `RobustZDetector(features: list[str])` name `stat.robust_z`, technique `["ANY"]`, score = max of listed z columns.
  - `CusumDetector()` name `stat.cusum`, technique `["MM-T001"]`, score = `max(cusum_ret, cusum_imb)`.
  - `IsolationForestDetector(features, seed)` (`baseline.iforest`, score = `−score_samples`), `PCADetector(features, n_components=0.95)` (`baseline.pca`, reconstruction MSE on robust-scaled inputs), `RandomDetector(seed)` (`baseline.random`, uniform from `hash(symbol, t_end_us, seed)`), all `["ANY"]`.
  - `build_detectors(names: list[str], cfg: Settings) -> list[Detector]`; name `"rules"` expands to every signature in `configs/signatures.yaml`; unknown name → `KeyError` listing valid names.
- `configs/signatures.yaml` initial content (`layer: rule` unless noted):
  - `100101` MM-T001.001 pump burst: `z_quote_vol_60 >= 6`, `ret_60 >= 0.02`, `z_rush_60 >= 4`, severity 3.
  - `100102` MM-T001.002 pump stealth: `cusum_ret >= 8`, `ret_1800 >= 0.03`, `buy_imb_1800 >= 0.2`, severity 2.
  - `100103` MM-T001 Kamps–Kleinberg (adapted to 30-min candles, `layer: baseline`): `kk_vol_ratio >= 4.0`, `kk_price_ratio >= 1.05`, severity 1.
  - `100501` MM-T005.001 wash-like: `z_pingpong_300 >= 5`, `ret_300 abs<= 0.002`, `z_quote_vol_300 >= 3`, severity 1.
- `taxonomy.yaml`: MM-T000 (unclassified anomaly) and MM-T001…T005 with `name_th`, `definition_th`, `reference` (§13), `observability` text, sub-techniques.

- [ ] **Step 1: Failing tests:** signature margin math (row exactly at thresholds ⇒ score 0; abs<= case); NaN in a required feature ⇒ NaN score; robust_z picks max; random detector reproducible per seed and differs across seeds; iforest/pca score an injected outlier row above 99th percentile of normal rows; `build_detectors(["rules"], cfg)` yields 4 signature detectors with the sids above; unknown name raises `KeyError`; `evidence()` returns ≤ 5 items sorted desc.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(detection): detector contract, signatures, stats and baselines`.

### Task 8: Autoencoder detector + model registry

**Files:**
- Create: `idsai/ai/autoencoder.py`, `idsai/ai/learning.py` (registry part)
- Test: `tests/ai/test_autoencoder.py`

**Interfaces:**
- Consumes: `Detector`, `REGISTRY` (for `log1p` flags).
- Produces:
  - `AutoencoderDetector(features: list[str], hidden=64, bottleneck=16, epochs=50, lr=1e-3, weight_decay=1e-5, batch_size=512, patience=5, clip=10.0, seed=0)` — name `ai.autoencoder_trade`, layer `ai`, technique `["ANY"]`. `fit`: drop NaN rows, log1p where flagged, `RobustScaler` fit, clip ±`clip`, last 10 % (time order) as validation for early stopping, `torch.manual_seed(seed)` + `torch.use_deterministic_algorithms(True)`. `score`: per-row MSE. `evidence`: per-feature squared error → top 5.
  - `training_mask(windows: pl.DataFrame, labels: pl.DataFrame, margin_s: int = 3600) -> tuple[pl.Series, float]` (in `ai/learning.py`) — boolean keep-mask (False only inside a label ± margin for the same symbol) and excluded fraction. Alerts are never an input.
  - `ModelRegistry(models_dir: Path)`: `save(det: Detector, meta: dict) -> str` (version `v{n}` under `models/{name}/`, writes `state.pt`/`state.json` + `meta.json` with `data_hash, config_hash, seed, created_at, train_range`), `load(name: str, version: str | None = None) -> tuple[dict, dict]`, `versions(name) -> list[str]`.

- [ ] **Step 1: Failing tests:** on 20 k correlated-Gaussian normal rows + 200 shifted rows, ROC-AUC of AE scores > 0.9 (`sklearn.metrics.roc_auc_score`); same seed ⇒ identical scores (`np.array_equal`); save→load→score identical; NaN row ⇒ NaN; `training_mask` excludes only the labeled symbol ± margin and has no alert parameter (`inspect.signature`); fit on 100 k × 40 features finishes < 120 s on CPU.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(ai): dense autoencoder detector and model registry`.

### Task 9: Chief (calibration, fusion, budget, episodes)

**Files:**
- Create: `idsai/ai/chief.py`
- Test: `tests/ai/test_chief.py`

**Interfaces:**
- Consumes: `Detector` list, `ChiefCfg`, `Alert`.
- Produces:
  - `Chief(detectors: list[Detector], cfg: ChiefCfg)`.
  - `calibrate(cal: pl.DataFrame) -> None` — `cal` has `symbol, t_end_us` and `score__<name>` per detector; stores sorted finite scores per detector.
  - `pvalues(name: str, scores: np.ndarray) -> np.ndarray` — `(1 + #{s_i ≥ s}) / (n + 1)`; NaN score ⇒ 1.0.
  - `fuse(scored: pl.DataFrame) -> pl.DataFrame` — adds `p__<name>`, `F = max_d w_d·(−log10 p_d)` (w = 1), `best_detector`, `technique` (argmax over specific-technique detectors if its −log10 p ≥ `attribution_min`, else `"MM-T000"`), `agreement` (distinct layers with p < `agreement_p`).
  - `fit_threshold(cal_fused: pl.DataFrame) -> float` — smallest τ such that `episodes(cal_fused, τ)` per symbol-hour ≤ `budget_per_hour / universe_size` (binary search over unique F values); stores `self.tau`.
  - `episodes(fused: pl.DataFrame, tau: float | None = None, provenance="replay") -> list[Alert]` — rows with `F ≥ τ`, grouped per `(symbol, technique)` where gaps ≤ `suppress_for_s` merge; alert `ts_event_us` = first row, `fused_p = 10**−max F`, evidence from best detector at peak row, `severity = 1 + (agreement ≥ 2) + (agreement ≥ 3 or fused_p < 1e-4)`, `dedup_key = f"{symbol}|{technique}|{first_ts // 300e6}"`, `config_hash` filled.
  - `ChiefStream(chief: Chief)`: `step(row: dict) -> Alert | None` — returns a new `Alert(status="open")` on first exceedance, `Alert(status="updated")` when peak rises within an open episode, closes after `suppress_for_s` quiet.
  - `state_dict()/load_state_dict()` (calibration arrays + τ).

- [ ] **Step 1: Failing tests:**

```python
def test_pvalue_formula():
    ch.calibrate(cal_with("d", [1, 2, 3, 4]))
    assert ch.pvalues("d", np.array([5.0, 2.0, np.nan])).tolist() == [0.2, 0.8, 1.0]

def test_threshold_respects_budget():  # 50 symbols × 24 h of iid scores
    tau = ch.fit_threshold(ch.fuse(iid_scores(seed=0)))
    rate = len(ch.episodes(ch.fuse(iid_scores(seed=1)), tau)) / (50 * 24)
    assert rate <= 2.0 / 5 * 1.5

def test_episode_merge_and_split():  # exceedances at t=0,200s → one; t=0,400s → two
def test_unclassified_when_only_any_detectors_fire(): assert alerts[0].technique_id == "MM-T000"
def test_agreement_counts_layers_not_detectors(): ...
def test_stream_matches_batch_first_times(): ...  # ChiefStream over rows == episodes() first ts
```
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(ai): chief with conformal calibration and alert budget`.

### Task 10: Batch pipeline per symbol-day

**Files:**
- Create: `idsai/pipeline/{__init__,batch}.py`
- Test: `tests/pipeline/test_batch.py`

**Interfaces:**
- Consumes: `Store`, `trades_to_bars`, `compute_windows`, `inject`, detectors.
- Produces:
  - `DayResult(symbol: str, day: date, windows: pl.DataFrame, labels: pl.DataFrame, skipped: str | None)`.
  - `load_day_windows(store, symbol, day, cfg, scenario: Scenario | None = None) -> DayResult` — reads trades for `[day − 1, day + 1)`, injects if scenario given (base rate = median quote/s of D−1), builds bars + windows, keeps rows with `t_end_us` inside `day`; `skipped="missing_day"` if D absent, `"no_baseline"` if all z are NaN. Caches windows to `data/features/{feature_hash}/{symbol}/{day}.parquet` where `feature_hash` = hash of `FeatureCfg` + scenario JSON.
  - `score_windows(windows: pl.DataFrame, detectors) -> pl.DataFrame` — adds `score__<name>`.

- [ ] **Step 1: Failing tests** on a synthetic store (two days of random trades for `TESTBTC`): result covers exactly 8640 stride rows of D; missing D ⇒ `skipped == "missing_day"`; missing D−1 ⇒ `skipped == "no_baseline"`; injected burst raises `score__rule.100101` max above 0 in the label window; second call hits the cache (patch `compute_windows` to assert not called).
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(pipeline): batch symbol-day processing with cache`.

### Task 11: Splits, metrics, experiment runner, report

**Files:**
- Create: `configs/splits.yaml`, `idsai/eval/{splits,metrics,experiments,report}.py`
- Test: `tests/eval/test_metrics.py`, `tests/eval/test_splits.py`, `tests/eval/test_experiments_smoke.py`

**Interfaces:**
- Consumes: `Store`, labels, `load_day_windows`, `score_windows`, `build_detectors`, `AutoencoderDetector`, `Chief`, `ModelRegistry`.
- Produces:
  - `configs/splits.yaml`: `train: [2018-01-01, 2019-06-30]`, `cal: [2019-07-01, 2019-12-31]`, `test: [2020-01-01, 2022-01-31]`, `normal_days_per_symbol: 4`, `min_gap_days: 2`.
  - `Split(train: list[SymDay], cal: list[SymDay], test_events: list[SymDay], test_normal: list[SymDay])`; `make_split(labels: pl.DataFrame, cfg, seed: int) -> Split` — normal days are days of labeled symbols ≥ `min_gap_days` away from any label of that symbol, sampled with `seed`, each assigned to the period containing it; asserts `max(train) < min(cal) ≤ max(cal) < min(test)`.
  - `match_events(alerts: list[Alert], labels: pl.DataFrame, before_s: int, after_s: int) -> MatchResult` with fields `n_events, n_detected_any, n_detected_correct, n_alerts, n_alerts_matched, ttd_s: list[float]`; `summarize(mr: MatchResult, symbol_hours: float, universe_size: int) -> dict` → `recall_any, recall_correct, precision, f1, fa_per_symbol_day, fa_per_hour_universe, ttd_median_s, ttd_p90_s`.
  - `bootstrap_ci(per_event: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]` (95 %, percentile).
  - `run_experiment(name: Literal["real_t001","semisyn","ablation","baselines","ref_supervised","retrain_drift"], cfg: Settings, store: Store, seed: int) -> list[dict]` — each returned record has `run_id, experiment, variant, seed, provenance, git_sha, config_hash, model_versions, split_def, metrics` and is inserted via `store.insert_run`. Variants:
    - `real_t001`: full stack (rules + stat + AE + baselines in config) on `test_events ∪ test_normal`, provenance `REAL`.
    - `semisyn`: one test-region scenario per test-normal day per family, provenance `SEMI-SYN`, metrics per family.
    - `ablation`: detector subsets `rules`, `stats`, `ai`, `rules+stats`, `all`.
    - `baselines`: each of `baseline.random`, `rule.100103` (KK), `baseline.iforest`, `stat.robust_z`, `ai.autoencoder_trade` alone through Chief.
    - `ref_supervised`: RandomForest (200 trees, max_depth 5, seed) trained on `train`-period labeled windows (positives = windows inside label ± margin) with our Track T features, evaluated on test through the same episode/match code; reported as "supervised reference", provenance `REAL`.
    - `retrain_drift`: AE v1 trained on `train` days ≤ 2018-06-30 vs v2 on all `train`; same cal/test.
  - Chief is always calibrated on `cal` days only; AE fit on `train` normal windows only, with label windows ±1 h masked via `learning.training_mask` (Task 8).
  - `build_report(store: Store, out_md: Path, out_json: Path) -> None` — tables per experiment × variant: mean ± std across seeds and bootstrap CI; header states total run count; every row has provenance; JSON consumed by the dashboard.

- [ ] **Step 1: Failing metric tests:**

```python
def test_match_window_edges():  # label at t0; alerts at t0-121s (miss), t0-119s (hit)
def test_correct_vs_any_technique(): ...  # MM-T000 alert counts for recall_any only
def test_fa_rate_units(): assert summarize(mr_with_unmatched(4), symbol_hours=48, universe_size=5)["fa_per_hour_universe"] == pytest.approx(4 / 48 * 5)
def test_ttd_negative_allowed(): ...
def test_bootstrap_ci_contains_mean_and_deterministic(): ...
```
- [ ] **Step 2: Failing split tests:** periods strictly ordered; no normal day within 2 days of a label; same seed ⇒ same split.
- [ ] **Step 3: Failing smoke test:** synthetic store with 3 symbols × 10 days spanning fake train/cal/test dates (override splits), `run_experiment("semisyn", seed=0)` returns records with provenance `SEMI-SYN` and finite recall; `build_report` writes both files and the markdown contains `"runs: "`.
- [ ] **Step 4: Run** → FAIL. **Step 5: Implement.** **Step 6: Run** → PASS.
- [ ] **Step 7: Commit** `feat(eval): time splits, event metrics, experiment runner and report`.

### Task 12: Learning loop (champion/challenger acceptance)

**Files:**
- Modify: `idsai/ai/learning.py`, `idsai/eval/report.py` (add the `retrain_drift` Decision: v2 vs v1 on `recall_any`, paired by seed, shown in report + JSON)
- Test: `tests/ai/test_learning.py`

**Interfaces:**
- Produces:
  - `Decision(accept: bool, diff_mean: float, ci: tuple[float, float], reason_th: str)`; `compare(champion: list[float], challenger: list[float], seed: int, n_boot: int) -> Decision` — paired across seeds; accept iff CI lower bound of (challenger − champion) > 0.

- [ ] **Step 1: Failing tests:** `compare([0.5]*5, [0.6]*5)` accepts, `compare([0.5,0.6,0.5,0.6,0.5], [0.55,0.5,0.6,0.5,0.6])` rejects.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(ai): champion-challenger acceptance`.

### Task 13: Auditor + leakage guard

**Files:**
- Create: `idsai/ai/auditor.py`
- Test: `tests/ai/test_auditor.py`, `tests/test_leakage_guard.py`

**Interfaces:**
- Produces:
  - `Check(name: str, status: Literal["ok","warn","fail"], value: float | None, detail_th: str)`; `HealthReport(checks: list[Check], ts_us: int)` with `overall` = worst status.
  - `Auditor(cfg: Settings)`: `alert_rate(n_alerts: int, symbol_hours: float) -> Check` (Poisson upper-tail p < 0.01 vs `budget_per_hour/universe_size` ⇒ warn); `calibration(pvals: np.ndarray) -> Check` (KS vs uniform, p < 0.01 ⇒ warn; < 200 values ⇒ ok with note); `latency(samples_ms: list[float]) -> Check` (p99 > 2000 ⇒ warn); `trade_gaps(trades: pl.DataFrame) -> Check` (count of `trade_id` jumps > 1 among `source == "archive"|"live"`); `exclusion(frac: float) -> Check` (> 0.02 ⇒ warn); `staleness(meta: dict, now_us: int, max_age_days: int = 30) -> Check`; `report(**inputs) -> HealthReport`.
  - Leakage guard test: AST-scan every module under `idsai/detection/` and `idsai/ai/` except `ai/learning.py`; none may import `idsai.eval.labels` or reference `frames.LABELS`.

- [ ] **Step 1–4:** failing tests for each check's ok/warn boundary and the AST guard → implement → PASS.
- [ ] **Step 5: Commit** `feat(ai): auditor health checks and label leakage guard`.

### Task 14: Template explainer

**Files:**
- Create: `idsai/product/explainer/{__init__,template}.py`
- Test: `tests/product/test_explainer.py`

**Interfaces:**
- Consumes: `Alert`, `REGISTRY`, taxonomy.
- Produces: `Explanation(text_th: str, facts: list[tuple[str, str]])`; `explain(alert: Alert) -> Explanation`; `numbers_in(text: str) -> list[float]`; `validate_numbers(text: str, allowed: list[float], rel_tol: float = 0.01) -> bool`; `BANNED_TH: list[str]` (e.g. `"โกง"`, `"มิจฉาชีพ"`, `"ผู้ปั่น"`, `"คนร้าย"`).
- Copy (exact skeleton): `"พบความผิดปกติทางสถิติ ({technique_id} {name_th}) ที่ {symbol} เวลา {HH:MM:SS} UTC: {facts}; {agreement} จาก 3 ชั้นตรวจจับเห็นพ้อง; ความหายากเทียบตลาดปกติ p ≈ {fused_p:.1e}"`. Facts use `desc_th` from the registry and evidence values (z with 1 decimal, returns as % with 2 decimals).

- [ ] **Step 1: Failing tests:** text contains `"ความผิดปกติทางสถิติ"`; `validate_numbers(text, allowed_from(alert))` is True (allowed = evidence values, their % forms, agreement, 3, fused_p, time digits); no `BANNED_TH` word; MM-T000 alert renders "ความผิดปกติที่ยังไม่จัดประเภท".
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(product): thai template explainer with number validator`.

### Task 15: Bus, StreamingWindows, Time Machine, StreamPipeline

**Files:**
- Create: `idsai/bus.py`, `idsai/detection/features/streaming.py`, `idsai/data/replay.py`, `idsai/pipeline/stream.py`
- Test: `tests/test_bus.py`, `tests/detection/test_streaming_parity.py`, `tests/data/test_replay.py`, `tests/pipeline/test_stream.py`

**Interfaces:**
- Produces:
  - `Bus(maxsize: int = 1000)`: `subscribe(topic) -> asyncio.Queue`, `publish(topic, msg) -> None` (drop-oldest when full, increments `dropped[topic]`). Topics: `"bars"`, `"scores"`, `"alerts"`, `"health"`.
  - `StreamingWindows(symbol: str, cfg: FeatureCfg)`: `push_bar(bar: dict) -> dict | None` returns a window row when `bar.t_us + 1 s` is a stride boundary; same columns/values as `compute_windows`.
  - `Replayer(trades: pl.DataFrame, speed: float = 1.0, start_us: int | None = None)`: async iterator yielding `("trade", Trade)` and `("tick", t_us)` once per event-time second; methods `pause()`, `resume()`, `set_speed(x)` (`0` = as fast as possible), `seek(t_us)` (clamped to data range), `inject(sc: Scenario, base_quote_per_s: float)` (only affects trades with `ts_us >` current cursor; calls `assert_injection_allowed("replay")`), `cursor_us`.
  - `BarBuilder(symbol)`: `on_trade(t: Trade) -> None`, `on_tick(t_us) -> list[dict]` (closed bars, identical to `trades_to_bars` rows).
  - `StreamPipeline(cfg, store, detectors, chief, bus, mode: Literal["replay","live"])`: `async run(source)` wires source → BarBuilder → StreamingWindows → `score_windows` (1-row frame) → `ChiefStream.step` → `store.insert_alert` + bus; records per-window latency (`perf_counter` from tick receipt to alert decision) into `latency_ms` deque(maxlen=10_000).

- [ ] **Step 1: Failing parity test:** fixture of 30 h random trades; batch = `compute_windows(trades_to_bars(...))`; stream = BarBuilder→StreamingWindows; for all columns `np.allclose(batch, stream, rtol=1e-9, atol=1e-9, equal_nan=True)` over the last 6 h.
- [ ] **Step 2: Failing replay tests:** event order preserved; ticks every second including empty ones; `seek` beyond end clamps; `pause` stops yields until `resume`; `inject` after cursor changes only future prices; live-mode injection raises `EthicsError`.
- [ ] **Step 3: Failing pipeline test:** replay 2 days of fixture trades with an injected burst at speed 0 → at least one alert with `technique_id == "MM-T001"` within the label window; `p99(latency_ms) < 2000`; bus `"alerts"` received the same alert.
- [ ] **Step 4: Run** → FAIL. **Step 5: Implement** (StreamingWindows keeps a ring buffer of `max(windows)+baseline` bars as NumPy arrays and computes baselines with `np.quantile(method="linear")` on prior stride rows). **Step 6: Run** → PASS.
- [ ] **Step 7: Commit** `feat(stream): bus, streaming features with parity, time machine and stream pipeline`.

### Task 16: LIVE trade sensor

**Files:**
- Create: `idsai/data/live.py`
- Test: `tests/data/test_live.py`

**Interfaces:**
- Produces: `LiveTradeSensor(symbols: list[str], url: str = "wss://data-stream.binance.vision/stream", rotate_s: int = 23*3600)`: async iterator yielding the same `("trade", Trade)` / `("tick", t_us)` tuples as `Replayer` (ticks from wall clock); combined stream path `?streams=<s>@trade/...&timeUnit=MICROSECOND`; `parse_trade(msg: dict) -> Trade` (fields `e,E,s,t,p,q,T,m`; `recv_us` = local clock); dedup by `(symbol, trade_id)` over the last 10 000 ids; reconnect with exponential backoff 1→60 s; proactive reconnect after `rotate_s`; counts `reconnects`, `dupes`.

- [ ] **Step 1: Failing tests:** `parse_trade` on a recorded payload (µs `T`, `m=true` ⇒ `"sell"`); duplicates after a simulated reconnect are dropped; against a local `websockets.serve` fake server that closes after 3 messages, the sensor reconnects and continues; URL built with lowercase symbols and `timeUnit=MICROSECOND`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Manual live check (not in CI):** `uv run python -m idsai.data.live BTCUSDT` prints ≥ 10 trades within 10 s.
- [ ] **Step 6: Commit** `feat(data): live trade sensor with reconnect and dedup`.

### Task 17: API (FastAPI)

**Files:**
- Create: `idsai/product/api/{__init__,state,app}.py`
- Test: `tests/product/test_api.py`

**Interfaces:**
- Consumes: `Store`, `StreamPipeline`, `Replayer`, `Auditor`, `explain`, taxonomy, report JSON.
- Produces:
  - `AppState(cfg, store, mode, pipeline: StreamPipeline | None, replayer: Replayer | None, auditor)`.
  - `create_app(state: AppState) -> FastAPI` serving `/` static (`product/web`) and:
    - `GET /api/alerts?symbol=&limit=` → list of Alert JSON (+ `explanation_th`); `GET /api/alerts/{id}` → Alert + Explanation + feedback list; 404 unknown.
    - `POST /api/alerts/{id}/feedback {verdict, analyst}` → 201.
    - `GET /api/symbols`, `GET /api/health` (HealthReport), `GET /api/experiments` (report JSON or `{"runs": 0}`), `GET /api/taxonomy`.
    - `POST /api/replay/control {action: play|pause|speed|seek, value?}` → 409 when mode is live.
    - `POST /api/redteam/inject {family, region, offset_s}` → 403 with Thai message when mode is live; else 202 + Scenario JSON.
    - `WS /ws/stream` → messages `{type: "bar"|"score"|"alert"|"health", data}`; per-client throttle 4 Hz for `bar`/`score` (alerts never dropped).

- [ ] **Step 1: Failing tests (TestClient):** each endpoint shape; feedback persisted; redteam 403 in live; replay control 409 in live; ws receives an `alert` message after publishing one on the bus.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5: Commit** `feat(api): rest and websocket api`.

### Task 18: CLI, demo pack, docs, CI

**Files:**
- Create: `idsai/cli.py` (+ `[project.scripts] idsai = "idsai.cli:main"`), `README.md`, `NOTICE.md`, `docs/{ETHICS,DATA_LIMITATIONS,TEAM_GUIDE}.md`, `.github/workflows/ci.yml`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces subcommands (argparse):
  - `idsai labels` (load Sapienza + Hu if present → `store.write_labels`), `idsai download --split research|--symbols S --days N` (D−1 included, verified), `idsai train [--seed N] [--profile research|live]` (live profile: last 14 days of `cfg.live.symbols`; trains AE, calibrates Chief on the last 3 days, saves via ModelRegistry), `idsai eval --experiment NAME|all --seeds 0,1,2,3,4`, `idsai report`, `idsai demo-pack` (3 labeled pump days with D−1 + 2 normal days into `data/demo/`), `idsai serve --mode replay|live [--symbol S --day D] [--port 8800]`.
  - Default port 8800 (avoids the user's 8765 clash).
- Docs: ETHICS.md (§15 expanded), DATA_LIMITATIONS.md (§1 table + label caveats + "claims we do not make"), TEAM_GUIDE.md (role ownership, how to add a detector in 3 steps, how to run tests, glossary for first-years), README (quickstart in Thai: `uv sync`, `idsai demo-pack`, `idsai serve`).
- CI: ubuntu + windows, Python 3.13, `uv sync --frozen`, `uv run ruff check .`, `uv run pytest -q -m "not network"`; tests touching the network are marked `network`.

- [ ] **Step 1: Failing tests:** `main(["--help"])` exits 0 and lists all subcommands; `serve` default port 8800; `train --profile live` rejects empty `cfg.live.symbols`.
- [ ] **Step 2–4:** FAIL → implement → PASS. Run `uv run pytest -q` (full suite) → all PASS.
- [ ] **Step 5: Commit** `feat: cli, demo pack, docs and ci`.

### Task 19: SOC dashboard (static)

**Files:**
- Create: `idsai/product/web/{index.html,app.js,styles.css}`, `idsai/product/web/vendor/lightweight-charts.standalone.production.js` (from `https://unpkg.com/lightweight-charts@5.2.1/dist/lightweight-charts.standalone.production.js`, exact version recorded in `vendor/VERSION`), `idsai/product/web/vendor/NOTICE-lightweight-charts`
- Test: `tests/product/test_web_static.py` + browser verification

**Interfaces:**
- Consumes: Task 17 endpoints and WS message types only.
- Produces: single-page app with hash routes `#/monitor`, `#/alerts`, `#/alerts/{id}`, `#/redteam`, `#/eval`, `#/taxonomy`, `#/about`.
  - Top bar: mode badge (LIVE/REPLAY), event clock, Auditor overall status dot, alert-budget meter (alerts last hour vs budget).
  - Monitor: candlestick/line price chart with alert markers; fused score lane with τ line; L1/L2/L3 lanes; symbol picker; replay controls (play/pause/speed/seek) hidden in LIVE.
  - Alerts: triage table (severity, technique, symbol, time, status) → detail page (Thai explanation, evidence table, detector p-values + agreement, technique card, TP/FP/unsure buttons).
  - Red-Team: family + region pickers, inject button, live time-to-detect timer; disabled with notice in LIVE.
  - Eval: tables from `/api/experiments` with provenance badges (REAL / SEMI-SYN / SYN) on every metric, run count, before/after retrain chart with CI whiskers.
  - Taxonomy: MM-T000–T005 cards with observability limits. About: data limitations summary, ethics, attribution "Binance Vision (CC BY-NC-SA 4.0)" and TradingView link (`attributionLogo: true`).
  - Visual language: clean Apple-like, system font stack, CSS custom properties for light/dark via `prefers-color-scheme`, responsive to 360 px width, no build step.

- [ ] **Step 1: Failing test:** `GET /` returns HTML containing `"Market IDS"`, `"Binance Vision"`, `"tradingview"`; vendor file served with JS content type.
- [ ] **Step 2: Implement.** **Step 3: Run test** → PASS.
- [ ] **Step 4: Browser verification:** start `idsai serve --mode replay` on the demo pack (Task 18) via preview tools; check console has no errors, Monitor renders chart, injecting a burst produces an alert visible in Alerts within the label window, dark mode and 375 px layout render without horizontal scroll; capture screenshots.
- [ ] **Step 5: Commit** `feat(web): soc dashboard`.

### Task 20: Real run, report, and honest results

**Files:**
- Create: `docs/REPORT.md` (generated tables + Thai discussion), `data/` (gitignored)

- [ ] **Step 1:** `uv run idsai labels && uv run idsai download --split research` — record downloaded symbol-days, 404 count, and disk used in REPORT.md.
- [ ] **Step 2:** `uv run idsai eval --experiment all --seeds 0,1,2,3,4` (run in background; record wall time).
- [ ] **Step 3:** `uv run idsai report` → `docs/REPORT.md` tables + `data/report.json`.
- [ ] **Step 4:** Write the discussion section: which targets were met/missed (recall ≥ 0.8 per technique, FA ≤ 2/h, p99 latency < 2 s), whether AE beats baselines (CI overlap), ablation reading, retrain_drift decision, and the limitations from DATA_LIMITATIONS.md. Every number must come from the generated tables.
- [ ] **Step 5:** `uv run idsai serve --mode replay` and verify Eval page shows the same numbers.
- [ ] **Step 6: Commit** `docs: experiment report with real and semi-synthetic results`.
