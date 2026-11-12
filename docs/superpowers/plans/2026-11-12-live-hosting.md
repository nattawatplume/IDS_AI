# Live mode on Render free (512 MB, 0.1 CPU)

Date: 2026-11-12. Owner: "ผมอยากให้มันเป็นการใช้งานแบบ realtime ไปเลย ... ถ้าไม่ไหวจริงๆปรับเหรียญลงได้นะ ... หาทางที่ดีที่สุด"

## Spike results (local, POLARS_MAX_THREADS=1)

- **Stream:** `wss://data-stream.binance.vision/stream` works, about 10 trades/s for 5 coins. Steady live CPU is about 1 % of one core.
- **Training:** `download --days 7` plus `train --profile live` for 5 coins took 99 s and 124 s. The bundle is 4 MB.
- **RAM, 5 coins:** about 1.1 GB steady and 1.5 GB peak. One BTC day ingest alone peaks at 1.5 GB RSS, ETH at 1.0 GB and SOL at 376 MB.
- **RAM, 3 coins (SOL, DOGE, PEPE):** a 24 h preroll read in one frame peaks at 499 MB. Most of that comes from reading one large frame. Freed memory is not given back to the OS.
- **Conclusion:** 5 coins cannot fit 512 MB. 3 coins fit if:
  - the preroll is read in chunks;
  - the archive download runs in its own process before `serve`;
  - the model is trained at image build.
- **Why these coins:** SOL, DOGE and PEPE are the pump-prone coins, and BTC/ETH are hardly manipulated, so the choice also fits the use case.
- **Unknown:** whether the stream is reachable from Render (Oregon, US). `start.sh` probes it and falls back to replay.

## Task 1: app code (TDD)

1. **Config:** in `idsai/config.py` and `configs/default.yaml`, set `live.symbols: [SOLUSDT, DOGEUSDT, PEPEUSDT]` and `live.universe_size: 3`. Update `tests/test_config.py`.
2. **Chunked archive events (`idsai/data/live.py`):** add `chunked_archive(read, symbols, start_us, end_us, chunk_s=3600)`.
   - `read(symbol, a, b)` returns the TRADES in `[a, b)` (`Store.read_trades`).
   - It is an async iterator that yields exactly the events of `Replayer(all_trades_in_[start,end), speed=0, start_us=start_us)`, without `preroll_done`. That means every `tick` from `start+1s` through `end_us` and every trade in order, while holding only one chunk (all symbols, one hour) in memory at a time.
   - Build it from one `Replayer(chunk_frame, speed=0, start_us=a)` per chunk, plus the ticks the Replayer does not emit after the chunk's last trade up to `b` (an empty chunk gives ticks only).
   - It must yield to the event loop, as the Replayer does.
   - Tests:
     - equality with a single Replayer on a small multi-symbol frame, with chunk boundaries that fall inside the data, on an empty chunk, and on chunk_s that does not divide the span;
     - only the requested symbols are read.
3. **`warmup_then_live(archive, sensor)`:** `archive` may also be an async iterable of events (the chunked source). It plays exactly as a frame does today: drop `preroll_done`, track the last tick, then the gap, then the sensor. Existing DataFrame behaviour and tests are unchanged.
4. **`idsai/cli.py`:**
   - Split `_live_preroll` into `live_prep_ingest(cfg, store, symbols, now_us)`, the existing download loop, and a read part.
   - The read part returns the chunked source over `[start, end)`:
     - `end = min(now, 00:00 UTC after the newest stored day <= today among the symbols)`;
     - `start = max(now - 24 h, end - LIVE_PREROLL_H h)`, with `LIVE_PREROLL_H = 12`;
     - none if `start >= end`.
   - The stale warning uses `end` in place of the max ts (no full read just to find it).
   - `prepare_live` still calls the ingest first (cheap when the days are already stored).
5. **New subcommands** (Thai help and messages, like the others):
   - `idsai live-prep`: runs `live_prep_ingest` for `live.symbols`. It is run in its own process before `serve`, so the parse memory is freed.
   - `idsai live-check [--timeout S]` (default 20): connects with `LiveTradeSensor(live.symbols, live.ws_url)` and exits 0 at the first trade, or 1 with a Thai message on timeout or error. It sends nothing. The test uses the existing loopback fake server pattern from the live tests.
   - `idsai download --split live`: `live.symbols` × the latest `live.train_days` published days, with each first day's D-1, the same as `--symbols ... --days N`.
6. **Gates:** `uv run ruff check .` and `uv run pytest -q -m "not network and not slow"` are green.

## Task 2: deploy

1. **Dockerfile build stage:** after `demo-pack`, run `idsai download --split live && idsai train --profile live`. Copy `models/` into the runtime stage (`COPY --from=build --chown=user /home/user/app/models /home/user/app/models`). `check_no_hu.py data` still runs.
2. **`deploy/start.sh`:** `MODE="${IDSAI_MODE:-live}"`.
   - **Live:** when the mode is live, `models/live/bundle.json` exists and `idsai live-check` passes, run `idsai live-prep` (on failure, print a Thai warning and go on), then `exec idsai serve --mode live` with the same host, port, auth, cookie, sign-up and public-host flags (no `--loop` or `--speed`).
   - **Otherwise:** print the reason in Thai and exec the existing replay line.
   - `user seed-env` still runs first.
3. **`render.yaml`:** add the env var `IDSAI_MODE` with value `live`, so setting it to `replay` in Render goes back.
4. **README hosting section:** explain in plain Thai:
   - live mode on SOL, DOGE and PEPE;
   - why 3 coins (512 MB);
   - after each deploy or restart, about 10–20 min of catch-up, then the first live row after about 30 min;
   - red-team simulation works only in replay;
   - the automatic replay fallback, and `IDSAI_MODE=replay`.
5. **Tests:** update `tests/test_deploy_files.py` for all of the above.
6. **Gates:** same as Task 1.

## Rules

- Never commit `data/`, `models/`, `.claude/` or `.superpowers/`, and never `git add -f`.
- No order or key code. Live stays read-only: market data only.
- Tests never touch the real `data/app.duckdb`. No long `--basetemp`.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push.
