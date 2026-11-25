# Track record page (2.3) and the web alert card history line (1.3)

Source: `docs/ROADMAP.md` 2.3 and 1.3. Branch `feat/track-record`, cut from `main` (e14c84f). Product layer only: detection, configs and the config hash (9aff5e3482f2) do not change. The `Alert` contract does not change either (schema_v stays 1).

## 1.3 History line on web alert cards
- The Telegram message already carries `HISTORY_TH` (`idsai/product/notify/telegram.py`). It is left off sell-phase alerts, because it was measured on the rise of real pumps only.
- Move `HISTORY_TH` and its source comment to a shared place, e.g. `idsai/product/explainer/`. Import it from there in Telegram, so the text does not change; the Telegram tests must still pass.
- `GET /api/alerts/{id}` gains `history_th`: the string for pump-phase or phase-less alerts, and null for `phase == "sell"`.
- The web alert detail shows it as a small, muted line, with "ที่มา: ผลเฟส 0 ใน docs/ROADMAP.md".
- It stays history, not a forecast. Never phrase it as advice.

## 2.3 Track record (ผลงานย้อนหลัง)
**Outcome store.** Add a new table `alert_outcomes` in `Store`:
```
alert_outcomes(alert_id PK, symbol, ts_event_us, price_at, price_1h, t_1h_us, status, updated_us)
```
- `status` is one of: `pending` (not yet 1 h after), `done`, `unknown` (the price was not recoverable).

**Filling outcomes.** Write a small pure function plus a periodic task (`OutcomeTracker`):
- Run it every 60 s, beside the existing app tasks (find where `serve` starts its asyncio tasks, e.g. the session purge or Telegram tasks).
- For each alert with provenance `live` or `replay` (never `sim`) that has no `done`/`unknown` outcome:
  - `price_at` is the close of the latest ring point with `t_end_us <= ts_event_us`. Use the pipeline ring via a new read-only `StreamPipeline.close_at(symbol, t_us)` or similar. Fall back to `store.last_trade_before(symbol, ts_event_us)` (it exists for replay archives).
  - `price_1h` is the first ring close with `t_end_us >= ts_event_us + 3600 s`. Use it only if the pipeline's `latest_us` has passed that time. Fall back to the trade archive if that day is stored.
  - If neither the ring nor the archive can give a value and `latest_us > ts_event_us + 24 h`, set `status = unknown`.
- It must be cheap. Only look at alerts from the last 25 h by `ts_event_us`, plus pending rows. Never scan everything each minute.

**API.** `GET /api/track-record` (signed-in) returns:
- `items`: newest first, limited to 200 by a `limit` query. Each item has:
  - `alert_id`, `symbol`, `ts_event_us`, `technique_id`, `phase`, `severity`;
  - the outcome: `price_at`, `price_1h`, `change_1h_pct`, `status`;
  - feedback counts `{tp, fp, unsure}` from the existing feedback table.
- `summary`:
  - `n_alerts`, `n_done`;
  - `share_lower_1h`: of the done items, the share with `price_1h < price_at`;
  - `median_change_1h_pct`;
  - `feedback_totals`;
  - `since_us`: the earliest alert in the store.
- `note_th` explains these points:
  - The numbers count only alerts this server has stored. The server's disk is not permanent, so the count restarts after a redeploy. Say it plainly.
  - Price after 1 h is history, not a forecast.
  - 👎 means a user judged it a false alarm.

**Web.** Add a nav route `#/track`, "ผลงานย้อนหลัง", built like the `#/check` route:
- Summary tiles: the alert count, the share lower 1 h later with n, the median change, and the 👍/👎 totals.
- A list or table of alerts, including the false alarms; never hide them.
  - Each row: time in Bangkok time, coin, technique, phase, price at alert, then +1 h and the % change (or "รอครบ 1 ชม." / "ไม่มีข้อมูลราคา").
  - Each row also shows 👍/👎 counts and links to the alert detail, where voting already exists.
- Make it work at phone width. Insert all text with textContent; no innerHTML. Poll every 60 s while on the page only.

## Wording rules
- Use "ความผิดปกติทางสถิติ". Accuse no one.
- Never call F or p a probability. No accuracy metric: share_lower_1h is described as "สัดส่วนที่ราคาต่ำกว่าตอนเตือนใน 1 ชม.", not accuracy.
- No buy/sell advice. Include "ข้อมูลนี้ไม่ใช่คำแนะนำการลงทุน".

## Tests (temp stores only, never `data/app.duckdb`)
- Outcome function, each case:
  - a ring hit;
  - the archive fallback;
  - pending before 1 h;
  - unknown after 24 h;
  - sim excluded;
  - idempotent re-runs.
- Track-record endpoint: auth, shape, summary maths (the share and the median) and feedback counts.
- `history_th`: present for pump and null phase, null for sell. The Telegram text is unchanged.
- Web static tests for the new route (follow the existing `test_web_static.py` patterns).
- `/api/series` output unchanged. Ruff clean. Full suite `-m "not network and not slow"` green.

## Out of scope
- Phase 3, news, and public access without sign-in (the page stays behind sign-in for now).
