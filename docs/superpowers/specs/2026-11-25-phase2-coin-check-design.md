# Phase 2: coin history (2.1) and check before buying (2.2)

Source: `docs/ROADMAP.md` phase 2 (answers critique C3). Branch `feat/phase2`, cut from `main` (f579f58). Product layer only: detection, configs and config hashes do not change.

## Goal
Before a user trades a coin, show four facts about it on one page:
- how anomalous it is right now;
- the alerts in the last 24 h;
- its liquidity compared with normal;
- its alert history over 7 and 30 days.

Plus one status label. The labels are **ปกติ / เฝ้าระวัง / ผิดปกติมาก**. Never "ซื้อได้/ห้ามซื้อ". The page carries the line "ข้อมูลนี้ไม่ใช่คำแนะนำการลงทุน".

## Wording rules (project-wide)
- Say "ความผิดปกติทางสถิติ"; never accuse anyone.
- F and p are never called a probability. Describe F as "คะแนนความผิดปกติ (F)" against the alert threshold τ.
- No accuracy metric.

## 2.1 Coin history
- `Store.count_alerts(symbol, since_us, until_us, provenances)`: counts alerts by `ts_event_us`. An episode counts once, because `upsert_alert` keeps one row per `alert_id`.
- Count only provenance `live` and `replay`. Never count `sim` (Red-Team injects).
- **"Now":** the pipeline's latest event time (`_latest_us`; add a read-only property if needed) when a pipeline exists. Otherwise it is wall-clock now. In replay, history is relative to the replay clock.
- Windows are 7 days and 30 days. Also return `by_phase` counts (pump/sell/unknown) when the alerts carry `phase`.
- **API:** `GET /api/symbols/{symbol}/history` (signed-in) → `{symbol, now_us, d7, d30, by_phase_d30}`. Unknown symbol → 404 in Thai, like `/api/series`.
- **Web:** on the coin view, show one line such as "ถูกเตือน 5 ครั้งใน 30 วัน (7 วัน: 1)". With zero alerts, say so plainly. Add a short note that the count only covers the period this system has watched.

## 2.2 Check before buying
- **API:** `GET /api/symbols/{symbol}/check` (signed-in) → JSON with these parts:
  - `score`: `{F_now, tau, t_end_us}` from the last ring point. Null when there is no point yet (warm-up).
  - `alerts_24h`: the count over the same provenances, plus the newest alert's `{alert_id, ts_event_us, technique_id, phase, status}` or null.
  - `liquidity`:
    - `qvol_5m`: the mean quote volume per minute over the last 5 min.
    - `qvol_median_24h`: the median per-minute quote volume over the ring (up to 24 h).
    - `ratio`.
    - Any of these is null when there is not enough data (fewer than 60 whole one-minute buckets in the ring, `LIQ_MIN_MINUTES`).
    - A median of 0 (with enough data) keeps `ratio` null and adds the thin-market reason "สภาพคล่องบาง: ช่วงที่ผ่านมาส่วนใหญ่แทบไม่มีการซื้อขาย" (→ เฝ้าระวัง).
    - To get the data, add `quote_vol` to the ring point in `StreamPipeline._point`. Keep it out of the existing `/api/series` output unless a test shows the extra key is harmless; prefer leaving `/api/series` byte-identical.
  - `history`: the 2.1 object.
  - `label` and `reasons_th`: a list of short Thai reasons.
- **Label rule (fixed; test each branch):**
  - **ผิดปกติมาก** if F_now ≥ τ, or there is an alert with status open/updated whose last exceedance (`window[1]`) is in the last 60 min.
  - Else **เฝ้าระวัง** if any of these hold:
    - alerts_24h ≥ 1;
    - F_now ≥ 0.5·τ;
    - liquidity ratio < 0.3 (thin market), or ratio > 5 (unusual surge).
  - Else **ปกติ**.
  - When there is no score yet: label "ยังไม่มีข้อมูลพอ" and the reason "ระบบกำลังเก็บข้อมูลพื้นฐาน".
  - Put the thresholds as named constants in one place, with no config change.
- **Web:** a new route/page "เช็กก่อนซื้อ" in the nav, following how existing routes (`data-route`) are built.
  - Pick a coin from the known symbols. Four tiles: score with τ, 24 h alerts with a link to the alert, liquidity vs normal, and history.
  - Above them, a big label chip with reasons and the disclaimer line. Colours: green / amber / red.
  - It refreshes on new `scores`/`alerts` bus messages, or by polling every 15 s if that is simpler.
  - Make it work at phone width. Keep the current HUD style. Never call it a buy/sell signal.

## Tests (no real `data/app.duckdb`; use temp stores)
- `count_alerts`: windows, provenance filter (sim excluded), and an upserted episode counted once.
- History and check endpoints: 404 for an unknown symbol, auth required, and shapes.
- Every label branch, including warm-up and the liquidity edges.
- The ring point `quote_vol` is present, and `/api/series` output is unchanged.
- Existing suite green: `uv run ruff check .` and `uv run pytest -q -m "not network and not slow"`.

## Out of scope
- 2.3 (the public track-record page), 1.3 on web, phase 3. No order-sending code, no API keys, no news API.
