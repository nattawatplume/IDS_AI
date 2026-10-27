# Market IDS — Design Spec

- วันที่: 2026-10-27 · สถานะ: Draft รอทีม review
- โปรเจกต์: CPE102 (ทีม 4 คน) · ต้นแบบเพื่อการศึกษา: ไม่ส่งคำสั่งซื้อขาย ไม่ระบุตัวบุคคล ไม่ใช่คำแนะนำการลงทุน
- ป้ายสถานะข้อเท็จจริงในเอกสารนี้: **[C]** ยืนยันจากแหล่งหลัก/ทดสอบจริงแล้ว · **[U]** ยังไม่ยืนยัน/สมมติฐาน

---

## 0. เป้าหมายและเกณฑ์ความสำเร็จ

**เป้าหมาย:** ระบบ IDS สำหรับตลาดคริปโต ที่ตรวจจับ "ความผิดปกติทางสถิติที่สอดคล้องกับรูปแบบการปั่นตลาด" จาก Binance public market data โดยมี AI (autoencoder) เป็นชั้นหลักร่วมกับชั้นกฎและสถิติ แล้วมี Chief รวมผลและตัดสิน พร้อม dashboard แบบ SOC

**เกณฑ์ความสำเร็จ (ตั้งต้น ปรับตามผลจริงและต้องรายงานทุกครั้งแม้ไม่ถึง):**

| ตัวชี้วัด | เป้า | วัดบน |
|---|---|---|
| Event recall ต่อ technique | ≥ 0.8 | test split (ระบุ provenance: REAL / SEMI-SYN / SYN) |
| False alerts | ≤ 2 ต่อชั่วโมง ต่อทั้ง universe ที่เฝ้า | clean normal ช่วง test |
| Compute latency (event → alert พร้อมใช้) | p99 < 2 s | replay ที่ความเร็ว 1× |
| Time-to-detect | รายงาน median / p90 (ไม่มีเป้าตายตัว ขึ้นกับขนาดหน้าต่าง) | test split |
| AI ต้องชนะ baseline | AE (หรือ fusion ที่มี AE) ดีกว่า IsolationForest, robust-z, Kamps–Kleinberg อย่างมีนัย (CI ไม่ทับกัน) | ถ้าไม่ชนะ ต้องเขียนลงรายงานตรงๆ |

**ไม่อยู่ในขอบเขต:** ส่งคำสั่งซื้อขาย, ระบุตัวบุคคล/บัญชี, ใช้เป็นหลักฐานทางกฎหมาย, คำแนะนำการลงทุน

---

## 1. ข้อจำกัดของข้อมูลจริง (ต้องอยู่ในรายงานทุกฉบับ)

| ข้อจำกัด | ผลต่อการออกแบบ |
|---|---|
| Spot archive (data.binance.vision) มีแค่ trades / aggTrades / klines — **ไม่มี L2 depth ย้อนหลังฟรี** [C] | spoofing/layering/quote stuffing ต้องใช้ L2 ที่บันทึกเองหรือข้อมูลสังเคราะห์ |
| ทีมบันทึก live ได้ครั้งละไม่กี่ชั่วโมง | Track B (order book) ใช้ real เพื่อ sanity check และวัด false alert เท่านั้น ส่วนการวัด recall ใช้ข้อมูลสังเคราะห์ |
| L2 ของ Binance เป็นระดับราคารวม ไม่มี order ID/อายุคำสั่ง, diff stream batch ทุก 100 ms [C] | "cancel" ต้องอนุมานจาก qty ลดลงโดยไม่มี trade; quote stuffing ต่ำกว่า 100 ms มองไม่เห็น ใช้ proxy เป็นจำนวน level ที่เปลี่ยนต่อ update |
| ไม่มี account ID | wash trading พิสูจน์ไม่ได้ ทำได้แค่ "wash-like pattern" + การทดสอบเชิงสถิติระดับคู่เทรด |
| ป้าย pump จริงมาจาก Telegram (announcement time, ไม่ครบ) [C] | label เป็นช่วง [t0−2m, t0+5m]; negative มีสัญญาณรบกวน (pump ที่ไม่ถูกบันทึก) → false alert ที่วัดได้อาจสูงเกินจริง ต้องระบุ |
| ป้าย pump จริงเป็นคู่ BTC ช่วง 2018–2022 เท่านั้น | ห้ามอ้างว่าผลใช้กับคู่ USDT ปัจจุบันได้ |
| bookTicker ของ spot ไม่มี timestamp [C] | stamp เวลาเครื่องรับ (`recv_us`) และระบุว่าเป็นเวลาเครื่อง |

ทั้งหมดนี้ไปอยู่ใน `docs/DATA_LIMITATIONS.md` และ dashboard แสดงป้าย provenance บนทุกตัวเลขผลลัพธ์

---

## 2. สถาปัตยกรรม (Approach A: Two-Track)

```
Sources ── Archive (zip + CHECKSUM)  │  Live WS (data-stream.binance.vision)  │  Red-Team Simulator
              └──────────────── normalize → Event schema (Trade / DepthDiff / BookTop) ────────────┘
Store ──── DuckDB + Parquet (partition: source/symbol/date, ts เป็น int64 µs)
Replay ─── Time Machine: event-time, deterministic, speed 1×/10×/max, seek  ──┐  (live ใช้ code path เดียวกัน)
Features ─ 1s base bars → windows 10s / 60s / 5m / 30m (baseline จากอดีตเท่านั้น) ◄┘
Detectors  [L1 Rules/signatures] [L2 Stats: robust-z, CUSUM, Page-Hinkley] [L3 AI: dense AE] (+ baselines)
Chief ──── conformal p-value ต่อ detector → fusion ต่อ technique → alert budget → dedup/lifecycle
Alert ──── Suricata-EVE-like record + evidence + technique MM-T00x
Explainer  template (ภาษาไทย) → optional Ollama rephrase + number/wording validator
Auditor ── ตรวจระบบเอง: alert-rate vs budget, calibration drift, data gaps, latency, leakage, staleness
Eval ───── event-wise metrics, multi-seed, ablation, leave-one-family-out, baselines, run registry
API/UI ─── FastAPI (REST + WebSocket) + static HTML/JS + lightweight-charts
```

**Agent model ภายในระบบ:** แต่ละหน้าที่เป็น "agent" = โมดูลที่รับ/ส่งข้อความผ่าน in-process event bus (asyncio queues ที่มีขนาดจำกัด) ภายใน process เดียว — เหตุผล: ทีมปี 1 debug ง่าย, ไม่มี infra เพิ่ม, แต่เพราะทุก agent คุยผ่าน contract จึงแยกเป็น process/บริการภายหลังได้โดยไม่แก้ logic

| Agent | หน้าที่ | Input → Output |
|---|---|---|
| Sensor (Archive/Live/Sim) | นำเข้าและ normalize | raw → `Event` |
| BookBuilder | สร้าง order book ตามกฎ U/u ของ Binance, resync เมื่อ gap | `DepthDiff` → `BookTop` |
| FeatureAgent | คำนวณ base bars และ windows แบบ incremental (ring buffer) | `Event` → `FeatureWindow` |
| Specialists (L1/L2/L3) | ให้คะแนนตาม contract `Detector` | `FeatureWindow` → `DetectorResult` |
| Chief | calibrate, fuse, ตัดสิน, dedup | `DetectorResult[]` → `Alert` |
| Explainer | สร้างคำอธิบายจาก evidence | `Alert` → `Explanation` |
| Auditor | เฝ้าสุขภาพระบบ (meta-IDS) | metrics ทุกตัว → `HealthReport` |
| RedTeam | ฉีดการโจมตีสังเคราะห์ (REPLAY/SIM เท่านั้น) | scenario → `Event` + `Label` |

---

## 3. Contracts (สัญญาระหว่างโมดูล)

ทั้งหมดเป็น pydantic v2 model อยู่ใน `idsai/contracts/` และมี schema version (`schema_v`) เมื่อเปลี่ยน field ต้องเพิ่มเวอร์ชันและผ่าน contract test

### 3.1 Events
- `Trade`: `ts_us, recv_us, venue, symbol, price, qty, quote_qty, taker_side ("buy"|"sell"), trade_id, n_fills, source ("archive"|"live"|"sim")`
  - `taker_side` = `"sell"` เมื่อ `isBuyerMaker = true` (ผู้ซื้อเป็น maker ⇒ ผู้ขายเป็นฝ่ายรุก) [C]
  - normalize: spot archive ก่อน 2025-01-01 เป็น ms, ตั้งแต่ 2025-01-01 เป็น µs; futures เป็น ms; boolean `True`/`true` [C]
- `DepthDiff`: `ts_us, recv_us, symbol, first_update_id (U), last_update_id (u), bids[(p,q)], asks[(p,q)]` (q=0 = ลบ level)
- `BookTop`: `ts_us, symbol, bids[≤20], asks[≤20], seq` — sample ทุก 100 ms หลัง book sync

**กฎสำคัญ:** event ไม่มีฟิลด์ ground truth ใดๆ (simulator เขียน label แยกตาราง) — ป้องกัน detector "แอบดูเฉลย"

### 3.2 Features
- `FeatureWindow`: `symbol, track ("T"|"B"), t_end_us, window_s, values: dict[str, float], feature_set_v`
- Feature registry (`idsai/detection/features/registry.py`) ประกาศชื่อ, หน่วย, คำอธิบายภาษาไทย, track — Explainer ใช้ข้อมูลนี้แปลเป็นข้อความ

### 3.3 Detector
```python
class Detector(Protocol):
    name: str                 # "ai.autoencoder_trade"
    layer: Literal["rule", "stat", "ai", "baseline"]
    track: Literal["T", "B"]
    technique_ids: list[str]  # ที่ detector นี้ให้หลักฐานได้
    def fit(self, normal: Iterable[FeatureWindow]) -> None: ...
    def score(self, w: FeatureWindow) -> DetectorResult: ...
    def state_dict(self) -> dict: ...        # สำหรับ versioning
```
- `DetectorResult`: `detector, t_end_us, symbol, score (สูง = ผิดปกติมาก), evidence: dict[str, float], top_features: list[(name, contribution)], technique_ids`
- เพิ่ม detector ใหม่ = เขียน class + ใส่ชื่อใน `configs/*.yaml` — ไม่ต้องแก้ Chief/API/UI (registry ค้นจากชื่อ)

### 3.4 Alert (ยึดโครง Suricata EVE + Zeek notice)
`alert_id, ts_event_us, symbol, technique_id, signature_id, rev, severity (1–3), status ("open"|"updated"|"closed"), fused_p, detectors: [{name, p, score}], agreement (จำนวน layer ที่เห็นด้วย), evidence, window, dedup_key, suppress_for_s, config_hash, model_versions, provenance ("live"|"replay"|"sim")`

### 3.5 DuckDB tables
`trades, book_top, features_1s, scores, alerts, labels(source, symbol, t_start_us, t_end_us, technique_id, family, scenario_seed), feedback(alert_id, verdict, analyst, ts), runs(run_id, git_sha, config_hash, seed, model_versions, split_def, metrics_json)`

---

## 4. Features

### 4.1 Track T (จาก trades)
Base bar 1 วินาที: `n_trades, buy_vol, sell_vol, quote_vol, vwap, ret, high, low, max_trade_qty, rush_orders` (rush order = กลุ่ม fill ฝั่งซื้อที่ timestamp เดียวกัน ≈ market buy หนึ่งคำสั่ง ตาม La Morgia et al.), `n_round_size` (สำหรับ wash-like)

Window features (10s / 60s / 5m / 30m): ผลรวม/อัตรา + **z-score เทียบ baseline 24 ชม. ก่อนหน้าของ symbol เดียวกัน (robust: median/MAD)** + `buy_imbalance`, `price_jump_max`, `ret_cum`, `vol_accel` และ CUSUM state ของ return/imbalance สำหรับ stealth pump

**ตัดออกโดยตั้งใจ:** hour-of-day/minute-of-day (เข้ารหัส "pump เริ่มที่ :00" ทำให้คะแนนเฟ้อ)

### 4.2 Track B (จาก book)
Base bar 1 วินาที: `spread_bps, mid, imbalance_top5, depth_l1..l10 (bid/ask), adds, cancels_inferred (qty ลดโดยไม่มี trade ที่ราคานั้น), levels_changed_per_update, update_rate, flicker_count (qty ใหญ่ ≥ k×median ที่ปรากฏแล้วหายใน T วินาทีโดยไม่ถูก fill), far_level_churn, cancel_to_trade_ratio`

### 4.3 กันข้อมูลอนาคตรั่ว
- baseline, scaler, percentile ทุกตัวคำนวณจากข้อมูลก่อน `t_end` เท่านั้น
- scaler/model fit บน train split เท่านั้น และบันทึก data hash ไว้ใน model metadata
- test อัตโนมัติ: เลื่อนข้อมูลอนาคตแล้ว feature ของอดีตต้องไม่เปลี่ยน (property test)

---

## 5. Red-Team Simulator

ทุก scenario = `(family, params, seed)` → สร้าง event สังเคราะห์ + label ในตาราง `labels`; ใช้ได้เฉพาะ source = archive replay / sim — **โค้ดปฏิเสธการฉีดเมื่อ mode = LIVE** (มี test)

| Technique | Family | วิธีฉีด |
|---|---|---|
| MM-T001 Pump-and-dump | `burst` (30 s–3 m), `stealth` (30–120 m) | ลงบน trade stream จริง: เพิ่ม synthetic taker-buy (มี rush orders), ปรับราคาต่อจากนั้นด้วย impact curve แล้วตามด้วย dump/decay |
| MM-T002 Spoofing | `single_wall` | order ใหญ่ห่าง touch 2–5 level ถือ 1–10 s ยกเลิกก่อนโดน fill มักตามด้วย trade ฝั่งตรงข้าม |
| MM-T003 Layering | `ladder` | หลายคำสั่ง 3–8 level ฝั่งเดียว ขึ้น/หายพร้อมกัน |
| MM-T004 Quote stuffing | `burst_churn` | การเพิ่ม/ลบถี่ที่ level ไกล ⇒ levels_changed_per_update พุ่ง |
| MM-T005 Wash-like | `pingpong` | คู่ซื้อ/ขายขนาดเท่ากัน ราคาเดียวกัน ห่างกันไม่กี่ ms, ขนาดกลม, volume ขึ้นแต่ราคาไม่ขยับ |

**Hard negatives (ต้องไม่แจ้งเตือน):** วันที่ผันผวนสูงสุดในข้อมูลจริง (เลือกด้วยสถิติ ไม่อ้างว่าเป็นข่าว), synthetic market maker quote 2 ฝั่งและ refresh, กำแพงคำสั่งจริงที่อยู่นานและถูก fill ในที่สุด, organic momentum (ราคาขึ้นต่อเนื่องพร้อม volume กระจายทั้งสองฝั่ง)

**Normal book สังเคราะห์:** zero-intelligence Poisson order flow (Cont, Stoikov & Talreja 2010) โดย calibrate arrival rate และ size distribution คร่าวๆ จาก recording จริงไม่กี่ชั่วโมง — รายงานว่าเป็นแบบจำลองอย่างง่าย และผลของ Track B เป็นขอบบน ไม่ใช่ประสิทธิภาพในตลาดจริง

**Unseen families:** ทุก family มี parameter region ที่กันไว้เฉพาะ test (เช่น stealth ที่ยาวกว่า 90 m, layering 7–8 levels) + การทดลอง leave-one-family-out

**Adversary co-evolution (โบนัส):** red team ค้นหา params ที่หลบ detector ได้ (random search ในงบที่กำหนด) แล้วรายงานจุดบอด — ไม่ใช้ผลนี้ train โดยตรงเพื่อกันการ overfit ต่อ simulator

---

## 6. Detectors (Defense in Depth)

### L1 — Rules / Signatures (`configs/signatures.yaml`)
แต่ละ signature: `sid, rev, technique_id, track, window, condition (expression บน feature), severity, description_th`
- ตัวอย่าง `sid 100101 MM-T001 burst`: `vol_z_60s > 6 and ret_60s > 0.02 and rush_z_60s > 4`
- รวม **Kamps & Kleinberg (2018)** เป็น signature baseline (volume และ price เหนือ rolling mean เกินเกณฑ์)
- score ของ rule = ระยะเกินเกณฑ์ที่ normalize แล้ว (ไม่ใช่แค่ 0/1) เพื่อให้ Chief calibrate ได้

### L2 — Statistics
- `stat.robust_z`: max |robust z| ข้าม feature ของ track (median/MAD จากอดีต)
- `stat.cusum`: one-sided CUSUM บน standardized return และ buy imbalance — สำหรับ stealth pump
- `stat.page_hinkley`: drift บน update rate / cancel ratio (Track B)
- `stat.wash_tests` (ระดับคู่เทรด รายสัปดาห์, ไม่ใช่ alert ต่อเหตุการณ์): Benford first-digit (χ² + ระวัง over-rejection ที่ n ใหญ่), round-size clustering, power-law tail ตาม Cong et al. (2023) → แสดงเป็น "posture indicator" ใน dashboard

### L3 — AI
- `ai.autoencoder_trade` / `ai.autoencoder_book`: dense AE (in → 64 → 16 → 64 → out), ReLU, MSE, weight decay, early stopping บน val
  - input: log1p สำหรับปริมาณ heavy-tailed → RobustScaler (fit บน train) → clip ±10
  - train บน normal เท่านั้น (ตัดช่วงที่มี label ±margin ออก)
  - evidence: reconstruction error รายฟีเจอร์ → top-k features (ใช้ใน Explainer)
  - ความเสี่ยงที่รู้: AE อาจ reconstruct anomaly ได้ดี (Bouman & Heskes 2025) → bottleneck เล็ก + clip + มี L1/L2 คอยคุม
- Baselines (ต้องมีเสมอ): `baseline.isolation_forest`, `baseline.pca_recon`, `baseline.random` (alert rate เท่ากัน)
- Supervised reference (เฉพาะ Track T จริง): reproduce La Morgia RF **ด้วย time split** — ใช้เป็นจุดอ้างอิง ไม่ใช่ detector หลัก (เพราะต้องใช้ป้ายจริงซึ่งมีน้อยและหมดอายุ)
- โบนัส: LSTM-AE / 1D-conv AE ถ้าเวลาพอและชนะ dense AE บน val

---

## 7. Chief (ผู้ตัดสิน)

1. **Calibration:** สำหรับแต่ละ detector เก็บคะแนนบน calibration set (normal ช่วงหลัง train, ก่อน test) แล้วแปลงคะแนนใหม่เป็น conformal p-value: `p = (1 + #{s_i ≥ s}) / (n + 1)` (Bates et al. 2023)
2. **Fusion ต่อ technique:** `S = max_d w_d · (−log10 p_d)` ข้าม detector ที่ประกาศ technique นั้น (default w = 1); ทางเลือก Fisher-weighted ใน config
3. **Re-calibrate fused score** บน calibration set แล้วตั้ง threshold ให้อัตราแจ้งเตือนบน normal = **alert budget** (เช่น 2/ชม. ทั้ง universe ⇒ แปลงเป็นอัตราต่อ window ต่อ symbol อัตโนมัติ)
4. **Severity:** จาก fused p และ `agreement` (จำนวน layer ที่ p < 0.01) — เช่น 3 layer เห็นพ้อง = severity 3
5. **Dedup/lifecycle:** `dedup_key = (symbol, technique, bucket 5 นาที)`, `suppress_for` แบบ Zeek, alert เปลี่ยนสถานะ open → updated → closed
6. **ถ้อยคำ:** fused p เรียกว่า "anomaly p-value / ความหายากเทียบตลาดปกติ" **ไม่ใช่ความน่าจะเป็นของการปั่น** — guarantee เป็นแบบ marginal และเสื่อมเมื่อ regime drift (Auditor เฝ้า)

---

## 8. Learning Loop ("พัฒนาตัวเอง" ที่พิสูจน์ได้)

- **ปรับได้ด้วยป้ายน้อย:** น้ำหนัก fusion `w_d` และ threshold ต่อ technique — จาก feedback (TP/FP) + label สังเคราะห์ บน **tuning split**
- **Champion/Challenger:** challenger ถูกยอมรับเมื่อดีกว่า champion บน **acceptance split** (กันไว้ ไม่เคยใช้ tune) เกินความแปรปรวนข้าม seed (CI ไม่ทับ) — ไม่งั้นคง champion
- **Retrain AE:** บน normal ใหม่ (รับมือ drift) ตามรอบ; ทุกเวอร์ชันเก็บใน `models/<name>/<version>/` พร้อม metadata (data hash, config hash, seed, metrics)
- **กัน feedback loop:** ไม่ตัดช่วงที่ "เคยแจ้งเตือน" ออกจาก train ทั้งหมด ตัดเฉพาะช่วงที่มี label ยืนยัน (± margin); Auditor แจ้งเตือนถ้าสัดส่วนที่ถูกตัด > 2% หรือ normal หดลงต่อเนื่อง
- **หลักฐาน:** กราฟก่อน/หลัง (v1 vs v2) บน acceptance split, หลาย seed พร้อม CI — สร้างจากตาราง `runs` เท่านั้น ไม่มีการวาดมือ

---

## 9. Auditor (ตรวจตัวระบบเอง)

| ตรวจ | วิธี | ผล |
|---|---|---|
| Alert rate vs budget | binomial test ต่อชั่วโมง/วัน | warn เมื่อเกิน budget อย่างมีนัย |
| Calibration drift | KS test ความ uniform ของ p-value บน normal ล่าสุด | แนะนำ recalibrate |
| Feature drift | PSI ต่อฟีเจอร์ | แสดงฟีเจอร์ที่ drift |
| Data integrity | gap ของ trade_id, book resync count, out-of-order events | ลด confidence ของ alert ช่วงนั้น |
| Latency SLO | p50/p99 ของ event → alert | warn เมื่อ p99 > 2 s |
| Leakage guard | detector ห้าม import โมดูล labels (import-linter test), train/test ไม่ทับเวลา | fail CI |
| Model staleness | อายุโมเดลเทียบ config | warn |
| Claim guard | ตัวเลขทุกตัวในรายงาน/หน้า Eval ต้องมี provenance tag | fail report build ถ้าไม่มี |

---

## 10. Explainer

- **Template (หลัก, deterministic):** เช่น "พบความผิดปกติทางสถิติ (MM-T001 รูปแบบ pump) ที่ XYZBTC เวลา 14:00:12 UTC: ปริมาณซื้อใน 60 วินาทีสูงกว่าค่ากลาง 24 ชม. 8.3 เท่า (robust z = 7.1), ราคาขึ้น 4.2% ใน 60 วินาที; 3 จาก 3 ชั้นตรวจจับเห็นพ้อง; ความหายากเทียบตลาดปกติ p ≈ 0.0004" ตัวเลขทุกตัวดึงจาก `evidence`
- **Ollama (เสริม, ออฟไลน์):** ส่ง evidence JSON + template ให้เรียบเรียงภาษาไทยให้อ่านง่าย แล้ว **validator** ตรวจ: (1) ทุกตัวเลขในผลลัพธ์ต้องตรงกับค่าใน evidence (ยอมคลาดเคลื่อนจากการปัด) (2) ห้ามคำกล่าวหาบุคคล/คำต้องห้าม (3) ต้องมีวลี "ความผิดปกติทางสถิติ" — ไม่ผ่านข้อใด ⇒ ใช้ template
- LLM ไม่มีสิทธิ์เปลี่ยน severity/การตัดสินใดๆ

---

## 11. Evaluation Protocol

### 11.1 Splits (ตามเวลาเสมอ)
- **Track T REAL:** ป้าย Sapienza (MIT) + Hu et al. (ไม่มี license: ใช้เพื่อวิจัย อ้างอิง ไม่แจกจ่ายซ้ำ) รวม ~559 events, ~525 มีข้อมูลใน archive [C]; dedup (symbol, ±5 นาที)
  - train normal: 2018-01 → 2019-06 · calibration: 2019-07 → 2019-12 · test: 2020-01 → 2022-01 (ค่าตั้งต้น ปรับได้ใน `configs/splits.yaml` แต่ต้องประกาศก่อนดูผล test)
  - normal = วันของ symbol เดียวกันที่ห่างจาก label ≥ 2 วัน + symbol BTC-pair อื่นที่สุ่มด้วย seed
- **Track T SEMI-SYN:** ฉีด pump สังเคราะห์ลงบน normal จริงช่วง test
- **Track B SYN:** book สังเคราะห์ seed ชุด A สำหรับ fit/calibrate, seed ชุด B + parameter region ที่กันไว้สำหรับ test; recording จริงใช้วัด false alert และ sanity

### 11.2 Metrics (ห้ามใช้ accuracy และ point-adjust F1)
- **Event recall:** event ถูกจับ ถ้ามี alert technique ที่ถูกต้องใน [t_start − 2m, t_end + 5m] (configurable)
- **Alert precision:** สัดส่วน alert ที่ทับ label ใดๆ (± margin) → F1 จากสองค่านี้
- **False alerts / ชั่วโมง** บน clean normal และ **บน hard negatives แยก**
- **Time-to-detect:** median / p90 นับจาก t_start
- **PR-AUC ระดับ window** (รอง), **compute latency** p50/p99
- ทุกตัวแยกตาม technique, family, provenance

### 11.3 ความน่าเชื่อถือ
- ≥ 5 seeds (simulator + model init) → mean ± std และ bootstrap 95% CI ข้าม events
- Ablation: L1 / L2 / L3 / L1+L2 / ทั้งหมด → พิสูจน์ (หรือหักล้าง) defense in depth
- Leave-one-family-out สำหรับ Chief tuning
- Baselines เสมอ: random, Kamps–Kleinberg, IsolationForest, robust-z, (ref) La Morgia RF time-split
- **ทุก run บันทึกลง `runs`** และรายงานสร้างจากตารางนี้ทั้งหมด (ห้ามเลือกเฉพาะผลสวย — report แสดงจำนวน run ทั้งหมด)

---

## 12. API, Dashboard, Modes

### 12.1 Modes
- **LIVE:** `wss://data-stream.binance.vision` combined stream (`@trade`, `@aggTrade`, `@depth@100ms`, `@bookTicker`, `timeUnit=MICROSECOND`), REST snapshot จาก `data-api.binance.vision`; จัดการ ping/pong, reconnect ก่อน 24 ชม., ≤ 5 control msg/s, depth snapshot 5000 = weight 250 (จาก 6000/นาที) ⇒ จำกัด resync [C]
- **REPLAY (Time Machine):** เล่นจาก DuckDB/Parquet ตาม event time, play/pause/speed/seek, ฉีด Red-Team ได้
- **VIDEO:** วิดีโอเดโมที่อัดไว้ + "demo pack" ข้อมูลเล็กที่สร้างในเครื่อง (สำรองเมื่อเน็ต/Binance มีปัญหาวันนำเสนอ)

### 12.2 API (FastAPI)
`GET /api/alerts`, `GET /api/alerts/{id}`, `POST /api/alerts/{id}/feedback`, `GET /api/symbols`, `GET /api/health` (Auditor), `GET /api/experiments`, `GET /api/taxonomy`, `POST /api/replay/control`, `POST /api/redteam/inject` (ปฏิเสธใน LIVE), `WS /ws/stream` (bars, scores, alerts, health ที่ 2–4 Hz)

### 12.3 Dashboard (static HTML/JS + lightweight-charts 5.x, ไม่มี build step)
สไตล์ SOC ที่สะอาด อ่านง่าย รองรับ light/dark:
- **แถบบน:** mode (LIVE/REPLAY/SIM), event clock, สถานะ Auditor, มาตรวัด alert budget
- **Monitor:** กราฟราคา + marker แจ้งเตือน, เลนคะแนน fused พร้อมเส้น threshold, เลน L1/L2/L3 แยก
- **Alert queue (triage):** severity, MM-T00x, symbol, เวลา → **Alert detail:** evidence, detector ที่เห็นพ้อง + p-value, คำอธิบายไทย, การ์ด technique, ปุ่ม TP/FP
- **Book heatmap (Track B):** depth ตามเวลา ไฮไลต์ flicker/cancel
- **Red-Team:** เลือก family/params → ฉีด → ดูการตรวจจับพร้อมตัวจับเวลา time-to-detect
- **Evaluation:** ตาราง metrics ต่อ technique, ablation, กราฟก่อน/หลัง retrain พร้อม CI, **ป้าย REAL / SEMI-SYN / SYN บนทุกตัวเลข**
- **Taxonomy:** ATT&CK for Markets (MM-T001–T005) พร้อมนิยามและข้อจำกัดการตรวจจับ
- ต้องแสดง attribution: TradingView (lightweight-charts, Apache-2.0 NOTICE) และ "Binance Vision" (CC BY-NC-SA 4.0)

---

## 13. ATT&CK for Markets (กรอบที่ทีมสร้าง)

อ้างอิงนิยามจาก EU Delegated Regulation 2016/522 Annex II และ MiCA Art. 91 [C] (ไม่มี "ATT&CK for markets" ทางการ — ทีมสร้างเอง)

| ID | ชื่อ | อ้างอิง | สิ่งที่ข้อมูลสาธารณะให้ได้ |
|---|---|---|---|
| MM-T001 | Pump-and-dump | 2016/522 Annex II §1(4)(c) | trades: ✅ ทดสอบกับป้ายจริงได้ |
| MM-T002 | Spoofing | §1(6)(a) | L2 ที่บันทึกเอง: ⚠️ heuristic, ป้ายสังเคราะห์เท่านั้น |
| MM-T003 | Layering | §1(5)(e) | เหมือน T002 |
| MM-T004 | Quote stuffing | §1(4)(e), MiCA 91(3)(b) | ⚠️ เห็นได้ที่ความละเอียด 100 ms เท่านั้น |
| MM-T005 | Wash-like trading | §1(3)(a),(c) | ⚠️ ไม่มี account ID → pattern + สถิติระดับคู่เทรด |

แต่ละ technique มี sub-technique ตาม family (เช่น MM-T001.001 burst, MM-T001.002 stealth) อยู่ใน `idsai/detection/taxonomy.yaml`

---

## 14. Repo, Ops, Reproducibility

```
IdsAI/
  idsai/
    contracts/            # events, features, detector, alert (ทุกคนใช้ร่วม)
    data/                 # Role 1: archive, live, book, store, replay
    detection/            # Role 2: features/, rules/, stats/, simulator/, taxonomy.yaml
    ai/                   # Role 3: autoencoder, baselines, chief, learning, auditor
    eval/                 # Role 3: splits, metrics, experiments, report
    product/              # Role 4: api/, web/, explainer/
    bus.py, config.py, cli.py
  configs/                # default.yaml, splits.yaml, signatures.yaml, scenarios.yaml
  tests/                  # unit, contract, property (no-leak), e2e (replay ขนาดเล็ก)
  docs/                   # ETHICS.md, DATA_LIMITATIONS.md, TEAM_GUIDE.md, REPORT.md, specs/
  scripts/                # download archive/labels, build demo pack
```
- Python 3.13 + uv, dependencies pin ใน `pyproject.toml` + `uv.lock`; torch จาก CPU index
- config เป็น YAML → pydantic settings; ทุก run บันทึก `config_hash`, `git_sha`, `seed`
- CI (GitHub Actions): ruff + pytest บน fixture ขนาดเล็ก (ไม่ดาวน์โหลดจาก Binance ใน CI)
- ข้อมูลดิบไม่ commit (`.gitignore`); มีสคริปต์ดาวน์โหลด + ตรวจ CHECKSUM; label Sapienza (MIT) vendor ได้พร้อม license; Hu et al. ดาวน์โหลดผ่านสคริปต์เท่านั้น
- CLI: `idsai download`, `idsai ingest`, `idsai record` (live), `idsai train`, `idsai eval`, `idsai serve`, `idsai report`

## 15. จริยธรรม (สรุป — รายละเอียดใน ETHICS.md)
- ไม่มีโค้ดส่งคำสั่ง/ใช้ API key; Red-Team ทำงานกับ replay/sim เท่านั้น (บังคับในโค้ด + test)
- ถ้อยคำ "พบความผิดปกติทางสถิติ" ไม่กล่าวหาบุคคล; ไม่พยายามระบุตัวตน (ตามเงื่อนไข Binance Vision ด้วย)
- ใช้ข้อมูลแบบ non-commercial พร้อม attribution; ไม่ใช่คำแนะนำการลงทุน

---

## 16. Milestones (MVP ก่อนเสมอ — หยุดที่จุดไหนก็ยังได้ระบบที่ครบ)

| M | ผลลัพธ์ | นับเป็น MVP |
|---|---|---|
| M0 | repo, contracts, config, CI, fixtures | ✅ |
| M1 | archive download + CHECKSUM + ingest + store + Time Machine replay | ✅ |
| M2 | Track T features + Red-Team T001/T005 + rules + robust-z/CUSUM + AE + baselines | ✅ |
| M3 | Chief (conformal + budget) + eval harness (real + semi-syn) + run registry | ✅ |
| M4 | API + dashboard (REPLAY) + Explainer template + กราฟก่อน/หลัง retrain | ✅ |
| M5 | live recorder + BookBuilder + Track B sim (T002–T004) + book detectors | โบนัส |
| M6 | learning loop เต็ม + Auditor เต็ม + Ollama explainer + wash tests | โบนัส |
| M7 | co-evolution, LSTM-AE, VIDEO pack, รายงานฉบับสมบูรณ์ | โบนัส |

## 17. แหล่งอ้างอิงหลัก
- Binance: github.com/binance/binance-spot-api-docs (web-socket-streams.md, rest-api.md), github.com/binance/binance-public-data (README, TERMS_AND_CONDITIONS.md — CC BY-NC-SA 4.0)
- La Morgia et al. 2023, ACM TOIT 23(1) — github.com/SystemsLab-Sapienza/pump-and-dump-dataset (MIT)
- Hu et al. 2023, SIGMOD — github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency (ไม่มี license)
- Kamps & Kleinberg 2018, Crime Science 7:18 · Cong et al. 2023, Management Science 69(11) (NBER w30783)
- Kim et al. 2022 AAAI (arXiv 2109.05257) · Tatbul et al. 2018 NeurIPS (arXiv 1803.03639)
- Bates et al. 2023, Annals of Statistics (arXiv 2104.08279) · Sarfraz et al. 2024 ICML (arXiv 2405.02678) · Wu & Keogh (arXiv 2009.13807) · Bouman & Heskes 2025 (arXiv 2501.13864)
- Cont, Stoikov & Talreja 2010, Operations Research 58(3) · EU Delegated Regulation 2016/522 Annex II · MiCA Art. 91
