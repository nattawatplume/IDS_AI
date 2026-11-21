# รายงานผลการทดลอง Market IDS (เพื่อการศึกษา)

## ที่มาของตัวเลข

- ตารางในภาคผนวกคัดลอกตรงตัวจาก `data/report.md` ที่สร้างด้วย `uv run idsai report` (อ่านทุก run ใน `data/research.duckdb`) ตัวเลขผลการทดลองในบทอภิปรายมาจากตารางนั้นและ `data/report.json` ส่วนตัวเลขอื่นมาจาก: ผลวัด pump จริงช่วง train (`scripts/measure_pumps.py`, log `measure.log`), บันทึก inject เดโมของ Task 19 ใน `progress.md` (F 3.54 / 2.92, τ 3.87), log ของ eval (`eval_seed0.log`, `eval_seeds1to4.log`, `eval_seeds1to4b.log`) และ `live_check.log`
- git_sha `b1e37136421e4e569a71723c6acc59c3867611ef` · config_hash `53e3b662ab96` · signatures_sha `0fe49a5924d7` (ไม่ได้ปรับ signature ตาม Ruling 46 L3) · split `7fb702a841c5` · รอบเติมข้อมูล (2026-11-19/20) รัน real_t001 และ semisyn ใหม่ที่ commit 9aecfee ด้วย `configs/eval_prev_live.yaml` (config_hash เดิม) ตัวเลขทุกค่าที่มีอยู่เดิมตรงกับรอบก่อนทุกตัว (เทียบ metrics 80 runs ต่างกัน 0 ค่า) รอบนี้เพิ่มเฉพาะ CI ของ FA, จำนวนวัน และตารางสับสน · 2026-11-20/21 เพิ่มการทดลอง m1_fusion (commit 59179d7, 300 runs) ตารางในภาคผนวกจึงขึ้นคำเตือนว่ามีหลาย git_sha
- `uv run idsai eval` 6 การทดลอง (real_t001, semisyn, ablation, baselines, ref_supervised, retrain_drift) × seed 0–4 รวม 500 runs ไม่มี error งบการแจ้งเตือน 0.1 / 0.5 / 2 / 10 ต่อ symbol-day งบหลักคือ 2
- เครื่อง: โน้ตบุ๊ก Windows เครื่องเดียว รันทีละการทดลอง เวลารวมจากบรรทัด `wall` ใน log: seed 0 = 16,156 s (ราว 4.49 ชม.), seed 1–4 = 34,366 s (ราว 9.55 ชม.), รวม 50,522 s (ราว 14.0 ชม.) การรัน semisyn seed 1–4 ครั้งแรก (เริ่ม 2026-11-04 10:36:38Z) หยุดก่อนจบโดยไม่มีผลใน log แล้วเริ่มใหม่ 12:56:33Z เวลาของครั้งที่หยุดไม่นับรวม
- ป้ายจริง 601 รายการ ใช้ได้ 477 ตรวจเวลาในป้ายผ่านทุกแหล่งจึงไม่ได้ตั้ง `label_offset_s` (ดู `docs/DATA_LIMITATIONS.md` หัวข้อ 2)
- ช่วง train ของ pump_burst ใน `configs/scenarios.yaml` ตั้งจาก 10–90 % ของ pump จริง 159 รายการที่วัดได้ในช่วง train (จาก 188) commit b1e3713 ก่อนรัน eval ช่วง test คงตามที่ออกแบบ (impact 0.03–0.05, vol_mult 5–30) และการทดลอง semisyn สุ่มเหตุการณ์จากช่วง test เท่านั้น

## บทอภิปราย

ถ้าไม่ระบุอื่น ตัวเลขคือ budget 2 ต่อ symbol-day, ค่าเฉลี่ยข้าม 5 seed [95 % CI], n = จำนวนหน่วย bootstrap (event สำหรับ REAL, symbol-day สำหรับ SEMI-SYN)

### 1. เป้าหมาย

| เป้า | ชุด | ค่า | ผล |
|---|---|---|---|
| recall ≥ 0.8 ต่อ technique | REAL pump จริง (real_t001 full) | recall_any 0.739 [0.680, 0.801], n = 200; recall_correct 0.399 [0.334, 0.472] | ไม่ถึง |
| | SEMI-SYN pump_burst | recall_any 0.295 [0.264, 0.328], n = 395; recall_correct 0.186 [0.163, 0.208] | ไม่ถึง |
| | SEMI-SYN pump_stealth | recall_any 0.351 [0.328, 0.375] (std ข้าม seed 0.191), n = 394; recall_correct 0.101 [0.085, 0.118] | ไม่ถึง |
| | SEMI-SYN wash_pingpong | recall_any 1.000 [1.000, 1.000] (rule of three: ขอบล่างราว 99 %), n = 401; recall_correct 0.001 [0.000, 0.003] | recall_any ถึง, ระบุเทคนิคไม่ถึง |
| FA ต่อ symbol-day นอกชุดสอบเทียบ ≤ งบ | REAL test (real_t001 full, 579 symbol-day ในชุด; FA นับเฉพาะวันปกติ 401 symbol-day) | 0.121 [0.079, 0.170] / 0.400 [0.263, 0.574] / 1.650 [1.240, 2.166] / 6.477 [5.670, 7.387] ที่งบ 0.1 / 0.5 / 2 / 10 | ค่าเฉลี่ยถึงที่งบ 0.5, 2, 10 เกินที่งบ 0.1; ที่งบ 2 ขอบบน CI เกินงบเล็กน้อย |
| | REAL accept (retrain_drift v1, 456 symbol-day ในชุด; FA นับเฉพาะวันปกติ จำนวนวันไม่อยู่ในรายงาน) | 0.204 / 0.924 / 3.578 / 10.557 | เกินทุกงบ |
| FA บน hard negative (spec: ต้องไม่แจ้งเตือน) | REAL (real_t001 full, 20 symbol-day) | 11.170 [6.460, 17.021] ต่อ symbol-day ที่งบ 2 (0.730 / 3.010 / 23.160 ที่งบ 0.1 / 0.5 / 10) | ไม่ถึง |
| compute latency p99 < 2 s | replay STEEMBTC 2021-01-18 ทั้งวัน ชุดเดโม เครื่องนี้ (`scripts/bench_latency.py`, n = 8,637 แถว) | p99 5.9 ms, median 4.0 ms, max 13.7 ms (เฉพาะเวลาประมวลผล ไม่รวมเครือข่าย การรอปิดแท่ง stride และการเขียน store) | ถึง (วัด 1 symbol ยังไม่ได้วัดหลาย symbol พร้อมกัน) |

- CI ของ FA ได้จากการ bootstrap ทีละ symbol-day (เฉลี่ยข้าม seed ในแต่ละวันก่อน) hard negative มีแค่ 20 symbol-day CI จึงกว้าง
- REAL ได้ recall_any ≥ 0.8 เฉพาะที่งบ 10 (0.808 [0.751, 0.858]) แลกกับ FA 6.477 และ hard 23.160 ต่อ symbol-day
- ตัวเลขทุกตัวคือความผิดปกติทางสถิติ ไม่ใช่ข้อกล่าวหาต่อบุคคลหรือตลาดใด

### 2. AE เทียบ baseline (REAL, paired bootstrap, budget 2, 200 events, 5 seed)

| baseline | ผลต่าง recall_any (AE − baseline) [95 % CI] | อ่านว่า |
|---|---|---|
| baseline.random | +0.639 [0.575, 0.700] | AE ดีกว่า |
| baseline.pca | +0.075 [0.035, 0.122] | AE ดีกว่า |
| stat.robust_z | +0.005 [−0.033, 0.043] | ไม่พบความต่าง (CI คร่อม 0) |
| baseline.iforest | −0.158 [−0.208, −0.111] | iforest สูงกว่า AE |
| baseline.kk | −0.170 [−0.222, −0.121] | KK สูงกว่า AE |

เป้าใน spec ("AE ชนะ IsolationForest, robust-z, Kamps–Kleinberg อย่างมีนัย") **ไม่ถึง** ตารางในภาคผนวกเขียน iforest และ KK ว่า "ไม่มีนัยสำคัญ" เพราะทดสอบทางเดียวว่า AE ดีกว่าหรือไม่ แต่ CI ทั้งช่วงต่ำกว่า 0 แปลว่าในการทดลองนี้สองตัวนี้จับเหตุการณ์จริงได้มากกว่า AE (ไม่ได้ทดสอบทิศนี้โดยตรง) อ่านคู่กับ: iforest FA 2.066 (เกินงบ 2) hard 0.650, PR-AUC 0.253; KK τ saturated (FA 0.601 ใช้งบไม่เต็ม) hard 1.150, PR-AUC 0.074; AE FA 1.556 hard 11.300, PR-AUC 0.154

### 3. Ablation (budget 2)

| ชั้น | REAL recall_any (n = 200) | REAL FA / hard | SEMI-SYN recall_any (n = 1188) |
|---|---|---|---|
| ai | 0.650 [0.588, 0.710] | 1.556 / 11.300 | 0.519 [0.505, 0.535] |
| stats | 0.650 [0.580, 0.715] | 0.773 / 3.700 | 0.542 [0.526, 0.557] |
| rules (τ saturated) | 0.405 [0.340, 0.475] | 1.172 / 2.800 | 0.396 [0.382, 0.409] |
| rules+stats | 0.745 [0.685, 0.800] | 1.516 / 5.550 | 0.710 [0.693, 0.726] |
| all (rules+stats+ai) | 0.739 [0.680, 0.801] | 1.650 / 11.170 | 0.552 [0.539, 0.567] |

- การเพิ่ม AE บน rules+stats ไม่ทำให้ recall สูงขึ้น: REAL 0.745 → 0.739, recall_correct 0.435 → 0.399, แต่ FA บน hard negative เพิ่มจาก 5.550 เป็น 11.170 บน SEMI-SYN ที่งบ 2 ชุด all ได้ 0.552 เทียบ rules+stats 0.710 (ที่งบ 10 ใกล้กัน 0.923 กับ 0.927)
- paired bootstrap ที่งบ 2 (ภาคผนวก): REAL all − rules+stats = −0.006 [−0.039, 0.028] ไม่มีนัย คือ AE ไม่เพิ่ม recall บนข้อมูลจริง; SEMI-SYN = −0.157 [−0.177, −0.138] rules+stats ตรวจพบมากกว่าอย่างมีนัย (หน่วยสุ่ม symbol-day 401 กลุ่ม) ส่วน all ดีกว่า rules, stats และ ai เดี่ยวอย่างมีนัยบน REAL
- rules ถึงเพดานที่งบ 2 (τ saturated) งบจึงไม่ได้ถูกทดสอบจริง; stats ระบุเทคนิคไม่ได้ (recall_correct 0 จาก 200 events REAL, rule of three ขอบบนราว 2 %) recall_correct จึงมาจากชั้น rule

### 4. retrain_drift Decision (REAL, split accept, 94 events, 5 seed, budget 2)

- full stack (deployed): v1 0.760 [0.670, 0.840] → v2 0.774 [0.681, 0.853], ผลต่าง +0.015 [−0.006, 0.043] → **ไม่ยอมรับ challenger คง v1**
- AE alone (diagnostic): v1:ai 0.721 → v2:ai 0.728, ผลต่าง +0.006 [−0.021, 0.040] → ไม่ยอมรับ
- ทั้ง v1 และ v2 มี FA เกินงบบน split นี้ (3.578 และ 3.611)

### 5. ช่องว่างกับ supervised reference (REAL, ใช้ป้าย ใช้เทียบเท่านั้น)

ref.supervised_rf: recall_any 0.851 [0.802, 0.896], FA 2.121, hard 1.610, PR-AUC 0.354 เทียบระบบเต็ม 0.739 [0.680, 0.801], FA 1.650, hard 11.170, PR-AUC 0.243 ผลต่างของค่าเฉลี่ย recall 0.112 ไม่ได้ทดสอบแบบ paired และ CI แทบชนกัน (ขอบล่าง 0.802 กับขอบบน 0.801) ทั้งนี้ RF มี FA 2.121 เกินงบ 2 ผลนี้ชี้ว่าข้อมูลอาจมีสัญญาณที่แยกเหตุการณ์ได้มากกว่าที่ระบบแบบไม่ใช้ป้ายจับได้ โดยเฉพาะบน hard negative แต่ยังสรุปไม่ได้

### 6. เรื่องที่ยกมาจากงานก่อน (Ruling 46 L5)

- **ความเฉพาะเจาะจงของ T001 บนข้อมูลจริง และตารางสับสนรายตระกูล (budget 2, ภาคผนวก):** REAL telegram_pump ระบุเป็น MM-T001 0.399, MM-T000 (ระบุไม่ได้) 0.339, MM-T005 0.001, ตรวจไม่พบ 0.261 ส่วน SEMI-SYN wash_pingpong ถูกระบุเป็น MM-T000 0.988 และ MM-T005 เพียง 0.001 แปลว่าระบบเห็นความผิดปกติในวันที่มี wash แต่ชั้น rule แทบไม่ให้เทคนิคที่ถูก (งาน M3 ใน `docs/ROADMAP.md`) pump_burst ตรวจไม่พบ 0.704 และ pump_stealth 0.648 เหตุการณ์ที่ถูกระบุหลายเทคนิคนับทุกคอลัมน์
- **Ruling 30 (เริ่ม batch เร็วขึ้น 60 นาที) ไม่ได้ทำ:** batch เริ่มนับหน้าต่างใหม่ทุกวัน ส่วน replay/live เป็นสายต่อเนื่อง τ ที่สอบเทียบจากแถว batch จึงอาจคลาดเล็กน้อยในชั่วโมงแรกหลังเที่ยงคืนเมื่อรันแบบ stream ทางแก้ต้องใช้ข้อมูล D−2, ล้าง cache และสอบเทียบใหม่ จึงไม่ทำในรอบนี้ ตัวเลขในรายงานเป็นของ batch ทั้งหมด
- **inject เดโมที่จับไม่ได้:** ใน Task 19 ฉีด pump_burst ช่วง test สองครั้ง (impact 3.5 % vol ×6 และ 4.1 % vol ×25) ได้ F สูงสุด 3.54 และ 2.92 ต่ำกว่า τ 3.87 ที่งบ 2 ของชุดเดโม ผลนี้สอดคล้องกับ SEMI-SYN pump_burst ที่งบ 2 ซึ่งจับได้ 0.295 [0.264, 0.328] (n = 395) คือพลาดเป็นส่วนใหญ่ ช่วง test ตั้งใจให้ยากและอยู่นอกการกระจายของ pump จริง: impact 3–5 % ต่ำกว่าหรือใกล้เคียงขอบ 10 % ของ pump จริงช่วง train (0.0499) และ vol_mult 5–30 ต่ำกว่าขอบ 10 % (45.39) (มัธยฐานของ pump จริง: impact 0.1543, vol_mult 738.9) ส่วน pump จริงบนข้อมูล REAL จับได้ 0.739 ช่วง train ของ Red-Team ไม่ได้ถูกประเมินในรายงานนี้

### 7. ข้อจำกัด

ดูทั้งหมดใน `docs/DATA_LIMITATIONS.md` ที่กระทบการอ่านรายงานนี้โดยตรง:

- ผล REAL ใช้ได้กับคู่ BTC ช่วง 2018–2022 เท่านั้น ป้ายบางส่วน (Hu-only) มีเวลาห่างจุดปริมาณซื้อขายสูงสุดมาก
- SEMI-SYN วัดเฉพาะรูปแบบที่จำลองในช่วง test ไม่ได้วัดการปั่นจริง
- หน้าต่างทับซ้อนมี autocorrelation ทำให้การรับประกันของ conformal p-value อ่อนลง อัตรา FA จริงจึงอาจเกินงบ (เห็นที่งบ 0.1 และบน split accept)
- latency วัดบน 1 symbol-day ของเครื่องนี้เท่านั้น; CI ของ FA บน ablation, baselines, ref_supervised และ retrain_drift ยังไม่มีเพราะยังไม่ได้รันการทดลองเหล่านั้นใหม่ (ก่อนรัน ablation ใหม่ต้องแก้การรวม fa_by_day ของ semisyn หลายตระกูล ดู `docs/ROADMAP.md` M0)
- live check รอบนี้เป็นการตรวจ sensor อย่างเดียว ไม่มีการตรวจจับบนข้อมูลสด
- ไม่ได้ปรับ signature บนข้อมูล train (L3) recall ของชั้น rule อาจต่ำกว่าชุดที่ปรับแล้ว

### 8. หลังการแจ้งเตือน: เตือนก่อนจุดสูงสุดหรือไม่ และราคาเป็นอย่างไร (REAL, real_t001 full, budget 2, 5 seed)

ที่มา `scripts/post_alert_outcomes.py` → `data/post_alert_outcomes.json` (อ่านอย่างเดียว) เวลาเตือน = เวลาเริ่ม episode แรกที่จับคู่กับป้าย (t0 + TTD) ไม่รวมเวลารอปิดแท่ง 1 s และ stride 10 s

- จับได้ 151 จาก 200 ป้าย (739 คู่ ป้าย × seed) วัดจุดสูงสุดได้ 136 ป้าย (ตัดออก late_peak 9, no_rise 6)
- เตือนก่อนจุดสูงสุด 0.847 [0.788, 0.901] (เฉลี่ยรายป้ายข้าม seed ที่จับได้, bootstrap รายป้าย) ค่ามัธยฐานเตือนก่อน 40 s (q10 −1.6 s, q90 355 s); เฉพาะคู่ที่ episode เริ่มไม่เกิน 120 s ก่อนป้าย: 0.822 [0.757, 0.889], มัธยฐาน 31 s
- ราคา 1 ชม. หลังเตือนต่ำกว่าราคาตอนเตือน 0.888 [0.838, 0.934] ของป้าย, มัธยฐานผลต่าง −11.3 % แต่ภายในชั่วโมงนั้นราคายังขึ้นได้อีก (มัธยฐานของจุดสูงสุด +15.5 %) ต้องรายงานคู่กันเสมอ
- ผล 24 ชม. มีเพียง 26 จาก 739 คู่ (ไม่ได้เก็บ trades วันถัดไป) จึงไม่ใช้
- เป็นสถิติย้อนหลังของ pump ที่ระบบจับได้ ไม่ใช่การพยากรณ์ราคาและไม่ใช่คำแนะนำการลงทุน

### 9. M1: ตัด AE ออกจากการรวมคะแนนหรือไม่ (REAL, budget 2, 5 seed)

ที่มา การทดลอง `m1_fusion` (commit 59179d7, `configs/eval_prev_live.yaml`) ตัวตรวจจับ fit และ score ครั้งเดียวต่อ seed ต่างกันเฉพาะวิธีรวมคะแนน และ τ ที่ fit ใหม่ของแต่ละแบบ (spec `docs/superpowers/specs/2026-11-20-m1-ae-false-alarms-design.md`)

- **กฎเลือกกำหนดก่อนดูผล test และใช้ split accept (94 events, accept-hard 20 วัน) เท่านั้น:** ตัดแบบที่ recall_any ต่ำกว่า V0 อย่างมีนัย แล้วเลือกแบบที่ FA ของวันผันผวนสูง (accept-hard) ต่ำสุด
- **ผลที่ accept:** FA hard ของ V0 all 12.47, V1 no_ai 7.45, V3 ai_shift2 7.62, V3 ai_shift1 10.75, V2 ai_mondrian 11.39 ไม่มีแบบใดที่ recall ต่ำกว่า V0 อย่างมีนัย **แบบที่ถูกเลือกคือ V1 no_ai** (AE ยังคำนวณและแสดงค่า p แต่ไม่เข้า F)
- **ผลที่ test (ใช้ดูอย่างเดียว ไม่ได้ใช้เลือก), V1 no_ai เทียบ V0 all:**
  - recall_any 0.745 [0.685, 0.800] เทียบ 0.739 [0.680, 0.801] ผลต่างจับคู่ (V0 − V1) −0.006 [−0.039, 0.028] ไม่ต่างอย่างมีนัย
  - FA hard 5.55 [3.35, 8.30] เทียบ 11.17 [6.46, 17.02] ลดลงราวครึ่งหนึ่ง และถึงเป้า M1 (≤ 5.55) พอดี
  - FA ทั้งหมด 1.516 [1.229, 1.840] เทียบ 1.650 [1.240, 2.166]; recall_correct 0.435 เทียบ 0.399
- **SEMI-SYN (ใช้ดูอย่างเดียว):** recall_any V1 0.710 เทียบ V0 0.552 (สูงกว่าอย่างมีนัย) แต่ ai_shift1 / ai_shift2 ได้ 0.881 / 0.874 สูงกว่า V1 มาก โดยที่ FA hard บน REAL test ใกล้ V1 (shift2 5.62) แปลว่า AE ยังช่วยกับ pump สังเคราะห์ถ้าบังคับให้ต้องหายากกว่า rules/stats ราว 2 หลัก ประเด็นนี้ส่งต่อไป M2 ไม่ได้ใช้เปลี่ยนผลการเลือก
- ยังไม่ได้เปลี่ยนค่าเริ่มต้นของระบบ live (`ai_mode`) การเปลี่ยนเป็นขั้นตอนแยกที่ต้องตัดสินใจก่อน

### 10. M4: จับช่วงกำลังเทขาย และ bear raid (REAL + SEMI-SYN, budget 2, 5 seed)

ที่มา การทดลอง `m4_sell` (branch feat/m4, `configs/eval_prev_live.yaml`) เทียบ `base` (ระบบเดิม) กับ `sell` (เปิด `features.sell_side`: feature ฝั่งขาย + กฎ dump phase MM-T001.003 และ bear raid MM-T006.001 จาก `configs/signatures_sell.yaml` ที่ตั้งเกณฑ์จากข้อมูล train เท่านั้น) spec `docs/superpowers/specs/2026-11-21-m4-sell-side-design.md`

- **ช่วงเทขายของ pump จริง** = (จุดสูงสุด + 60 s, จุดสูงสุด + dump_tau] วัดจากราคาจริง วัดได้ 147 จาก 200 ป้าย ใช้ได้ 85 ป้าย (อีก 62 เทขายจบภายใน 60 s จึงไม่นับในตัวหาร)
- **ระบุว่าเป็น "กำลังเทขาย" ได้ถูก:** sell 0.544 [0.447, 0.642] เทียบ base 0 (เดิมระบบบอกช่วงไม่ได้เลย) สูงขึ้นอย่างมีนัย
- **มีการแจ้งเตือนในช่วงเทขาย (ไม่สนว่าระบุช่วงถูกไหม):** 0.833 เทียบ 0.805 ไม่ต่างอย่างมีนัย ระบบเดิมเตือนในช่วงนี้อยู่แล้ว สิ่งที่เพิ่มคือบอกได้ว่าเป็นช่วงไหน
- **ไม่เสียของเดิม:** recall ช่วงพุ่ง 0.751 เทียบ 0.739 (ผลต่าง +0.012 [0.000, 0.029] ไม่ลดลง), FA 1.636 เทียบ 1.650, FA hard 10.94 เทียบ 11.17
- **bear raid (SEMI-SYN) ยังจับไม่ได้:** recall_any 0.171 ทั้งสองแบบ และระบุเป็น bear raid ถูกเพียง 0.006 กฎ bear raid แทบไม่ทำงานกับการฉีดแรงขายที่ใช้ช่วงพารามิเตอร์เดียวกับ pump_burst ต้องแก้ต่อ (ดู ROADMAP M4)
- ยังไม่ได้เปิด `sell_side` ในระบบ live

---

## ภาคผนวก: รายงานที่สร้างอัตโนมัติ (คัดลอกตรงตัวจาก `data/report.md`)

# รายงานผลการประเมิน Market IDS (เพื่อการศึกษา)

runs: 880

signatures_sha: 0fe49a5924d7

git_sha: 3682667c3664f57f1543ae2fe4fb3866c031216b-dirty, 59179d7d6a81dffff0cdb55dc2ffd39d4cd27a69-dirty, 83c783a9fa784a80d425b9287850daada177e677, 9aecfee1f199ee2695d2d99182c13eb29b688e45-dirty, b1e37136421e4e569a71723c6acc59c3867611ef, ec98769e69b20cf155892fb33118c85307b09e04-dirty · config_hash: 53e3b662ab96

labels (split 7fb702a841c5): eligible 477 of 601; dropped: out_of_period 0, cal_period 91, missing_day 33, no_baseline 0; symbol-days skipped at window loading: none

ทุกตัวเลขมาจากตาราง runs ทั้งหมด; ค่าเฉลี่ยข้าม seed และช่วงความเชื่อมั่น 95 % จากการ bootstrap (หน่วยตามคอลัมน์ bootstrap unit)

**คำเตือน:** ตาราง runs รวมผลจากหลาย git_sha (3682667c3664f57f1543ae2fe4fb3866c031216b-dirty, 59179d7d6a81dffff0cdb55dc2ffd39d4cd27a69-dirty, 83c783a9fa784a80d425b9287850daada177e677, 9aecfee1f199ee2695d2d99182c13eb29b688e45-dirty, b1e37136421e4e569a71723c6acc59c3867611ef, ec98769e69b20cf155892fb33118c85307b09e04-dirty) ผลอาจมาจากโค้ดคนละรุ่น (รุ่น -dirty นับเป็นคนละรุ่น)

## real_t001

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day [95% CI] | FA/symbol-day hard [95% CI] | TTD median / p90 (s) | PR-AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0.1 | REAL | 5 | 200 | 0.540 ± 0.025 [0.473, 0.605] | 0.229 ± 0.011 [0.180, 0.285] | event | 0.500 (prevalence 0.0017) | 0.121 [0.079, 0.170] (401 วัน) | 0.730 [0.340, 1.190] (20 วัน) | 9 / 61 | 0.243 |  |
| full | 0.5 | REAL | 5 | 200 | 0.638 ± 0.016 [0.574, 0.699] | 0.301 ± 0.017 [0.241, 0.366] | event | 0.300 (prevalence 0.0017) | 0.400 [0.263, 0.574] (401 วัน) | 3.010 [1.150, 5.601] (20 วัน) | 8 / 72 | 0.243 |  |
| full | 2 | REAL | 5 | 200 | 0.739 ± 0.011 [0.680, 0.801] | 0.399 ± 0.008 [0.334, 0.472] | event | 0.113 (prevalence 0.0017) | 1.650 [1.240, 2.166] (401 วัน) | 11.170 [6.460, 17.021] (20 วัน) | 6 / 60 | 0.243 |  |
| full | 10 | REAL | 5 | 200 | 0.808 ± 0.004 [0.751, 0.858] | 0.481 ± 0.009 [0.412, 0.555] | event | 0.037 (prevalence 0.0017) | 6.477 [5.670, 7.387] (401 วัน) | 23.160 [16.150, 31.231] (20 วัน) | -0 / 46 | 0.243 |  |

## semisyn

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day [95% CI] | FA/symbol-day hard [95% CI] | TTD median / p90 (s) | PR-AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pump_burst | 0.1 | SEMI-SYN | 5 | 395 | 0.049 ± 0.018 [0.035, 0.066] | 0.018 ± 0.009 [0.011, 0.026] | symbol-day | 0.283 (prevalence 0.0061) | 0.122 [0.085, 0.167] (395 วัน) | – | 208 / 320 | 0.134 |  |
| pump_burst | 0.5 | SEMI-SYN | 5 | 395 | 0.106 ± 0.026 [0.085, 0.131] | 0.047 ± 0.013 [0.034, 0.061] | symbol-day | 0.229 (prevalence 0.0061) | 0.353 [0.257, 0.500] (395 วัน) | – | 152 / 278 | 0.134 |  |
| pump_burst | 2 | SEMI-SYN | 5 | 395 | 0.295 ± 0.047 [0.264, 0.328] | 0.186 ± 0.039 [0.163, 0.208] | symbol-day | 0.171 (prevalence 0.0061) | 1.425 [1.134, 1.983] (395 วัน) | – | 93 / 224 | 0.134 |  |
| pump_burst | 10 | SEMI-SYN | 5 | 395 | 0.769 ± 0.026 [0.744, 0.792] | 0.609 ± 0.020 [0.584, 0.633] | symbol-day | 0.117 (prevalence 0.0061) | 6.079 [5.515, 7.088] (395 วัน) | – | 82 / 161 | 0.134 |  |
| pump_stealth | 0.1 | SEMI-SYN | 5 | 394 | 0.049 ± 0.020 [0.039, 0.059] | 0.006 ± 0.001 [0.003, 0.010] | symbol-day | 0.283 (prevalence 0.0775) | 0.122 [0.087, 0.165] (394 วัน) | – | 6103 / 6935 | 0.851 |  |
| pump_stealth | 0.5 | SEMI-SYN | 5 | 394 | 0.099 ± 0.006 [0.085, 0.114] | 0.019 ± 0.001 [0.012, 0.026] | symbol-day | 0.249 (prevalence 0.0775) | 0.317 [0.234, 0.425] (394 วัน) | – | 6055 / 6882 | 0.851 |  |
| pump_stealth | 2 | SEMI-SYN | 5 | 394 | 0.351 ± 0.191 [0.328, 0.375] | 0.101 ± 0.020 [0.085, 0.118] | symbol-day | 0.274 (prevalence 0.0775) | 1.100 [0.889, 1.424] (394 วัน) | – | 5579 / 6863 | 0.851 |  |
| pump_stealth | 10 | SEMI-SYN | 5 | 394 | 0.999 ± 0.001 [0.998, 1.000] | 0.485 ± 0.023 [0.453, 0.517] | symbol-day | 0.185 (prevalence 0.0775) | 5.333 [4.833, 6.017] (394 วัน) | – | 533 / 1008 | 0.851 |  |
| wash_pingpong | 0.1 | SEMI-SYN | 5 | 401 | 1.000 ± 0.000 [1.000, 1.000] | 0.001 ± 0.002 [0.000, 0.003] | symbol-day | 0.896 (prevalence 0.0171) | 0.116 [0.078, 0.162] (401 วัน) | – | 17 / 62 | 0.835 | recall_any 401 จาก 401 symbol-day: ช่วงจริงอาจต่ำถึงราว 99% (rule of three) |
| wash_pingpong | 0.5 | SEMI-SYN | 5 | 401 | 1.000 ± 0.000 [1.000, 1.000] | 0.001 ± 0.002 [0.000, 0.003] | symbol-day | 0.766 (prevalence 0.0171) | 0.306 [0.215, 0.420] (401 วัน) | – | 14 / 44 | 0.835 | recall_any 401 จาก 401 symbol-day: ช่วงจริงอาจต่ำถึงราว 99% (rule of three) |
| wash_pingpong | 2 | SEMI-SYN | 5 | 401 | 1.000 ± 0.000 [1.000, 1.000] | 0.001 ± 0.002 [0.000, 0.003] | symbol-day | 0.449 (prevalence 0.0171) | 1.241 [0.941, 1.609] (401 วัน) | – | 11 / 33 | 0.835 | recall_any 401 จาก 401 symbol-day: ช่วงจริงอาจต่ำถึงราว 99% (rule of three) |
| wash_pingpong | 10 | SEMI-SYN | 5 | 401 | 1.000 ± 0.000 [1.000, 1.000] | 0.001 ± 0.002 [0.000, 0.003] | symbol-day | 0.165 (prevalence 0.0171) | 5.171 [4.540, 5.897] (401 วัน) | – | 8 / 16 | 0.835 | recall_any 401 จาก 401 symbol-day: ช่วงจริงอาจต่ำถึงราว 99% (rule of three) |

## ablation

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day [95% CI] | FA/symbol-day hard [95% CI] | TTD median / p90 (s) | PR-AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real_t001:ai | 0.1 | REAL | 5 | 200 | 0.505 ± 0.047 [0.445, 0.566] | – | event | 0.503 (prevalence 0.0017) | 0.110 [0.074, 0.155] (401 วัน) | 0.660 [0.130, 1.460] (20 วัน) | 40 / 127 | 0.154 | technique-agnostic: recall_correct n/a |
| real_t001:ai | 0.5 | REAL | 5 | 200 | 0.598 ± 0.033 [0.536, 0.661] | – | event | 0.275 (prevalence 0.0017) | 0.429 [0.276, 0.639] (401 วัน) | 3.160 [1.029, 6.180] (20 วัน) | 34 / 94 | 0.154 | technique-agnostic: recall_correct n/a |
| real_t001:ai | 2 | REAL | 5 | 200 | 0.650 ± 0.023 [0.588, 0.710] | – | event | 0.111 (prevalence 0.0017) | 1.556 [1.099, 2.118] (401 วัน) | 11.300 [5.518, 17.772] (20 วัน) | 21 / 83 | 0.154 | technique-agnostic: recall_correct n/a |
| real_t001:ai | 10 | REAL | 5 | 200 | 0.820 ± 0.014 [0.772, 0.866] | – | event | 0.020 (prevalence 0.0017) | 12.396 [11.006, 13.964] (401 วัน) | 26.230 [17.358, 34.971] (20 วัน) | -1 / 39 | 0.154 | technique-agnostic: recall_correct n/a |
| real_t001:all | 0.1 | REAL | 5 | 200 | 0.540 ± 0.025 [0.473, 0.605] | 0.229 ± 0.011 [0.180, 0.285] | event | 0.500 (prevalence 0.0017) | 0.121 [0.079, 0.170] (401 วัน) | 0.730 [0.340, 1.190] (20 วัน) | 9 / 61 | 0.243 |  |
| real_t001:all | 0.5 | REAL | 5 | 200 | 0.638 ± 0.016 [0.574, 0.699] | 0.301 ± 0.017 [0.241, 0.366] | event | 0.300 (prevalence 0.0017) | 0.400 [0.263, 0.574] (401 วัน) | 3.010 [1.150, 5.601] (20 วัน) | 8 / 72 | 0.243 |  |
| real_t001:all | 2 | REAL | 5 | 200 | 0.739 ± 0.011 [0.680, 0.801] | 0.399 ± 0.008 [0.334, 0.472] | event | 0.113 (prevalence 0.0017) | 1.650 [1.240, 2.166] (401 วัน) | 11.170 [6.460, 17.021] (20 วัน) | 6 / 60 | 0.243 |  |
| real_t001:all | 10 | REAL | 5 | 200 | 0.808 ± 0.004 [0.751, 0.858] | 0.481 ± 0.009 [0.412, 0.555] | event | 0.037 (prevalence 0.0017) | 6.477 [5.670, 7.387] (401 วัน) | 23.160 [16.150, 31.231] (20 วัน) | -0 / 46 | 0.243 |  |
| real_t001:rules | 0.1 | REAL | 5 | 200 | 0.215 ± 0.000 [0.165, 0.275] | 0.210 ± 0.000 [0.160, 0.270] | event | 0.183 (prevalence 0.0017) | 0.209 [0.142, 0.277] (401 วัน) | 1.250 [0.350, 2.550] (20 วัน) | 6 / 236 | 0.013 | deterministic: CI from event bootstrap |
| real_t001:rules | 0.5 | REAL | 5 | 200 | 0.370 ± 0.000 [0.305, 0.440] | 0.365 ± 0.000 [0.300, 0.435] | event | 0.088 (prevalence 0.0017) | 0.945 [0.741, 1.152] (401 วัน) | 2.550 [0.950, 4.650] (20 วัน) | 4 / 137 | 0.013 | deterministic: CI from event bootstrap |
| real_t001:rules | 2 | REAL | 5 | 200 | 0.405 ± 0.000 [0.340, 0.475] | 0.400 ± 0.000 [0.330, 0.470] | event | 0.080 (prevalence 0.0017) | 1.172 [0.940, 1.401] (401 วัน) | 2.800 [1.099, 5.100] (20 วัน) | 6 / 170 | 0.013 | τ saturated; deterministic: CI from event bootstrap |
| real_t001:rules | 10 | REAL | 5 | 200 | 0.405 ± 0.000 [0.340, 0.475] | 0.400 ± 0.000 [0.330, 0.470] | event | 0.080 (prevalence 0.0017) | 1.172 [0.940, 1.401] (401 วัน) | 2.800 [1.099, 5.100] (20 วัน) | 6 / 170 | 0.013 | τ saturated; deterministic: CI from event bootstrap |
| real_t001:rules+stats | 0.1 | REAL | 5 | 200 | 0.565 ± 0.000 [0.495, 0.630] | 0.240 ± 0.000 [0.185, 0.300] | event | 0.568 (prevalence 0.0017) | 0.107 [0.060, 0.162] (401 วัน) | 0.900 [0.400, 1.450] (20 วัน) | 7 / 45 | 0.250 | deterministic: CI from event bootstrap |
| real_t001:rules+stats | 0.5 | REAL | 5 | 200 | 0.655 ± 0.000 [0.585, 0.720] | 0.320 ± 0.000 [0.260, 0.390] | event | 0.335 (prevalence 0.0017) | 0.369 [0.257, 0.509] (401 วัน) | 2.700 [1.250, 4.551] (20 วัน) | 9 / 60 | 0.250 | deterministic: CI from event bootstrap |
| real_t001:rules+stats | 2 | REAL | 5 | 200 | 0.745 ± 0.000 [0.685, 0.800] | 0.435 ± 0.000 [0.370, 0.505] | event | 0.118 (prevalence 0.0017) | 1.516 [1.229, 1.840] (401 วัน) | 5.550 [3.350, 8.300] (20 วัน) | 5 / 60 | 0.250 | deterministic: CI from event bootstrap |
| real_t001:rules+stats | 10 | REAL | 5 | 200 | 0.820 ± 0.000 [0.765, 0.870] | 0.470 ± 0.000 [0.400, 0.540] | event | 0.034 (prevalence 0.0017) | 6.878 [6.062, 7.711] (401 วัน) | 20.150 [14.649, 26.250] (20 วัน) | 1 / 50 | 0.250 | deterministic: CI from event bootstrap |
| real_t001:stats | 0.1 | REAL | 5 | 200 | 0.545 ± 0.000 [0.480, 0.610] | 0.000 ± 0.000 [0.000, 0.000] | event | 0.605 (prevalence 0.0017) | 0.097 [0.052, 0.152] (401 วัน) | 0.550 [0.100, 1.100] (20 วัน) | 7 / 40 | 0.254 | deterministic: CI from event bootstrap; recall_correct 0 จาก 200 เหตุการณ์: ช่วงจริงอาจสูงถึงราว 2% (rule of three) |
| real_t001:stats | 0.5 | REAL | 5 | 200 | 0.615 ± 0.000 [0.545, 0.675] | 0.000 ± 0.000 [0.000, 0.000] | event | 0.362 (prevalence 0.0017) | 0.342 [0.224, 0.484] (401 วัน) | 2.350 [0.750, 4.300] (20 วัน) | 8 / 40 | 0.254 | deterministic: CI from event bootstrap; recall_correct 0 จาก 200 เหตุการณ์: ช่วงจริงอาจสูงถึงราว 2% (rule of three) |
| real_t001:stats | 2 | REAL | 5 | 200 | 0.650 ± 0.000 [0.580, 0.715] | 0.000 ± 0.000 [0.000, 0.000] | event | 0.205 (prevalence 0.0017) | 0.773 [0.556, 1.052] (401 วัน) | 3.700 [1.550, 6.351] (20 วัน) | 7 / 31 | 0.254 | deterministic: CI from event bootstrap; recall_correct 0 จาก 200 เหตุการณ์: ช่วงจริงอาจสูงถึงราว 2% (rule of three) |
| real_t001:stats | 10 | REAL | 5 | 200 | 0.770 ± 0.000 [0.710, 0.825] | 0.000 ± 0.000 [0.000, 0.000] | event | 0.035 (prevalence 0.0017) | 6.429 [5.608, 7.270] (401 วัน) | 19.550 [12.800, 26.901] (20 วัน) | 5 / 36 | 0.254 | deterministic: CI from event bootstrap; recall_correct 0 จาก 200 เหตุการณ์: ช่วงจริงอาจสูงถึงราว 2% (rule of three) |
| semisyn:ai | 0.1 | SEMI-SYN | 5 | 1188 | 0.374 ± 0.010 [0.367, 0.381] | – | symbol-day | 0.790 (prevalence 0.0334) | 0.100 [0.069, 0.137] (1188 วัน) | – | 18 / 154 | 0.383 | technique-agnostic: recall_correct n/a |
| semisyn:ai | 0.5 | SEMI-SYN | 5 | 1188 | 0.411 ± 0.018 [0.402, 0.422] | – | symbol-day | 0.551 (prevalence 0.0334) | 0.340 [0.230, 0.483] (1188 วัน) | – | 16 / 274 | 0.383 | technique-agnostic: recall_correct n/a |
| semisyn:ai | 2 | SEMI-SYN | 5 | 1188 | 0.519 ± 0.080 [0.505, 0.535] | – | symbol-day | 0.321 (prevalence 0.0334) | 1.135 [0.841, 1.499] (1188 วัน) | – | 19 / 4945 | 0.383 | technique-agnostic: recall_correct n/a |
| semisyn:ai | 10 | SEMI-SYN | 5 | 1188 | 0.888 ± 0.026 [0.875, 0.901] | – | symbol-day | 0.099 (prevalence 0.0334) | 11.342 [10.084, 12.734] (1188 วัน) | – | 31 / 5736 | 0.383 | technique-agnostic: recall_correct n/a |
| semisyn:all | 0.1 | SEMI-SYN | 5 | 1188 | 0.371 ± 0.009 [0.365, 0.377] | 0.009 ± 0.003 [0.006, 0.011] | symbol-day | 0.756 (prevalence 0.0334) | 0.120 [0.082, 0.166] (1188 วัน) | – | 20 / 141 | 0.748 |  |
| semisyn:all | 0.5 | SEMI-SYN | 5 | 1188 | 0.406 ± 0.010 [0.397, 0.415] | 0.022 ± 0.005 [0.018, 0.027] | symbol-day | 0.558 (prevalence 0.0334) | 0.325 [0.232, 0.443] (1188 วัน) | – | 17 / 226 | 0.748 |  |
| semisyn:all | 2 | SEMI-SYN | 5 | 1188 | 0.552 ± 0.078 [0.539, 0.567] | 0.095 ± 0.019 [0.087, 0.104] | symbol-day | 0.316 (prevalence 0.0334) | 1.256 [0.980, 1.620] (1188 วัน) | – | 23 / 4848 | 0.748 |  |
| semisyn:all | 10 | SEMI-SYN | 5 | 1188 | 0.923 ± 0.008 [0.916, 0.931] | 0.363 ± 0.012 [0.350, 0.376] | symbol-day | 0.155 (prevalence 0.0334) | 5.525 [4.911, 6.225] (1188 วัน) | – | 68 / 621 | 0.748 |  |
| semisyn:rules | 0.1 | SEMI-SYN | 5 | 1188 | 0.046 ± 0.006 [0.039, 0.054] | 0.045 ± 0.006 [0.038, 0.053] | symbol-day | 0.222 (prevalence 0.0334) | 0.199 [0.134, 0.263] (1188 วัน) | – | 1373 / 5104 | 0.053 |  |
| semisyn:rules | 0.5 | SEMI-SYN | 5 | 1188 | 0.334 ± 0.014 [0.321, 0.349] | 0.330 ± 0.014 [0.317, 0.343] | symbol-day | 0.331 (prevalence 0.0334) | 0.892 [0.698, 1.086] (1188 วัน) | – | 135 / 4036 | 0.053 |  |
| semisyn:rules | 2 | SEMI-SYN | 5 | 1188 | 0.396 ± 0.009 [0.382, 0.409] | 0.390 ± 0.009 [0.378, 0.403] | symbol-day | 0.330 (prevalence 0.0334) | 1.109 [0.895, 1.326] (1188 วัน) | – | 131 / 3965 | 0.053 | τ saturated |
| semisyn:rules | 10 | SEMI-SYN | 5 | 1188 | 0.396 ± 0.009 [0.382, 0.409] | 0.390 ± 0.009 [0.378, 0.403] | symbol-day | 0.330 (prevalence 0.0334) | 1.109 [0.895, 1.326] (1188 วัน) | – | 131 / 3965 | 0.053 | τ saturated |
| semisyn:rules+stats | 0.1 | SEMI-SYN | 5 | 1188 | 0.017 ± 0.003 [0.012, 0.022] | 0.008 ± 0.002 [0.005, 0.010] | symbol-day | 0.158 (prevalence 0.0334) | 0.090 [0.055, 0.134] (1188 วัน) | – | 175 / 3044 | 0.726 |  |
| semisyn:rules+stats | 0.5 | SEMI-SYN | 5 | 1188 | 0.111 ± 0.004 [0.098, 0.124] | 0.032 ± 0.004 [0.027, 0.038] | symbol-day | 0.282 (prevalence 0.0334) | 0.293 [0.212, 0.389] (1188 วัน) | – | 303 / 1688 | 0.726 |  |
| semisyn:rules+stats | 2 | SEMI-SYN | 5 | 1188 | 0.710 ± 0.006 [0.693, 0.726] | 0.354 ± 0.014 [0.341, 0.367] | symbol-day | 0.358 (prevalence 0.0334) | 1.325 [1.089, 1.598] (1188 วัน) | – | 471 / 847 | 0.726 |  |
| semisyn:rules+stats | 10 | SEMI-SYN | 5 | 1188 | 0.927 ± 0.002 [0.918, 0.936] | 0.390 ± 0.009 [0.378, 0.403] | symbol-day | 0.145 (prevalence 0.0334) | 5.660 [5.030, 6.296] (1188 วัน) | – | 102 / 631 | 0.726 |  |
| semisyn:stats | 0.1 | SEMI-SYN | 5 | 1188 | 0.016 ± 0.002 [0.011, 0.021] | 0.000 ± 0.000 [0.000, 0.000] | symbol-day | 0.165 (prevalence 0.0334) | 0.080 [0.047, 0.124] (1188 วัน) | – | 126 / 1259 | 0.805 | recall_correct 0 จาก 401 symbol-day: ช่วงจริงอาจสูงถึงราว 1% (rule of three) |
| semisyn:stats | 0.5 | SEMI-SYN | 5 | 1188 | 0.112 ± 0.005 [0.099, 0.125] | 0.000 ± 0.000 [0.000, 0.000] | symbol-day | 0.306 (prevalence 0.0334) | 0.259 [0.179, 0.359] (1188 วัน) | – | 276 / 906 | 0.805 | recall_correct 0 จาก 401 symbol-day: ช่วงจริงอาจสูงถึงราว 1% (rule of three) |
| semisyn:stats | 2 | SEMI-SYN | 5 | 1188 | 0.542 ± 0.006 [0.526, 0.557] | 0.000 ± 0.000 [0.000, 0.000] | symbol-day | 0.477 (prevalence 0.0334) | 0.619 [0.453, 0.819] (1188 วัน) | – | 513 / 936 | 0.805 | recall_correct 0 จาก 401 symbol-day: ช่วงจริงอาจสูงถึงราว 1% (rule of three) |
| semisyn:stats | 10 | SEMI-SYN | 5 | 1188 | 0.835 ± 0.004 [0.822, 0.848] | 0.000 ± 0.000 [0.000, 0.000] | symbol-day | 0.142 (prevalence 0.0334) | 5.161 [4.522, 5.778] (1188 วัน) | – | 91 / 642 | 0.805 | recall_correct 0 จาก 401 symbol-day: ช่วงจริงอาจสูงถึงราว 1% (rule of three) |

### ablation: all เทียบชุดย่อย (paired bootstrap, budget 2)

| เทียบ | provenance | seeds | events | bootstrap unit (กลุ่ม) | ผลต่าง recall_any (all − X) [95 % CI] | อ่านว่า |
|---|---|---|---|---|---|---|
| all vs real_t001:rules+stats | REAL | 5 | 200 | event (200) | -0.006 [-0.039, 0.028] | ไม่มีนัยสำคัญ |
| all vs real_t001:rules | REAL | 5 | 200 | event (200) | 0.334 [0.261, 0.414] | all ดีกว่า rules อย่างมีนัย |
| all vs real_t001:stats | REAL | 5 | 200 | event (200) | 0.089 [0.054, 0.127] | all ดีกว่า stats อย่างมีนัย |
| all vs real_t001:ai | REAL | 5 | 200 | event (200) | 0.089 [0.054, 0.130] | all ดีกว่า ai อย่างมีนัย |
| all vs semisyn:rules+stats | SEMI-SYN | 5 | 1188 | symbol-day (401) | -0.157 [-0.177, -0.138] | rules+stats ดีกว่า all อย่างมีนัย |
| all vs semisyn:rules | SEMI-SYN | 5 | 1188 | symbol-day (401) | 0.157 [0.137, 0.177] | all ดีกว่า rules อย่างมีนัย |
| all vs semisyn:stats | SEMI-SYN | 5 | 1188 | symbol-day (401) | 0.010 [-0.005, 0.028] | ไม่มีนัยสำคัญ |
| all vs semisyn:ai | SEMI-SYN | 5 | 1188 | symbol-day (401) | 0.033 [0.024, 0.042] | all ดีกว่า ai อย่างมีนัย |

- ผลต่าง > 0 แปลว่าชุดเต็ม (all) ตรวจพบมากกว่า; ผลต่าง < 0 แปลว่าชุดย่อยตรวจพบมากกว่า
- เกณฑ์ (ทั้งสองทิศ): CI ล่างของผลต่าง > 0 และผลต่าง ≥ 0.02 (ห้ามใช้ CI ไม่ทับกันแทน)
- REAL: สุ่มซ้ำทีละเหตุการณ์ด้วยดัชนีเดียวกันทั้งสองฝั่ง; SEMI-SYN: จับคู่ (seed, เหตุการณ์) ของแต่ละ seed แล้วสุ่มซ้ำทีละ symbol-day ด้วยดัชนีเดียวกันทั้งสองฝั่ง (ตัวประมาณแบบอัตราส่วนเดียวกับตาราง)

## baselines

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day [95% CI] | FA/symbol-day hard [95% CI] | TTD median / p90 (s) | PR-AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ai.autoencoder_trade | 0.1 | REAL | 5 | 200 | 0.505 ± 0.047 [0.445, 0.566] | – | event | 0.503 (prevalence 0.0017) | 0.110 | 0.660 | 40 / 127 | 0.154 | technique-agnostic: recall_correct n/a |
| ai.autoencoder_trade | 0.5 | REAL | 5 | 200 | 0.598 ± 0.033 [0.536, 0.661] | – | event | 0.275 (prevalence 0.0017) | 0.429 | 3.160 | 34 / 94 | 0.154 | technique-agnostic: recall_correct n/a |
| ai.autoencoder_trade | 2 | REAL | 5 | 200 | 0.650 ± 0.023 [0.588, 0.710] | – | event | 0.111 (prevalence 0.0017) | 1.556 | 11.300 | 21 / 83 | 0.154 | technique-agnostic: recall_correct n/a |
| ai.autoencoder_trade | 10 | REAL | 5 | 200 | 0.820 ± 0.014 [0.772, 0.866] | – | event | 0.020 (prevalence 0.0017) | 12.396 | 26.230 | -1 / 39 | 0.154 | technique-agnostic: recall_correct n/a |
| baseline.iforest | 0.1 | REAL | 5 | 200 | 0.601 ± 0.022 [0.538, 0.657] | – | event | 0.480 (prevalence 0.0017) | 0.135 | 0.050 | 10 / 63 | 0.253 | technique-agnostic: recall_correct n/a |
| baseline.iforest | 0.5 | REAL | 5 | 200 | 0.753 ± 0.009 [0.693, 0.808] | – | event | 0.232 (prevalence 0.0017) | 0.624 | 0.130 | 6 / 30 | 0.253 | technique-agnostic: recall_correct n/a |
| baseline.iforest | 2 | REAL | 5 | 200 | 0.808 ± 0.004 [0.752, 0.860] | – | event | 0.097 (prevalence 0.0017) | 2.066 | 0.650 | 0 / 20 | 0.253 | technique-agnostic: recall_correct n/a |
| baseline.iforest | 10 | REAL | 5 | 200 | 0.862 ± 0.003 [0.812, 0.907] | – | event | 0.026 (prevalence 0.0017) | 10.023 | 5.290 | -80 / 12 | 0.253 | technique-agnostic: recall_correct n/a |
| baseline.kk | 0.1 | REAL | 5 | 200 | 0.705 ± 0.000 [0.640, 0.770] | – | event | 0.391 (prevalence 0.0017) | 0.212 | 0.500 | 5 / 25 | 0.074 | deterministic: CI from event bootstrap; technique-agnostic: recall_correct n/a |
| baseline.kk | 0.5 | REAL | 5 | 200 | 0.820 ± 0.000 [0.770, 0.870] | – | event | 0.231 (prevalence 0.0017) | 0.601 | 1.150 | -7 / 20 | 0.074 | τ saturated; deterministic: CI from event bootstrap; technique-agnostic: recall_correct n/a |
| baseline.kk | 2 | REAL | 5 | 200 | 0.820 ± 0.000 [0.770, 0.870] | – | event | 0.231 (prevalence 0.0017) | 0.601 | 1.150 | -7 / 20 | 0.074 | τ saturated; deterministic: CI from event bootstrap; technique-agnostic: recall_correct n/a |
| baseline.kk | 10 | REAL | 5 | 200 | 0.820 ± 0.000 [0.770, 0.870] | – | event | 0.231 (prevalence 0.0017) | 0.601 | 1.150 | -7 / 20 | 0.074 | τ saturated; deterministic: CI from event bootstrap; technique-agnostic: recall_correct n/a |
| baseline.pca | 0.1 | REAL | 5 | 200 | 0.275 ± 0.000 [0.220, 0.335] | – | event | 0.556 (prevalence 0.0017) | 0.082 | 0.500 | 28 / 73 | 0.197 | deterministic: CI from event bootstrap; technique-agnostic: recall_correct n/a |
| baseline.pca | 0.5 | REAL | 5 | 200 | 0.350 ± 0.000 [0.290, 0.415] | – | event | 0.297 (prevalence 0.0017) | 0.287 | 2.450 | 22 / 90 | 0.197 | deterministic: CI from event bootstrap; technique-agnostic: recall_correct n/a |
| baseline.pca | 2 | REAL | 5 | 200 | 0.575 ± 0.000 [0.510, 0.640] | – | event | 0.114 (prevalence 0.0017) | 1.377 | 11.950 | 10 / 86 | 0.197 | deterministic: CI from event bootstrap; technique-agnostic: recall_correct n/a |
| baseline.pca | 10 | REAL | 5 | 200 | 0.700 ± 0.000 [0.640, 0.760] | – | event | 0.031 (prevalence 0.0017) | 6.387 | 25.650 | 7 / 61 | 0.197 | deterministic: CI from event bootstrap; technique-agnostic: recall_correct n/a |
| baseline.random | 0.1 | REAL | 5 | 200 | 0.002 ± 0.003 [0.000, 0.005] | – | event | 0.006 (prevalence 0.0017) | 0.112 | 0.110 | 92 / 92 | 0.002 | technique-agnostic: recall_correct n/a |
| baseline.random | 0.5 | REAL | 5 | 200 | 0.003 ± 0.004 [0.000, 0.007] | – | event | 0.002 (prevalence 0.0017) | 0.538 | 0.500 | 186 / 260 | 0.002 | technique-agnostic: recall_correct n/a |
| baseline.random | 2 | REAL | 5 | 200 | 0.011 ± 0.007 [0.005, 0.017] | – | event | 0.002 (prevalence 0.0017) | 2.157 | 1.990 | 137 / 250 | 0.002 | technique-agnostic: recall_correct n/a |
| baseline.random | 10 | REAL | 5 | 200 | 0.049 ± 0.031 [0.036, 0.062] | – | event | 0.002 (prevalence 0.0017) | 10.211 | 9.880 | 99 / 261 | 0.002 | technique-agnostic: recall_correct n/a |
| stat.robust_z | 0.1 | REAL | 5 | 200 | 0.545 ± 0.000 [0.480, 0.610] | – | event | 0.605 (prevalence 0.0017) | 0.097 | 0.550 | 7 / 40 | 0.269 | deterministic: CI from event bootstrap; technique-agnostic: recall_correct n/a |
| stat.robust_z | 0.5 | REAL | 5 | 200 | 0.615 ± 0.000 [0.545, 0.675] | – | event | 0.362 (prevalence 0.0017) | 0.342 | 2.350 | 8 / 40 | 0.269 | deterministic: CI from event bootstrap; technique-agnostic: recall_correct n/a |
| stat.robust_z | 2 | REAL | 5 | 200 | 0.645 ± 0.000 [0.575, 0.710] | – | event | 0.263 (prevalence 0.0017) | 0.574 | 3.350 | 7 / 32 | 0.269 | deterministic: CI from event bootstrap; technique-agnostic: recall_correct n/a |
| stat.robust_z | 10 | REAL | 5 | 200 | 0.765 ± 0.000 [0.705, 0.820] | – | event | 0.035 (prevalence 0.0017) | 6.426 | 19.650 | 5 / 36 | 0.269 | deterministic: CI from event bootstrap; technique-agnostic: recall_correct n/a |

## ref_supervised

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day [95% CI] | FA/symbol-day hard [95% CI] | TTD median / p90 (s) | PR-AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref.supervised_rf | 0.1 | REAL | 5 | 200 | 0.631 ± 0.011 [0.563, 0.693] | – | event | 0.653 (prevalence 0.0017) | 0.079 | 0.080 | 13 / 81 | 0.354 | supervised reference (uses labels); technique-agnostic: recall_correct n/a |
| ref.supervised_rf | 0.5 | REAL | 5 | 200 | 0.801 ± 0.002 [0.745, 0.853] | – | event | 0.306 (prevalence 0.0017) | 0.514 | 0.370 | 8 / 35 | 0.354 | supervised reference (uses labels); technique-agnostic: recall_correct n/a |
| ref.supervised_rf | 2 | REAL | 5 | 200 | 0.851 ± 0.004 [0.802, 0.896] | – | event | 0.101 (prevalence 0.0017) | 2.121 | 1.610 | -8 / 19 | 0.354 | supervised reference (uses labels); technique-agnostic: recall_correct n/a |
| ref.supervised_rf | 10 | REAL | 5 | 200 | 0.916 ± 0.002 [0.880, 0.955] | – | event | 0.028 (prevalence 0.0017) | 9.592 | 9.880 | -335 / 10 | 0.354 | supervised reference (uses labels); technique-agnostic: recall_correct n/a |

## retrain_drift

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day [95% CI] | FA/symbol-day hard [95% CI] | TTD median / p90 (s) | PR-AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | 0.1 | REAL | 5 | 94 | 0.594 ± 0.012 [0.489, 0.685] | 0.143 ± 0.014 [0.077, 0.211] | event | 0.384 (prevalence 0.0010) | 0.204 | – | 12 / 61 | 0.210 |  |
| v1 | 0.5 | REAL | 5 | 94 | 0.696 ± 0.010 [0.598, 0.781] | 0.221 ± 0.020 [0.138, 0.306] | event | 0.138 (prevalence 0.0010) | 0.924 | – | 10 / 61 | 0.210 |  |
| v1 | 2 | REAL | 5 | 94 | 0.760 ± 0.018 [0.670, 0.840] | 0.313 ± 0.010 [0.223, 0.409] | event | 0.043 (prevalence 0.0010) | 3.578 | – | 10 / 52 | 0.210 |  |
| v1 | 10 | REAL | 5 | 94 | 0.823 ± 0.012 [0.740, 0.900] | 0.377 ± 0.006 [0.283, 0.475] | event | 0.016 (prevalence 0.0010) | 10.557 | – | 0 / 30 | 0.210 |  |
| v1:ai | 0.1 | REAL | 5 | 94 | 0.540 ± 0.026 [0.443, 0.634] | – | event | 0.336 (prevalence 0.0010) | 0.236 | – | 50 / 197 | 0.120 | technique-agnostic: recall_correct n/a |
| v1:ai | 0.5 | REAL | 5 | 94 | 0.660 ± 0.025 [0.560, 0.755] | – | event | 0.119 (prevalence 0.0010) | 1.064 | – | 42 / 163 | 0.120 | technique-agnostic: recall_correct n/a |
| v1:ai | 2 | REAL | 5 | 94 | 0.721 ± 0.036 [0.632, 0.806] | – | event | 0.042 (prevalence 0.0010) | 3.578 | – | 26 / 108 | 0.120 | technique-agnostic: recall_correct n/a |
| v1:ai | 10 | REAL | 5 | 94 | 0.872 ± 0.008 [0.800, 0.934] | – | event | 0.013 (prevalence 0.0010) | 13.148 | – | 0 / 30 | 0.120 | technique-agnostic: recall_correct n/a |
| v2 | 0.1 | REAL | 5 | 94 | 0.572 ± 0.014 [0.470, 0.666] | 0.132 ± 0.010 [0.066, 0.202] | event | 0.394 (prevalence 0.0010) | 0.172 | – | 12 / 61 | 0.208 |  |
| v2 | 0.5 | REAL | 5 | 94 | 0.687 ± 0.012 [0.589, 0.775] | 0.219 ± 0.018 [0.138, 0.304] | event | 0.142 (prevalence 0.0010) | 0.884 | – | 10 / 62 | 0.208 |  |
| v2 | 2 | REAL | 5 | 94 | 0.774 ± 0.009 [0.681, 0.853] | 0.313 ± 0.006 [0.223, 0.409] | event | 0.043 (prevalence 0.0010) | 3.611 | – | 10 / 55 | 0.208 |  |
| v2 | 10 | REAL | 5 | 94 | 0.832 ± 0.014 [0.753, 0.906] | 0.368 ± 0.006 [0.274, 0.466] | event | 0.018 (prevalence 0.0010) | 9.497 | – | 3 / 37 | 0.208 |  |
| v2:ai | 0.1 | REAL | 5 | 94 | 0.483 ± 0.071 [0.389, 0.572] | – | event | 0.312 (prevalence 0.0010) | 0.210 | – | 69 / 191 | 0.106 | technique-agnostic: recall_correct n/a |
| v2:ai | 0.5 | REAL | 5 | 94 | 0.623 ± 0.034 [0.523, 0.717] | – | event | 0.112 (prevalence 0.0010) | 1.054 | – | 46 / 171 | 0.106 | technique-agnostic: recall_correct n/a |
| v2:ai | 2 | REAL | 5 | 94 | 0.728 ± 0.019 [0.634, 0.809] | – | event | 0.040 (prevalence 0.0010) | 3.763 | – | 32 / 117 | 0.106 | technique-agnostic: recall_correct n/a |
| v2:ai | 10 | REAL | 5 | 94 | 0.881 ± 0.014 [0.817, 0.938] | – | event | 0.011 (prevalence 0.0010) | 15.121 | – | 2 / 43 | 0.106 | technique-agnostic: recall_correct n/a |

## m1_fusion

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day [95% CI] | FA/symbol-day hard [95% CI] | TTD median / p90 (s) | PR-AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accept:ai_mondrian | 0.1 | REAL | 5 | 94 | 0.553 ± 0.017 [0.451, 0.645] | 0.121 ± 0.006 [0.057, 0.189] | event | 0.389 (prevalence 0.0010) | 0.178 [0.121, 0.249] (368 วัน) | 0.960 [0.280, 1.870] (20 วัน) | 16 / 61 | 0.205 |  |
| accept:ai_mondrian | 0.5 | REAL | 5 | 94 | 0.662 ± 0.009 [0.562, 0.747] | 0.204 ± 0.009 [0.128, 0.289] | event | 0.140 (prevalence 0.0010) | 0.897 [0.653, 1.181] (368 วัน) | 4.720 [1.890, 7.801] (20 วัน) | 10 / 60 | 0.205 |  |
| accept:ai_mondrian | 2 | REAL | 5 | 94 | 0.743 ± 0.005 [0.651, 0.830] | 0.298 ± 0.008 [0.206, 0.394] | event | 0.041 (prevalence 0.0010) | 3.683 [2.951, 4.486] (368 วัน) | 11.390 [5.630, 17.503] (20 วัน) | 10 / 55 | 0.205 |  |
| accept:ai_mondrian | 10 | REAL | 5 | 94 | 0.819 ± 0.011 [0.734, 0.889] | 0.360 ± 0.005 [0.264, 0.455] | event | 0.019 (prevalence 0.0010) | 8.602 [7.488, 9.806] (368 วัน) | 20.900 [13.370, 29.221] (20 วัน) | 10 / 51 | 0.205 |  |
| accept:ai_shift1 | 0.1 | REAL | 5 | 94 | 0.581 ± 0.010 [0.481, 0.672] | 0.162 ± 0.005 [0.085, 0.238] | event | 0.425 (prevalence 0.0010) | 0.145 [0.091, 0.208] (368 วัน) | 0.890 [0.200, 1.840] (20 วัน) | 10 / 59 | 0.216 |  |
| accept:ai_shift1 | 0.5 | REAL | 5 | 94 | 0.655 ± 0.012 [0.557, 0.743] | 0.226 ± 0.022 [0.143, 0.311] | event | 0.135 (prevalence 0.0010) | 0.853 [0.629, 1.106] (368 วัน) | 3.570 [1.590, 5.851] (20 วัน) | 10 / 60 | 0.216 |  |
| accept:ai_shift1 | 2 | REAL | 5 | 94 | 0.770 ± 0.010 [0.681, 0.853] | 0.347 ± 0.006 [0.251, 0.443] | event | 0.050 (prevalence 0.0010) | 2.934 [2.402, 3.541] (368 วัน) | 10.750 [5.578, 16.201] (20 วัน) | 10 / 57 | 0.216 |  |
| accept:ai_shift1 | 10 | REAL | 5 | 94 | 0.819 ± 0.000 [0.734, 0.894] | 0.383 ± 0.000 [0.287, 0.479] | event | 0.018 (prevalence 0.0010) | 9.207 [8.143, 10.332] (368 วัน) | 21.770 [15.180, 29.482] (20 วัน) | 0 / 30 | 0.216 |  |
| accept:ai_shift2 | 0.1 | REAL | 5 | 94 | 0.574 ± 0.000 [0.479, 0.670] | 0.160 ± 0.000 [0.085, 0.234] | event | 0.431 (prevalence 0.0010) | 0.144 [0.089, 0.210] (368 วัน) | 0.920 [0.210, 1.960] (20 วัน) | 10 / 57 | 0.216 |  |
| accept:ai_shift2 | 0.5 | REAL | 5 | 94 | 0.626 ± 0.005 [0.528, 0.717] | 0.206 ± 0.006 [0.128, 0.291] | event | 0.129 (prevalence 0.0010) | 0.843 [0.615, 1.096] (368 วัน) | 3.340 [1.460, 5.560] (20 วัน) | 10 / 55 | 0.216 |  |
| accept:ai_shift2 | 2 | REAL | 5 | 94 | 0.745 ± 0.008 [0.655, 0.832] | 0.336 ± 0.006 [0.240, 0.436] | event | 0.059 (prevalence 0.0010) | 2.305 [1.887, 2.772] (368 วัน) | 7.620 [4.000, 11.591] (20 วัน) | 10 / 50 | 0.216 |  |
| accept:ai_shift2 | 10 | REAL | 5 | 94 | 0.819 ± 0.000 [0.734, 0.894] | 0.381 ± 0.005 [0.287, 0.479] | event | 0.018 (prevalence 0.0010) | 9.071 [8.024, 10.144] (368 วัน) | 20.650 [14.299, 28.076] (20 วัน) | 0 / 30 | 0.216 |  |
| accept:all | 0.1 | REAL | 5 | 94 | 0.585 ± 0.015 [0.483, 0.681] | 0.132 ± 0.006 [0.068, 0.202] | event | 0.400 (prevalence 0.0010) | 0.173 [0.118, 0.235] (368 วัน) | 0.910 [0.300, 1.730] (20 วัน) | 16 / 65 | 0.210 |  |
| accept:all | 0.5 | REAL | 5 | 94 | 0.681 ± 0.017 [0.583, 0.768] | 0.211 ± 0.009 [0.132, 0.294] | event | 0.141 (prevalence 0.0010) | 0.911 [0.637, 1.213] (368 วัน) | 5.690 [2.360, 9.382] (20 วัน) | 10 / 60 | 0.210 |  |
| accept:all | 2 | REAL | 5 | 94 | 0.755 ± 0.000 [0.664, 0.840] | 0.302 ± 0.010 [0.211, 0.398] | event | 0.042 (prevalence 0.0010) | 3.657 [2.888, 4.497] (368 วัน) | 12.470 [6.380, 18.972] (20 วัน) | 10 / 58 | 0.210 |  |
| accept:all | 10 | REAL | 5 | 94 | 0.830 ± 0.008 [0.749, 0.904] | 0.370 ± 0.005 [0.274, 0.468] | event | 0.018 (prevalence 0.0010) | 9.272 [8.149, 10.508] (368 วัน) | 22.760 [15.290, 31.117] (20 วัน) | 5 / 39 | 0.210 |  |
| accept:no_ai | 0.1 | REAL | 5 | 94 | 0.574 ± 0.000 [0.479, 0.670] | 0.160 ± 0.000 [0.085, 0.234] | event | 0.439 (prevalence 0.0010) | 0.141 [0.087, 0.207] (368 วัน) | 0.900 [0.200, 1.950] (20 วัน) | 10 / 57 | 0.215 |  |
| accept:no_ai | 0.5 | REAL | 5 | 94 | 0.617 ± 0.000 [0.511, 0.713] | 0.202 ± 0.000 [0.128, 0.287] | event | 0.128 (prevalence 0.0010) | 0.837 [0.609, 1.090] (368 วัน) | 3.300 [1.450, 5.451] (20 วัน) | 10 / 53 | 0.215 |  |
| accept:no_ai | 2 | REAL | 5 | 94 | 0.734 ± 0.000 [0.638, 0.819] | 0.319 ± 0.000 [0.223, 0.415] | event | 0.059 (prevalence 0.0010) | 2.272 [1.851, 2.736] (368 วัน) | 7.450 [3.900, 11.255] (20 วัน) | 10 / 50 | 0.215 |  |
| accept:no_ai | 10 | REAL | 5 | 94 | 0.819 ± 0.000 [0.734, 0.894] | 0.372 ± 0.000 [0.277, 0.468] | event | 0.018 (prevalence 0.0010) | 9.065 [8.013, 10.136] (368 วัน) | 20.600 [14.250, 28.052] (20 วัน) | 0 / 30 | 0.215 |  |
| semisyn:ai_mondrian | 0.1 | SEMI-SYN | 5 | 1188 | 0.359 ± 0.009 [0.354, 0.364] | 0.006 ± 0.003 [0.004, 0.008] | symbol-day | 0.751 (prevalence 0.0334) | 0.119 [0.079, 0.173] (1188 วัน) | – | 18 / 101 | 0.749 |  |
| semisyn:ai_mondrian | 0.5 | SEMI-SYN | 5 | 1188 | 0.393 ± 0.009 [0.385, 0.402] | 0.017 ± 0.004 [0.014, 0.022] | symbol-day | 0.567 (prevalence 0.0334) | 0.302 [0.222, 0.413] (1188 วัน) | – | 16 / 189 | 0.749 |  |
| semisyn:ai_mondrian | 2 | SEMI-SYN | 5 | 1188 | 0.524 ± 0.044 [0.511, 0.538] | 0.084 ± 0.012 [0.076, 0.093] | symbol-day | 0.294 (prevalence 0.0334) | 1.355 [1.085, 1.693] (1188 วัน) | – | 19 / 4246 | 0.749 |  |
| semisyn:ai_mondrian | 10 | SEMI-SYN | 5 | 1188 | 0.907 ± 0.009 [0.898, 0.916] | 0.357 ± 0.014 [0.344, 0.371] | symbol-day | 0.166 (prevalence 0.0334) | 4.982 [4.351, 5.716] (1188 วัน) | – | 66 / 624 | 0.749 |  |
| semisyn:ai_shift1 | 0.1 | SEMI-SYN | 5 | 1188 | 0.356 ± 0.006 [0.350, 0.361] | 0.009 ± 0.003 [0.006, 0.011] | symbol-day | 0.771 (prevalence 0.0334) | 0.105 [0.068, 0.153] (1188 วัน) | – | 22 / 113 | 0.776 |  |
| semisyn:ai_shift1 | 0.5 | SEMI-SYN | 5 | 1188 | 0.398 ± 0.008 [0.390, 0.406] | 0.033 ± 0.005 [0.028, 0.039] | symbol-day | 0.554 (prevalence 0.0334) | 0.323 [0.239, 0.422] (1188 วัน) | – | 22 / 184 | 0.776 |  |
| semisyn:ai_shift1 | 2 | SEMI-SYN | 5 | 1188 | 0.881 ± 0.007 [0.872, 0.890] | 0.354 ± 0.014 [0.341, 0.367] | symbol-day | 0.374 (prevalence 0.0334) | 1.531 [1.275, 1.832] (1188 วัน) | – | 90 / 670 | 0.776 |  |
| semisyn:ai_shift1 | 10 | SEMI-SYN | 5 | 1188 | 0.937 ± 0.004 [0.929, 0.944] | 0.390 ± 0.009 [0.378, 0.403] | symbol-day | 0.146 (prevalence 0.0334) | 5.688 [5.067, 6.321] (1188 วัน) | – | 61 / 623 | 0.776 |  |
| semisyn:ai_shift2 | 0.1 | SEMI-SYN | 5 | 1188 | 0.346 ± 0.007 [0.341, 0.351] | 0.008 ± 0.002 [0.006, 0.011] | symbol-day | 0.784 (prevalence 0.0334) | 0.096 [0.060, 0.140] (1188 วัน) | – | 28 / 133 | 0.768 |  |
| semisyn:ai_shift2 | 0.5 | SEMI-SYN | 5 | 1188 | 0.379 ± 0.003 [0.373, 0.386] | 0.033 ± 0.004 [0.027, 0.038] | symbol-day | 0.557 (prevalence 0.0334) | 0.304 [0.223, 0.401] (1188 วัน) | – | 28 / 159 | 0.768 |  |
| semisyn:ai_shift2 | 2 | SEMI-SYN | 5 | 1188 | 0.874 ± 0.006 [0.865, 0.882] | 0.354 ± 0.014 [0.341, 0.367] | symbol-day | 0.400 (prevalence 0.0334) | 1.356 [1.112, 1.631] (1188 วัน) | – | 97 / 671 | 0.768 |  |
| semisyn:ai_shift2 | 10 | SEMI-SYN | 5 | 1188 | 0.936 ± 0.003 [0.928, 0.944] | 0.390 ± 0.009 [0.378, 0.403] | symbol-day | 0.146 (prevalence 0.0334) | 5.670 [5.036, 6.299] (1188 วัน) | – | 62 / 620 | 0.768 |  |
| semisyn:all | 0.1 | SEMI-SYN | 5 | 1188 | 0.371 ± 0.009 [0.365, 0.377] | 0.009 ± 0.003 [0.006, 0.011] | symbol-day | 0.756 (prevalence 0.0334) | 0.120 [0.082, 0.166] (1188 วัน) | – | 20 / 141 | 0.748 |  |
| semisyn:all | 0.5 | SEMI-SYN | 5 | 1188 | 0.406 ± 0.010 [0.397, 0.415] | 0.022 ± 0.005 [0.018, 0.027] | symbol-day | 0.558 (prevalence 0.0334) | 0.325 [0.232, 0.443] (1188 วัน) | – | 17 / 226 | 0.748 |  |
| semisyn:all | 2 | SEMI-SYN | 5 | 1188 | 0.552 ± 0.078 [0.539, 0.567] | 0.095 ± 0.019 [0.087, 0.104] | symbol-day | 0.316 (prevalence 0.0334) | 1.256 [0.980, 1.620] (1188 วัน) | – | 23 / 4848 | 0.748 |  |
| semisyn:all | 10 | SEMI-SYN | 5 | 1188 | 0.923 ± 0.008 [0.916, 0.931] | 0.363 ± 0.012 [0.350, 0.376] | symbol-day | 0.155 (prevalence 0.0334) | 5.525 [4.911, 6.225] (1188 วัน) | – | 68 / 621 | 0.748 |  |
| semisyn:no_ai | 0.1 | SEMI-SYN | 5 | 1188 | 0.017 ± 0.003 [0.012, 0.022] | 0.008 ± 0.002 [0.005, 0.010] | symbol-day | 0.158 (prevalence 0.0334) | 0.090 [0.055, 0.134] (1188 วัน) | – | 175 / 3044 | 0.726 |  |
| semisyn:no_ai | 0.5 | SEMI-SYN | 5 | 1188 | 0.111 ± 0.004 [0.098, 0.124] | 0.032 ± 0.004 [0.027, 0.038] | symbol-day | 0.282 (prevalence 0.0334) | 0.293 [0.212, 0.389] (1188 วัน) | – | 303 / 1688 | 0.726 |  |
| semisyn:no_ai | 2 | SEMI-SYN | 5 | 1188 | 0.710 ± 0.006 [0.693, 0.726] | 0.354 ± 0.014 [0.341, 0.367] | symbol-day | 0.358 (prevalence 0.0334) | 1.325 [1.089, 1.598] (1188 วัน) | – | 471 / 847 | 0.726 |  |
| semisyn:no_ai | 10 | SEMI-SYN | 5 | 1188 | 0.927 ± 0.002 [0.918, 0.936] | 0.390 ± 0.009 [0.378, 0.403] | symbol-day | 0.145 (prevalence 0.0334) | 5.660 [5.030, 6.296] (1188 วัน) | – | 102 / 631 | 0.726 |  |
| test:ai_mondrian | 0.1 | REAL | 5 | 200 | 0.509 ± 0.007 [0.440, 0.575] | 0.213 ± 0.007 [0.163, 0.270] | event | 0.509 (prevalence 0.0017) | 0.121 [0.076, 0.182] (401 วัน) | 0.790 [0.350, 1.330] (20 วัน) | 8 / 31 | 0.244 |  |
| test:ai_mondrian | 0.5 | REAL | 5 | 200 | 0.616 ± 0.019 [0.552, 0.682] | 0.278 ± 0.008 [0.221, 0.340] | event | 0.323 (prevalence 0.0017) | 0.364 [0.252, 0.502] (401 วัน) | 2.600 [0.970, 4.813] (20 วัน) | 8 / 67 | 0.244 |  |
| test:ai_mondrian | 2 | REAL | 5 | 200 | 0.739 ± 0.007 [0.679, 0.800] | 0.390 ± 0.004 [0.325, 0.464] | event | 0.114 (prevalence 0.0017) | 1.692 [1.307, 2.160] (401 วัน) | 9.410 [5.270, 14.853] (20 วัน) | 6 / 62 | 0.244 |  |
| test:ai_mondrian | 10 | REAL | 5 | 200 | 0.795 ± 0.008 [0.738, 0.848] | 0.459 ± 0.004 [0.389, 0.534] | event | 0.042 (prevalence 0.0017) | 5.701 [4.948, 6.607] (401 วัน) | 19.780 [13.139, 27.412] (20 วัน) | 2 / 54 | 0.244 |  |
| test:ai_shift1 | 0.1 | REAL | 5 | 200 | 0.563 ± 0.003 [0.493, 0.628] | 0.242 ± 0.003 [0.188, 0.301] | event | 0.523 (prevalence 0.0017) | 0.117 [0.069, 0.172] (401 วัน) | 0.880 [0.400, 1.440] (20 วัน) | 7 / 42 | 0.249 |  |
| test:ai_shift1 | 0.5 | REAL | 5 | 200 | 0.663 ± 0.007 [0.597, 0.726] | 0.328 ± 0.003 [0.266, 0.397] | event | 0.316 (prevalence 0.0017) | 0.395 [0.274, 0.543] (401 วัน) | 2.630 [1.160, 4.480] (20 วัน) | 9 / 60 | 0.249 |  |
| test:ai_shift1 | 2 | REAL | 5 | 200 | 0.765 ± 0.004 [0.706, 0.821] | 0.443 ± 0.003 [0.375, 0.516] | event | 0.106 (prevalence 0.0017) | 1.770 [1.455, 2.162] (401 วัน) | 7.260 [4.348, 11.000] (20 วัน) | 5 / 60 | 0.249 |  |
| test:ai_shift1 | 10 | REAL | 5 | 200 | 0.818 ± 0.004 [0.763, 0.867] | 0.477 ± 0.004 [0.408, 0.549] | event | 0.033 (prevalence 0.0017) | 6.945 [6.136, 7.749] (401 วัน) | 21.500 [15.609, 28.062] (20 วัน) | 0 / 50 | 0.249 |  |
| test:ai_shift2 | 0.1 | REAL | 5 | 200 | 0.566 ± 0.002 [0.499, 0.631] | 0.240 ± 0.000 [0.185, 0.300] | event | 0.545 (prevalence 0.0017) | 0.112 [0.064, 0.167] (401 วัน) | 0.900 [0.400, 1.450] (20 วัน) | 7 / 46 | 0.250 |  |
| test:ai_shift2 | 0.5 | REAL | 5 | 200 | 0.655 ± 0.000 [0.585, 0.720] | 0.321 ± 0.002 [0.261, 0.391] | event | 0.326 (prevalence 0.0017) | 0.379 [0.265, 0.522] (401 วัน) | 2.700 [1.250, 4.551] (20 วัน) | 9 / 60 | 0.250 |  |
| test:ai_shift2 | 2 | REAL | 5 | 200 | 0.751 ± 0.005 [0.691, 0.809] | 0.437 ± 0.003 [0.371, 0.507] | event | 0.117 (prevalence 0.0017) | 1.546 [1.257, 1.880] (401 วัน) | 5.620 [3.360, 8.410] (20 วัน) | 5 / 60 | 0.250 |  |
| test:ai_shift2 | 10 | REAL | 5 | 200 | 0.821 ± 0.002 [0.766, 0.871] | 0.474 ± 0.004 [0.406, 0.543] | event | 0.034 (prevalence 0.0017) | 6.887 [6.072, 7.719] (401 วัน) | 20.140 [14.649, 26.220] (20 วัน) | 1 / 51 | 0.250 |  |
| test:all | 0.1 | REAL | 5 | 200 | 0.540 ± 0.025 [0.473, 0.605] | 0.229 ± 0.011 [0.180, 0.285] | event | 0.500 (prevalence 0.0017) | 0.121 [0.079, 0.170] (401 วัน) | 0.730 [0.340, 1.190] (20 วัน) | 9 / 61 | 0.243 |  |
| test:all | 0.5 | REAL | 5 | 200 | 0.638 ± 0.016 [0.574, 0.699] | 0.301 ± 0.017 [0.241, 0.366] | event | 0.300 (prevalence 0.0017) | 0.400 [0.263, 0.574] (401 วัน) | 3.010 [1.150, 5.601] (20 วัน) | 8 / 72 | 0.243 |  |
| test:all | 2 | REAL | 5 | 200 | 0.739 ± 0.011 [0.680, 0.801] | 0.399 ± 0.008 [0.334, 0.472] | event | 0.113 (prevalence 0.0017) | 1.650 [1.240, 2.166] (401 วัน) | 11.170 [6.460, 17.021] (20 วัน) | 6 / 60 | 0.243 |  |
| test:all | 10 | REAL | 5 | 200 | 0.808 ± 0.004 [0.751, 0.858] | 0.481 ± 0.009 [0.412, 0.555] | event | 0.037 (prevalence 0.0017) | 6.477 [5.670, 7.387] (401 วัน) | 23.160 [16.150, 31.231] (20 วัน) | -0 / 46 | 0.243 |  |
| test:no_ai | 0.1 | REAL | 5 | 200 | 0.565 ± 0.000 [0.495, 0.630] | 0.240 ± 0.000 [0.185, 0.300] | event | 0.568 (prevalence 0.0017) | 0.107 [0.060, 0.162] (401 วัน) | 0.900 [0.400, 1.450] (20 วัน) | 7 / 45 | 0.250 |  |
| test:no_ai | 0.5 | REAL | 5 | 200 | 0.655 ± 0.000 [0.585, 0.720] | 0.320 ± 0.000 [0.260, 0.390] | event | 0.335 (prevalence 0.0017) | 0.369 [0.257, 0.509] (401 วัน) | 2.700 [1.250, 4.551] (20 วัน) | 9 / 60 | 0.250 |  |
| test:no_ai | 2 | REAL | 5 | 200 | 0.745 ± 0.000 [0.685, 0.800] | 0.435 ± 0.000 [0.370, 0.505] | event | 0.118 (prevalence 0.0017) | 1.516 [1.229, 1.840] (401 วัน) | 5.550 [3.350, 8.300] (20 วัน) | 5 / 60 | 0.250 |  |
| test:no_ai | 10 | REAL | 5 | 200 | 0.820 ± 0.000 [0.765, 0.870] | 0.470 ± 0.000 [0.400, 0.540] | event | 0.034 (prevalence 0.0017) | 6.878 [6.062, 7.711] (401 วัน) | 20.150 [14.649, 26.250] (20 วัน) | 1 / 50 | 0.250 |  |

### m1_fusion: ตัวแปรการรวมคะแนนของ AE (budget 2)

#### accept (REAL)

| ตัวแปร | seeds | events | recall_any [95% CI] | recall_correct | FA/symbol-day [95% CI] | FA hard [95% CI] | ผลต่าง recall_any (V0 − X) [95% CI] | อ่านว่า |
|---|---|---|---|---|---|---|---|---|
| V0 all | 5 | 94 | 0.755 [0.664, 0.840] | 0.302 | 3.657 [2.888, 4.497] | 12.470 [6.380, 18.972] | – | ตัวอ้างอิง |
| V1 no_ai | 5 | 94 | 0.734 [0.638, 0.819] | 0.319 | 2.272 [1.851, 2.736] | 7.450 [3.900, 11.255] | 0.021 [-0.034, 0.072] | ไม่ต่างจาก V0 อย่างมีนัย |
| V3 ai_shift1 | 5 | 94 | 0.770 [0.681, 0.853] | 0.347 | 2.934 [2.402, 3.541] | 10.750 [5.578, 16.201] | -0.015 [-0.060, 0.017] | ไม่ต่างจาก V0 อย่างมีนัย |
| V3 ai_shift2 | 5 | 94 | 0.745 [0.655, 0.832] | 0.336 | 2.305 [1.887, 2.772] | 7.620 [4.000, 11.591] | 0.011 [-0.040, 0.055] | ไม่ต่างจาก V0 อย่างมีนัย |
| V2 ai_mondrian | 5 | 94 | 0.743 [0.651, 0.830] | 0.298 | 3.683 [2.951, 4.486] | 11.390 [5.630, 17.503] | 0.013 [0.002, 0.026] | ไม่ต่างจาก V0 อย่างมีนัย |

#### test (REAL)

| ตัวแปร | seeds | events | recall_any [95% CI] | recall_correct | FA/symbol-day [95% CI] | FA hard [95% CI] | ผลต่าง recall_any (V0 − X) [95% CI] | อ่านว่า |
|---|---|---|---|---|---|---|---|---|
| V0 all | 5 | 200 | 0.739 [0.680, 0.801] | 0.399 | 1.650 [1.240, 2.166] | 11.170 [6.460, 17.021] | – | ตัวอ้างอิง |
| V1 no_ai | 5 | 200 | 0.745 [0.685, 0.800] | 0.435 | 1.516 [1.229, 1.840] | 5.550 [3.350, 8.300] | -0.006 [-0.039, 0.028] | ไม่ต่างจาก V0 อย่างมีนัย |
| V3 ai_shift1 | 5 | 200 | 0.765 [0.706, 0.821] | 0.443 | 1.770 [1.455, 2.162] | 7.260 [4.348, 11.000] | -0.026 [-0.054, 0.000] | ไม่ต่างจาก V0 อย่างมีนัย |
| V3 ai_shift2 | 5 | 200 | 0.751 [0.691, 0.809] | 0.437 | 1.546 [1.257, 1.880] | 5.620 [3.360, 8.410] | -0.012 [-0.043, 0.019] | ไม่ต่างจาก V0 อย่างมีนัย |
| V2 ai_mondrian | 5 | 200 | 0.739 [0.679, 0.800] | 0.390 | 1.692 [1.307, 2.160] | 9.410 [5.270, 14.853] | 0.000 [-0.005, 0.004] | ไม่ต่างจาก V0 อย่างมีนัย |

#### semisyn (SEMI-SYN)

| ตัวแปร | seeds | events | recall_any [95% CI] | recall_correct | FA/symbol-day [95% CI] | FA hard [95% CI] | ผลต่าง recall_any (V0 − X) [95% CI] | อ่านว่า |
|---|---|---|---|---|---|---|---|---|
| V0 all | 5 | 1188 | 0.552 [0.539, 0.567] | 0.095 | 1.256 [0.980, 1.620] | – | – | ตัวอ้างอิง |
| V1 no_ai | 5 | 1188 | 0.710 [0.693, 0.726] | 0.354 | 1.325 [1.089, 1.598] | – | -0.157 [-0.177, -0.138] | สูงกว่า V0 อย่างมีนัย |
| V3 ai_shift1 | 5 | 1188 | 0.881 [0.872, 0.890] | 0.354 | 1.531 [1.275, 1.832] | – | -0.329 [-0.342, -0.315] | สูงกว่า V0 อย่างมีนัย |
| V3 ai_shift2 | 5 | 1188 | 0.874 [0.865, 0.882] | 0.354 | 1.356 [1.112, 1.631] | – | -0.321 [-0.336, -0.306] | สูงกว่า V0 อย่างมีนัย |
| V2 ai_mondrian | 5 | 1188 | 0.524 [0.511, 0.538] | 0.084 | 1.355 [1.085, 1.693] | – | 0.028 [0.023, 0.034] | ต่ำกว่า V0 อย่างมีนัย |

- กฎเลือก (กำหนดไว้ก่อนดูผล test และใช้ split accept เท่านั้น): ตัดตัวแปรที่ recall_any บน accept ต่ำกว่า V0 all อย่างมีนัย (paired bootstrap ข้ามเหตุการณ์, CI ล่างของ (V0 − X) > 0 และผลต่าง ≥ min_effect) และตัวแปรที่ตัดสิน recall เทียบ V0 ไม่ได้ (เหตุการณ์น้อยเกินไป) แล้วเลือกตัวที่ FA ของวันผันผวนสูงของ accept (accept-hard) ต่ำสุดที่งบ 2; ถ้าเท่ากันเลือกตัวที่ง่ายกว่า: all < no_ai < ai_shift1 < ai_shift2 < ai_mondrian (min_effect = 0.02)
- **ผลการเลือก (accept):** V1 no_ai
- ผล test และ semisyn รายงานไว้ดูเท่านั้น ไม่ได้ใช้เลือกตัวแปร
- ทุกตัวแปรใช้คะแนนของตัวตรวจจับชุดเดียวกัน (fit และ score ครั้งเดียวต่อ seed) ต่างกันเฉพาะการรวมคะแนน และ τ ที่ fit ใหม่บนข้อมูลสอบเทียบของแต่ละตัวแปร
- no_ai: AE ยังถูกคำนวณและแสดง p แต่ไม่เข้า F, agreement และเทคนิค; ai_shiftδ: AE ต้องหายากขึ้น (ส่วนของ AE ใน F = max(0, −log10 p − δ)); ai_mondrian: p ของ AE เทียบกับคะแนนสอบเทียบในช่วงเดียวกันของช่วงราคาขาขึ้นในหน้าต่าง 1800 วินาที (jump_1800) (3 ช่วงจาก tercile บนข้อมูลสอบเทียบ); jump_1800 ไม่ใช่ความผันผวนสัมบูรณ์ และแถวที่มีการปั่นราคาขึ้นมีค่า jump สูงโดยธรรมชาติ ai_mondrian จึงอาจลด recall ของ AE ซึ่งเกณฑ์ recall ของกฎเลือกจะคัดออกเอง
- F คือ −log10 ของ anomaly p-value (ความหายากเทียบตลาดปกติ) ไม่ใช่ความน่าจะเป็นว่ามีการปั่นราคา
- accept-hard = วัน accept-normal ที่ผันผวนสูงสุดตามเกณฑ์เดียวกับ hard negative ของ test (realized volatility 1 นาที) และยังนับรวมใน FA ของ accept ด้วย

## m4_sell

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day [95% CI] | FA/symbol-day hard [95% CI] | TTD median / p90 (s) | PR-AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bear_raid:base | 0.1 | SEMI-SYN | 5 | 395 | 0.024 ± 0.013 [0.016, 0.034] | 0.000 ± 0.000 [0.000, 0.000] | symbol-day | 0.165 (prevalence 0.0061) | 0.121 [0.082, 0.167] (395 วัน) | – | 152 / 208 | 0.052 | recall_correct 0 จาก 397 symbol-day: ช่วงจริงอาจสูงถึงราว 1% (rule of three) |
| bear_raid:base | 0.5 | SEMI-SYN | 5 | 395 | 0.068 ± 0.038 [0.053, 0.082] | 0.000 ± 0.000 [0.000, 0.000] | symbol-day | 0.150 (prevalence 0.0061) | 0.369 [0.252, 0.528] (395 วัน) | – | 134 / 245 | 0.052 | recall_correct 0 จาก 397 symbol-day: ช่วงจริงอาจสูงถึงราว 1% (rule of three) |
| bear_raid:base | 2 | SEMI-SYN | 5 | 395 | 0.172 ± 0.059 [0.147, 0.196] | 0.000 ± 0.000 [0.000, 0.000] | symbol-day | 0.099 (prevalence 0.0061) | 1.575 [1.202, 2.129] (395 วัน) | – | 142 / 325 | 0.052 | recall_correct 0 จาก 397 symbol-day: ช่วงจริงอาจสูงถึงราว 1% (rule of three) |
| bear_raid:base | 10 | SEMI-SYN | 5 | 395 | 0.550 ± 0.040 [0.518, 0.584] | 0.000 ± 0.000 [0.000, 0.000] | symbol-day | 0.083 (prevalence 0.0061) | 6.319 [5.649, 7.295] (395 วัน) | – | 56 / 210 | 0.052 | recall_correct 0 จาก 397 symbol-day: ช่วงจริงอาจสูงถึงราว 1% (rule of three) |
| bear_raid:sell | 0.1 | SEMI-SYN | 5 | 395 | 0.019 ± 0.010 [0.012, 0.027] | 0.002 ± 0.002 [0.000, 0.004] | symbol-day | 0.158 (prevalence 0.0061) | 0.103 [0.069, 0.145] (395 วัน) | – | 143 / 212 | 0.052 |  |
| bear_raid:sell | 0.5 | SEMI-SYN | 5 | 395 | 0.063 ± 0.032 [0.049, 0.077] | 0.006 ± 0.005 [0.002, 0.010] | symbol-day | 0.143 (prevalence 0.0061) | 0.361 [0.257, 0.502] (395 วัน) | – | 133 / 233 | 0.052 |  |
| bear_raid:sell | 2 | SEMI-SYN | 5 | 395 | 0.171 ± 0.057 [0.145, 0.194] | 0.006 ± 0.005 [0.002, 0.010] | symbol-day | 0.099 (prevalence 0.0061) | 1.565 [1.210, 2.118] (395 วัน) | – | 141 / 324 | 0.052 |  |
| bear_raid:sell | 10 | SEMI-SYN | 5 | 395 | 0.549 ± 0.040 [0.518, 0.583] | 0.006 ± 0.005 [0.002, 0.010] | symbol-day | 0.083 (prevalence 0.0061) | 6.317 [5.639, 7.297] (395 วัน) | – | 56 / 210 | 0.052 |  |
| test:base | 0.1 | REAL | 5 | 200 | 0.540 ± 0.025 [0.473, 0.605] | 0.229 ± 0.011 [0.180, 0.285] | event | 0.500 (prevalence 0.0017) | 0.121 [0.079, 0.170] (401 วัน) | 0.730 [0.340, 1.190] (20 วัน) | 9 / 61 | 0.243 |  |
| test:base | 0.5 | REAL | 5 | 200 | 0.638 ± 0.016 [0.574, 0.699] | 0.301 ± 0.017 [0.241, 0.366] | event | 0.300 (prevalence 0.0017) | 0.400 [0.263, 0.574] (401 วัน) | 3.010 [1.150, 5.601] (20 วัน) | 8 / 72 | 0.243 |  |
| test:base | 2 | REAL | 5 | 200 | 0.739 ± 0.011 [0.680, 0.801] | 0.399 ± 0.008 [0.334, 0.472] | event | 0.113 (prevalence 0.0017) | 1.650 [1.240, 2.166] (401 วัน) | 11.170 [6.460, 17.021] (20 วัน) | 6 / 60 | 0.243 |  |
| test:base | 10 | REAL | 5 | 200 | 0.808 ± 0.004 [0.751, 0.858] | 0.481 ± 0.009 [0.412, 0.555] | event | 0.037 (prevalence 0.0017) | 6.477 [5.670, 7.387] (401 วัน) | 23.160 [16.150, 31.231] (20 วัน) | -0 / 46 | 0.243 |  |
| test:sell | 0.1 | REAL | 5 | 200 | 0.556 ± 0.013 [0.490, 0.617] | 0.468 ± 0.014 [0.404, 0.531] | event | 0.520 (prevalence 0.0017) | 0.105 [0.068, 0.150] (401 วัน) | 0.530 [0.250, 0.870] (20 วัน) | 10 / 87 | 0.252 |  |
| test:sell | 0.5 | REAL | 5 | 200 | 0.681 ± 0.008 [0.619, 0.743] | 0.585 ± 0.008 [0.516, 0.656] | event | 0.310 (prevalence 0.0017) | 0.389 [0.266, 0.547] (401 วัน) | 2.800 [1.090, 5.231] (20 วัน) | 9 / 77 | 0.252 |  |
| test:sell | 2 | REAL | 5 | 200 | 0.751 ± 0.009 [0.691, 0.810] | 0.641 ± 0.004 [0.573, 0.706] | event | 0.115 (prevalence 0.0017) | 1.636 [1.232, 2.140] (401 วัน) | 10.940 [6.280, 16.741] (20 วัน) | 6 / 60 | 0.252 |  |
| test:sell | 10 | REAL | 5 | 200 | 0.807 ± 0.003 [0.749, 0.858] | 0.697 ± 0.006 [0.630, 0.759] | event | 0.037 (prevalence 0.0017) | 6.475 [5.672, 7.387] (401 วัน) | 23.140 [16.140, 31.221] (20 วัน) | -0 / 45 | 0.252 |  |

### m4_sell: ตรวจช่วงกำลังพุ่งและช่วงกำลังเทขาย (budget 2)

| ตัวแปร | seeds | events (REAL) | recall ช่วงพุ่ง [95% CI] | recall ช่วงเทขาย [95% CI] | ระบุช่วงเทขายถูก [95% CI] | FA/symbol-day [95% CI] | FA hard [95% CI] | bear raid recall_any [95% CI] (SEMI-SYN) | bear raid recall_correct [95% CI] (SEMI-SYN) | สถานะการวัดช่วงเทขาย |
|---|---|---|---|---|---|---|---|---|---|---|
| base (ไม่มีกฎฝั่งขาย) | 5 | 200 | 0.739 [0.680, 0.801] | 0.805 [0.727, 0.878] | 0.000 [0.000, 0.000] | 1.650 [1.240, 2.166] | 11.170 [6.460, 17.021] | 0.172 [0.147, 0.196] | 0.000 [0.000, 0.000] | late_peak 18, measured 147, no_rise 35 |
| sell (มีกฎฝั่งขาย) | 5 | 200 | 0.751 [0.691, 0.810] | 0.833 [0.758, 0.904] | 0.544 [0.447, 0.642] | 1.636 [1.232, 2.140] | 10.940 [6.280, 16.741] | 0.171 [0.145, 0.194] | 0.006 [0.002, 0.010] | late_peak 18, measured 147, no_rise 35 |

- ผลต่าง recall ช่วงพุ่ง (sell − base): 0.012 [0.000, 0.029] (200 event, paired bootstrap) → recall ช่วงพุ่งไม่ลดลงอย่างมีนัย (min_effect = 0.02)
- ผลต่าง recall ช่วงเทขาย (sell − base): 0.028 [-0.002, 0.061] (85 event, paired bootstrap) → recall ช่วงเทขายไม่ต่างกันอย่างมีนัย (min_effect = 0.02)
- ผลต่าง ระบุช่วงเทขายถูก (sell − base): 0.544 [0.447, 0.642] (85 event, paired bootstrap) → ระบุช่วงเทขายถูกสูงขึ้นอย่างมีนัย (min_effect = 0.02)
- base (ไม่มีกฎฝั่งขาย): เหตุการณ์ที่เทขายจบภายใน 60 วินาที 62 รายการ ไม่นับในตัวหารของ recall ช่วงเทขาย (และผลต่าง sell − base)
- sell (มีกฎฝั่งขาย): เหตุการณ์ที่เทขายจบภายใน 60 วินาที 62 รายการ ไม่นับในตัวหารของ recall ช่วงเทขาย (และผลต่าง sell − base)
- ช่วงพุ่ง (pump phase) = recall_any บนหน้าต่างรอบเวลาประกาศของเหตุการณ์ (เหมือน real_t001)
- ช่วงเทขาย (sell phase) = มีตอนแจ้งเตือนใดก็ได้ที่มีแถวเกินเกณฑ์ในช่วง (จุดสูงสุด + 60 วินาที, จุดสูงสุด + dump_tau] ที่วัดจากราคาจริง (dump_tau อย่างน้อย 60 วินาที ไม่เกิน 1 ชั่วโมง; ถ้าราคาไม่กลับลงภายใน 1 ชั่วโมงใช้ 1 ชั่วโมงเต็ม) ส่วนหางของตอนแจ้งเตือนช่วงพุ่งที่เลยจุดสูงสุดไม่ถึง 60 วินาทีจึงไม่นับ และ 'ระบุช่วงถูก' = ตอนแจ้งเตือนนั้นระบุว่าเป็นช่วงกำลังเทขาย (แถวล่าสุดที่กฎจัดประเภทเป็น MM-T001.003 หรือ MM-T006)
- เหตุการณ์ที่วัดช่วงเทขายไม่ได้ไม่นับใน recall ช่วงเทขาย (นับจำนวนตามสถานะไว้ในคอลัมน์สุดท้าย)
- bear raid เป็นข้อมูลกึ่งสังเคราะห์ (SEMI-SYN) ทั้งหมด: ฉีดแรงขายเข้าไปในวัน test-normal จริง
- ทุกตัวเลขเป็นความผิดปกติทางสถิติ ไม่ใช่ความน่าจะเป็นว่ามีการปั่นหรือทุบราคา และไม่ได้บอกว่าใครทำอะไร
- เกณฑ์ผลต่าง sell − base: ลดลงอย่างมีนัย = CI บนของผลต่าง < 0 และผลต่าง ≤ −min_effect

## AE vs baselines (paired bootstrap, budget 2, real_t001 events)

| baseline | provenance | seeds | events | diff recall_any (AE − baseline) | 95% CI | ผล |
|---|---|---|---|---|---|---|
| baseline.random | REAL | 5 | 200 | 0.639 | [0.575, 0.700] | AE ดีกว่าอย่างมีนัย |
| baseline.kk | REAL | 5 | 200 | -0.170 | [-0.222, -0.121] | ไม่มีนัยสำคัญ |
| baseline.iforest | REAL | 5 | 200 | -0.158 | [-0.208, -0.111] | ไม่มีนัยสำคัญ |
| baseline.pca | REAL | 5 | 200 | 0.075 | [0.035, 0.122] | AE ดีกว่าอย่างมีนัย |
| stat.robust_z | REAL | 5 | 200 | 0.005 | [-0.033, 0.043] | ไม่มีนัยสำคัญ |

เกณฑ์: CI ล่าง > 0 และผลต่าง ≥ 0.02

## retrain_drift: champion v1 vs challenger v2 (paired bootstrap, budget 2, accept split)

| comparison | champion → challenger | provenance | seeds | events | recall_any champion [95% CI] | recall_any challenger [95% CI] | diff (challenger − champion) | 95% CI | ผล |
|---|---|---|---|---|---|---|---|---|---|
| full stack (deployed) | v1 → v2 | REAL | 5 | 94 | 0.760 [0.670, 0.840] | 0.774 [0.681, 0.853] | 0.015 | [-0.006, 0.043] | ไม่ยอมรับ |
| AE alone (diagnostic) | v1:ai → v2:ai | REAL | 5 | 94 | 0.721 [0.632, 0.806] | 0.728 [0.634, 0.809] | 0.006 | [-0.021, 0.040] | ไม่ยอมรับ |

- full stack (deployed): ไม่ยอมรับ challenger: ผลต่าง recall +0.015 (95% CI -0.006 ถึง 0.043) CI คร่อม 0
- AE alone (diagnostic): ไม่ยอมรับ challenger: ผลต่าง recall +0.006 (95% CI -0.021 ถึง 0.040) CI คร่อม 0

challenger แทน champion เมื่อ CI ล่างของผลต่าง > 0 และผลต่าง ≥ 0.02 (bootstrap แบบจับคู่ข้ามเหตุการณ์ ใช้ดัชนีสุ่มซ้ำเดียวกันทั้งสองฝั่ง)

## confusion: family ของป้ายกำกับ × เทคนิคที่ระบุ (budget 2)

### real_t001 full

| family | seeds | events | MM-T000 | MM-T001 | MM-T005 | none |
|---|---|---|---|---|---|---|
| telegram_pump | 5 | 200 | 0.339 | 0.399 | 0.001 | 0.261 |

- ค่าในตาราง = สัดส่วนเหตุการณ์ของ family นั้น (เฉลี่ยข้าม seed) ที่ episode ที่จับคู่ได้ระบุเทคนิคหลัก (parent) ตามคอลัมน์; MM-T000 = ระบุเทคนิคไม่ได้; none = ตรวจไม่พบ; เหตุการณ์ที่ถูกระบุหลายเทคนิคนับในทุกคอลัมน์ของเทคนิคเหล่านั้น ผลรวมของแถวจึงอาจเกิน 1

### semisyn

| family | seeds | events | MM-T000 | MM-T001 | MM-T005 | none |
|---|---|---|---|---|---|---|
| pump_burst | 5 | 395 | 0.110 | 0.186 | 0.000 | 0.704 |
| pump_stealth | 5 | 394 | 0.270 | 0.101 | 0.000 | 0.648 |
| wash_pingpong | 5 | 401 | 0.988 | 0.011 | 0.001 | 0.000 |

- ค่าในตาราง = สัดส่วนเหตุการณ์ของ family นั้น (เฉลี่ยข้าม seed) ที่ episode ที่จับคู่ได้ระบุเทคนิคหลัก (parent) ตามคอลัมน์; MM-T000 = ระบุเทคนิคไม่ได้; none = ตรวจไม่พบ; เหตุการณ์ที่ถูกระบุหลายเทคนิคนับในทุกคอลัมน์ของเทคนิคเหล่านั้น ผลรวมของแถวจึงอาจเกิน 1

## หมายเหตุ

- หน้าต่างเวลาที่ทับซ้อนกันมี autocorrelation สูง จึงทำให้การรับประกันของ conformal p-value อ่อนลง: อัตราการแจ้งเตือนจริงอาจสูงกว่างบที่ตั้งไว้
- เวลาตรวจจับ (TTD, latency) ไม่รวมเวลารอปิดแท่ง 1 วินาทีและเวลารอ stride 10 วินาที
- precision แสดงคู่กับ prevalence (สัดส่วนหน้าต่างที่เป็นบวก) เสมอ; F1 เก็บไว้ใน JSON แต่ไม่ใช่ตัวชี้วัดหลัก
- คะแนนรวม (fused) คือ anomaly p-value หรือ “ความหายากเทียบตลาดปกติ” ไม่ใช่ความน่าจะเป็นว่ามีการปั่นราคา
- ทุกการแจ้งเตือนคือความผิดปกติทางสถิติ ไม่ได้กล่าวหาบุคคลใด
- τ saturated: งบการแจ้งเตือนสูงกว่าอัตราสูงสุดที่ข้อมูลสอบเทียบทำได้ τ จึงถูกตั้งที่ค่า F บวกที่เล็กที่สุด หรือเหนือค่า F ทั้งหมดเมื่อไม่มีค่า F บวกเลย (เช่น กฎที่ไม่เคยทำงานในช่วงสอบเทียบ) งบจึงไม่ได้ถูกทดสอบกับข้อมูลจริง
- deterministic: CI from event bootstrap: ตัวตรวจจับที่ไม่ขึ้นกับ seed ให้ผลเหมือนกันทุก seed ช่วงความเชื่อมั่นจึงมาจากการ bootstrap ข้ามเหตุการณ์เท่านั้น
- ref_supervised เป็น supervised reference (uses labels) ใช้เทียบเท่านั้น
- technique-agnostic: recall_correct n/a: ชุดตัวตรวจจับที่ไม่มี rule หรือ stat.cusum ระบุเทคนิคไม่ได้ (ได้ MM-T000 เสมอ) จึงไม่แสดง recall_correct
- bootstrap unit: event = สุ่มซ้ำทีละเหตุการณ์จริง; symbol-day = สุ่มซ้ำทีละวันของสัญลักษณ์ เพราะเหตุการณ์ SEMI-SYN ทุก seed และทุก family ถูกฉีดลงในวันเดียวกัน; event (legacy, optimistic) = ผล SEMI-SYN รุ่นเก่าที่ไม่มีวันกำกับ ช่วงความเชื่อมั่นจึงแคบเกินจริง
- recall และช่วงความเชื่อมั่นใช้ตัวประมาณแบบอัตราส่วนเดียวกัน: รวมจำนวนที่ตรวจพบ / จำนวนทั้งหมด ของทุก (seed, เหตุการณ์) และสุ่มซ้ำทีละกลุ่ม (cluster)
- rule of three: เมื่อตรวจพบ 0 ครั้งหรือครบทุกครั้ง bootstrap CI จะแคบเหลือจุดเดียวทั้งที่ข้อมูลมีน้อย จึงแสดงขอบเขตโดยประมาณเพิ่ม: ขอบบน 3/n เมื่อตรวจไม่พบเลย หรือขอบล่าง 1 − 3/n เมื่อตรวจพบครบ โดย n คือจำนวนกลุ่มอิสระตามคอลัมน์ bootstrap unit (ไม่นับซ้ำข้าม seed)
- ± std คือส่วนเบี่ยงเบนมาตรฐานของ recall ข้าม seed (– เมื่อมี seed เดียว)
- การเปรียบเทียบสองฝั่งต้องมีอย่างน้อย 10 เหตุการณ์ ถ้าน้อยกว่านั้นจะไม่สรุปว่าดีกว่า
- TTD ติดลบหมายความว่าการแจ้งเตือนเริ่มก่อนเวลาเริ่มของป้ายกำกับ (label)
- งบการแจ้งเตือนเป็นเพดาน ไม่ใช่เป้าหมาย: อัตรา FA ที่วัดได้จึงอาจต่ำกว่างบที่ตั้งไว้
- FA/symbol-day [95% CI]: เมื่อผลมีจำนวนการแจ้งเตือนเท็จรายวัน (fa_by_day) จะเฉลี่ยข้าม seed ในแต่ละ symbol-day แล้ว bootstrap สุ่มซ้ำทีละ symbol-day (นับวันที่ไม่มีการแจ้งเตือนด้วย); ผลรุ่นเก่าที่ไม่มีข้อมูลนี้แสดงเฉพาะค่าเฉลี่ย; (n วัน) คือจำนวน symbol-day ที่ใช้นับ
- recall_correct นับเหตุการณ์ที่มีการแจ้งเตือนทับช่วงเหตุการณ์และระบุเทคนิคตรงกับป้ายกำกับ: เทคนิคของการแจ้งเตือนมาจากแถวที่จัดประเภทได้และมีค่า F สูงสุดของ episode ส่วนหลักฐานมาจากแถวจุดสูงสุดของ episode ซึ่งอาจเป็นคนละแถวกัน
