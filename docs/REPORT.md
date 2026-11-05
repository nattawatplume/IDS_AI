# รายงานผลการทดลอง Market IDS (เพื่อการศึกษา)

## ที่มาของตัวเลข

- ตารางในภาคผนวกคัดลอกตรงตัวจาก `data/report.md` ที่สร้างด้วย `uv run idsai report` (อ่านทุก run ใน `data/research.duckdb`) ตัวเลขผลการทดลองในบทอภิปรายมาจากตารางนั้นและ `data/report.json` ส่วนตัวเลขอื่นมาจาก: ผลวัด pump จริงช่วง train (`scripts/measure_pumps.py`, log `measure.log`), บันทึก inject เดโมของ Task 19 ใน `progress.md` (F 3.54 / 2.92, τ 3.87), log ของ eval (`eval_seed0.log`, `eval_seeds1to4.log`, `eval_seeds1to4b.log`) และ `live_check.log`
- git_sha `b1e37136421e4e569a71723c6acc59c3867611ef` · config_hash `53e3b662ab96` · signatures_sha `0fe49a5924d7` (ไม่ได้ปรับ signature ตาม Ruling 46 L3) · split `7fb702a841c5`
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
| FA ต่อ symbol-day นอกชุดสอบเทียบ ≤ งบ | REAL test (real_t001 full, 579 symbol-day ในชุด; FA นับเฉพาะวันปกติ จำนวนวันไม่อยู่ในรายงาน) | 0.121 / 0.400 / 1.650 / 6.477 ที่งบ 0.1 / 0.5 / 2 / 10 | ถึงที่งบ 0.5, 2, 10; เกินที่งบ 0.1 |
| | REAL accept (retrain_drift v1, 456 symbol-day ในชุด; FA นับเฉพาะวันปกติ จำนวนวันไม่อยู่ในรายงาน) | 0.204 / 0.924 / 3.578 / 10.557 | เกินทุกงบ |
| FA บน hard negative (spec: ต้องไม่แจ้งเตือน) | REAL (real_t001 full) | 11.170 ต่อ symbol-day ที่งบ 2 (0.730 / 3.010 / 23.160 ที่งบ 0.1 / 0.5 / 10) | ไม่ถึง |
| compute latency p99 < 2 s | – | ไม่มีในรายงานนี้ (eval ไม่ได้วัด latency) | ยังไม่ได้วัด |

- รายงานไม่มี CI ของ FA และไม่มีจำนวนวัน hard negative
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
- รายงานนี้ไม่มี paired bootstrap ของ ablation จึงไม่สรุปว่าความต่างใดมีนัย (ห้ามใช้ CI ไม่ทับกันแทน)
- rules ถึงเพดานที่งบ 2 (τ saturated) งบจึงไม่ได้ถูกทดสอบจริง; stats ระบุเทคนิคไม่ได้ (recall_correct 0 จาก 200 events REAL, rule of three ขอบบนราว 2 %) recall_correct จึงมาจากชั้น rule

### 4. retrain_drift Decision (REAL, split accept, 94 events, 5 seed, budget 2)

- full stack (deployed): v1 0.760 [0.670, 0.840] → v2 0.774 [0.681, 0.853], ผลต่าง +0.015 [−0.006, 0.043] → **ไม่ยอมรับ challenger คง v1**
- AE alone (diagnostic): v1:ai 0.721 → v2:ai 0.728, ผลต่าง +0.006 [−0.021, 0.040] → ไม่ยอมรับ
- ทั้ง v1 และ v2 มี FA เกินงบบน split นี้ (3.578 และ 3.611)

### 5. ช่องว่างกับ supervised reference (REAL, ใช้ป้าย ใช้เทียบเท่านั้น)

ref.supervised_rf: recall_any 0.851 [0.802, 0.896], FA 2.121, hard 1.610, PR-AUC 0.354 เทียบระบบเต็ม 0.739 [0.680, 0.801], FA 1.650, hard 11.170, PR-AUC 0.243 ผลต่างของค่าเฉลี่ย recall 0.112 ไม่ได้ทดสอบแบบ paired และ CI แทบชนกัน (ขอบล่าง 0.802 กับขอบบน 0.801) ทั้งนี้ RF มี FA 2.121 เกินงบ 2 ผลนี้ชี้ว่าข้อมูลอาจมีสัญญาณที่แยกเหตุการณ์ได้มากกว่าที่ระบบแบบไม่ใช้ป้ายจับได้ โดยเฉพาะบน hard negative แต่ยังสรุปไม่ได้

### 6. เรื่องที่ยกมาจากงานก่อน (Ruling 46 L5)

- **ความเฉพาะเจาะจงของ T001 บนข้อมูลจริง และตารางสับสนรายตระกูล:** `data/report.json` ไม่มี per_event จึงทำตารางไม่ได้ ไม่มีในรายงานนี้ (บันทึกเป็นข้อจำกัด) สิ่งที่เห็นได้: REAL recall_correct 0.399 ต่ำกว่า recall_any 0.739 มาก และ SEMI-SYN wash_pingpong recall_any 1.000 แต่ recall_correct 0.001 แปลว่าระบบเห็นความผิดปกติในวันที่มี wash แต่แทบไม่เคยให้เทคนิคตรงป้าย ส่วนว่าให้เป็นเทคนิคอะไรแทน ต้องรอ per_event
- **Ruling 30 (เริ่ม batch เร็วขึ้น 60 นาที) ไม่ได้ทำ:** batch เริ่มนับหน้าต่างใหม่ทุกวัน ส่วน replay/live เป็นสายต่อเนื่อง τ ที่สอบเทียบจากแถว batch จึงอาจคลาดเล็กน้อยในชั่วโมงแรกหลังเที่ยงคืนเมื่อรันแบบ stream ทางแก้ต้องใช้ข้อมูล D−2, ล้าง cache และสอบเทียบใหม่ จึงไม่ทำในรอบนี้ ตัวเลขในรายงานเป็นของ batch ทั้งหมด
- **inject เดโมที่จับไม่ได้:** ใน Task 19 ฉีด pump_burst ช่วง test สองครั้ง (impact 3.5 % vol ×6 และ 4.1 % vol ×25) ได้ F สูงสุด 3.54 และ 2.92 ต่ำกว่า τ 3.87 ที่งบ 2 ของชุดเดโม ผลนี้สอดคล้องกับ SEMI-SYN pump_burst ที่งบ 2 ซึ่งจับได้ 0.295 [0.264, 0.328] (n = 395) คือพลาดเป็นส่วนใหญ่ ช่วง test ตั้งใจให้ยากและอยู่นอกการกระจายของ pump จริง: impact 3–5 % ต่ำกว่าหรือใกล้เคียงขอบ 10 % ของ pump จริงช่วง train (0.0499) และ vol_mult 5–30 ต่ำกว่าขอบ 10 % (45.39) (มัธยฐานของ pump จริง: impact 0.1543, vol_mult 738.9) ส่วน pump จริงบนข้อมูล REAL จับได้ 0.739 ช่วง train ของ Red-Team ไม่ได้ถูกประเมินในรายงานนี้

### 7. ข้อจำกัด

ดูทั้งหมดใน `docs/DATA_LIMITATIONS.md` ที่กระทบการอ่านรายงานนี้โดยตรง:

- ผล REAL ใช้ได้กับคู่ BTC ช่วง 2018–2022 เท่านั้น ป้ายบางส่วน (Hu-only) มีเวลาห่างจุดปริมาณซื้อขายสูงสุดมาก
- SEMI-SYN วัดเฉพาะรูปแบบที่จำลองในช่วง test ไม่ได้วัดการปั่นจริง
- หน้าต่างทับซ้อนมี autocorrelation ทำให้การรับประกันของ conformal p-value อ่อนลง อัตรา FA จริงจึงอาจเกินงบ (เห็นที่งบ 0.1 และบน split accept)
- ไม่มี CI ของ FA, ไม่มีจำนวนวัน hard negative, ไม่มี latency p99, ไม่มี per_event
- live check รอบนี้เป็นการตรวจ sensor อย่างเดียว ไม่มีการตรวจจับบนข้อมูลสด
- ไม่ได้ปรับ signature บนข้อมูล train (L3) recall ของชั้น rule อาจต่ำกว่าชุดที่ปรับแล้ว

---

## ภาคผนวก: รายงานที่สร้างอัตโนมัติ (คัดลอกตรงตัวจาก `data/report.md`)

# รายงานผลการประเมิน Market IDS (เพื่อการศึกษา)

runs: 500

signatures_sha: 0fe49a5924d7

git_sha: b1e37136421e4e569a71723c6acc59c3867611ef · config_hash: 53e3b662ab96

labels (split 7fb702a841c5): eligible 477 of 601; dropped: out_of_period 0, cal_period 91, missing_day 33, no_baseline 0; symbol-days skipped at window loading: none

ทุกตัวเลขมาจากตาราง runs ทั้งหมด; ค่าเฉลี่ยข้าม seed และช่วงความเชื่อมั่น 95 % จากการ bootstrap (หน่วยตามคอลัมน์ bootstrap unit)

## real_t001

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day | FA/symbol-day hard | TTD median / p90 (s) | PR-AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0.1 | REAL | 5 | 200 | 0.540 ± 0.025 [0.473, 0.605] | 0.229 ± 0.011 [0.180, 0.285] | event | 0.500 (prevalence 0.0017) | 0.121 | 0.730 | 9 / 61 | 0.243 |  |
| full | 0.5 | REAL | 5 | 200 | 0.638 ± 0.016 [0.574, 0.699] | 0.301 ± 0.017 [0.241, 0.366] | event | 0.300 (prevalence 0.0017) | 0.400 | 3.010 | 8 / 72 | 0.243 |  |
| full | 2 | REAL | 5 | 200 | 0.739 ± 0.011 [0.680, 0.801] | 0.399 ± 0.008 [0.334, 0.472] | event | 0.113 (prevalence 0.0017) | 1.650 | 11.170 | 6 / 60 | 0.243 |  |
| full | 10 | REAL | 5 | 200 | 0.808 ± 0.004 [0.751, 0.858] | 0.481 ± 0.009 [0.412, 0.555] | event | 0.037 (prevalence 0.0017) | 6.477 | 23.160 | -0 / 46 | 0.243 |  |

## semisyn

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day | FA/symbol-day hard | TTD median / p90 (s) | PR-AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pump_burst | 0.1 | SEMI-SYN | 5 | 395 | 0.049 ± 0.018 [0.035, 0.066] | 0.018 ± 0.009 [0.011, 0.026] | symbol-day | 0.283 (prevalence 0.0061) | 0.122 | – | 208 / 320 | 0.134 |  |
| pump_burst | 0.5 | SEMI-SYN | 5 | 395 | 0.106 ± 0.026 [0.085, 0.131] | 0.047 ± 0.013 [0.034, 0.061] | symbol-day | 0.229 (prevalence 0.0061) | 0.353 | – | 152 / 278 | 0.134 |  |
| pump_burst | 2 | SEMI-SYN | 5 | 395 | 0.295 ± 0.047 [0.264, 0.328] | 0.186 ± 0.039 [0.163, 0.208] | symbol-day | 0.171 (prevalence 0.0061) | 1.425 | – | 93 / 224 | 0.134 |  |
| pump_burst | 10 | SEMI-SYN | 5 | 395 | 0.769 ± 0.026 [0.744, 0.792] | 0.609 ± 0.020 [0.584, 0.633] | symbol-day | 0.117 (prevalence 0.0061) | 6.079 | – | 82 / 161 | 0.134 |  |
| pump_stealth | 0.1 | SEMI-SYN | 5 | 394 | 0.049 ± 0.020 [0.039, 0.059] | 0.006 ± 0.001 [0.003, 0.010] | symbol-day | 0.283 (prevalence 0.0775) | 0.122 | – | 6103 / 6935 | 0.851 |  |
| pump_stealth | 0.5 | SEMI-SYN | 5 | 394 | 0.099 ± 0.006 [0.085, 0.114] | 0.019 ± 0.001 [0.012, 0.026] | symbol-day | 0.249 (prevalence 0.0775) | 0.317 | – | 6055 / 6882 | 0.851 |  |
| pump_stealth | 2 | SEMI-SYN | 5 | 394 | 0.351 ± 0.191 [0.328, 0.375] | 0.101 ± 0.020 [0.085, 0.118] | symbol-day | 0.274 (prevalence 0.0775) | 1.100 | – | 5579 / 6863 | 0.851 |  |
| pump_stealth | 10 | SEMI-SYN | 5 | 394 | 0.999 ± 0.001 [0.998, 1.000] | 0.485 ± 0.023 [0.453, 0.517] | symbol-day | 0.185 (prevalence 0.0775) | 5.333 | – | 533 / 1008 | 0.851 |  |
| wash_pingpong | 0.1 | SEMI-SYN | 5 | 401 | 1.000 ± 0.000 [1.000, 1.000] | 0.001 ± 0.002 [0.000, 0.003] | symbol-day | 0.896 (prevalence 0.0171) | 0.116 | – | 17 / 62 | 0.835 | recall_any 401 จาก 401 symbol-day: ช่วงจริงอาจต่ำถึงราว 99% (rule of three) |
| wash_pingpong | 0.5 | SEMI-SYN | 5 | 401 | 1.000 ± 0.000 [1.000, 1.000] | 0.001 ± 0.002 [0.000, 0.003] | symbol-day | 0.766 (prevalence 0.0171) | 0.306 | – | 14 / 44 | 0.835 | recall_any 401 จาก 401 symbol-day: ช่วงจริงอาจต่ำถึงราว 99% (rule of three) |
| wash_pingpong | 2 | SEMI-SYN | 5 | 401 | 1.000 ± 0.000 [1.000, 1.000] | 0.001 ± 0.002 [0.000, 0.003] | symbol-day | 0.449 (prevalence 0.0171) | 1.241 | – | 11 / 33 | 0.835 | recall_any 401 จาก 401 symbol-day: ช่วงจริงอาจต่ำถึงราว 99% (rule of three) |
| wash_pingpong | 10 | SEMI-SYN | 5 | 401 | 1.000 ± 0.000 [1.000, 1.000] | 0.001 ± 0.002 [0.000, 0.003] | symbol-day | 0.165 (prevalence 0.0171) | 5.171 | – | 8 / 16 | 0.835 | recall_any 401 จาก 401 symbol-day: ช่วงจริงอาจต่ำถึงราว 99% (rule of three) |

## ablation

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day | FA/symbol-day hard | TTD median / p90 (s) | PR-AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real_t001:ai | 0.1 | REAL | 5 | 200 | 0.505 ± 0.047 [0.445, 0.566] | – | event | 0.503 (prevalence 0.0017) | 0.110 | 0.660 | 40 / 127 | 0.154 | technique-agnostic: recall_correct n/a |
| real_t001:ai | 0.5 | REAL | 5 | 200 | 0.598 ± 0.033 [0.536, 0.661] | – | event | 0.275 (prevalence 0.0017) | 0.429 | 3.160 | 34 / 94 | 0.154 | technique-agnostic: recall_correct n/a |
| real_t001:ai | 2 | REAL | 5 | 200 | 0.650 ± 0.023 [0.588, 0.710] | – | event | 0.111 (prevalence 0.0017) | 1.556 | 11.300 | 21 / 83 | 0.154 | technique-agnostic: recall_correct n/a |
| real_t001:ai | 10 | REAL | 5 | 200 | 0.820 ± 0.014 [0.772, 0.866] | – | event | 0.020 (prevalence 0.0017) | 12.396 | 26.230 | -1 / 39 | 0.154 | technique-agnostic: recall_correct n/a |
| real_t001:all | 0.1 | REAL | 5 | 200 | 0.540 ± 0.025 [0.473, 0.605] | 0.229 ± 0.011 [0.180, 0.285] | event | 0.500 (prevalence 0.0017) | 0.121 | 0.730 | 9 / 61 | 0.243 |  |
| real_t001:all | 0.5 | REAL | 5 | 200 | 0.638 ± 0.016 [0.574, 0.699] | 0.301 ± 0.017 [0.241, 0.366] | event | 0.300 (prevalence 0.0017) | 0.400 | 3.010 | 8 / 72 | 0.243 |  |
| real_t001:all | 2 | REAL | 5 | 200 | 0.739 ± 0.011 [0.680, 0.801] | 0.399 ± 0.008 [0.334, 0.472] | event | 0.113 (prevalence 0.0017) | 1.650 | 11.170 | 6 / 60 | 0.243 |  |
| real_t001:all | 10 | REAL | 5 | 200 | 0.808 ± 0.004 [0.751, 0.858] | 0.481 ± 0.009 [0.412, 0.555] | event | 0.037 (prevalence 0.0017) | 6.477 | 23.160 | -0 / 46 | 0.243 |  |
| real_t001:rules | 0.1 | REAL | 5 | 200 | 0.215 ± 0.000 [0.165, 0.275] | 0.210 ± 0.000 [0.160, 0.270] | event | 0.183 (prevalence 0.0017) | 0.209 | 1.250 | 6 / 236 | 0.013 | deterministic: CI from event bootstrap |
| real_t001:rules | 0.5 | REAL | 5 | 200 | 0.370 ± 0.000 [0.305, 0.440] | 0.365 ± 0.000 [0.300, 0.435] | event | 0.088 (prevalence 0.0017) | 0.945 | 2.550 | 4 / 137 | 0.013 | deterministic: CI from event bootstrap |
| real_t001:rules | 2 | REAL | 5 | 200 | 0.405 ± 0.000 [0.340, 0.475] | 0.400 ± 0.000 [0.330, 0.470] | event | 0.080 (prevalence 0.0017) | 1.172 | 2.800 | 6 / 170 | 0.013 | τ saturated; deterministic: CI from event bootstrap |
| real_t001:rules | 10 | REAL | 5 | 200 | 0.405 ± 0.000 [0.340, 0.475] | 0.400 ± 0.000 [0.330, 0.470] | event | 0.080 (prevalence 0.0017) | 1.172 | 2.800 | 6 / 170 | 0.013 | τ saturated; deterministic: CI from event bootstrap |
| real_t001:rules+stats | 0.1 | REAL | 5 | 200 | 0.565 ± 0.000 [0.495, 0.630] | 0.240 ± 0.000 [0.185, 0.300] | event | 0.568 (prevalence 0.0017) | 0.107 | 0.900 | 7 / 45 | 0.250 | deterministic: CI from event bootstrap |
| real_t001:rules+stats | 0.5 | REAL | 5 | 200 | 0.655 ± 0.000 [0.585, 0.720] | 0.320 ± 0.000 [0.260, 0.390] | event | 0.335 (prevalence 0.0017) | 0.369 | 2.700 | 9 / 60 | 0.250 | deterministic: CI from event bootstrap |
| real_t001:rules+stats | 2 | REAL | 5 | 200 | 0.745 ± 0.000 [0.685, 0.800] | 0.435 ± 0.000 [0.370, 0.505] | event | 0.118 (prevalence 0.0017) | 1.516 | 5.550 | 5 / 60 | 0.250 | deterministic: CI from event bootstrap |
| real_t001:rules+stats | 10 | REAL | 5 | 200 | 0.820 ± 0.000 [0.765, 0.870] | 0.470 ± 0.000 [0.400, 0.540] | event | 0.034 (prevalence 0.0017) | 6.878 | 20.150 | 1 / 50 | 0.250 | deterministic: CI from event bootstrap |
| real_t001:stats | 0.1 | REAL | 5 | 200 | 0.545 ± 0.000 [0.480, 0.610] | 0.000 ± 0.000 [0.000, 0.000] | event | 0.605 (prevalence 0.0017) | 0.097 | 0.550 | 7 / 40 | 0.254 | deterministic: CI from event bootstrap; recall_correct 0 จาก 200 เหตุการณ์: ช่วงจริงอาจสูงถึงราว 2% (rule of three) |
| real_t001:stats | 0.5 | REAL | 5 | 200 | 0.615 ± 0.000 [0.545, 0.675] | 0.000 ± 0.000 [0.000, 0.000] | event | 0.362 (prevalence 0.0017) | 0.342 | 2.350 | 8 / 40 | 0.254 | deterministic: CI from event bootstrap; recall_correct 0 จาก 200 เหตุการณ์: ช่วงจริงอาจสูงถึงราว 2% (rule of three) |
| real_t001:stats | 2 | REAL | 5 | 200 | 0.650 ± 0.000 [0.580, 0.715] | 0.000 ± 0.000 [0.000, 0.000] | event | 0.205 (prevalence 0.0017) | 0.773 | 3.700 | 7 / 31 | 0.254 | deterministic: CI from event bootstrap; recall_correct 0 จาก 200 เหตุการณ์: ช่วงจริงอาจสูงถึงราว 2% (rule of three) |
| real_t001:stats | 10 | REAL | 5 | 200 | 0.770 ± 0.000 [0.710, 0.825] | 0.000 ± 0.000 [0.000, 0.000] | event | 0.035 (prevalence 0.0017) | 6.429 | 19.550 | 5 / 36 | 0.254 | deterministic: CI from event bootstrap; recall_correct 0 จาก 200 เหตุการณ์: ช่วงจริงอาจสูงถึงราว 2% (rule of three) |
| semisyn:ai | 0.1 | SEMI-SYN | 5 | 1188 | 0.374 ± 0.010 [0.367, 0.381] | – | symbol-day | 0.790 (prevalence 0.0334) | 0.100 | – | 18 / 154 | 0.383 | technique-agnostic: recall_correct n/a |
| semisyn:ai | 0.5 | SEMI-SYN | 5 | 1188 | 0.411 ± 0.018 [0.402, 0.422] | – | symbol-day | 0.551 (prevalence 0.0334) | 0.340 | – | 16 / 274 | 0.383 | technique-agnostic: recall_correct n/a |
| semisyn:ai | 2 | SEMI-SYN | 5 | 1188 | 0.519 ± 0.080 [0.505, 0.535] | – | symbol-day | 0.321 (prevalence 0.0334) | 1.135 | – | 19 / 4945 | 0.383 | technique-agnostic: recall_correct n/a |
| semisyn:ai | 10 | SEMI-SYN | 5 | 1188 | 0.888 ± 0.026 [0.875, 0.901] | – | symbol-day | 0.099 (prevalence 0.0334) | 11.342 | – | 31 / 5736 | 0.383 | technique-agnostic: recall_correct n/a |
| semisyn:all | 0.1 | SEMI-SYN | 5 | 1188 | 0.371 ± 0.009 [0.365, 0.377] | 0.009 ± 0.003 [0.006, 0.011] | symbol-day | 0.756 (prevalence 0.0334) | 0.120 | – | 20 / 141 | 0.748 |  |
| semisyn:all | 0.5 | SEMI-SYN | 5 | 1188 | 0.406 ± 0.010 [0.397, 0.415] | 0.022 ± 0.005 [0.018, 0.027] | symbol-day | 0.558 (prevalence 0.0334) | 0.325 | – | 17 / 226 | 0.748 |  |
| semisyn:all | 2 | SEMI-SYN | 5 | 1188 | 0.552 ± 0.078 [0.539, 0.567] | 0.095 ± 0.019 [0.087, 0.104] | symbol-day | 0.316 (prevalence 0.0334) | 1.256 | – | 23 / 4848 | 0.748 |  |
| semisyn:all | 10 | SEMI-SYN | 5 | 1188 | 0.923 ± 0.008 [0.916, 0.931] | 0.363 ± 0.012 [0.350, 0.376] | symbol-day | 0.155 (prevalence 0.0334) | 5.525 | – | 68 / 621 | 0.748 |  |
| semisyn:rules | 0.1 | SEMI-SYN | 5 | 1188 | 0.046 ± 0.006 [0.039, 0.054] | 0.045 ± 0.006 [0.038, 0.053] | symbol-day | 0.222 (prevalence 0.0334) | 0.199 | – | 1373 / 5104 | 0.053 |  |
| semisyn:rules | 0.5 | SEMI-SYN | 5 | 1188 | 0.334 ± 0.014 [0.321, 0.349] | 0.330 ± 0.014 [0.317, 0.343] | symbol-day | 0.331 (prevalence 0.0334) | 0.892 | – | 135 / 4036 | 0.053 |  |
| semisyn:rules | 2 | SEMI-SYN | 5 | 1188 | 0.396 ± 0.009 [0.382, 0.409] | 0.390 ± 0.009 [0.378, 0.403] | symbol-day | 0.330 (prevalence 0.0334) | 1.109 | – | 131 / 3965 | 0.053 | τ saturated |
| semisyn:rules | 10 | SEMI-SYN | 5 | 1188 | 0.396 ± 0.009 [0.382, 0.409] | 0.390 ± 0.009 [0.378, 0.403] | symbol-day | 0.330 (prevalence 0.0334) | 1.109 | – | 131 / 3965 | 0.053 | τ saturated |
| semisyn:rules+stats | 0.1 | SEMI-SYN | 5 | 1188 | 0.017 ± 0.003 [0.012, 0.022] | 0.008 ± 0.002 [0.005, 0.010] | symbol-day | 0.158 (prevalence 0.0334) | 0.090 | – | 175 / 3044 | 0.726 |  |
| semisyn:rules+stats | 0.5 | SEMI-SYN | 5 | 1188 | 0.111 ± 0.004 [0.098, 0.124] | 0.032 ± 0.004 [0.027, 0.038] | symbol-day | 0.282 (prevalence 0.0334) | 0.293 | – | 303 / 1688 | 0.726 |  |
| semisyn:rules+stats | 2 | SEMI-SYN | 5 | 1188 | 0.710 ± 0.006 [0.693, 0.726] | 0.354 ± 0.014 [0.341, 0.367] | symbol-day | 0.358 (prevalence 0.0334) | 1.325 | – | 471 / 847 | 0.726 |  |
| semisyn:rules+stats | 10 | SEMI-SYN | 5 | 1188 | 0.927 ± 0.002 [0.918, 0.936] | 0.390 ± 0.009 [0.378, 0.403] | symbol-day | 0.145 (prevalence 0.0334) | 5.660 | – | 102 / 631 | 0.726 |  |
| semisyn:stats | 0.1 | SEMI-SYN | 5 | 1188 | 0.016 ± 0.002 [0.011, 0.021] | 0.000 ± 0.000 [0.000, 0.000] | symbol-day | 0.165 (prevalence 0.0334) | 0.080 | – | 126 / 1259 | 0.805 | recall_correct 0 จาก 401 symbol-day: ช่วงจริงอาจสูงถึงราว 1% (rule of three) |
| semisyn:stats | 0.5 | SEMI-SYN | 5 | 1188 | 0.112 ± 0.005 [0.099, 0.125] | 0.000 ± 0.000 [0.000, 0.000] | symbol-day | 0.306 (prevalence 0.0334) | 0.259 | – | 276 / 906 | 0.805 | recall_correct 0 จาก 401 symbol-day: ช่วงจริงอาจสูงถึงราว 1% (rule of three) |
| semisyn:stats | 2 | SEMI-SYN | 5 | 1188 | 0.542 ± 0.006 [0.526, 0.557] | 0.000 ± 0.000 [0.000, 0.000] | symbol-day | 0.477 (prevalence 0.0334) | 0.619 | – | 513 / 936 | 0.805 | recall_correct 0 จาก 401 symbol-day: ช่วงจริงอาจสูงถึงราว 1% (rule of three) |
| semisyn:stats | 10 | SEMI-SYN | 5 | 1188 | 0.835 ± 0.004 [0.822, 0.848] | 0.000 ± 0.000 [0.000, 0.000] | symbol-day | 0.142 (prevalence 0.0334) | 5.161 | – | 91 / 642 | 0.805 | recall_correct 0 จาก 401 symbol-day: ช่วงจริงอาจสูงถึงราว 1% (rule of three) |

## baselines

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day | FA/symbol-day hard | TTD median / p90 (s) | PR-AUC | notes |
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

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day | FA/symbol-day hard | TTD median / p90 (s) | PR-AUC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref.supervised_rf | 0.1 | REAL | 5 | 200 | 0.631 ± 0.011 [0.563, 0.693] | – | event | 0.653 (prevalence 0.0017) | 0.079 | 0.080 | 13 / 81 | 0.354 | supervised reference (uses labels); technique-agnostic: recall_correct n/a |
| ref.supervised_rf | 0.5 | REAL | 5 | 200 | 0.801 ± 0.002 [0.745, 0.853] | – | event | 0.306 (prevalence 0.0017) | 0.514 | 0.370 | 8 / 35 | 0.354 | supervised reference (uses labels); technique-agnostic: recall_correct n/a |
| ref.supervised_rf | 2 | REAL | 5 | 200 | 0.851 ± 0.004 [0.802, 0.896] | – | event | 0.101 (prevalence 0.0017) | 2.121 | 1.610 | -8 / 19 | 0.354 | supervised reference (uses labels); technique-agnostic: recall_correct n/a |
| ref.supervised_rf | 10 | REAL | 5 | 200 | 0.916 ± 0.002 [0.880, 0.955] | – | event | 0.028 (prevalence 0.0017) | 9.592 | 9.880 | -335 / 10 | 0.354 | supervised reference (uses labels); technique-agnostic: recall_correct n/a |

## retrain_drift

| variant | budget | provenance | seeds | events | recall_any ± std [95% CI] | recall_correct ± std [95% CI] | bootstrap unit | precision (prevalence) | FA/symbol-day | FA/symbol-day hard | TTD median / p90 (s) | PR-AUC | notes |
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
- recall_correct นับเหตุการณ์ที่มีการแจ้งเตือนทับช่วงเหตุการณ์และระบุเทคนิคตรงกับป้ายกำกับ: เทคนิคของการแจ้งเตือนมาจากแถวที่จัดประเภทได้และมีค่า F สูงสุดของ episode ส่วนหลักฐานมาจากแถวจุดสูงสุดของ episode ซึ่งอาจเป็นคนละแถวกัน
