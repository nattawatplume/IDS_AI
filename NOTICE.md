# NOTICE

โครงงานนี้ใช้ข้อมูลและซอฟต์แวร์ของผู้อื่นตามรายการด้านล่าง ใช้เพื่อการศึกษาเท่านั้น ไม่ใช้เชิงพาณิชย์

## ข้อมูลการซื้อขาย: Binance Vision

- ที่มา: https://data.binance.vision (ข้อมูลย้อนหลังรายวัน) และ `data-stream.binance.vision` (ข้อมูลสด)
- สัญญาอนุญาต: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)
- ข้อมูลถูกดาวน์โหลดลงเครื่องของผู้ใช้แต่ละคน ไม่ได้เก็บไว้ใน git และทุกไฟล์ตรวจ CHECKSUM ก่อนใช้
- หน้าจอและรายงานต้องระบุที่มา "Binance Vision (CC BY-NC-SA 4.0)"

## บันทึกเหตุการณ์ปั่นราคา: Sapienza

- ไฟล์ `idsai/eval/data/pump_telegram.csv` มาจาก SystemsLab-Sapienza (Sapienza University of Rome)
  งาน "Pump and Dumps in the Bitcoin Era" (La Morgia et al.)
- Copyright (c) 2020 SystemsLab-Sapienza สัญญาอนุญาต MIT ข้อความเต็มอยู่ที่ `idsai/eval/data/LICENSE-sapienza`

## บันทึกเหตุการณ์ปั่นราคา: Hu et al.

- ที่มา: https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency
- repository ต้นทางไม่ได้ระบุสัญญาอนุญาต เราจึง **ไม่แจกจ่ายต่อ** ผู้ใช้โหลดเองด้วย
  `uv run python scripts/fetch_labels.py` ไฟล์จะอยู่ใน `data/labels/` ซึ่งไม่เข้า git

## TradingView Lightweight Charts

- ใช้วาดกราฟในหน้า Dashboard (จะนำมาเก็บไว้ในโปรเจกต์ในขั้นทำ Dashboard)
- Copyright TradingView, Inc. สัญญาอนุญาต Apache License 2.0
- หน้าจอต้องแสดงข้อความ attribution และลิงก์ https://www.tradingview.com/ ตามที่สัญญากำหนด

## โปรแกรมเสริม Python

โปรแกรมเสริมที่ติดตั้งผ่าน `uv sync` (เช่น polars, DuckDB, FastAPI, PyTorch, scikit-learn) มีสัญญาอนุญาตของตัวเอง
ดูรายการทั้งหมดได้ใน `pyproject.toml` และ `uv.lock`
