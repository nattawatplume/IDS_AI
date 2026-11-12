#!/bin/sh
set -eu

PUBLIC_HOST="${RENDER_EXTERNAL_HOSTNAME:-${SPACE_HOST:-}}"
PORT="${PORT:-7860}"
SPEED="${IDSAI_SPEED:-20}"
MODE="${IDSAI_MODE:-live}"

if [ -z "$PUBLIC_HOST" ]; then
  echo "ข้อผิดพลาด: ไม่พบตัวแปร RENDER_EXTERNAL_HOSTNAME (ต้องรันบน Render)" >&2
  exit 1
fi

idsai user seed-env

REASON=""
if [ "$MODE" != "live" ]; then
  REASON="IDSAI_MODE=$MODE (ไม่ใช่ live)"
elif [ ! -f models/live/bundle.json ]; then
  REASON="ไม่พบโมเดล live (models/live/bundle.json)"
elif ! idsai live-check; then
  REASON="ต่อสายข้อมูลสดของ Binance จากเซิร์ฟเวอร์นี้ไม่ได้"
fi

if [ -z "$REASON" ]; then
  if ! idsai live-prep; then
    echo "คำเตือน: เตรียมข้อมูลย้อนหลังของ live ไม่สำเร็จทั้งหมด จะเปิดต่อโดยใช้ข้อมูลเท่าที่มี" >&2
  fi
  exec idsai serve --mode live --host 0.0.0.0 --port "$PORT" --auth --secure-cookies --allow-signup --public-host "$PUBLIC_HOST"
fi

echo "เปิดโหมดย้อนดู (replay) แทน live เพราะ: $REASON" >&2
exec idsai serve --mode replay --loop --speed "$SPEED" --host 0.0.0.0 --port "$PORT" --auth --secure-cookies --allow-signup --public-host "$PUBLIC_HOST"
