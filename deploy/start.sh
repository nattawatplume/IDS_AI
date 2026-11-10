#!/bin/sh
set -eu

PUBLIC_HOST="${RENDER_EXTERNAL_HOSTNAME:-${SPACE_HOST:-}}"
PORT="${PORT:-7860}"
SPEED="${IDSAI_SPEED:-20}"

if [ -z "$PUBLIC_HOST" ]; then
  echo "ข้อผิดพลาด: ไม่พบตัวแปร RENDER_EXTERNAL_HOSTNAME (ต้องรันบน Render)" >&2
  exit 1
fi

idsai user seed-env
exec idsai serve --mode replay --loop --speed "$SPEED" --host 0.0.0.0 --port "$PORT" --auth --secure-cookies --public-host "$PUBLIC_HOST"
