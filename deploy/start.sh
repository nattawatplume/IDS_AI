#!/bin/sh
set -eu

if [ -z "${SPACE_HOST:-}" ]; then
  echo "ข้อผิดพลาด: ไม่พบตัวแปร SPACE_HOST (ต้องรันบน Hugging Face Space)" >&2
  exit 1
fi

idsai user seed-env
exec idsai serve --mode replay --loop --host 0.0.0.0 --port 7860 --auth --secure-cookies --public-host "$SPACE_HOST"
