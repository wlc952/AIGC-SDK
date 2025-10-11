#!/bin/bash
set -euo pipefail

export NO_ALBUMENTATIONS_UPDATE=1
SDK_TOP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)

mkdir -p "$SDK_TOP/tmpdir"

cleanup() {
  echo "Cleaning up temporary files..."
  rm -rf "$SDK_TOP"/tmpdir/* 2>/dev/null || true
}

trap cleanup EXIT

HOST="${AIGC_HOST:-0.0.0.0}"
START_PORT="${AIGC_PORT:-8000}"

if ! [[ "$START_PORT" =~ ^[0-9]+$ ]] || [ "$START_PORT" -le 0 ] || [ "$START_PORT" -ge 65536 ]; then
  echo "Invalid starting port: $START_PORT" >&2
  exit 2
fi

PORT=$START_PORT
while ss -tuln | grep -q ":$PORT"; do
  echo "Port $PORT is in use. Trying next port..."
  PORT=$((PORT + 1))
  if [ "$PORT" -ge 65536 ]; then
    echo "No free ports available." >&2
    exit 3
  fi
done

echo "Launching API on $HOST:$PORT"
uv run python main.py --host "$HOST" --port "$PORT" "$@"