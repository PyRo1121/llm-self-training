#!/usr/bin/env bash
# Stop project-local Redis started by scripts/redis-local.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$ROOT/data/redis/redis.pid"
if [[ -f "$PID_FILE ]]; then
  kill "$(cat "$PID_FILE")" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "redis-local stopped"
else
  echo "redis-local not running"
fi
