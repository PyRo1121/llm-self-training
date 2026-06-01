#!/usr/bin/env bash
# Start project-local Valkey/Redis on 127.0.0.1:6380 (does not touch system valkey.service)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${REDIS_PASSWORD:?Set REDIS_PASSWORD in .env (see .env.example)}"

DATA_DIR="$ROOT/data/redis"
CONF_OUT="$ROOT/config/redis.local.conf"
PID_FILE="$DATA_DIR/redis.pid"
mkdir -p "$DATA_DIR"

sed "s|REDIS_DATA_DIR|$DATA_DIR|g" "$ROOT/config/redis.local.conf.example" | \
  sed "s|^dir .*|dir \"$DATA_DIR\"|" > "$CONF_OUT"
echo "requirepass $REDIS_PASSWORD" >> "$CONF_OUT"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "redis-local already running (pid $(cat "$PID_FILE")) on port ${REDIS_PORT:-6380}"
  exit 0
fi

PORT="${REDIS_PORT:-6380}"
echo "Starting valkey/redis on 127.0.0.1:$PORT ..."
redis-server "$CONF_OUT" --port "$PORT" --daemonize yes --pidfile "$PID_FILE"
sleep 0.5
redis-cli -p "$PORT" -a "$REDIS_PASSWORD" ping
echo "OK — REDIS_URL=redis://:****@127.0.0.1:$PORT/0"
