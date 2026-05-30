#!/usr/bin/env bash
# Phase 1.5 sign-off: API health + overview + dashboard build
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== warehouse smoke =="
uv run --package llm-core warehouse-smoke

echo "== API health (start llm-api in another terminal if this fails) =="
curl -sf "http://127.0.0.1:8080/health" | head -c 200
echo
curl -sf "http://127.0.0.1:8080/api/v1/overview" | head -c 400
echo

echo "== dashboard build =="
cd apps/dashboard
bun install --frozen-lockfile 2>/dev/null || bun install
bun run build
echo "Phase 1.5 verify OK"
