#!/usr/bin/env bash
# Phase 2 completion: register bootstrap run + eval gate (placeholder suites OK)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN_NAME="${1:-pyro-coder-bootstrap}"

echo "== register training run: $RUN_NAME =="
uv run --package llm-train train-register --run-name "$RUN_NAME"

echo "== eval (bootstrap / placeholder suites) =="
uv run --package llm-eval run-eval \
  --train-run "$RUN_NAME" \
  --no-smoke-chat \
  --model qwen2.5-coder:7b

echo "== optional export (needs CUDA) =="
echo "  uv run --package llm-train train-export --adapter-dir runs/$RUN_NAME/adapter"
echo "Phase 2 pipeline OK for $RUN_NAME"
