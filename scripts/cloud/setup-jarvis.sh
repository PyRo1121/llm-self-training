#!/usr/bin/env bash
# Jarvis H100 one-time / per-run setup (flash-attn, HF auth, cloud profile).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export LLM_CONFIG_PROFILE="${LLM_CONFIG_PROFILE:-cloud-h100}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-24}"

if [[ -n "${LLM_DATA_DIR:-}" ]]; then
  mkdir -p "${LLM_DATA_DIR}/hf_cache" "${LLM_DATA_DIR}/raw" "${LLM_DATA_DIR}/curated" \
    "${LLM_DATA_DIR}/train" "${LLM_DATA_DIR}/warehouse"
fi

echo "=== cloud profile: ${LLM_CONFIG_PROFILE} ==="
make sync-all

if [[ -f /home/hf_token ]]; then
  echo "=== Hugging Face auth from /home/hf_token ==="
  HF_TOKEN="$(tr -d '[:space:]' < /home/hf_token)"
  export HF_TOKEN
  uv run huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null \
    || uv run hf auth login --token "${HF_TOKEN}" 2>/dev/null \
    || true
elif [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "=== Hugging Face auth from .env ==="
    uv run huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null \
      || uv run hf auth login --token "${HF_TOKEN}" 2>/dev/null \
      || true
  fi
elif [[ -n "${HF_TOKEN:-}" ]]; then
  uv run huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null \
    || uv run hf auth login --token "${HF_TOKEN}" 2>/dev/null \
    || true
fi

if [[ "${SKIP_FLASH_ATTN:-0}" != "1" ]]; then
  echo "=== flash-attn (10–20 min) ==="
  MAX_JOBS="${MAX_JOBS:-8}" bash scripts/install-flash-attn.sh
fi

echo "=== setup-jarvis OK ==="
