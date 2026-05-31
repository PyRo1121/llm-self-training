#!/usr/bin/env bash
# Cloud GPU setup: uv sync, HF auth, flash-attn (Jarvis / Vast / any SSH box).
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

# Private-repo secrets file (optional — track in git when repo is private)
if [[ -f "${ROOT}/config/cloud.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/config/cloud.env"
  set +a
fi

echo "=== cloud profile: ${LLM_CONFIG_PROFILE} ==="

if ! command -v uv >/dev/null 2>&1; then
  echo "=== installing uv ==="
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

make sync-all

_hf_login() {
  local token="$1"
  export HF_TOKEN="${token}"
  uv run huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null \
    || uv run hf auth login --token "${HF_TOKEN}" 2>/dev/null \
    || true
}

if [[ -f /home/hf_token ]]; then
  echo "=== Hugging Face auth from /home/hf_token ==="
  _hf_login "$(tr -d '[:space:]' < /home/hf_token)"
elif [[ -n "${HF_TOKEN:-}" ]]; then
  echo "=== Hugging Face auth from HF_TOKEN env ==="
  _hf_login "${HF_TOKEN}"
elif [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "=== Hugging Face auth from .env ==="
    _hf_login "${HF_TOKEN}"
  fi
fi

if [[ "${SKIP_FLASH_ATTN:-0}" != "1" ]]; then
  if uv run --package llm-train --extra unsloth python3 -c \
    "from llm_train.flash_attn import flash_attn_available; raise SystemExit(0 if flash_attn_available() else 1)" \
    2>/dev/null; then
    echo "=== flash-attn already installed ==="
  else
    echo "=== flash-attn build (10–20 min first time) ==="
    MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 8)}" bash scripts/install-flash-attn.sh
  fi
else
  echo "=== flash-attn skipped (SKIP_FLASH_ATTN=1) ==="
fi

echo "=== setup-cloud OK ==="
