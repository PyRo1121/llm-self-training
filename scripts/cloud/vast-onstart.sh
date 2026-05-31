#!/usr/bin/env bash
# In-repo Vast runner (after clone). Setup deps then full train pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

export LLM_CONFIG_PROFILE="${LLM_CONFIG_PROFILE:-cloud-h100}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-24}"
export PATH="${HOME}/.local/bin:${PATH}"

if [[ -f "${ROOT}/config/cloud.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/config/cloud.env"
  set +a
fi

bash scripts/cloud/setup-cloud.sh

TRAIN_ARGS=(--run "${RUN:-pyro-coder-h100-v1}" --personal-ratio "${PERSONAL_RATIO:-0.75}")

if [[ -n "${CLOUD_TRAIN_MAX_STEPS:-}" ]]; then
  TRAIN_ARGS+=(--max-steps "${CLOUD_TRAIN_MAX_STEPS}")
fi

if [[ "${VAST_SMOKE:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--smoke-only)
fi

if [[ "${VAST_SKIP_INGEST:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--skip-ingest)
fi

if [[ -n "${VAST_INGEST_MODE:-}" ]]; then
  TRAIN_ARGS+=(--ingest-mode "${VAST_INGEST_MODE}")
fi

if [[ -n "${VAST_TRAIN_EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( ${VAST_TRAIN_EXTRA} )
  TRAIN_ARGS+=("${EXTRA[@]}")
fi

exec bash scripts/cloud/train-cloud.sh "${TRAIN_ARGS[@]}"
