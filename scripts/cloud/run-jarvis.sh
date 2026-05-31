#!/usr/bin/env bash
# Local: launch Jarvis H100 managed run (jl CLI).
#
# Personal data (pick one):
#   A) Commit data/cloud/personal/personal-tier1.jsonl in a PRIVATE git repo (see CLOUD-TRAIN.md)
#   B) export CLOUD_DATA_REPO_URL=https://github.com/You/llm-self-training-data.git
#   C) export HF_DATASET=YourOrg/private-bundle  + HF_TOKEN in .env
#
# HF token (never commit): cp .env.example .env  →  HF_TOKEN=hf_...
#
# Usage:
#   ./scripts/cloud/run-jarvis.sh --smoke-only
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

GPU="${JARVIS_GPU:-H100}"
REGION="${JARVIS_REGION:-IN2}"
STORAGE="${JARVIS_STORAGE:-500}"
FS_ID="${JARVIS_FS_ID:-}"
RUN="${RUN:-pyro-coder-h100-v1}"
HF_DATASET="${HF_DATASET:-}"
CLOUD_DATA_REPO_URL="${CLOUD_DATA_REPO_URL:-}"
PERSONAL_RATIO="${PERSONAL_RATIO:-0.75}"
KEEP="${JARVIS_KEEP:-1}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  EXTRA_ARGS+=("$1")
  shift
done

if ! command -v jl >/dev/null 2>&1; then
  echo "Install Jarvis CLI: uv tool install jarvislabs && jl setup" >&2
  exit 1
fi

FS_FLAG=()
if [[ -n "${FS_ID}" ]]; then
  FS_FLAG=(--fs-id "${FS_ID}")
fi

KEEP_FLAG=(--keep)
if [[ "${KEEP}" == "0" ]]; then
  KEEP_FLAG=()
fi

echo "=== jl run: GPU=${GPU} region=${REGION} storage=${STORAGE}GB run=${RUN} ==="

TRAIN_ARGS=(--run "${RUN}" --personal-ratio "${PERSONAL_RATIO}")
if [[ -n "${HF_DATASET}" ]]; then
  TRAIN_ARGS+=(--personal-dataset "${HF_DATASET}")
fi
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

jl run . \
  --script scripts/cloud/train-cloud.sh \
  --gpu "${GPU}" \
  --region "${REGION}" \
  --storage "${STORAGE}" \
  --name "${RUN}" \
  "${FS_FLAG[@]}" \
  --setup "bash scripts/cloud/setup-jarvis.sh" \
  "${KEEP_FLAG[@]}" \
  --yes \
  -- \
  "${TRAIN_ARGS[@]}"

echo ""
echo "After run: jl list"
echo "  jl download <machine_id> runs/${RUN} ./runs/${RUN} -r"
echo "  jl download <machine_id> exports/${RUN} ./exports/${RUN} -r"
echo "  jl pause <machine_id>"
