#!/usr/bin/env bash
# Local: rent Vast H100, onstart clones repo and runs train-cloud.sh immediately.
#
# One-time:
#   pip install vastai && vastai set api-key $VAST_API_KEY
#   vastai create ssh-key ~/.ssh/id_ed25519.pub
#   cp config/cloud.env.example config/cloud.env   # HF_TOKEN, GITHUB_TOKEN (private repo)
#
# Launch:
#   make cloud-vast-smoke    # 5 train steps (~cheap validation)
#   make cloud-vast          # full ingest + train + export
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
_load_env() {
  [[ -f "${ROOT}/config/cloud.env" ]] && source "${ROOT}/config/cloud.env"
  [[ -f "${ROOT}/.env" ]] && source "${ROOT}/.env"
}
_load_env

SMOKE=0
SKIP_SEARCH=0
OFFER_ID="${VAST_OFFER_ID:-}"
WAIT=1
EXTRA_TRAIN=()

usage() {
  cat <<'EOF'
Usage: run-vast.sh [OPTIONS]

  Rent a Vast.ai GPU and start training on boot (charged from instance create).

Options:
  --smoke-only          Smoke train (5 steps) after ingest
  --skip-ingest         Re-use HF cache on instance/volume
  --ingest-mode MODE    full (default) or bootstrap
  --offer-id ID         Skip search; use this offer ID
  --no-wait             Return after create (don't poll for running)
  --max-steps N         Cap training steps
  -h, --help            This help

Environment (config/cloud.env or .env):
  VAST_API_KEY          Required — https://cloud.vast.ai/manage-keys/
  HF_TOKEN              Gated HF datasets (also set as Vast template secret)
  GITHUB_TOKEN          Private repo clone (repo read token)
  CLOUD_GIT_REPO        default PyRo1121/llm-self-training
  CLOUD_GIT_REF         default main
  VAST_GPU              default H100_SXM
  VAST_DISK_GB          default 600
  VAST_IMAGE            default pytorch devel (flash-attn nvcc)
  VAST_SEARCH_QUERY     override offer search filter
  VAST_VOLUME           optional: VOLUME_ID:/mount/path
  RUN                   train run name (default pyro-coder-h100-v1)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-only) SMOKE=1; shift ;;
    --skip-ingest) export VAST_SKIP_INGEST=1; shift ;;
    --ingest-mode) export VAST_INGEST_MODE="$2"; shift 2 ;;
    --offer-id) OFFER_ID="$2"; SKIP_SEARCH=1; shift 2 ;;
    --no-wait) WAIT=0; shift ;;
    --max-steps) export CLOUD_TRAIN_MAX_STEPS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_TRAIN+=("$@"); break ;;
    *) EXTRA_TRAIN+=("$1"); shift ;;
  esac
done

if ! command -v vastai >/dev/null 2>&1; then
  echo "Install: pip install vastai  (or: uv tool install vastai)" >&2
  echo "Then: vastai set api-key \$VAST_API_KEY" >&2
  exit 1
fi

if [[ -z "${VAST_API_KEY:-}" ]]; then
  echo "VAST_API_KEY not set. Add to config/cloud.env or export it." >&2
  exit 1
fi

export VAST_API_KEY

GPU="${VAST_GPU:-H100_SXM}"
DISK="${VAST_DISK_GB:-600}"
IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}"
RUN_NAME="${RUN:-pyro-coder-h100-v1}"
REF="${CLOUD_GIT_REF:-main}"
REPO_SLUG="${CLOUD_GIT_REPO:-PyRo1121/llm-self-training}"

if [[ "${SMOKE}" -eq 1 ]]; then
  export VAST_SMOKE=1
fi

if [[ "${#EXTRA_TRAIN[@]}" -gt 0 ]]; then
  export VAST_TRAIN_EXTRA="${EXTRA_TRAIN[*]}"
fi

# Build --env for Docker (Vast passes through to container)
ENV_BLOCK="-e TZ=UTC"
ENV_BLOCK+=" -e CLOUD_GIT_REPO=${REPO_SLUG}"
ENV_BLOCK+=" -e CLOUD_GIT_REF=${REF}"
ENV_BLOCK+=" -e RUN=${RUN_NAME}"
ENV_BLOCK+=" -e PERSONAL_RATIO=${PERSONAL_RATIO:-0.75}"
ENV_BLOCK+=" -e LLM_CONFIG_PROFILE=cloud-h100"
ENV_BLOCK+=" -e HF_XET_HIGH_PERFORMANCE=1"
ENV_BLOCK+=" -e HF_XET_NUM_CONCURRENT_RANGE_GETS=24"
[[ -n "${HF_TOKEN:-}" ]] && ENV_BLOCK+=" -e HF_TOKEN=${HF_TOKEN}"
[[ -n "${GITHUB_TOKEN:-}" ]] && ENV_BLOCK+=" -e GITHUB_TOKEN=${GITHUB_TOKEN}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set — add to config/cloud.env or .env" >&2
fi
echo "=== Passing to Vast container env: HF_TOKEN=$([[ -n "${HF_TOKEN:-}" ]] && echo set || echo MISSING) VAST_API_KEY=$([[ -n "${VAST_API_KEY:-}" ]] && echo set || echo MISSING) ==="
[[ -n "${LLM_DATA_DIR:-}" ]] && ENV_BLOCK+=" -e LLM_DATA_DIR=${LLM_DATA_DIR}"
[[ -n "${CLOUD_TRAIN_MAX_STEPS:-}" ]] && ENV_BLOCK+=" -e CLOUD_TRAIN_MAX_STEPS=${CLOUD_TRAIN_MAX_STEPS}"
[[ "${VAST_SMOKE:-0}" == "1" ]] && ENV_BLOCK+=" -e VAST_SMOKE=1"
[[ "${VAST_SKIP_INGEST:-0}" == "1" ]] && ENV_BLOCK+=" -e VAST_SKIP_INGEST=1"
[[ -n "${VAST_INGEST_MODE:-}" ]] && ENV_BLOCK+=" -e VAST_INGEST_MODE=${VAST_INGEST_MODE}"

if [[ "${SKIP_SEARCH}" -eq 0 ]]; then
  QUERY="${VAST_SEARCH_QUERY:-gpu_name=${GPU} num_gpus=1 disk_space>=${DISK} verified=true rentable=true direct_port_count>=1 inet_down>=500 reliability>=0.98}"
  echo "=== Searching Vast offers ==="
  echo "    query: ${QUERY}"
  SEARCH_OUT="$(vastai search offers "${QUERY}" 2>/dev/null | head -20 || true)"
  if [[ -z "${SEARCH_OUT}" ]]; then
    echo "Search returned nothing. Set VAST_OFFER_ID or rent manually on console.vast.ai" >&2
    exit 1
  fi
  OFFER_ID="$(echo "${SEARCH_OUT}" | awk 'NR==1 {print $1}' | tr -d '#')"
  if [[ -z "${OFFER_ID}" || ! "${OFFER_ID}" =~ ^[0-9]+$ ]]; then
    echo "${SEARCH_OUT}"
    echo "Could not parse offer ID. Export VAST_OFFER_ID=<id> from search above." >&2
    exit 1
  fi
  echo "${SEARCH_OUT}" | head -3
  echo "Selected offer ID: ${OFFER_ID}"
fi

mkdir -p "${ROOT}/.vast"

echo "=== Creating instance: GPU=${GPU} disk=${DISK}GB image=${IMAGE} ==="
echo "    onstart → vast-bootstrap.sh (clone + train)"

CREATE_ARGS=(
  create instance "${OFFER_ID}"
  --image "${IMAGE}"
  --disk "${DISK}"
  --ssh --direct
  --onstart "${ROOT}/scripts/cloud/vast-bootstrap.sh"
  --env "${ENV_BLOCK}"
)

if [[ -n "${VAST_VOLUME:-}" ]]; then
  CREATE_ARGS+=(--mount "${VAST_VOLUME}")
fi

CREATE_JSON="$(vastai "${CREATE_ARGS[@]}" 2>&1)" || {
  echo "${CREATE_JSON}" >&2
  exit 1
}

echo "${CREATE_JSON}"

INSTANCE_ID="$(echo "${CREATE_JSON}" | python3 -c "
import json, sys, re
text = sys.stdin.read()
try:
    d = json.loads(text)
    print(d.get('new_contract') or d.get('id') or '')
except json.JSONDecodeError:
    m = re.search(r'\"new_contract\"\\s*:\\s*(\\d+)', text)
    print(m.group(1) if m else '')
" 2>/dev/null)" || INSTANCE_ID=""

if [[ -n "${INSTANCE_ID}" ]]; then
  echo "${INSTANCE_ID}" > "${ROOT}/.vast/last-instance-id"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${ROOT}/.vast/last-instance-created"
fi

echo ""
echo "Instance ID: ${INSTANCE_ID:-unknown}"
echo "Track: cat .vast/last-instance-id"
echo ""
echo "Pull artifacts when done:"
echo "  make cloud-vast-pull INSTANCE=${INSTANCE_ID:-<id>}"
echo "  make cloud-vast-destroy INSTANCE=${INSTANCE_ID:-<id>}"

if [[ "${WAIT}" -eq 1 && -n "${INSTANCE_ID}" ]]; then
  echo ""
  echo "=== Waiting for instance running (poll 15s) ==="
  for _ in $(seq 1 40); do
    STATUS="$(vastai show instance "${INSTANCE_ID}" --raw 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('actual_status') or d.get('status') or 'unknown')
except Exception:
    print('unknown')
" 2>/dev/null || echo unknown)"
    echo "  status: ${STATUS}"
    if [[ "${STATUS}" == "running" ]]; then
      echo ""
      vastai ssh-url "${INSTANCE_ID}" 2>/dev/null || true
      echo ""
      echo "Tail logs: vastai ssh-url ${INSTANCE_ID}  then  tail -f /var/log/onstart*.log  (or ssh and watch process)"
      break
    fi
    if [[ "${STATUS}" == "exited" || "${STATUS}" == "offline" ]]; then
      echo "Instance failed to start (${STATUS}). Destroy and retry with another offer." >&2
      exit 1
    fi
    sleep 15
  done
fi
