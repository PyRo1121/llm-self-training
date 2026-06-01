#!/usr/bin/env bash
# Cloud GPU full pipeline: personal → public ingest → train → export (Jarvis / Vast / SSH).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export LLM_CONFIG_PROFILE="${LLM_CONFIG_PROFILE:-cloud-h100}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-24}"

RUN="pyro-coder-h100-v1"
PERSONAL_DATASET=""
DATA_REPO_URL="${CLOUD_DATA_REPO_URL:-}"
PERSONAL_RATIO="0.75"
MANIFEST="personal-first"
SKIP_INGEST=0
SKIP_TRAIN=0
SMOKE_ONLY=0
INGEST_MODE="full"
HF_TOKEN_FILE="/home/hf_token"
MAX_STEPS="${CLOUD_TRAIN_MAX_STEPS:-}"

usage() {
  sed -n '2,20p' "$0"
  echo "Options:"
  echo "  --run NAME              Run name (default: pyro-coder-h100-v1)"
  echo "  --personal-dataset REPO   HF private dataset (optional if data in repo)"
  echo "  CLOUD_DATA_REPO_URL=…     Clone private git repo into data/cloud/personal/"
  echo "  --personal-ratio F        Manifest personal ratio (default: 0.75)"
  echo "  --skip-ingest             Re-use cached public HF data on disk"
  echo "  --skip-train              Ingest + manifest only"
  echo "  --smoke-only              Smoke train (5 steps) then exit"
  echo "  --ingest-mode full|bootstrap   full=all enabled datasets (default)"
  echo "  --hf-token-file PATH      HF token file on instance (default: /home/hf_token)"
  echo "  --max-steps N             Cap training steps (else full epoch, no cap on cloud)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN="$2"; shift 2 ;;
    --personal-dataset) PERSONAL_DATASET="$2"; shift 2 ;;
    --personal-ratio) PERSONAL_RATIO="$2"; shift 2 ;;
    --skip-ingest) SKIP_INGEST=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --smoke-only) SMOKE_ONLY=1; shift ;;
    --ingest-mode) INGEST_MODE="$2"; shift 2 ;;
    --hf-token-file) HF_TOKEN_FILE="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

UV=(uv run --package)
mkdir -p data/raw data/curated data/train data/warehouse runs exports

if [[ -f "${HF_TOKEN_FILE}" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "${HF_TOKEN_FILE}")"
elif [[ -n "${HF_TOKEN:-}" ]]; then
  :
elif [[ -f "${ROOT}/config/cloud.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/config/cloud.env"
  set +a
elif [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set — gated public datasets will be skipped" >&2
fi

echo "=== [1/11] Environment ==="
echo "ROOT=${ROOT}"
echo "LLM_DATA_DIR=${LLM_DATA_DIR:-${ROOT}/data}"
echo "LLM_CONFIG_PROFILE=${LLM_CONFIG_PROFILE}"
echo "RUN=${RUN} PERSONAL_RATIO=${PERSONAL_RATIO} INGEST_MODE=${INGEST_MODE}"

echo "=== [2/11] Sync deps ==="
make sync-all

if [[ "${SKIP_FLASH_ATTN:-0}" != "1" ]]; then
  echo "=== [3/11] flash-attn ==="
  MAX_JOBS="${MAX_JOBS:-8}" bash scripts/install-flash-attn.sh || {
    echo "WARNING: flash-attn failed — continuing (seq may cap lower)" >&2
  }
else
  echo "=== [3/11] flash-attn skipped (SKIP_FLASH_ATTN=1) ==="
fi

REPO_PERSONAL="${ROOT}/data/cloud/personal/personal-tier1.jsonl"
PERSONAL_IMPORT="${ROOT}/data/curated/imported-personal.jsonl"

_import_personal_bundle() {
  local label="$1"
  shift
  echo "=== [4/11] Personal data: ${label} ==="
  : > "${PERSONAL_IMPORT}"
  local part n=0
  for part in "$@"; do
    if [[ -f "${part}" ]]; then
      cat "${part}" >> "${PERSONAL_IMPORT}"
      n=$((n + $(wc -l < "${part}" | tr -d ' ')))
    fi
  done
  echo "Personal rows merged: ${n} → ${PERSONAL_IMPORT}"
}

if [[ -f "${REPO_PERSONAL}" ]]; then
  _import_personal_bundle "repo personal-tier1.jsonl" "${REPO_PERSONAL}"
elif compgen -G "${ROOT}/data/cloud/personal/harnesses/"'*.jsonl' > /dev/null; then
  PARTS=()
  while IFS= read -r -d '' f; do PARTS+=("$f"); done < <(
    find "${ROOT}/data/cloud/personal/harnesses" -name '*.jsonl' -type f -print0 2>/dev/null | sort -z
  )
  _import_personal_bundle "harness shards (${#PARTS[@]} files)" "${PARTS[@]}"
elif [[ -n "${DATA_REPO_URL}" ]]; then
  echo "=== [4/11] Clone private data repo: ${DATA_REPO_URL} ==="
  rm -rf data/cloud/personal/_clone
  git clone --depth 1 "${DATA_REPO_URL}" data/cloud/personal/_clone
  PERSONAL_FILE="$(find data/cloud/personal/_clone -name 'personal-tier1.jsonl' -o -name '*.jsonl' | head -1)"
  if [[ -z "${PERSONAL_FILE}" || ! -f "${PERSONAL_FILE}" ]]; then
    echo "No JSONL found in ${DATA_REPO_URL}" >&2
    exit 1
  fi
  cp "${PERSONAL_FILE}" data/curated/imported-personal.jsonl
  echo "Personal rows file: data/curated/imported-personal.jsonl"
elif [[ -n "${PERSONAL_DATASET}" ]]; then
  echo "=== [4/11] Download personal bundle: ${PERSONAL_DATASET} ==="
  PERSONAL_DIR="$(mktemp -d)"
  uv run huggingface-cli download "${PERSONAL_DATASET}" --repo-type dataset --local-dir "${PERSONAL_DIR}" \
    || uv run hf download "${PERSONAL_DATASET}" --repo-type dataset --local-dir "${PERSONAL_DIR}"
  PERSONAL_FILE="${PERSONAL_DIR}/personal-tier1.jsonl"
  if [[ ! -f "${PERSONAL_FILE}" ]]; then
    PERSONAL_FILE="$(find "${PERSONAL_DIR}" -name '*.jsonl' -type f | head -1)"
  fi
  if [[ -z "${PERSONAL_FILE}" || ! -f "${PERSONAL_FILE}" ]]; then
    echo "No personal JSONL in HF dataset ${PERSONAL_DATASET}" >&2
    exit 1
  fi
  cp "${PERSONAL_FILE}" data/curated/imported-personal.jsonl
  echo "Personal rows file: data/curated/imported-personal.jsonl"
else
  echo "=== [4/11] No personal source (add data/cloud/personal/personal-tier1.jsonl or --personal-dataset) ==="
fi

if [[ "${SKIP_INGEST}" -eq 0 ]]; then
  echo "=== [5/11] Public HF ingest (${INGEST_MODE}) ==="
  if [[ "${INGEST_MODE}" == "bootstrap" ]]; then
    make public-ingest PUBLIC_DATASETS="cooper_qwen9b_coop_claude,swe_next,nemotron_opencode,agentic_sft_new"
  else
    make public-ingest
  fi
else
  echo "=== [5/11] Public ingest skipped (--skip-ingest) ==="
fi

echo "=== [6/11] Curate public raw (fast bulk) ==="
make curate-fast

echo "=== [7/11] Warehouse index ==="
CURATED_FILES=(data/curated/*.jsonl)
if [[ ! -e "${CURATED_FILES[0]}" ]]; then
  echo "No curated JSONL after curate" >&2
  exit 1
fi
"${UV[@]}" llm-dataprep warehouse-load --tier 1 "${CURATED_FILES[@]}"

echo "=== [8/11] Training manifest ${PERSONAL_RATIO} personal ==="
make manifest-mixed MANIFEST="${MANIFEST}" PERSONAL_RATIO="${PERSONAL_RATIO}"
make extract MANIFEST="${MANIFEST}" TRAIN_FILE="data/train/${MANIFEST}.jsonl"

TRAIN_ROWS="$(wc -l < "data/train/${MANIFEST}.jsonl" | tr -d ' ')"
echo "Train file rows: ${TRAIN_ROWS}"
if [[ "${TRAIN_ROWS}" -lt 200 ]]; then
  echo "WARNING: <200 train rows — check personal bundle + public ingest" >&2
fi

STEPS_EST=$((TRAIN_ROWS / 16))
if [[ "${STEPS_EST}" -gt 500000 ]]; then
  echo "WARNING: ~${STEPS_EST} steps at eff_batch=16 for one epoch."
  echo "  Set CLOUD_TRAIN_MAX_STEPS or --max-steps to cap, or expect multi-day run."
  echo "  Tier-1 curation usually shrinks raw 20M+ → much smaller train set."
fi

echo "=== [9/11] Preflight ==="
"${UV[@]}" llm-train train-preflight --promote

if [[ "${SKIP_TRAIN}" -eq 1 ]]; then
  echo "=== Skip train (--skip-train). Manifest ready: data/train/${MANIFEST}.jsonl ==="
  exit 0
fi

TRAIN_FLAGS=(--cloud --run-name "${RUN}" --train-file "data/train/${MANIFEST}.jsonl")
if [[ -n "${MAX_STEPS}" ]]; then
  TRAIN_FLAGS+=(--max-steps "${MAX_STEPS}")
fi

if [[ "${SMOKE_ONLY}" -eq 1 ]]; then
  echo "=== [10/11] Smoke train ==="
  "${UV[@]}" llm-train train-qlora --smoke --cloud --run-name "smoke-${RUN}" \
    --train-file "data/train/${MANIFEST}.jsonl"
  echo "Smoke OK."
  exit 0
fi

echo "=== [10/11] Smoke gate ==="
"${UV[@]}" llm-train train-qlora --smoke --cloud --run-name "smoke-${RUN}" \
  --train-file "data/train/${MANIFEST}.jsonl"

echo "=== [11/11] Full promote train + export ==="
"${UV[@]}" llm-train train-qlora "${TRAIN_FLAGS[@]}"
"${UV[@]}" llm-train train-export \
  --adapter-dir "runs/${RUN}/adapter" \
  --out "exports/${RUN}" --unsloth

echo ""
echo "=== DONE ==="
echo "  Adapter: runs/${RUN}/adapter"
echo "  Export:  exports/${RUN}/"
echo "  Download: make cloud-vast-pull INSTANCE=<id>  (Vast)"
echo "            jl download <machine_id> runs/${RUN} ./runs/${RUN} -r  (Jarvis)"
