#!/usr/bin/env bash
# Ingest → gitleaks+Presidio scan → curate public HF datasets smallest→largest.
# Checkpoints per dataset; final --latest-per-prefix curate dedupes the full lake.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-24}"
# Parallel scan/curate — tune up if CPU/RAM headroom (14+ GB free typical).
export SCAN_WORKERS="${SCAN_WORKERS:-8}"
export CURATE_WORKERS="${CURATE_WORKERS:-8}"
export PRESIDIO_N_PROCESS="${PRESIDIO_N_PROCESS:-4}"
export PRESIDIO_SESSION_BATCH="${PRESIDIO_SESSION_BATCH:-128}"
export HF_SNAPSHOT_MAX_WORKERS="${HF_SNAPSHOT_MAX_WORKERS:-2}"

if [[ -f "${ROOT}/config/cloud.env" ]]; then
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

LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}" data/raw data/curated data/warehouse
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG_DIR}/parse-public-${STAMP}.log"
PROGRESS="${LOG_DIR}/parse-public-progress.jsonl"
PIDFILE="${LOG_DIR}/parse-public.pid"

# Enabled datasets — must match config/default.yaml public_datasets.*.enabled (true-data policy).
DATASETS=(
  swe_chat                    # wild Cursor / Claude Code (gated); only default public ingest
)

UV=(uv run --package llm-dataprep)

usage() {
  sed -n '2,5p' "$0"
  echo "Env: HF_TOKEN, FORCE_RECONVERT=0|1, PUBLIC_REFRESH=1"
  echo "Perf: SCAN_WORKERS CURATE_WORKERS PRESIDIO_N_PROCESS HF_SNAPSHOT_MAX_WORKERS"
  echo "Log:  logs/parse-public-*.log"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

echo "$$" > "${PIDFILE}"

log_progress() {
  local msg="$*"
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${msg}" | tee -a "${PROGRESS}"
}

id_to_prefix() {
  echo "public-${1//_/-}"
}

id_to_glob() {
  echo "$(id_to_prefix "$1")*.jsonl"
}

force_reconvert() {
  local id="$1"
  local prefix
  prefix="$(id_to_prefix "${id}")"
  rm -f "data/hf_cache/${id}/.ingest_state.json"
  find data/raw -maxdepth 1 -name "${prefix}-*.jsonl" -delete 2>/dev/null || true
  echo "force reconvert: cleared ingest state + raw for ${id}"
}

exec > >(tee -a "${LOG}") 2>&1

echo "=== parse-public-ordered ==="
echo "ROOT=${ROOT}"
echo "LOG=${LOG}"
echo "HF_TOKEN=${HF_TOKEN:+set}"
echo "SCAN_WORKERS=${SCAN_WORKERS}"
echo "CURATE_WORKERS=${CURATE_WORKERS}"
echo "PRESIDIO_N_PROCESS=${PRESIDIO_N_PROCESS}"
echo "HF_SNAPSHOT_MAX_WORKERS=${HF_SNAPSHOT_MAX_WORKERS}"
echo "FORCE_RECONVERT=${FORCE_RECONVERT:-0}"
echo "PUBLIC_REFRESH=${PUBLIC_REFRESH:-0}"
log_progress "START pipeline datasets=${#DATASETS[@]}"

echo "=== sync deps (dataprep full extra) ==="
uv sync --package llm-dataprep --extra full

for id in "${DATASETS[@]}"; do
  glob="$(id_to_glob "${id}")"
  log_progress "START ${id}"

  if [[ "${FORCE_RECONVERT:-0}" == "1" ]]; then
    force_reconvert "${id}"
  fi

  echo "--- [${id}] public-ingest ---"
  make public-ingest \
    PUBLIC_DATASETS="${id}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    $(if [[ "${PUBLIC_REFRESH:-0}" == "1" ]]; then echo "PUBLIC_REFRESH=1"; fi)

  echo "--- [${id}] scan-raw (gitleaks per-file + Presidio, ${SCAN_WORKERS} workers) ---"
  "${UV[@]}" scan-raw --glob "${glob}" --gitleaks --gitleaks-per-file --workers "${SCAN_WORKERS}"

  echo "--- [${id}] curate-raw (batch Presidio, ${CURATE_WORKERS} workers, no session gitleaks) ---"
  "${UV[@]}" curate-raw \
    --glob "${glob}" \
    --out-suffix "${id}" \
    --honor-safety-failures \
    --workers "${CURATE_WORKERS}"

  log_progress "DONE ${id}"
done

echo "=== final curate: all raw, latest-per-prefix, dedupe (${CURATE_WORKERS} workers) ==="
"${UV[@]}" curate-raw --latest-per-prefix --honor-safety-failures --workers "${CURATE_WORKERS}"

echo "=== warehouse-load ==="
make warehouse-load

echo "=== lake-stats (curated only) ==="
make lake-stats

log_progress "COMPLETE"
echo "Done. Log: ${LOG}"
