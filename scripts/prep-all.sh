#!/usr/bin/env bash
# One command: personal ingest → scan → public parse → train JSONL.
# HF_TOKEN loaded from config/cloud.env or .env — no manual exports needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-24}"
export SCAN_WORKERS="${SCAN_WORKERS:-10}"
export CURATE_WORKERS="${CURATE_WORKERS:-10}"
export PERSONAL_SCAN_WORKERS="${PERSONAL_SCAN_WORKERS:-6}"
export PRESIDIO_N_PROCESS="${PRESIDIO_N_PROCESS:-8}"
export PRESIDIO_BATCH_SIZE="${PRESIDIO_BATCH_SIZE:-256}"
export PRESIDIO_SESSION_BATCH="${PRESIDIO_SESSION_BATCH:-256}"
export HF_SNAPSHOT_MAX_WORKERS="${HF_SNAPSHOT_MAX_WORKERS:-2}"
export FORCE_RECONVERT="${FORCE_RECONVERT:-0}"

REPO=""
INCLUDE_SUBAGENTS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --no-subagents) INCLUDE_SUBAGENTS=0; shift ;;
    -h|--help)
      sed -n '2,4p' "$0"
      echo "Options: --repo PATH  --no-subagents"
      echo "Override: FORCE_RECONVERT=1 PUBLIC_REFRESH=1 SCAN_WORKERS=… PERSONAL_SCAN_WORKERS=…"
      exit 0
      ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

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
mkdir -p "${LOG_DIR}" data/raw data/curated data/train data/warehouse
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG_DIR}/prep-${STAMP}.log"

echo "=== prep-all ==="
echo "LOG=${LOG} (use tee manually if you want a duplicate on disk)"
echo "HF_TOKEN=${HF_TOKEN:+set}"
echo "SCAN_WORKERS=${SCAN_WORKERS} CURATE_WORKERS=${CURATE_WORKERS} PERSONAL_SCAN_WORKERS=${PERSONAL_SCAN_WORKERS}"
echo "PRESIDIO_N_PROCESS=${PRESIDIO_N_PROCESS} PRESIDIO_BATCH_SIZE=${PRESIDIO_BATCH_SIZE} PRESIDIO_SESSION_BATCH=${PRESIDIO_SESSION_BATCH}"
echo "FORCE_RECONVERT=${FORCE_RECONVERT}"

echo ""
echo "=== [1/4] sync deps ==="
uv sync \
  --package llm-core \
  --package llm-train \
  --package llm-eval \
  --package llm-api \
  --package llm-dataprep \
  --extra full

echo ""
echo "=== [2/4] personal agent ingest ==="
ingest_args=()
[[ "${INCLUDE_SUBAGENTS}" == "1" ]] && ingest_args+=(--include-subagents)
[[ -n "${REPO}" ]] && ingest_args+=(--repo "${REPO}")
uv run --package llm-dataprep agent-ingest "${ingest_args[@]}"

echo ""
echo "=== [3/4] scan personal raw (non-public) ==="
UV=(uv run --package llm-dataprep)

cleanup_gitleaks_scratch() {
  rm -rf "${ROOT}/data/.gitleaks-scratch" /tmp/gitleaks-rows-* 2>/dev/null || true
}

cleanup_gitleaks_scratch
safety_ver="$("${UV[@]}" python -c "from llm_dataprep.safety_policy import safety_policy_version; print(safety_policy_version())")"
echo "safety_policy_version=${safety_ver}"

personal_files=()
shopt -s nullglob
for f in data/raw/*.jsonl; do
  base="$(basename "$f")"
  [[ "${base}" == safety-failures* ]] && continue
  [[ "${base}" == public-* ]] && continue
  personal_files+=("$f")
done
shopt -u nullglob
if ((${#personal_files[@]} > 0)); then
  echo "Warming Presidio/spaCy (PRESIDIO_N_PROCESS=${PRESIDIO_N_PROCESS})..."
  "${UV[@]}" python -c "
from llm_dataprep.perf import presidio_n_process
from presidio_analyzer import AnalyzerEngine
AnalyzerEngine()
print(f'presidio ok (n_process={presidio_n_process()})')
"
  echo "Scanning ${#personal_files[@]} personal file(s) with ${PERSONAL_SCAN_WORKERS} workers"
  "${UV[@]}" scan-raw \
    --gitleaks --gitleaks-per-file \
    --workers "${PERSONAL_SCAN_WORKERS}" \
    --files "${personal_files[@]}"
  cleanup_gitleaks_scratch
  echo "gitleaks scratch cleaned"
else
  echo "No personal raw files yet — skipping personal scan"
fi

echo ""
echo "=== [4/4] public datasets + final curate + warehouse ==="
bash scripts/parse-public-ordered.sh

echo ""
echo "=== training manifest + extract ==="
make prepare-mixed

echo ""
echo "=== prep complete ==="
echo "Train file: data/train/personal-first.jsonl"
echo "Next:       make train"
echo "Cloud GPU:  make train-cloud"
echo "Log:        ${LOG}"
