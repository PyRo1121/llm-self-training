#!/usr/bin/env bash
# Vast.ai onstart entry (uploaded by run-vast.sh). Clones repo → train immediately.
# Secrets: set HF_TOKEN (+ GITHUB_TOKEN for private clone) in Vast env or config/cloud.env.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PATH="${HOME}/.local/bin:/usr/local/bin:${PATH}"

log() { echo "[vast-bootstrap $(date -u +%H:%M:%S)] $*" >&2; }

log "boot — charge clock starts; minimizing idle time"

# Minimal OS deps (pytorch devel images usually have build-essential + git)
if ! command -v git >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq git curl ca-certificates
fi

if ! command -v uv >/dev/null 2>&1; then
  log "install uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

REPO_SLUG="${CLOUD_GIT_REPO:-PyRo1121/llm-self-training}"
REF="${CLOUD_GIT_REF:-main}"
WORKDIR="${CLOUD_WORKDIR:-/workspace/llm-self-training}"

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  CLONE_URL="https://${GITHUB_TOKEN}@github.com/${REPO_SLUG}.git"
elif [[ -n "${CLOUD_GIT_URL:-}" ]]; then
  CLONE_URL="${CLOUD_GIT_URL}"
else
  CLONE_URL="https://github.com/${REPO_SLUG}.git"
fi

rm -rf "${WORKDIR}"
log "clone ${REPO_SLUG}@${REF}"
git clone --depth 1 --branch "${REF}" "${CLONE_URL}" "${WORKDIR}"
cd "${WORKDIR}"

# Secrets from committed config/cloud.env (public/private repo) + Vast instance env
if [[ -f "${WORKDIR}/config/cloud.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${WORKDIR}/config/cloud.env"
  set +a
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  printf '%s' "${HF_TOKEN}" > /home/hf_token
  chmod 600 /home/hf_token
  export HF_TOKEN
  log "HF_TOKEN ready (/home/hf_token + env)"
else
  log "WARNING: HF_TOKEN missing — gated HF datasets will skip"
fi

if [[ -n "${LLM_DATA_DIR:-}" ]]; then
  mkdir -p "${LLM_DATA_DIR}"
fi

log "handoff → vast-onstart.sh"
exec bash scripts/cloud/vast-onstart.sh
