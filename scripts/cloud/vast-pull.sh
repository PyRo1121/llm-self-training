#!/usr/bin/env bash
# Pull train artifacts from a Vast instance to local runs/ and exports/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

INSTANCE="${INSTANCE:-${VAST_INSTANCE:-}}"
RUN="${RUN:-pyro-coder-h100-v1}"
REMOTE="${VAST_REMOTE_ROOT:-/workspace/llm-self-training}"

if [[ -z "${INSTANCE}" && -f "${ROOT}/.vast/last-instance-id" ]]; then
  INSTANCE="$(tr -d '[:space:]' < "${ROOT}/.vast/last-instance-id")"
fi

if [[ -z "${INSTANCE}" ]]; then
  echo "Usage: INSTANCE=<id> make cloud-vast-pull  (or set .vast/last-instance-id)" >&2
  exit 1
fi

if ! command -v vastai >/dev/null 2>&1; then
  echo "pip install vastai" >&2
  exit 1
fi

mkdir -p runs exports

echo "=== Pull runs/${RUN} ==="
vastai copy "${INSTANCE}:${REMOTE}/runs/${RUN}" "./runs/${RUN}" -r 2>/dev/null \
  || echo "WARN: runs copy failed (train may still be running)"

echo "=== Pull exports/${RUN} ==="
vastai copy "${INSTANCE}:${REMOTE}/exports/${RUN}" "./exports/${RUN}" -r 2>/dev/null \
  || echo "WARN: exports copy failed (export may not exist yet)"

echo "Done. Local: runs/${RUN}  exports/${RUN}"
