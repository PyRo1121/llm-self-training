#!/usr/bin/env bash
# Destroy a Vast instance (stop billing).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

INSTANCE="${INSTANCE:-${VAST_INSTANCE:-}}"
if [[ -z "${INSTANCE}" && -f "${ROOT}/.vast/last-instance-id" ]]; then
  INSTANCE="$(tr -d '[:space:]' < "${ROOT}/.vast/last-instance-id")"
fi

if [[ -z "${INSTANCE}" ]]; then
  echo "Usage: INSTANCE=<id> make cloud-vast-destroy" >&2
  exit 1
fi

echo "Destroying instance ${INSTANCE}..."
vastai destroy instance "${INSTANCE}"
echo "Destroyed."
