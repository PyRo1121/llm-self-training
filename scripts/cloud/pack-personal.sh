#!/usr/bin/env bash
# Local: pack tier-1 curated personal data → private Hugging Face dataset.
# Usage:
#   ./scripts/cloud/pack-personal.sh
#   HF_DATASET=PyRo1121/pyro-coder-personal-bundle ./scripts/cloud/pack-personal.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

HF_DATASET="${HF_DATASET:-PyRo1121/pyro-coder-personal-bundle}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
OUT_DIR="${ROOT}/.cloud-staging"
BUNDLE="${OUT_DIR}/personal-tier1-${STAMP}.jsonl"

mkdir -p "${OUT_DIR}"

latest="$(ls -t data/curated/curated-*.jsonl 2>/dev/null | head -1 || true)"
if [[ -z "${latest}" ]]; then
  echo "No data/curated/curated-*.jsonl — run: make phase1 && make curate" >&2
  exit 1
fi

echo "=== Filtering tier-1 from ${latest} ==="
python3 - <<'PY' "${latest}" "${BUNDLE}"
import json, sys
src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(src, encoding="utf-8", errors="replace") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        tier = int((row.get("meta") or {}).get("train_tier", 0))
        if tier < 1:
            continue
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
print(f"tier-1 rows: {n} → {dst}")
if n < 200:
    print("WARNING: <200 tier-1 rows — consider more phase1 ingest before cloud train", file=sys.stderr)
PY

SIZE="$(du -h "${BUNDLE}" | awk '{print $1}')"
echo "Bundle size: ${SIZE}"

if [[ "${UPLOAD:-1}" == "0" ]]; then
  echo "Skip upload (UPLOAD=0). Bundle: ${BUNDLE}"
  exit 0
fi

echo "=== Uploading to ${HF_DATASET} (private dataset) ==="
uv run huggingface-cli upload "${HF_DATASET}" "${BUNDLE}" "personal-tier1.jsonl" \
  --repo-type dataset --private \
  || uv run hf upload "${HF_DATASET}" "${BUNDLE}" "personal-tier1.jsonl" --repo-type dataset --private

echo "Done. Cloud train: --personal-dataset ${HF_DATASET}"
