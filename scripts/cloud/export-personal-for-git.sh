#!/usr/bin/env bash
# Export tier-1 personal rows for git or private data repo (NOT for public GitHub).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT="${ROOT}/data/cloud/personal/personal-tier1.jsonl"
mkdir -p "$(dirname "${OUT}")"

latest="$(ls -t data/curated/curated-*.jsonl 2>/dev/null | head -1 || true)"
if [[ -z "${latest}" ]]; then
  echo "No curated file — run: make phase1 && make curate" >&2
  exit 1
fi

echo "=== Export personal tier-1 from ${latest} ==="
python3 - <<'PY' "${latest}" "${OUT}"
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
        meta = row.get("meta") or {}
        tier = int(meta.get("train_tier", 0))
        if tier < 1:
            continue
        harness = str(meta.get("harness") or "")
        data_source = meta.get("data_source") or (
            "public" if harness.startswith("public_") else "personal"
        )
        if data_source == "public":
            continue
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
print(f"Wrote {n} personal tier-1 rows → {dst}")
PY

du -h "${OUT}"
