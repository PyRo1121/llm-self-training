#!/usr/bin/env bash
# Export tier-1 personal rows: merged file + per-harness shards for cloud clone.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT="${ROOT}/data/cloud/personal/personal-tier1.jsonl"
HARNESS_DIR="${ROOT}/data/cloud/personal/harnesses"
MANIFEST="${ROOT}/data/cloud/personal/manifest.json"
mkdir -p "$(dirname "${OUT}")" "${HARNESS_DIR}"

latest="$(ls -t data/curated/curated-*.jsonl 2>/dev/null | head -1 || true)"
if [[ -z "${latest}" ]]; then
  echo "No curated file — run: make ingest && make curate" >&2
  exit 1
fi

echo "=== Export personal tier-1 from ${latest} ==="
python3 - <<'PY' "${latest}" "${OUT}" "${HARNESS_DIR}" "${MANIFEST}"
import json, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

src, dst, harness_dir, manifest_path = sys.argv[1:5]
harness_dir = Path(harness_dir)
by_harness: dict[str, list[str]] = defaultdict(list)
merged = 0

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
        harness = str(meta.get("harness") or meta.get("source") or "unknown")
        data_source = meta.get("data_source") or (
            "public" if harness.startswith("public_") else "personal"
        )
        if data_source == "public":
            continue
        blob = json.dumps(row, ensure_ascii=False) + "\n"
        fout.write(blob)
        by_harness[harness].append(blob)
        merged += 1

for harness, lines in sorted(by_harness.items()):
    safe = harness.replace("/", "_").replace(" ", "_")
    path = harness_dir / f"{safe}.jsonl"
    path.write_text("".join(lines), encoding="utf-8")

manifest = {
    "exported_at": datetime.now(timezone.utc).isoformat(),
    "source_curated": src,
    "total_tier1_personal_rows": merged,
    "harnesses": {
        h: {"rows": len(lines), "file": f"harnesses/{h.replace('/', '_').replace(' ', '_')}.jsonl"}
        for h, lines in sorted(by_harness.items())
    },
}
Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

# Redact secrets before git push (GitHub push protection)
import re
_hf = re.compile(r"hf_[A-Za-z0-9]{20,}")
_gh = re.compile(r"gho_[A-Za-z0-9]{20,}")
for p in [Path(dst), *harness_dir.glob("*.jsonl")]:
    text = p.read_text(encoding="utf-8")
    red = _hf.sub("[REDACTED_HF_TOKEN]", _gh.sub("[REDACTED_GITHUB_TOKEN]", text))
    if red != text:
        p.write_text(red, encoding="utf-8")
        print(f"redacted secrets in {p}")
print(f"Wrote {merged} personal tier-1 rows → {dst}")
print(f"Harness shards: {len(by_harness)} under {harness_dir}")
for h, lines in sorted(by_harness.items(), key=lambda x: -len(x[1])):
    print(f"  {h}: {len(lines)}")
PY

du -h "${OUT}" "${HARNESS_DIR}" 2>/dev/null || du -h "${OUT}"
