"""Generate a 50-row secrets+PII audit sample from curated JSONL (Phase 1 exit)."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_dataprep.filters import scan_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sample of curated rows for secrets/PII")
    parser.add_argument("--curated", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--use-presidio", action="store_true")
    parser.add_argument("--gitleaks", action="store_true")
    args = parser.parse_args()

    by_harness: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.curated.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            h = row.get("meta", {}).get("harness") or "unknown"
            by_harness[h].append(row)

    rng = random.Random(args.seed)
    pool: list[dict[str, Any]] = []
    harnesses = sorted(by_harness.keys())
    per = max(1, args.n // max(1, len(harnesses)))
    for h in harnesses:
        rows = by_harness[h]
        pool.extend(rng.sample(rows, min(per, len(rows))))
    if len(pool) < args.n:
        rest = [r for h in harnesses for r in by_harness[h] if r not in pool]
        pool.extend(rng.sample(rest, min(args.n - len(pool), len(rest))))

    pool = pool[: args.n]
    findings_count = 0
    report_rows: list[dict[str, Any]] = []

    for i, row in enumerate(pool):
        text = "\n\n".join(m.get("content", "") for m in row.get("messages", []))
        safety = scan_text(
            text,
            use_gitleaks=args.gitleaks,
            use_presidio=args.use_presidio,
        )
        if not safety.ok:
            findings_count += 1
        report_rows.append(
            {
                "audit_index": i,
                "harness": row.get("meta", {}).get("harness"),
                "session_id": row.get("meta", {}).get("session_id"),
                "chunk_index": row.get("meta", {}).get("chunk_index"),
                "safety": safety.to_dict(),
                "preview": text[:400],
            }
        )

    out_dir = args.out_dir or Path("docs/audits")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    jsonl_path = out_dir / f"phase1-audit-{stamp}.jsonl"
    md_path = out_dir / f"phase1-audit-{stamp}.md"

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in report_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Phase 1 audit sample ({stamp})\n\n")
        fh.write(f"Rows: **{len(report_rows)}** · Flagged: **{findings_count}**\n\n")
        fh.write("| # | harness | session | ok | findings |\n")
        fh.write("|---|---------|---------|----|---------|\n")
        for r in report_rows:
            ok = r["safety"]["ok"]
            n_find = len(r["safety"].get("findings") or [])
            sid = str(r.get("session_id") or "")[:12]
            fh.write(
                f"| {r['audit_index']} | {r.get('harness')} | {sid} | {ok} | {n_find} |\n"
            )
        fh.write("\nOperator: manually review flagged rows before Phase 2 train.\n")

    print(f"Audit sample → {jsonl_path} ({findings_count}/{len(report_rows)} flagged)")
    print(f"Summary → {md_path}")


if __name__ == "__main__":
    main()
