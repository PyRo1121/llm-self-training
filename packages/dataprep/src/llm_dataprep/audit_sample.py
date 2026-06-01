"""Generate a 50-row secrets+PII audit sample from curated JSONL (Phase 1 exit)."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_dataprep.filters import SafetyFinding, scan_text
from llm_dataprep.safety_policy import (
    Severity,
    classify_finding,
    findings_to_dicts,
    load_safety_policy,
)


def _safety_policy_report(text: str, *, use_gitleaks: bool, use_presidio: bool) -> dict[str, Any]:
    pol = load_safety_policy()
    report = scan_text(text, use_gitleaks=use_gitleaks, use_presidio=use_presidio)
    block: list[SafetyFinding] = []
    warn: list[SafetyFinding] = []
    for finding in report.findings:
        sev = classify_finding(finding, pol)
        if sev == Severity.BLOCK:
            block.append(finding)
        elif sev == Severity.WARN:
            warn.append(finding)
    sevs = [Severity.BLOCK] * len(block) + [Severity.WARN] * len(warn)
    return {
        "ok": report.ok,
        "block_count": len(block),
        "warn_count": len(warn),
        "findings": findings_to_dicts(block + warn, sevs),
    }


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
    rows_quarantine = 0
    rows_with_block = 0
    rows_with_warn = 0
    total_block_findings = 0
    total_warn_findings = 0
    report_rows: list[dict[str, Any]] = []

    for i, row in enumerate(pool):
        text = "\n\n".join(m.get("content", "") for m in row.get("messages", []))
        safety = _safety_policy_report(
            text,
            use_gitleaks=args.gitleaks,
            use_presidio=args.use_presidio,
        )
        n_block = int(safety["block_count"])
        n_warn = int(safety["warn_count"])
        total_block_findings += n_block
        total_warn_findings += n_warn
        if n_block:
            rows_with_block += 1
        if n_warn:
            rows_with_warn += 1
        if not safety["ok"]:
            rows_quarantine += 1
        report_rows.append(
            {
                "audit_index": i,
                "harness": row.get("meta", {}).get("harness"),
                "session_id": row.get("meta", {}).get("session_id"),
                "chunk_index": row.get("meta", {}).get("chunk_index"),
                "safety": safety,
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
        fh.write(f"Rows sampled: **{len(report_rows)}**\n\n")
        fh.write("## Safety policy summary\n\n")
        fh.write("| metric | count |\n")
        fh.write("|--------|------:|\n")
        fh.write(f"| rows quarantine (policy) | {rows_quarantine} |\n")
        fh.write(f"| rows with ≥1 block finding | {rows_with_block} |\n")
        fh.write(f"| rows with ≥1 warn finding | {rows_with_warn} |\n")
        fh.write(f"| total block findings | {total_block_findings} |\n")
        fh.write(f"| total warn findings | {total_warn_findings} |\n")
        fh.write("\n## Per-row sample\n\n")
        fh.write("| # | harness | session | ok | block | warn |\n")
        fh.write("|---|---------|---------|----|------:|-----:|\n")
        for r in report_rows:
            ok = r["safety"]["ok"]
            n_block = r["safety"]["block_count"]
            n_warn = r["safety"]["warn_count"]
            sid = str(r.get("session_id") or "")[:12]
            fh.write(
                f"| {r['audit_index']} | {r.get('harness')} | {sid} | {ok} | {n_block} | {n_warn} |\n"
            )
        fh.write("\nOperator: manually review quarantine rows before Phase 2 train.\n")

    print(
        f"Audit sample → {jsonl_path} "
        f"({rows_quarantine}/{len(report_rows)} quarantine, "
        f"{total_block_findings} block / {total_warn_findings} warn findings)"
    )
    print(f"Summary → {md_path}")


if __name__ == "__main__":
    main()
