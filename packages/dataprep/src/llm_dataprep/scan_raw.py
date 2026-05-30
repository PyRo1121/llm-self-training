"""Scan data/raw JSONL ingest rows for secrets and PII (Phase 1 safety)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_core import data_dir
from llm_dataprep.filters import SafetyReport, gitleaks_line_flags, scan_record_text_fields


def iter_raw_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield line_no, {"_parse_error": True, "raw": line[:500]}
                continue
            if isinstance(obj, dict):
                yield line_no, obj


def scan_file(
    path: Path,
    *,
    use_gitleaks: bool,
    gitleaks_per_file: bool,
    use_presidio: bool,
    limit: int | None,
) -> tuple[int, int, list[dict[str, Any]]]:
    scanned = 0
    failed = 0
    failure_rows: list[dict[str, Any]] = []

    buffered: list[tuple[int, dict[str, Any]]] = []
    for line_no, record in iter_raw_records(path):
        if limit is not None and scanned >= limit:
            break
        scanned += 1
        if record.get("_parse_error"):
            failed += 1
            failure_rows.append(
                {
                    "source_file": str(path),
                    "line_no": line_no,
                    "parse_error": True,
                    "safety": {"ok": False, "findings": [{"source": "json", "kind": "parse_error", "detail": "invalid json"}]},
                }
            )
            continue
        buffered.append((line_no, record))

    gitleaks_flags: dict[int, list] = {}
    if use_gitleaks and gitleaks_per_file and buffered:
        gitleaks_flags = gitleaks_line_flags(
            path,
            iter(buffered),
            max_rows=limit,
        )

    for line_no, record in buffered:
        per_row_gitleaks = use_gitleaks and not gitleaks_per_file
        report = scan_record_text_fields(
            record,
            use_gitleaks=per_row_gitleaks,
            use_presidio=use_presidio,
        )
        extra = gitleaks_flags.get(line_no)
        if extra:
            report.findings.extend(extra)
            report.ok = False
        if not report.ok:
            failed += 1
            failure_rows.append(_failure_row(path, line_no, record, report))
    return scanned, failed, failure_rows


def _failure_row(
    path: Path,
    line_no: int,
    record: dict[str, Any],
    report: SafetyReport,
) -> dict[str, Any]:
    return {
        "source_file": str(path),
        "line_no": line_no,
        "harness": record.get("harness") or record.get("source"),
        "session_id": record.get("session_id"),
        "role": record.get("role"),
        "source_path": record.get("source_path"),
        "safety": report.to_dict(),
        "text_preview": (record.get("text") or "")[:240],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safety-scan data/raw/*.jsonl (regex + optional gitleaks/Presidio)"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory of ingest JSONL (default: data/raw)",
    )
    parser.add_argument(
        "--glob",
        default="*.jsonl",
        help="Filename pattern under raw-dir",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows per file")
    parser.add_argument(
        "--gitleaks",
        action="store_true",
        help="Enable gitleaks (default off at scale; use --gitleaks-per-file)",
    )
    parser.add_argument(
        "--gitleaks-per-file",
        action="store_true",
        help="One gitleaks dir scan per JSONL file (recommended vs per-row)",
    )
    parser.add_argument("--no-presidio", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write failures JSONL (default: data/raw/safety-failures-YYYY-MM-DD.jsonl)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit files instead of glob",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir or (data_dir() / "raw")
    files = args.files or sorted(
        p
        for p in raw_dir.glob(args.glob)
        if p.is_file() and not p.name.startswith("safety-failures")
    )
    if not files:
        print(f"No files in {raw_dir} matching {args.glob}")
        return

    use_gitleaks = args.gitleaks
    gitleaks_per_file = args.gitleaks_per_file or args.gitleaks
    use_presidio = not args.no_presidio

    all_failures: list[dict[str, Any]] = []
    total_scanned = 0
    total_failed = 0

    for fpath in files:
        if not fpath.is_file():
            continue
        scanned, failed, failures = scan_file(
            fpath,
            use_gitleaks=use_gitleaks,
            gitleaks_per_file=gitleaks_per_file,
            use_presidio=use_presidio,
            limit=args.limit,
        )
        total_scanned += scanned
        total_failed += failed
        all_failures.extend(failures)
        status = "FAIL" if failed else "ok"
        print(f"{fpath.name}: {scanned} rows, {failed} flagged ({status})")

    print(f"Total: {total_scanned} scanned, {total_failed} flagged")

    if all_failures:
        out = args.out or (
            raw_dir / f"safety-failures-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in all_failures:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Failures written → {out}")
    elif args.out:
        print("No failures — output file not created")


if __name__ == "__main__":
    main()
