"""Scan data/raw JSONL ingest rows for secrets and PII (Phase 1 safety)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_core import data_dir
from llm_dataprep.diff_scan import extract_added_lines, scan_diff_record
from llm_dataprep.filters import (
    SafetyFinding,
    SafetyReport,
    gitleaks_sidecar_line_flags,
    record_combined_text,
    scan_presidio_batch,
    scan_regex,
)
from llm_dataprep.perf import presidio_batch_size, presidio_n_process, worker_count
from llm_dataprep.safety_policy import (
    Severity,
    apply_policy,
    is_diff_harness,
    load_safety_policy,
    should_quarantine,
)


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


def _max_severity(block: list[SafetyFinding], warn: list[SafetyFinding]) -> str:
    if block:
        return Severity.BLOCK.value
    if warn:
        return Severity.WARN.value
    return "none"


def _evaluate_row(
    record: dict[str, Any],
    line_no: int,
    *,
    use_gitleaks: bool,
    use_presidio: bool,
    presidio_findings: list[SafetyFinding] | None,
    gitleaks_flags: dict[int, list[SafetyFinding]],
) -> tuple[list[SafetyFinding], list[SafetyFinding], bool]:
    """Classify row findings; return (block, warn, quarantine)."""
    pol = load_safety_policy()
    combined = record_combined_text(record)

    if is_diff_harness(record, pol):
        diff_report = scan_diff_record(record)
        findings = list(diff_report.findings)
        text = record.get("text") or record.get("content") or ""
        policy_text = extract_added_lines(text if isinstance(text, str) else "") or combined
    else:
        findings: list[SafetyFinding] = []
        if combined.strip():
            findings.extend(scan_regex(combined))
            if use_presidio and presidio_findings is not None:
                findings.extend(presidio_findings)
        policy_text = combined

    if use_gitleaks:
        findings.extend(gitleaks_flags.get(line_no, []))

    block, warn = apply_policy(findings, policy_text, policy=pol)
    quarantine = should_quarantine(block, warn, pol)
    return block, warn, quarantine


def _scan_progress(path: Path, msg: str) -> None:
    print(f"scan: {path.name}: {msg}", flush=True)


def scan_file(
    path: Path,
    *,
    use_gitleaks: bool,
    gitleaks_per_file: bool,
    use_presidio: bool,
    limit: int | None,
) -> tuple[int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    _ = gitleaks_per_file  # sidecar scan always runs per file when use_gitleaks
    scanned = 0
    failed = 0
    failure_rows: list[dict[str, Any]] = []
    warn_rows: list[dict[str, Any]] = []
    pol = load_safety_policy()

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
                    "block_count": 1,
                    "warn_count": 0,
                    "max_severity": Severity.BLOCK.value,
                    "safety": {
                        "ok": False,
                        "findings": [
                            {"source": "json", "kind": "parse_error", "detail": "invalid json"}
                        ],
                    },
                }
            )
            continue
        buffered.append((line_no, record))

    _scan_progress(path, f"loaded {len(buffered)} rows — gitleaks…")
    gitleaks_flags: dict[int, list[SafetyFinding]] = {}
    if use_gitleaks and buffered:
        sidecar_rows: list[tuple[int, str]] = []
        for line_no, record in buffered:
            if is_diff_harness(record, pol):
                text = record.get("text") or record.get("content") or ""
                sidecar_text = extract_added_lines(text if isinstance(text, str) else "")
            else:
                sidecar_text = record_combined_text(record)
            sidecar_rows.append((line_no, sidecar_text))
        gitleaks_flags = gitleaks_sidecar_line_flags(sidecar_rows, path)
    _scan_progress(path, "gitleaks done — presidio…")

    presidio_by_idx: dict[int, list[SafetyFinding]] = {}
    if use_presidio and buffered:
        non_diff: list[tuple[int, str]] = []
        for idx, (_line_no, record) in enumerate(buffered):
            if is_diff_harness(record, pol):
                continue
            non_diff.append((idx, record_combined_text(record)))
        if non_diff:
            _scan_progress(path, f"presidio on {len(non_diff)} rows (slow on big files)…")
            texts = [text for _idx, text in non_diff]
            batches = scan_presidio_batch(
                texts,
                n_process=presidio_n_process(),
                batch_size=presidio_batch_size(),
            )
            for (idx, _text), pf in zip(non_diff, batches, strict=True):
                presidio_by_idx[idx] = pf
    _scan_progress(path, "presidio done — policy pass…")

    for idx, (line_no, record) in enumerate(buffered):
        pf = presidio_by_idx.get(idx) if use_presidio else None
        block, warn, quarantine = _evaluate_row(
            record,
            line_no,
            use_gitleaks=use_gitleaks,
            use_presidio=use_presidio,
            presidio_findings=pf if use_presidio else None,
            gitleaks_flags=gitleaks_flags,
        )
        if quarantine:
            failed += 1
            report = SafetyReport(ok=False, findings=block + warn)
            failure_rows.append(_failure_row(path, line_no, record, report, block, warn))
        elif warn:
            report = SafetyReport(ok=True, findings=warn)
            warn_rows.append(_failure_row(path, line_no, record, report, block, warn))
    return scanned, failed, failure_rows, warn_rows


def _failure_row(
    path: Path,
    line_no: int,
    record: dict[str, Any],
    report: SafetyReport,
    block: list[SafetyFinding],
    warn: list[SafetyFinding],
) -> dict[str, Any]:
    return {
        "source_file": str(path),
        "line_no": line_no,
        "harness": record.get("harness") or record.get("source"),
        "session_id": record.get("session_id"),
        "role": record.get("role"),
        "source_path": record.get("source_path"),
        "block_count": len(block),
        "warn_count": len(warn),
        "max_severity": _max_severity(block, warn),
        "safety": report.to_dict(),
        "text_preview": (record.get("text") or "")[:240],
    }


def _scan_file_worker(
    payload: tuple[str, bool, bool, bool, int | None, bool],
) -> tuple[str, int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    """ProcessPoolExecutor entrypoint (must be module-level for pickling)."""
    import os

    path_str, use_gitleaks, gitleaks_per_file, use_presidio, limit, file_parallel = payload
    # Nested spaCy n_process only when multiple raw files scan concurrently (RAM safety).
    if file_parallel:
        os.environ["LLM_SCAN_SUBPROCESS"] = "1"
    else:
        os.environ.pop("LLM_SCAN_SUBPROCESS", None)
    scanned, failed, failures, warns = scan_file(
        Path(path_str),
        use_gitleaks=use_gitleaks,
        gitleaks_per_file=gitleaks_per_file,
        use_presidio=use_presidio,
        limit=limit,
    )
    return Path(path_str).name, scanned, failed, failures, warns


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
        help="Enable gitleaks sidecar scan per file",
    )
    parser.add_argument(
        "--gitleaks-per-file",
        action="store_true",
        help="Alias for --gitleaks (one sidecar scan per JSONL file)",
    )
    parser.add_argument("--no-presidio", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write failures JSONL (default: data/raw/safety-failures-YYYY-MM-DD.jsonl)",
    )
    parser.add_argument(
        "--warn-out",
        type=Path,
        default=None,
        help="Write warn-only JSONL (default: data/raw/safety-warn-YYYY-MM-DD.jsonl)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit files instead of glob",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel raw files to scan (default: SCAN_WORKERS env or CPU-2)",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir or (data_dir() / "raw")
    files = args.files or sorted(
        p
        for p in raw_dir.glob(args.glob)
        if p.is_file()
        and not p.name.startswith("safety-failures")
        and not p.name.startswith("safety-warn")
    )
    if not files:
        print(f"No files in {raw_dir} matching {args.glob}")
        return

    use_gitleaks = args.gitleaks or args.gitleaks_per_file
    gitleaks_per_file = args.gitleaks_per_file or args.gitleaks
    use_presidio = not args.no_presidio
    n_workers = worker_count("SCAN_WORKERS") if args.workers is None else max(1, args.workers)

    all_failures: list[dict[str, Any]] = []
    all_warns: list[dict[str, Any]] = []
    total_scanned = 0
    total_failed = 0

    file_parallel = min(n_workers, len(files)) > 1

    if n_workers <= 1 or len(files) <= 1:
        for fpath in files:
            if not fpath.is_file():
                continue
            name, scanned, failed, failures, warns = _scan_file_worker(
                (
                    str(fpath.resolve()),
                    use_gitleaks,
                    gitleaks_per_file,
                    use_presidio,
                    args.limit,
                    False,
                )
            )
            total_scanned += scanned
            total_failed += failed
            all_failures.extend(failures)
            all_warns.extend(warns)
            status = "FAIL" if failed else "ok"
            print(f"{name}: {scanned} rows, {failed} flagged ({status})")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(
            f"scan-raw: {len(files)} file(s), {n_workers} worker(s)"
            f", presidio_mp={'nested-off' if file_parallel else 'on'}",
            flush=True,
        )
        payloads = [
            (
                str(fpath.resolve()),
                use_gitleaks,
                gitleaks_per_file,
                use_presidio,
                args.limit,
                file_parallel,
            )
            for fpath in files
            if fpath.is_file()
        ]
        with ProcessPoolExecutor(max_workers=min(n_workers, len(payloads))) as pool:
            futures = [pool.submit(_scan_file_worker, p) for p in payloads]
            for fut in as_completed(futures):
                name, scanned, failed, failures, warns = fut.result()
                total_scanned += scanned
                total_failed += failed
                all_failures.extend(failures)
                all_warns.extend(warns)
                status = "FAIL" if failed else "ok"
                print(f"{name}: {scanned} rows, {failed} flagged ({status})")

    print(f"Total: {total_scanned} scanned, {total_failed} flagged")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if all_failures:
        out = args.out or (raw_dir / f"safety-failures-{stamp}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in all_failures:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Failures written → {out}")
    elif args.out:
        print("No failures — output file not created")

    if all_warns:
        warn_out = args.warn_out or (raw_dir / f"safety-warn-{stamp}.jsonl")
        warn_out.parent.mkdir(parents=True, exist_ok=True)
        with warn_out.open("w", encoding="utf-8") as fh:
            for row in all_warns:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Warn-only rows written → {warn_out}")
    elif args.warn_out:
        print("No warn-only rows — warn output file not created")


if __name__ == "__main__":
    main()
