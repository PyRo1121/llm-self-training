"""Load scan-raw safety-failures and block affected raw rows/sessions at curate time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _failure_row_max_severity(row: dict[str, Any]) -> str | None:
    """Normalized max severity ('block' | 'warn') or None when absent/unknown."""
    raw = row.get("max_severity")
    if raw is not None:
        return str(raw).lower()

    safety = row.get("safety")
    if not isinstance(safety, dict):
        return None

    severities: list[str] = []
    for finding in safety.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        sev = finding.get("severity")
        if sev is not None:
            severities.append(str(sev).lower())
    if not severities:
        return None
    return "block" if "block" in severities else "warn"


def _should_quarantine_failure_row(row: dict[str, Any]) -> bool:
    """Quarantine block rows; fail-closed when severity is missing."""
    sev = _failure_row_max_severity(row)
    if sev is None:
        return True
    return sev == "block"


def load_safety_failure_keys(raw_dir: Path) -> set[tuple[str, int]]:
    """Keys are (resolved source_file path, 1-based line_no) from safety-failures-*.jsonl.

    Only rows with max_severity block (or missing severity) are included; warn-only rows
    are ignored so curate can honor scan-raw severity without dropping warn-only sessions.
    """
    keys: set[tuple[str, int]] = set()
    if not raw_dir.is_dir():
        return keys
    for path in sorted(raw_dir.glob("safety-failures-*.jsonl")):
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if not _should_quarantine_failure_row(row):
                    continue
                src = row.get("source_file")
                line_no = row.get("line_no")
                if src is None or line_no is None:
                    continue
                try:
                    keys.add((str(Path(str(src)).resolve()), int(line_no)))
                except (TypeError, ValueError):
                    continue
    return keys


def session_has_quarantined_row(
    rows: list[dict[str, Any]],
    failure_keys: set[tuple[str, int]],
) -> bool:
    """Fail-closed: drop entire session if any ingest row was flagged in scan-raw."""
    if not failure_keys:
        return False
    for row in rows:
        src = row.get("_source_file") or row.get("source_path")
        line_no = row.get("_line_no") or row.get("line_no")
        if src is None or line_no is None:
            continue
        try:
            key = (str(Path(str(src)).resolve()), int(line_no))
        except (TypeError, ValueError):
            continue
        if key in failure_keys:
            return True
    return False
