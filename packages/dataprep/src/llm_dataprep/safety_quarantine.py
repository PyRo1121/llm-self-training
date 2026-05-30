"""Load scan-raw safety-failures and block affected raw rows/sessions at curate time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_safety_failure_keys(raw_dir: Path) -> set[tuple[str, int]]:
    """Keys are (resolved source_file path, 1-based line_no) from safety-failures-*.jsonl."""
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
