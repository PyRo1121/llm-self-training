"""Safety quarantine: severity-aware failure key loading."""

from __future__ import annotations

import json
from pathlib import Path

from llm_dataprep.safety_quarantine import (
    _failure_row_max_severity,
    _should_quarantine_failure_row,
    load_safety_failure_keys,
    session_has_quarantined_row,
)


def _write_failure(tmp_path: Path, name: str, rows: list[dict]) -> None:
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_max_severity_block_is_quarantined(tmp_path: Path) -> None:
    raw = tmp_path / "cursor.jsonl"
    _write_failure(
        tmp_path,
        "safety-failures-2026-05-30.jsonl",
        [
            {
                "source_file": str(raw),
                "line_no": 1,
                "max_severity": "block",
            }
        ],
    )
    keys = load_safety_failure_keys(tmp_path)
    assert keys == {(str(raw.resolve()), 1)}


def test_max_severity_warn_is_not_quarantined(tmp_path: Path) -> None:
    raw = tmp_path / "cursor.jsonl"
    _write_failure(
        tmp_path,
        "safety-failures-2026-05-30.jsonl",
        [
            {
                "source_file": str(raw),
                "line_no": 2,
                "max_severity": "warn",
            }
        ],
    )
    assert load_safety_failure_keys(tmp_path) == set()


def test_missing_severity_quarantines_fail_closed(tmp_path: Path) -> None:
    raw = tmp_path / "legacy.jsonl"
    _write_failure(
        tmp_path,
        "safety-failures-2026-05-30.jsonl",
        [
            {
                "source_file": str(raw),
                "line_no": 3,
                "safety": {
                    "ok": False,
                    "findings": [{"source": "regex", "kind": "generic_api_key", "detail": "x"}],
                },
            }
        ],
    )
    keys = load_safety_failure_keys(tmp_path)
    assert keys == {(str(raw.resolve()), 3)}


def test_findings_severity_derives_max_block(tmp_path: Path) -> None:
    raw = tmp_path / "mixed.jsonl"
    _write_failure(
        tmp_path,
        "safety-failures-2026-05-30.jsonl",
        [
            {
                "source_file": str(raw),
                "line_no": 4,
                "safety": {
                    "ok": False,
                    "findings": [
                        {"source": "regex", "kind": "generic_api_key", "severity": "warn"},
                        {"source": "regex", "kind": "aws_access_key", "severity": "block"},
                    ],
                },
            },
            {
                "source_file": str(raw),
                "line_no": 5,
                "safety": {
                    "ok": False,
                    "findings": [
                        {"source": "presidio", "kind": "PERSON", "severity": "warn"},
                    ],
                },
            },
        ],
    )
    keys = load_safety_failure_keys(tmp_path)
    assert keys == {(str(raw.resolve()), 4)}


def test_failure_row_max_severity_helpers() -> None:
    assert _failure_row_max_severity({"max_severity": "BLOCK"}) == "block"
    assert _failure_row_max_severity({"max_severity": "warn"}) == "warn"
    assert _should_quarantine_failure_row({"max_severity": "block"}) is True
    assert _should_quarantine_failure_row({"max_severity": "warn"}) is False
    assert _should_quarantine_failure_row({}) is True


def test_session_has_quarantined_row_respects_keys(tmp_path: Path) -> None:
    raw = tmp_path / "file.jsonl"
    key = (str(raw.resolve()), 7)
    rows = [{"_source_file": str(raw), "_line_no": 7, "text": "secret"}]
    assert session_has_quarantined_row(rows, {key}) is True
    assert session_has_quarantined_row(rows, set()) is False
