"""Gitleaks JSONL scan — in-place file scan, no row temp explosion."""

from __future__ import annotations

import json
from pathlib import Path

from llm_dataprep.filters import _parse_gitleaks_report, gitleaks_jsonl_line_flags


def test_parse_gitleaks_report_start_line() -> None:
    report = Path("/tmp/test-gitleaks-report.json")
    report.write_text(
        json.dumps(
            [
                {
                    "RuleID": "generic-api-key",
                    "Match": "sk-test123",
                    "File": "data/raw/codex.jsonl",
                    "StartLine": 42,
                }
            ]
        ),
        encoding="utf-8",
    )
    try:
        findings = _parse_gitleaks_report(report)
        assert len(findings) == 1
        assert "42" in findings[0].detail
    finally:
        report.unlink(missing_ok=True)


def test_gitleaks_line_flags_delegates_without_row_files(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "tiny.jsonl"
    raw.write_text('{"text":"hello"}\n', encoding="utf-8")
    calls: list[Path] = []

    def _fake(path: Path, **kwargs: object) -> dict:
        calls.append(path)
        return {1: []}

    monkeypatch.setattr(
        "llm_dataprep.filters.gitleaks_jsonl_line_flags",
        _fake,
    )
    from llm_dataprep.filters import gitleaks_line_flags

    out = gitleaks_line_flags(raw, iter([]))
    assert calls == [raw]
    assert out == {1: []}
