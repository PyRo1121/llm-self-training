"""scan_raw policy integration — quarantine, diff harness, sidecar gitleaks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_dataprep.filters import SafetyFinding, SafetyReport
from llm_dataprep.scan_raw import scan_file


def test_block_finding_quarantined(tmp_path: Path) -> None:
    raw = tmp_path / "rows.jsonl"
    raw.write_text(
        json.dumps({"text": "key sk-" + "A" * 24}) + "\n",
        encoding="utf-8",
    )
    scanned, failed, failures, warns = scan_file(
        raw,
        use_gitleaks=False,
        gitleaks_per_file=False,
        presidio_mode="off",
        limit=None,
    )
    assert scanned == 1
    assert failed == 1
    assert not warns
    row = failures[0]
    assert row["block_count"] == 1
    assert row["warn_count"] == 0
    assert row["max_severity"] == "block"


def test_warn_only_not_quarantined(tmp_path: Path) -> None:
    raw = tmp_path / "rows.jsonl"
    raw.write_text(
        json.dumps({"text": "secret=AbCdEfGhIjKlMnOpQrSt"}) + "\n",
        encoding="utf-8",
    )
    scanned, failed, failures, warns = scan_file(
        raw,
        use_gitleaks=False,
        gitleaks_per_file=False,
        presidio_mode="off",
        limit=None,
    )
    assert scanned == 1
    assert failed == 0
    assert not failures
    assert len(warns) == 1
    assert warns[0]["block_count"] == 0
    assert warns[0]["warn_count"] == 1
    assert warns[0]["max_severity"] == "warn"


def test_diff_harness_uses_scan_diff_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "diffs.jsonl"
    raw.write_text(
        json.dumps({"harness": "git-diffs", "text": "+++ b/foo\n+secret"}) + "\n",
        encoding="utf-8",
    )
    calls: list[dict] = []

    def _fake(record: dict) -> SafetyReport:
        calls.append(record)
        return SafetyReport(ok=True)

    monkeypatch.setattr("llm_dataprep.scan_raw.scan_diff_record", _fake)
    scanned, _failed, _failures, _warns = scan_file(
        raw,
        use_gitleaks=False,
        gitleaks_per_file=False,
        presidio_mode="off",
        limit=None,
    )
    assert scanned == 1
    assert len(calls) == 1
    assert calls[0]["harness"] == "git-diffs"


def test_gitleaks_diff_harness_uses_added_lines_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "diffs.jsonl"
    diff_text = (
        "--- a/config\n"
        "+++ b/config\n"
        "-password=REMOVED_SECRET\n"
        "+normal_added_line\n"
    )
    raw.write_text(
        json.dumps({"harness": "git-diffs", "text": diff_text}) + "\n",
        encoding="utf-8",
    )
    calls: list[list[tuple[int, str]]] = []

    def _fake(rows: list, source_path: Path, **kwargs: object) -> dict:
        calls.append(rows)
        return {}

    monkeypatch.setattr("llm_dataprep.scan_raw.gitleaks_sidecar_line_flags", _fake)
    scanned, _failed, _failures, _warns = scan_file(
        raw,
        use_gitleaks=True,
        gitleaks_per_file=True,
        presidio_mode="off",
        limit=None,
    )
    assert scanned == 1
    assert calls
    assert calls[0] == [(1, "normal_added_line")]


def test_gitleaks_uses_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "sidecar.jsonl"
    raw.write_text('{"text":"clean"}\n', encoding="utf-8")
    calls: list[tuple[Path, list]] = []

    def _fake(rows: list, source_path: Path, **kwargs: object) -> dict:
        calls.append((source_path, rows))
        return {1: [SafetyFinding(source="gitleaks", kind="test-rule", detail="match")]}

    monkeypatch.setattr("llm_dataprep.scan_raw.gitleaks_sidecar_line_flags", _fake)
    scanned, failed, failures, _warns = scan_file(
        raw,
        use_gitleaks=True,
        gitleaks_per_file=True,
        presidio_mode="off",
        limit=None,
    )
    assert scanned == 1
    assert calls
    assert calls[0][0] == raw
    assert calls[0][1] == [(1, "clean")]
    assert failed == 1
    assert failures[0]["block_count"] == 1


def test_gitleaks_in_place_when_not_per_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "inplace.jsonl"
    raw.write_text('{"text":"clean"}\n', encoding="utf-8")
    calls: list[Path] = []

    def _fake(path: Path, **kwargs: object) -> dict:
        calls.append(path)
        return {1: [SafetyFinding(source="gitleaks", kind="test-rule", detail="match")]}

    monkeypatch.setattr("llm_dataprep.scan_raw.gitleaks_sidecar_line_flags", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sidecar should not run")))
    monkeypatch.setattr("llm_dataprep.filters.gitleaks_jsonl_line_flags", _fake)
    scanned, failed, failures, _warns = scan_file(
        raw,
        use_gitleaks=True,
        gitleaks_per_file=False,
        presidio_mode="off",
        limit=None,
    )
    assert scanned == 1
    assert calls == [raw]
    assert failed == 1
    assert failures[0]["block_count"] == 1
