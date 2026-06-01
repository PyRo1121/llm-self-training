"""scan_raw — parse-error quarantine, presidio path, CLI smoke."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from llm_dataprep.filters import SafetyFinding
from llm_dataprep.scan_raw import (
    _max_severity,
    _scan_file_worker,
    iter_raw_records,
    main,
    scan_file,
)


def test_iter_raw_records_skips_blank_and_parse_errors(tmp_path: Path) -> None:
    raw = tmp_path / "mixed.jsonl"
    raw.write_text(
        "\n"
        + json.dumps({"text": "ok"}) + "\n"
        + "{not json\n"
        + json.dumps([1, 2, 3]) + "\n",
        encoding="utf-8",
    )
    rows = list(iter_raw_records(raw))
    assert len(rows) == 2
    assert rows[0] == (2, {"text": "ok"})
    line_no, bad = rows[1]
    assert line_no == 3
    assert bad["_parse_error"] is True
    assert bad["raw"].startswith("{not json")


def test_parse_error_quarantined(tmp_path: Path) -> None:
    raw = tmp_path / "bad.jsonl"
    raw.write_text("{broken\n", encoding="utf-8")
    scanned, failed, failures, warns = scan_file(
        raw,
        use_gitleaks=False,
        gitleaks_per_file=False,
        use_presidio=False,
        limit=None,
    )
    assert scanned == 1
    assert failed == 1
    assert not warns
    row = failures[0]
    assert row["parse_error"] is True
    assert row["block_count"] == 1
    assert row["max_severity"] == "block"
    assert row["safety"]["findings"][0]["kind"] == "parse_error"


def test_scan_file_respects_limit(tmp_path: Path) -> None:
    raw = tmp_path / "many.jsonl"
    raw.write_text(
        "\n".join(json.dumps({"text": f"row {i}"}) for i in range(5)) + "\n",
        encoding="utf-8",
    )
    scanned, failed, failures, warns = scan_file(
        raw,
        use_gitleaks=False,
        gitleaks_per_file=False,
        use_presidio=False,
        limit=2,
    )
    assert scanned == 2
    assert failed == 0
    assert not failures
    assert not warns


def test_max_severity_none_when_no_findings() -> None:
    assert _max_severity([], []) == "none"


def test_presidio_batch_findings_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "pii.jsonl"
    raw.write_text(json.dumps({"text": "Contact Jane Doe please"}) + "\n", encoding="utf-8")

    def _fake_batch(
        texts: list[str],
        *,
        n_process: int | None = None,
        batch_size: int | None = None,
    ) -> list[list[SafetyFinding]]:
        assert texts == ["Contact Jane Doe please"]
        return [
            [
                SafetyFinding(
                    source="presidio",
                    kind="PERSON",
                    detail="score=0.95",
                )
            ]
        ]

    monkeypatch.setattr("llm_dataprep.scan_raw.scan_presidio_batch", _fake_batch)
    scanned, failed, failures, warns = scan_file(
        raw,
        use_gitleaks=False,
        gitleaks_per_file=False,
        use_presidio=True,
        limit=None,
    )
    assert scanned == 1
    assert failed == 0
    assert not failures
    assert len(warns) == 1
    assert warns[0]["warn_count"] == 1
    assert any(f["source"] == "presidio" for f in warns[0]["safety"]["findings"])


def test_presidio_skips_diff_harness_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "diffs.jsonl"
    raw.write_text(
        json.dumps({"harness": "git-diffs", "text": "+++ b/x\n+clean"}) + "\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def _fake_batch(
        texts: list[str],
        *,
        n_process: int | None = None,
        batch_size: int | None = None,
    ) -> list[list[SafetyFinding]]:
        calls.append(texts)
        return [[] for _ in texts]

    monkeypatch.setattr("llm_dataprep.scan_raw.scan_presidio_batch", _fake_batch)
    scanned, failed, failures, warns = scan_file(
        raw,
        use_gitleaks=False,
        gitleaks_per_file=False,
        use_presidio=True,
        limit=None,
    )
    assert scanned == 1
    assert failed == 0
    assert not failures
    assert not warns
    assert calls == []


def test_scan_file_worker_sets_subprocess_env(tmp_path: Path) -> None:
    raw = tmp_path / "clean.jsonl"
    raw.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")
    payload = (str(raw.resolve()), False, False, False, None, True)
    name, scanned, failed, failures, warns = _scan_file_worker(payload)
    assert name == "clean.jsonl"
    assert scanned == 1
    assert failed == 0
    assert not failures
    assert not warns
    assert os.environ.get("LLM_SCAN_SUBPROCESS") == "1"

    payload_serial = (str(raw.resolve()), False, False, False, None, False)
    _scan_file_worker(payload_serial)
    assert "LLM_SCAN_SUBPROCESS" not in os.environ


def test_main_no_matching_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_dir = tmp_path / "empty"
    raw_dir.mkdir()
    monkeypatch.setattr(sys, "argv", ["scan-raw", "--raw-dir", str(raw_dir)])
    main()
    assert "No files" in capsys.readouterr().out


def test_main_smoke_writes_failures_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    block_file = raw_dir / "block.jsonl"
    warn_file = raw_dir / "warn.jsonl"
    block_file.write_text(
        json.dumps({"text": "key sk-" + "A" * 24}) + "\n",
        encoding="utf-8",
    )
    warn_file.write_text(
        json.dumps({"text": "secret=AbCdEfGhIjKlMnOpQrSt"}) + "\n",
        encoding="utf-8",
    )
    fail_out = tmp_path / "failures.jsonl"
    warn_out = tmp_path / "warns.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scan-raw",
            "--raw-dir",
            str(raw_dir),
            "--files",
            str(block_file),
            str(warn_file),
            "--no-presidio",
            "--workers",
            "1",
            "--out",
            str(fail_out),
            "--warn-out",
            str(warn_out),
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "Total:" in out
    assert "Failures written" in out
    assert "Warn-only rows written" in out
    assert fail_out.is_file()
    assert warn_out.is_file()
    fail_rows = [json.loads(line) for line in fail_out.read_text().splitlines() if line.strip()]
    warn_rows = [json.loads(line) for line in warn_out.read_text().splitlines() if line.strip()]
    assert len(fail_rows) == 1
    assert len(warn_rows) == 1


def test_main_custom_out_without_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    clean = raw_dir / "clean.jsonl"
    clean.write_text(json.dumps({"text": "benign content"}) + "\n", encoding="utf-8")
    fail_out = tmp_path / "no-fail.jsonl"
    warn_out = tmp_path / "no-warn.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scan-raw",
            "--raw-dir",
            str(raw_dir),
            "--files",
            str(clean),
            "--no-presidio",
            "--workers",
            "1",
            "--out",
            str(fail_out),
            "--warn-out",
            str(warn_out),
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "No failures — output file not created" in out
    assert "No warn-only rows — warn output file not created" in out
    assert not fail_out.exists()
    assert not warn_out.exists()


def test_main_parallel_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    files = []
    for i in range(2):
        path = raw_dir / f"row{i}.jsonl"
        path.write_text(json.dumps({"text": f"row {i}"}) + "\n", encoding="utf-8")
        files.append(path)

    class _FakeFuture:
        def __init__(self, result: tuple) -> None:
            self._result = result

        def result(self) -> tuple:
            return self._result

    class _FakePool:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> _FakePool:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def submit(self, fn: object, payload: tuple) -> _FakeFuture:
            return _FakeFuture(fn(payload))  # type: ignore[operator,misc]

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scan-raw",
            "--raw-dir",
            str(raw_dir),
            "--no-presidio",
            "--workers",
            "2",
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "scan-raw: 2 file(s), 2 worker(s)" in out
    assert "presidio_mp='nested-off'" in out or "presidio_mp=nested-off" in out
    assert "Total:" in out
