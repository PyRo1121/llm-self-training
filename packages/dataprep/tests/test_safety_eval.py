"""safety-eval CLI metrics and fixture routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from llm_dataprep.safety_eval import (
    compute_metrics,
    evaluate_fixtures,
    load_fixtures,
    main,
    _scan_fixture,
    _use_diff_scan,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "safety_eval.jsonl"


def test_compute_metrics_perfect() -> None:
    m = compute_metrics([True, False], [True, False])
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.tp == 1 and m.tn == 1


def test_use_diff_scan() -> None:
    assert _use_diff_scan({"label": "diff", "mode": ""})
    assert _use_diff_scan({"label": "diff_aws_secret_added", "mode": "diff"})
    assert not _use_diff_scan({"label": "benign_code_python"})


def test_scan_fixture_routes_diff() -> None:
    diff = "--- a/x\n+++ b/x\n@@\n+ghp_abcdefghijklmnopqrstuvwxyz1234567890\n"
    assert _scan_fixture("def ok(): pass", {"label": "text"}).ok
    assert not _scan_fixture(diff, {"label": "diff_github_pat_added", "mode": "diff"}).ok
    assert not _scan_fixture(diff, {"label": "synthetic_github_pat"}).ok


def test_evaluate_fixtures_on_bundled_file() -> None:
    rows = load_fixtures(_FIXTURES)
    overall, _per_label = evaluate_fixtures(rows)
    assert overall.n == len(rows)
    assert overall.f1 >= 0.9


def test_load_fixtures_requires_fields(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
    try:
        load_fixtures(bad)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "expect_block" in str(e)


def test_compute_metrics_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        compute_metrics([True], [True, False])


def test_compute_metrics_fp_fn_and_all_negative() -> None:
    fp = compute_metrics([False], [True])
    assert fp.fp == 1 and fp.tp == 0 and fp.precision == 0.0 and fp.recall == 0.0 and fp.f1 == 0.0

    fn = compute_metrics([True], [False])
    assert fn.fn == 1 and fn.tp == 0 and fn.precision == 0.0 and fn.recall == 0.0 and fn.f1 == 0.0

    tn_only = compute_metrics([False, False], [False, False])
    assert tn_only.tn == 2 and tn_only.precision == 0.0 and tn_only.recall == 0.0 and tn_only.f1 == 0.0


def test_load_fixtures_edge_cases(tmp_path: Path) -> None:
    with_blank = tmp_path / "blank.jsonl"
    with_blank.write_text(
        "\n"
        + json.dumps({"text": "ok", "expect_block": False, "label": "benign"})
        + "\n\n",
        encoding="utf-8",
    )
    rows = load_fixtures(with_blank)
    assert len(rows) == 1

    missing_text = tmp_path / "missing_text.jsonl"
    missing_text.write_text(
        json.dumps({"expect_block": False, "label": "x"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing 'text'"):
        load_fixtures(missing_text)

    missing_label = tmp_path / "missing_label.jsonl"
    missing_label.write_text(
        json.dumps({"text": "x", "expect_block": False}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing 'label'"):
        load_fixtures(missing_label)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no fixture rows"):
        load_fixtures(empty)


def test_evaluate_fixtures_requires_string_text() -> None:
    with pytest.raises(TypeError, match="text"):
        evaluate_fixtures([{"text": 123, "expect_block": False, "label": "bad"}])


def test_main_prints_overall(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["safety-eval", "--fixtures", str(_FIXTURES)])
    main()
    out = capsys.readouterr().out
    assert "fixtures:" in out
    assert "overall:" in out
    assert "label=" not in out


def test_main_per_label(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["safety-eval", "--fixtures", str(_FIXTURES), "--per-label"])
    main()
    out = capsys.readouterr().out
    assert "overall:" in out
    assert "label=benign_code_python" in out


def test_main_default_fixtures(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["safety-eval"])
    main()
    out = capsys.readouterr().out
    assert "overall:" in out
    assert str(_FIXTURES.resolve()) in out


def test_main_missing_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing.jsonl"
    monkeypatch.setattr(sys, "argv", ["safety-eval", "--fixtures", str(missing)])
    with pytest.raises(SystemExit, match="fixtures not found"):
        main()
