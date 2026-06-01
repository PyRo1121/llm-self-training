"""safety-eval CLI metrics and fixture routing."""

from __future__ import annotations

import json
from pathlib import Path

from llm_dataprep.safety_eval import (
    compute_metrics,
    evaluate_fixtures,
    load_fixtures,
    _scan_fixture,
    _use_diff_scan,
)


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
    fixtures = Path(__file__).resolve().parent / "fixtures" / "safety_eval.jsonl"
    rows = load_fixtures(fixtures)
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
