"""suites.py — SUITE_FILES registry and placeholder detection."""

from __future__ import annotations

from llm_eval.suites import (
    SUITE_FILES,
    is_placeholder_task,
    suite_is_placeholder_only,
    suite_names,
)

_PLACEHOLDER = {
    "id": "example-001",
    "repo": "REPLACE_ME",
    "prompt": "Apply the patch.",
    "meta": {"note": "Replace with a real frozen snapshot"},
}


def test_suite_names_matches_suite_files_keys() -> None:
    assert suite_names() == list(SUITE_FILES)


def test_empty_suite_not_placeholder_only() -> None:
    assert suite_is_placeholder_only([]) is False


def test_placeholder_only_when_all_placeholder() -> None:
    assert suite_is_placeholder_only([_PLACEHOLDER, _PLACEHOLDER]) is True
    assert suite_is_placeholder_only([_PLACEHOLDER, {"id": "real-1", "repo": "acme/x"}]) is False


def test_is_placeholder_task_signals() -> None:
    assert is_placeholder_task(_PLACEHOLDER) is True
    assert is_placeholder_task({"id": "acme-1", "repo": "acme/app", "prompt": "fix"}) is False
