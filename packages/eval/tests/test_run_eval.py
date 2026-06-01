"""run_eval verdict rules — placeholder vs strict vs no-smoke-chat."""

from __future__ import annotations

from unittest.mock import patch

from llm_eval.run_eval import evaluate_suite

_PLACEHOLDER = {
    "id": "example-001",
    "repo": "REPLACE_ME",
    "prompt": "Apply the patch.",
    "meta": {"note": "Replace with a real frozen snapshot"},
}
_REAL = {
    "id": "acme-fix-001",
    "repo": "acme/app",
    "prompt": "Fix the null check in auth middleware.",
}


@patch("llm_eval.run_eval.load_suite")
def test_placeholder_no_smoke_non_strict_passes(mock_load) -> None:
    mock_load.return_value = [_PLACEHOLDER]
    row = evaluate_suite(
        "diff_apply",
        model="qwen2.5-coder:7b",
        ollama_host="http://127.0.0.1:11434",
        strict=False,
        smoke_chat=False,
    )
    assert row["verdict"] == "pass"
    assert row["reason"] == "placeholder_suite_skipped"
    assert row["passed"] == 1


@patch("llm_eval.run_eval.load_suite")
def test_placeholder_strict_fails(mock_load) -> None:
    mock_load.return_value = [_PLACEHOLDER]
    row = evaluate_suite(
        "style",
        model="qwen2.5-coder:7b",
        ollama_host="http://127.0.0.1:11434",
        strict=True,
        smoke_chat=False,
    )
    assert row["verdict"] == "fail"
    assert row["reason"] == "placeholder_tasks_only"
    assert row["passed"] == 0


@patch("llm_eval.run_eval.load_suite")
def test_real_suite_no_smoke_not_pass(mock_load) -> None:
    mock_load.return_value = [_REAL]
    row = evaluate_suite(
        "debug",
        model="qwen2.5-coder:7b",
        ollama_host="http://127.0.0.1:11434",
        strict=False,
        smoke_chat=False,
    )
    assert row["verdict"] == "incomplete"
    assert row["reason"] == "manual_tasks_required"
    assert row["passed"] == 0


@patch("llm_eval.run_eval._ollama_chat", return_value="OK")
@patch("llm_eval.run_eval.load_suite")
def test_real_suite_smoke_chat_passes(mock_load, _chat) -> None:
    mock_load.return_value = [_REAL]
    row = evaluate_suite(
        "retrieval_gold",
        model="qwen2.5-coder:7b",
        ollama_host="http://127.0.0.1:11434",
        strict=False,
        smoke_chat=True,
    )
    assert row["verdict"] == "pass"
    assert row["reason"] == "smoke_chat"
    assert row["passed"] == 1


@patch("llm_eval.run_eval._ollama_chat", return_value="")
@patch("llm_eval.run_eval.load_suite")
def test_real_suite_smoke_chat_empty_reply_fails(mock_load, _chat) -> None:
    mock_load.return_value = [_REAL]
    row = evaluate_suite(
        "diff_apply",
        model="qwen2.5-coder:7b",
        ollama_host="http://127.0.0.1:11434",
        strict=False,
        smoke_chat=True,
    )
    assert row["verdict"] == "fail"
    assert row["passed"] == 0
