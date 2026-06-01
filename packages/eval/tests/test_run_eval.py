"""run_eval verdict rules — placeholder vs strict vs no-smoke-chat."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from llm_eval.run_eval import _smoke_prompt, evaluate_suite, run_all
from llm_eval.suites import SUITE_FILES

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
def test_empty_suite_always_fails(mock_load) -> None:
    mock_load.return_value = []
    for strict, smoke_chat in ((False, False), (False, True), (True, False)):
        row = evaluate_suite(
            "diff_apply",
            model="qwen2.5-coder:7b",
            ollama_host="http://127.0.0.1:11434",
            strict=strict,
            smoke_chat=smoke_chat,
        )
        assert row["verdict"] == "fail"
        assert row["reason"] == "empty_suite"
        assert row["tasks"] == 0
        assert row["passed"] == 0


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


def test_smoke_prompt_retrieval_uses_query() -> None:
    task = {"query": "Where is config?", "prompt": "ignored for retrieval"}
    assert _smoke_prompt("retrieval_gold", task) == "Where is config?"
    assert _smoke_prompt("debug", task) == "ignored for retrieval"


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


@patch("llm_eval.run_eval._ollama_chat", return_value="OK")
@patch("llm_eval.run_eval.load_suite")
def test_retrieval_smoke_sends_query_not_prompt(mock_load, mock_chat) -> None:
    mock_load.return_value = [
        {
            "id": "rag-1",
            "query": "Where is training config?",
            "prompt": "wrong field for smoke",
        }
    ]
    evaluate_suite(
        "retrieval_gold",
        model="qwen2.5-coder:7b",
        ollama_host="http://127.0.0.1:11434",
        strict=False,
        smoke_chat=True,
    )
    mock_chat.assert_called_once()
    assert mock_chat.call_args[0][2] == "Where is training config?"


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


@patch("llm_eval.run_eval.evaluate_suite")
def test_run_all_iterates_suite_files_keys(mock_eval) -> None:
    mock_eval.return_value = {
        "suite": "x",
        "verdict": "pass",
        "reason": "smoke_chat",
        "tasks": 1,
        "passed": 1,
    }
    report = run_all(
        model="qwen2.5-coder:7b",
        ollama_host="http://127.0.0.1:11434",
        strict=False,
        smoke_chat=False,
        train_run=None,
    )
    called = [c.args[0] for c in mock_eval.call_args_list]
    assert called == list(SUITE_FILES)
    assert len(report["suites"]) == len(SUITE_FILES)


@patch("llm_eval.run_eval.register_benchmark_run")
@patch("llm_eval.run_eval.ensure_warehouse")
@patch("llm_eval.run_eval.evaluate_suite")
def test_run_all_registers_warehouse_when_train_run(
    mock_eval, mock_wh, mock_register
) -> None:
    mock_eval.return_value = {
        "suite": "debug",
        "verdict": "pass",
        "reason": "smoke_chat",
        "tasks": 1,
        "passed": 1,
    }
    conn = MagicMock()
    mock_wh.return_value = conn
    report = run_all(
        model="qwen2.5-coder:7b",
        ollama_host="http://127.0.0.1:11434",
        strict=False,
        smoke_chat=False,
        train_run="pyro-test-run",
    )
    assert report["train_run"] == "pyro-test-run"
    assert mock_register.call_count == len(SUITE_FILES)
    conn.close.assert_called_once()
    for call in mock_register.call_args_list:
        assert call.args[0] is conn
        assert call.kwargs["train_run_name"] == "pyro-test-run"
        assert call.kwargs["status"] == "completed"
