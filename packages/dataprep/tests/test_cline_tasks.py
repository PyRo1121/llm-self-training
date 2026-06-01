"""Cline task JSON parsers."""

from __future__ import annotations

from llm_dataprep.cline_tasks import _messages_from_ui


def test_messages_from_ui_say_ask_schema() -> None:
    data = [
        {"type": "say", "say": "task", "text": "Run pytest on packages/dataprep", "ts": 1},
        {"type": "say", "say": "text", "text": "I'll run the test suite now.", "ts": 2},
        {"type": "ask", "ask": "followup", "text": "Should I include integration tests?", "ts": 3},
        {"type": "say", "say": "user_feedback", "text": "Yes, run the full suite.", "ts": 4},
        {"type": "say", "say": "api_req_started", "text": "API request started", "ts": 5},
        {"type": "say", "say": "text", "text": "Streaming…", "partial": True, "ts": 6},
        {"role": "user", "text": "legacy user line"},
        {"role": "assistant", "text": "legacy assistant line"},
    ]
    msgs = list(_messages_from_ui(data))
    assert msgs == [
        ("user", "Run pytest on packages/dataprep"),
        ("assistant", "I'll run the test suite now."),
        ("assistant", "Should I include integration tests?"),
        ("user", "Yes, run the full suite."),
        ("user", "legacy user line"),
        ("assistant", "legacy assistant line"),
    ]
