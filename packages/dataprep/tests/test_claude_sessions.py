"""Claude Code session JSONL parsing."""

from __future__ import annotations

import json

from llm_dataprep.claude_sessions import _role_and_text


def test_human_message_string_maps_to_user() -> None:
    obj = {
        "type": "human",
        "message": "Add retry logic to the GitHub harvest client.",
    }
    role, text = _role_and_text(obj)
    assert role == "user"
    assert "retry logic" in text


def test_assistant_tool_only_turn_summarized() -> None:
    obj = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": "Read",
                    "input": {"path": "/tmp/foo.py"},
                }
            ]
        },
    }
    role, text = _role_and_text(obj)
    assert role == "assistant"
    assert text.startswith("[tool Read]")
    assert "foo.py" in text


def test_assistant_mixed_text_and_tool_use() -> None:
    obj = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "Checking repo."},
                {
                    "type": "tool_use",
                    "name": "Shell",
                    "input": {
                        "command": "rtk git status",
                        "description": "Check git status",
                    },
                },
            ]
        },
    }
    role, text = _role_and_text(obj)
    assert role == "assistant"
    assert "Checking repo." in text
    assert "[tool Shell]" in text
    assert "rtk git status" in text
