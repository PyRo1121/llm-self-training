"""Cursor transcript parsing — tool-heavy agent sessions."""

from __future__ import annotations

import json

from llm_dataprep.cursor_transcripts import parse_line, session_id_from_path


def test_tool_use_blocks_become_trainable_text() -> None:
    line = json.dumps(
        {
            "role": "assistant",
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
    )
    rec = parse_line(
        line,
        session_id="abc",
        source_path="/tmp/x.jsonl",
        line_no=1,
    )
    assert rec is not None
    assert "[tool Shell]" in rec.text
    assert "rtk git status" in rec.text
    assert len(rec.text) >= 40
    assert rec.has_tool_use is True


def test_tool_only_turn_is_kept() -> None:
    line = json.dumps(
        {
            "role": "assistant",
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
    )
    rec = parse_line(line, session_id="abc", source_path="/tmp/x.jsonl", line_no=2)
    assert rec is not None
    assert rec.text.startswith("[tool Read]")
    assert "foo.py" in rec.text


def test_session_id_from_path() -> None:
    path = __import__("pathlib").Path(
        "/home/u/.cursor/projects/p/agent-transcripts/uuid/uuid.jsonl"
    )
    assert session_id_from_path(path) == "uuid"


def test_human_role_normalized_to_user() -> None:
    line = json.dumps(
        {
            "role": "Human",
            "message": "Fix the auth module please.",
        }
    )
    rec = parse_line(line, session_id="abc", source_path="/tmp/x.jsonl", line_no=3)
    assert rec is not None
    assert rec.role == "user"
    assert "auth module" in rec.text


def test_type_human_with_string_message_content() -> None:
    line = json.dumps(
        {
            "type": "human",
            "message": {"content": "Explain this pytest failure."},
        }
    )
    rec = parse_line(line, session_id="abc", source_path="/tmp/x.jsonl", line_no=4)
    assert rec is not None
    assert rec.role == "user"
    assert "pytest failure" in rec.text
