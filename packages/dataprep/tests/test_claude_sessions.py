"""Claude Code session JSONL parsing."""

from __future__ import annotations

import json
from pathlib import Path

from llm_dataprep.claude_sessions import _role_and_text, _should_skip_line, ingest


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


def test_string_message_strips_user_query_tags() -> None:
    obj = {
        "type": "human",
        "message": "<user_query>\nFix auth middleware.\n</user_query>",
    }
    role, text = _role_and_text(obj)
    assert role == "user"
    assert text == "Fix auth middleware."
    assert "<user_query>" not in text


def test_string_content_strips_user_query_tags() -> None:
    obj = {
        "type": "human",
        "message": {"content": "<user_query>Wire harvest parser.</user_query>"},
    }
    role, text = _role_and_text(obj)
    assert role == "user"
    assert text == "Wire harvest parser."


def test_should_skip_line_flags() -> None:
    assert _should_skip_line({"isCompactSummary": True, "type": "assistant"})
    assert _should_skip_line({"isSidechain": True, "type": "user"})
    assert _should_skip_line({"type": "system", "subtype": "session_start"})
    assert not _should_skip_line({"type": "human", "message": "hello"})


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


def _write_session(tmp_path: Path, project: str, session_id: str, *lines: str) -> None:
    sdir = tmp_path / project
    sdir.mkdir(parents=True)
    (sdir / f"{session_id}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_ingest_writes_claude_records(tmp_path: Path) -> None:
    _write_session(
        tmp_path,
        "proj-hash",
        "sess-1",
        json.dumps({"type": "human", "message": "Add retry logic."}),
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "On it."}]},
            }
        ),
        json.dumps({"message": {"role": "user", "content": "Cursor-like shape."}}),
        json.dumps({"isCompactSummary": True, "type": "assistant", "message": "summary"}),
        json.dumps({"isSidechain": True, "type": "user", "message": "branch"}),
        json.dumps({"type": "system", "subtype": "session_start"}),
        "",
        "bad-json",
        json.dumps({"type": "assistant", "message": "x" * 200_001}),
    )
    out = tmp_path / "raw"
    path, n = ingest(root=tmp_path, out_dir=out)
    assert path is not None
    assert n == 3
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert rec["source"] == "claude_code"
    assert rec["session_id"] == "sess-1"
    assert rec["project_hash"] == "proj-hash"


def test_ingest_empty_root_returns_none(tmp_path: Path) -> None:
    path, n = ingest(root=tmp_path / "missing", out_dir=tmp_path / "raw")
    assert path is None
    assert n == 0


def test_ingest_limit_files(tmp_path: Path) -> None:
    _write_session(tmp_path, "p1", "s1", json.dumps({"type": "human", "message": "one"}))
    _write_session(tmp_path, "p2", "s2", json.dumps({"type": "human", "message": "two"}))
    path, n = ingest(root=tmp_path, out_dir=tmp_path / "raw", limit_files=1)
    assert path is not None
    assert n == 1
