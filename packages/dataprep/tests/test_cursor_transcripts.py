"""Cursor transcript parsing — tool-heavy agent sessions."""

from __future__ import annotations

import json
from pathlib import Path

from llm_dataprep.cursor_transcripts import ingest, parse_line, session_id_from_path


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


def _write_transcript(tmp_path: Path, session_id: str, *lines: str) -> None:
    tdir = tmp_path / "proj" / "agent-transcripts" / session_id
    tdir.mkdir(parents=True)
    (tdir / f"{session_id}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_ingest_writes_records(tmp_path: Path) -> None:
    sid = "sess-uuid"
    _write_transcript(
        tmp_path,
        sid,
        json.dumps({"role": "user", "message": "hello world"}),
        json.dumps(
            {
                "role": "assistant",
                "message": {"content": [{"type": "text", "text": "hi there"}]},
            }
        ),
        "",
        "not-json",
    )
    out = tmp_path / "raw"
    path, n = ingest(projects_root=tmp_path, out_dir=out)
    assert path is not None
    assert n == 2
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert rec["source"] == "cursor"
    assert rec["session_id"] == sid
    assert rec["harness"] == "cursor"


def test_ingest_empty_root_returns_none(tmp_path: Path) -> None:
    path, n = ingest(projects_root=tmp_path / "missing", out_dir=tmp_path / "raw")
    assert path is None
    assert n == 0


def test_ingest_limit_files_and_skips_subagents(tmp_path: Path) -> None:
    _write_transcript(tmp_path, "a", json.dumps({"role": "user", "message": "one"}))
    _write_transcript(tmp_path, "b", json.dumps({"role": "user", "message": "two"}))
    sub = tmp_path / "proj" / "agent-transcripts" / "subagents" / "c"
    sub.mkdir(parents=True)
    (sub / "c.jsonl").write_text(
        json.dumps({"role": "user", "message": "sub"}) + "\n", encoding="utf-8"
    )
    path, n = ingest(projects_root=tmp_path, out_dir=tmp_path / "raw", limit_files=1)
    assert path is not None
    assert n == 1


def test_session_id_from_path_when_stem_differs() -> None:
    path = Path("/tmp/agent-transcripts/uuid/other-name.jsonl")
    assert session_id_from_path(path) == "other-name"
