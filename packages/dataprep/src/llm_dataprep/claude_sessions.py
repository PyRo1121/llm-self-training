"""Ingest Claude Code JSONL from ~/.claude/projects (if present)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.cursor_transcripts import _summarize_tool_input
from llm_dataprep.raw_io import append_records

DEFAULT_ROOT = Path.home() / ".claude/projects"


def iter_session_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    yield from sorted(root.rglob("*.jsonl"))


def _normalize_type(kind: str | None) -> str | None:
    if kind == "human":
        return "user"
    if kind in ("user", "assistant"):
        return kind
    return None


def _role_and_text(obj: dict[str, Any]) -> tuple[str | None, str]:
    # Newer shape: type user/assistant/human + message string or object
    role = _normalize_type(obj.get("type"))
    if role:
        msg = obj.get("message")
        if isinstance(msg, str):
            return role, msg.strip()
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return role, content.strip()
            if isinstance(content, list):
                parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    kind = block.get("type")
                    if kind == "text":
                        parts.append(block.get("text") or "")
                    elif kind == "tool_use":
                        name = block.get("name") or "tool"
                        summary = _summarize_tool_input(block.get("input"))
                        parts.append(f"[tool {name}] {summary}")
                return role, "\n".join(p for p in parts if p).strip()
    # Alternate: message.role + message.content (Cursor-like)
    message = obj.get("message") or {}
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            return role, content.strip()
    return None, ""


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    root = root or DEFAULT_ROOT
    files = list(iter_session_files(root))
    if limit_files:
        files = files[:limit_files]
    ingested = datetime.now(timezone.utc).isoformat()

    def all_records() -> Iterator[dict[str, Any]]:
        for fpath in files:
            project = fpath.parent.name
            session_id = fpath.stem
            with fpath.open(encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    role, text = _role_and_text(obj)
                    if not role or not text or len(text) > 200_000:
                        continue
                    yield {
                        "source": "claude_code",
                        "harness": "claude_code",
                        "session_id": session_id,
                        "project_hash": project,
                        "source_path": str(fpath),
                        "line_no": i,
                        "role": role,
                        "text": text,
                        "ingested_at": ingested,
                    }

    if not files:
        return None, 0
    return append_records("claude-sessions", all_records(), out_dir=out_dir)
