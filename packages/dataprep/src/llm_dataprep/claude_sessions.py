"""Ingest Claude Code JSONL from ~/.claude/projects (if present)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.message_blocks import (
    USER_QUERY_RE,
    normalize_role,
    text_from_content_blocks,
)
from llm_dataprep.raw_io import append_records

DEFAULT_ROOT = Path.home() / ".claude/projects"


def _text_from_string(raw: str) -> str:
    text = raw.strip()
    m = USER_QUERY_RE.search(text)
    return m.group(1).strip() if m else text


def _should_skip_line(obj: dict[str, Any]) -> bool:
    if obj.get("isCompactSummary") or obj.get("isSidechain"):
        return True
    kind = obj.get("type")
    return isinstance(kind, str) and kind.lower() == "system"


def iter_session_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    yield from sorted(root.rglob("*.jsonl"))


def _role_and_text(obj: dict[str, Any]) -> tuple[str | None, str]:
    role = normalize_role(obj)
    if not role:
        return None, ""

    msg = obj.get("message")
    if isinstance(msg, str):
        return role, _text_from_string(msg)
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            return role, _text_from_string(content)
        if isinstance(content, list):
            text, _ = text_from_content_blocks(content, include_tool_use=True)
            return role, text

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
                    if _should_skip_line(obj):
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
