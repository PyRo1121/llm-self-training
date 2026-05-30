"""Ingest Continue.dev session JSON from ~/.continue/sessions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records


def _sessions_dir() -> Path:
    base = os.environ.get("CONTINUE_GLOBAL_DIR") or str(Path.home() / ".continue")
    return Path(base).expanduser() / "sessions"


def _text_from_history_item(item: dict[str, Any]) -> tuple[str | None, str]:
    msg = item.get("message") or item
    role = msg.get("role")
    content = msg.get("content")
    if isinstance(content, str):
        return role, content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return role, "\n".join(parts).strip()
    return role, ""


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    sessions = root or _sessions_dir()
    if not sessions.is_dir():
        return None, 0
    files = sorted(p for p in sessions.glob("*.json") if p.name != "sessions.json")
    if limit_files:
        files = files[:limit_files]
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        for fpath in files:
            try:
                session = json.loads(fpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            history = session.get("history") or []
            session_id = session.get("sessionId") or fpath.stem
            for item in history:
                if not isinstance(item, dict):
                    continue
                role, text = _text_from_history_item(item)
                if role not in ("user", "assistant") or not text:
                    continue
                if len(text) > 200_000:
                    continue
                yield {
                    "source": "continue",
                    "harness": "continue",
                    "session_id": session_id,
                    "source_path": str(fpath),
                    "role": role,
                    "text": text,
                    "ingested_at": ingested,
                }

    if not files:
        return None, 0
    return append_records("continue-sessions", records(), out_dir=out_dir)
