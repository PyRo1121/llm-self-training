"""Ingest GitHub Copilot CLI session events.jsonl (~/.copilot/session-state/)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

DEFAULT_ROOT = Path.home() / ".copilot/session-state"
USER_TYPES = frozenset({"user.message"})
ASSISTANT_TYPES = frozenset({"assistant.message"})


def record_from_copilot_event(
    ev: dict[str, Any],
    *,
    session_id: str,
    source_path: str,
    line_no: int,
    ingested_at: str,
    source: str = "copilot",
) -> dict[str, Any] | None:
    """One Copilot Chronicle events.jsonl line → raw record, or None if skipped."""
    if ev.get("ephemeral"):
        return None
    etype = ev.get("type")
    data = ev.get("data") or {}
    role = None
    if etype in USER_TYPES:
        role = "user"
    elif etype in ASSISTANT_TYPES:
        role = "assistant"
    if not role:
        return None
    text = (data.get("content") or "").strip()
    if not text or len(text) > 200_000:
        return None
    return {
        "source": source,
        "harness": "copilot",
        "session_id": session_id,
        "source_path": source_path,
        "line_no": line_no,
        "event_type": etype,
        "role": role,
        "text": text,
        "ingested_at": ingested_at,
    }


def iter_copilot_event_lines(
    lines: Iterator[str] | list[str],
    *,
    session_id: str,
    source_path: str,
    ingested_at: str,
    max_lines: int | None = None,
    source: str = "copilot",
) -> Iterator[dict[str, Any]]:
    for i, line in enumerate(lines, start=1):
        if max_lines is not None and i > max_lines:
            break
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        rec = record_from_copilot_event(
            ev,
            session_id=session_id,
            source_path=source_path,
            line_no=i,
            ingested_at=ingested_at,
            source=source,
        )
        if rec is not None:
            yield rec


def iter_event_files(root: Path | None = None) -> Iterator[tuple[Path, str]]:
    root = root or DEFAULT_ROOT
    if not root.is_dir():
        return
    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir():
            continue
        events = session_dir / "events.jsonl"
        if events.is_file():
            yield events, session_dir.name


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_sessions: int | None = None,
) -> tuple[Path | None, int]:
    sessions = list(iter_event_files(root))
    if limit_sessions:
        sessions = sessions[:limit_sessions]
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        for path, session_id in sessions:
            with path.open(encoding="utf-8", errors="replace") as fh:
                yield from iter_copilot_event_lines(
                    fh,
                    session_id=session_id,
                    source_path=str(path),
                    ingested_at=ingested,
                )

    if not sessions:
        return None, 0
    return append_records("copilot-sessions", records(), out_dir=out_dir)
