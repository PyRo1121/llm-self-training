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
                for i, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("ephemeral"):
                        continue
                    etype = ev.get("type")
                    data = ev.get("data") or {}
                    role = None
                    if etype in USER_TYPES:
                        role = "user"
                    elif etype in ASSISTANT_TYPES:
                        role = "assistant"
                    if not role:
                        continue
                    text = (data.get("content") or "").strip()
                    if not text or len(text) > 200_000:
                        continue
                    yield {
                        "source": "copilot",
                        "harness": "copilot",
                        "session_id": session_id,
                        "source_path": str(path),
                        "line_no": i,
                        "event_type": etype,
                        "role": role,
                        "text": text,
                        "ingested_at": ingested,
                    }

    if not sessions:
        return None, 0
    return append_records("copilot-sessions", records(), out_dir=out_dir)
