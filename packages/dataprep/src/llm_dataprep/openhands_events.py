"""Ingest OpenHands event JSON files (~/.openhands-state/sessions/)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

DEFAULT_ROOTS = (
    Path.home() / ".openhands-state",
    Path.home() / ".openhands",
)


def _event_dirs(root: Path) -> Iterator[tuple[Path, str]]:
    for base in (root / "sessions", root):
        if not base.is_dir():
            continue
        for conv_dir in sorted(base.iterdir()):
            if not conv_dir.is_dir():
                continue
            events = conv_dir / "events"
            if events.is_dir():
                yield events, conv_dir.name


def _text_from_event(ev: dict[str, Any]) -> tuple[str | None, str]:
    # OpenHands V0/V1 shapes vary; try common fields
    if ev.get("action") == "message" or ev.get("source") in ("user", "agent"):
        src = ev.get("source")
        role = "user" if src == "user" else "assistant" if src == "agent" else None
        args = ev.get("args") or ev.get("message") or {}
        if isinstance(args, str):
            return role, args.strip()
        if isinstance(args, dict):
            text = args.get("content") or args.get("thought") or args.get("message") or ""
            if isinstance(text, list):
                text = "\n".join(str(x) for x in text)
            return role, str(text).strip()
    msg = ev.get("message")
    if isinstance(msg, dict):
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            return role, content.strip()
    return None, ""


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_sessions: int | None = None,
) -> tuple[Path | None, int]:
    roots = [root] if root else list(DEFAULT_ROOTS)
    pairs: list[tuple[Path, str]] = []
    for r in roots:
        pairs.extend(_event_dirs(r))
    if limit_sessions:
        pairs = pairs[:limit_sessions]
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        for events_dir, session_id in pairs:
            for ev_path in sorted(events_dir.glob("*.json")):
                try:
                    ev = json.loads(ev_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                role, text = _text_from_event(ev)
                if role not in ("user", "assistant") or not text:
                    continue
                if len(text) > 200_000:
                    continue
                yield {
                    "source": "openhands",
                    "harness": "openhands",
                    "session_id": session_id,
                    "source_path": str(ev_path),
                    "role": role,
                    "text": text,
                    "ingested_at": ingested,
                }

    if not pairs:
        return None, 0
    return append_records("openhands-events", records(), out_dir=out_dir)
