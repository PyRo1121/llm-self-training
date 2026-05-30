"""Ingest Amp thread JSON from ~/.local/share/amp/threads/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

DEFAULT_ROOT = Path.home() / ".local/share/amp/threads"


def _text_from_message(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text") or "")
                elif "text" in block:
                    parts.append(str(block["text"]))
        return "\n".join(parts).strip()
    return (msg.get("text") or "").strip()


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    root = root or DEFAULT_ROOT
    if not root.is_dir():
        return None, 0
    files = sorted(root.glob("*.json"))
    if limit_files:
        files = files[:limit_files]
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        for fpath in files:
            try:
                thread = json.loads(fpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            session_id = thread.get("id") or fpath.stem
            messages = thread.get("messages") or []
            if not isinstance(messages, list):
                continue
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role not in ("user", "assistant"):
                    continue
                text = _text_from_message(msg)
                if not text or len(text) > 200_000:
                    continue
                yield {
                    "source": "amp",
                    "harness": "amp",
                    "session_id": session_id,
                    "source_path": str(fpath),
                    "role": role,
                    "text": text,
                    "ingested_at": ingested,
                }

    if not files:
        return None, 0
    return append_records("amp-threads", records(), out_dir=out_dir)
