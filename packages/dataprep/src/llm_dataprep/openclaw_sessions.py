"""Ingest OpenClaw transcript JSONL (~/.openclaw/agents/*/sessions/*.jsonl)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

DEFAULT_ROOT = Path.home() / ".openclaw"


def iter_transcripts(root: Path | None = None) -> Iterator[Path]:
    root = root or DEFAULT_ROOT
    agents = root / "agents"
    if agents.is_dir():
        yield from sorted(agents.rglob("*.jsonl"))
    legacy = root / "sessions"
    if legacy.is_dir():
        yield from sorted(legacy.glob("*.jsonl"))


def _message_text(msg: dict[str, Any]) -> str:
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
    return ""


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    files = list(iter_transcripts(root))
    if limit_files:
        files = files[:limit_files]
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        for fpath in files:
            if fpath.name == "sessions.jsonl":
                continue
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
                    if obj.get("type") not in ("message", "custom_message"):
                        continue
                    msg = obj.get("message") or obj
                    role = msg.get("role")
                    if role not in ("user", "assistant"):
                        continue
                    text = _message_text(msg)
                    if not text or len(text) > 200_000:
                        continue
                    yield {
                        "source": "openclaw",
                        "harness": "openclaw",
                        "session_id": session_id,
                        "source_path": str(fpath),
                        "line_no": i,
                        "role": role,
                        "text": text,
                        "ingested_at": ingested,
                    }

    if not files:
        return None, 0
    return append_records("openclaw-sessions", records(), out_dir=out_dir)
