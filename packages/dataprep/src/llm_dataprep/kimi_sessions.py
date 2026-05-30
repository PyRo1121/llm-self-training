"""Ingest Kimi CLI context.jsonl (preferred) and wire.jsonl fallback."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

DEFAULT_ROOT = Path.home() / ".kimi/sessions"


def _text_from_kimi_message(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "\n".join(parts).strip()
    return ""


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    root = root or DEFAULT_ROOT
    if not root.is_dir():
        return None, 0
    context_files = sorted(root.rglob("context.jsonl"))
    if limit_files:
        context_files = context_files[:limit_files]
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        for fpath in context_files:
            session_id = fpath.parent.name
            with fpath.open(encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    role = obj.get("role")
                    if role == "_system_prompt":
                        continue
                    if role not in ("user", "assistant"):
                        continue
                    text = _text_from_kimi_message(obj)
                    if not text or len(text) > 200_000:
                        continue
                    yield {
                        "source": "kimi",
                        "harness": "kimi",
                        "session_id": session_id,
                        "source_path": str(fpath),
                        "line_no": i,
                        "role": role,
                        "text": text,
                        "ingested_at": ingested,
                    }

    if not context_files:
        return None, 0
    return append_records("kimi-sessions", records(), out_dir=out_dir)
