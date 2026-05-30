"""Ingest Watchfire session JSONL transcripts from ~/.watchfire/logs/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

DEFAULT_ROOT = Path.home() / ".watchfire/logs"


def _text_from_obj(obj: dict[str, Any]) -> tuple[str | None, str]:
    role = obj.get("role") or obj.get("type")
    if role in ("human",):
        role = "user"
    if role not in ("user", "assistant"):
        return None, ""
    content = obj.get("content") or obj.get("message") or obj.get("text")
    if isinstance(content, str):
        return role, content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text") or "")
                elif "text" in block:
                    parts.append(str(block["text"]))
        return role, "\n".join(parts).strip()
    return role, ""


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    root = root or DEFAULT_ROOT
    if not root.is_dir():
        return None, 0
    files = sorted(root.rglob("*.jsonl"))
    if limit_files:
        files = files[:limit_files]
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        for fpath in files:
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
                    if not isinstance(obj, dict):
                        continue
                    role, text = _text_from_obj(obj)
                    if not role or not text or len(text) > 200_000:
                        continue
                    yield {
                        "source": "watchfire",
                        "harness": "watchfire",
                        "session_id": session_id,
                        "source_path": str(fpath),
                        "line_no": i,
                        "role": role,
                        "text": text,
                        "ingested_at": ingested,
                    }

    if not files:
        return None, 0
    return append_records("watchfire-logs", records(), out_dir=out_dir)
