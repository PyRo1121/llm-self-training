"""Ingest Pi coding-agent JSONL from ~/.pi/agent/sessions (pi-mono format)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.harnesses import _pi_sessions_root
from llm_dataprep.raw_io import append_records

SKIP_TYPES = frozenset({"session", "model_change", "thinking_level_change", "compaction", "branch_summary", "label", "session_info", "custom"})


def _text_from_message(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "\n".join(parts).strip()
    return ""


def iter_session_files(root: Path | None = None) -> Iterator[Path]:
    root = root or _pi_sessions_root()
    if not root.exists():
        return
    yield from sorted(root.rglob("*.jsonl"))


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    root = root or Path(os.environ.get("PI_CODING_AGENT_SESSION_DIR", _pi_sessions_root()))
    files = list(iter_session_files(root))
    if limit_files:
        files = files[:limit_files]
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        for fpath in files:
            session_id = fpath.stem.split("_")[-1] if "_" in fpath.stem else fpath.stem
            with fpath.open(encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") in SKIP_TYPES:
                        continue
                    if obj.get("type") == "custom_message" and not obj.get("display", True):
                        continue
                    if obj.get("type") not in ("message", "custom_message"):
                        continue
                    msg = obj.get("message") if obj.get("type") == "message" else None
                    if obj.get("type") == "custom_message":
                        role = "user"
                        text = obj.get("content")
                        if isinstance(text, list):
                            text = _text_from_message({"content": text})
                        elif not isinstance(text, str):
                            text = ""
                    else:
                        if not isinstance(msg, dict):
                            continue
                        role = msg.get("role")
                        if role not in ("user", "assistant"):
                            continue
                        text = _text_from_message(msg)
                    if not text or len(text) > 200_000:
                        continue
                    yield {
                        "source": "pi",
                        "harness": "pi",
                        "session_id": session_id,
                        "source_path": str(fpath),
                        "line_no": i,
                        "role": role,
                        "text": text,
                        "ingested_at": ingested,
                    }

    if not files:
        return None, 0
    return append_records("pi-sessions", records(), out_dir=out_dir)
