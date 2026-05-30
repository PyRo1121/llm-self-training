"""Ingest Factory Droid JSONL transcripts under ~/.factory/projects/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

DEFAULT_ROOT = Path.home() / ".factory"


def iter_transcripts(root: Path | None = None) -> Iterator[Path]:
    root = root or DEFAULT_ROOT
    projects = root / "projects"
    if projects.is_dir():
        yield from sorted(projects.rglob("*.jsonl"))
    sessions = root / "sessions"
    if sessions.is_dir():
        yield from sorted(sessions.rglob("*.jsonl"))


def _role_text(obj: dict[str, Any]) -> tuple[str | None, str]:
    role = obj.get("role") or obj.get("type")
    if role in ("human", "Human"):
        role = "user"
    if role not in ("user", "assistant"):
        return None, ""
    text = obj.get("content") or obj.get("text") or ""
    if isinstance(text, list):
        text = "\n".join(str(x) for x in text)
    return role, str(text).strip()


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
                    if isinstance(obj, dict) and "messages" in obj:
                        for msg in obj.get("messages") or []:
                            if isinstance(msg, dict):
                                role, text = _role_text(msg)
                                if role and text:
                                    yield _rec(session_id, fpath, i, role, text, ingested)
                        continue
                    role, text = _role_text(obj)
                    if role and text and len(text) <= 200_000:
                        yield _rec(session_id, fpath, i, role, text, ingested)

    if not files:
        return None, 0
    return append_records("factory-droid", records(), out_dir=out_dir)


def _rec(session_id: str, fpath: Path, line_no: int, role: str, text: str, ingested: str) -> dict[str, Any]:
    return {
        "source": "factory",
        "harness": "factory",
        "session_id": session_id,
        "source_path": str(fpath),
        "line_no": line_no,
        "role": role,
        "text": text,
        "ingested_at": ingested,
    }
