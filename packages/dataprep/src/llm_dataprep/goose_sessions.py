"""Ingest Goose messages from ~/.local/share/goose/sessions/sessions.db."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records


def _db_path() -> Path:
    env = os.environ.get("GOOSE_PATH_ROOT")
    if env:
        return Path(env).expanduser() / "sessions" / "sessions.db"
    return Path.home() / ".local/share/goose/sessions/sessions.db"


def _content_text(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, str):
                return obj.strip()
            if isinstance(obj, dict):
                return (obj.get("text") or obj.get("content") or json.dumps(obj))[:200_000]
            if isinstance(obj, list):
                return "\n".join(str(x) for x in obj)[:200_000]
        except json.JSONDecodeError:
            pass
    return raw[:200_000]


def ingest(
    *,
    out_dir: Path | None = None,
    limit_rows: int | None = None,
) -> tuple[Path | None, int]:
    db = _db_path()
    if not db.is_file():
        return None, 0
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            sql = """
                SELECT session_id, role, content
                FROM messages
                WHERE role IN ('user', 'assistant')
                ORDER BY timestamp
            """
            if limit_rows:
                sql += f" LIMIT {int(limit_rows)}"
            for session_id, role, content in conn.execute(sql):
                text = _content_text(content or "")
                if not text:
                    continue
                yield {
                    "source": "goose",
                    "harness": "goose",
                    "session_id": session_id,
                    "role": role,
                    "text": text,
                    "ingested_at": ingested,
                }
        except sqlite3.Error:
            return
        finally:
            conn.close()

    return append_records("goose-sessions", records(), out_dir=out_dir)
