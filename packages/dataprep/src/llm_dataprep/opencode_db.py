"""Ingest OpenCode sessions from ~/.local/share/opencode/opencode.db (SQLite)."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

DEFAULT_DB = Path.home() / ".local/share/opencode/opencode.db"
LEGACY_MESSAGE_ROOT = Path.home() / ".local/share/opencode/storage/message"


def _db_path(root: Path | None) -> Path:
    if root and (root / "opencode.db").is_file():
        return root / "opencode.db"
    env = os.environ.get("OPENCODE_DATA_DIR")
    if env:
        p = Path(env).expanduser() / "opencode.db"
        if p.is_file():
            return p
    return DEFAULT_DB


def _ingest_sqlite(db: Path, *, limit_messages: int | None) -> Iterator[dict[str, Any]]:
    ingested = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """
            SELECT m.id AS message_id, m.session_id, m.data AS msg_data,
                   (SELECT p.data FROM part p
                    WHERE p.message_id = m.id AND json_extract(p.data, '$.type') = 'text'
                    ORDER BY p.time_created LIMIT 1) AS part_data
            FROM message m
            WHERE json_extract(m.data, '$.role') IN ('user', 'assistant')
            ORDER BY m.time_created
            """
        )
        for i, row in enumerate(cur):
            if limit_messages and i >= limit_messages:
                break
            msg_data = json.loads(row["msg_data"])
            role = msg_data.get("role")
            text = ""
            if row["part_data"]:
                part = json.loads(row["part_data"])
                if part.get("type") == "text":
                    text = (part.get("text") or "").strip()
            if not text:
                continue
            if len(text) > 200_000:
                continue
            yield {
                "source": "opencode",
                "harness": "opencode",
                "session_id": row["session_id"],
                "message_id": row["message_id"],
                "role": role,
                "text": text,
                "ingested_at": ingested,
            }
    finally:
        conn.close()


def _ingest_legacy_json(root: Path, *, limit_files: int | None) -> Iterator[dict[str, Any]]:
    ingested = datetime.now(timezone.utc).isoformat()
    files = sorted(root.rglob("*.json"))
    if limit_files:
        files = files[:limit_files]
    for fpath in files:
        try:
            obj = json.loads(fpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        role = obj.get("role")
        parts = obj.get("parts") or obj.get("content")
        text = ""
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "text":
                    text += (p.get("text") or "") + "\n"
        elif isinstance(obj.get("text"), str):
            text = obj["text"]
        text = text.strip()
        if role not in ("user", "assistant") or not text:
            continue
        yield {
            "source": "opencode",
            "harness": "opencode",
            "session_id": fpath.parent.name,
            "source_path": str(fpath),
            "role": role,
            "text": text,
            "ingested_at": ingested,
        }


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    db = _db_path(root)
    if db.is_file():

        def gen() -> Iterator[dict[str, Any]]:
            yield from _ingest_sqlite(db, limit_messages=limit_files)

        return append_records("opencode-sessions", gen(), out_dir=out_dir)

    if LEGACY_MESSAGE_ROOT.is_dir():

        def gen_legacy() -> Iterator[dict[str, Any]]:
            yield from _ingest_legacy_json(LEGACY_MESSAGE_ROOT, limit_files=limit_files)

        return append_records("opencode-sessions", gen_legacy(), out_dir=out_dir)

    return None, 0
