"""Ingest Zed AI threads from ~/.local/share/zed/threads/threads.db (zstd JSON blobs)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

DEFAULT_DB = Path.home() / ".local/share/zed/threads/threads.db"


def _decompress_blob(data: bytes, data_type: str) -> str | None:
    if data_type == "json":
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if data_type != "zstd":
        return None
    try:
        import zstandard as zstd
    except ImportError:
        return None
    try:
        return zstd.ZstdDecompressor().decompress(data).decode("utf-8")
    except Exception:
        return None


def _messages_from_thread(obj: dict[str, Any]) -> Iterator[tuple[str, str]]:
    for key in ("messages", "entries", "conversation"):
        items = obj.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            role = item.get("role") or item.get("type")
            if role not in ("user", "assistant"):
                continue
            content = item.get("content") or item.get("text")
            if isinstance(content, str) and content.strip():
                yield role, content.strip()
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text") or "")
                text = "\n".join(parts).strip()
                if text:
                    yield role, text
        return


def ingest(
    db_path: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_threads: int | None = None,
) -> tuple[Path | None, int]:
    db = db_path or DEFAULT_DB
    if not db.is_file():
        return None, 0
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            sql = "SELECT id, data_type, data FROM threads"
            if limit_threads:
                sql += f" LIMIT {int(limit_threads)}"
            for thread_id, data_type, blob in conn.execute(sql):
                if not isinstance(blob, (bytes, memoryview)):
                    continue
                raw = bytes(blob)
                json_text = _decompress_blob(raw, str(data_type or "zstd"))
                if not json_text:
                    continue
                try:
                    obj = json.loads(json_text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                for role, text in _messages_from_thread(obj):
                    if len(text) > 200_000:
                        continue
                    yield {
                        "source": "zed_ai",
                        "harness": "zed_ai",
                        "session_id": str(thread_id),
                        "role": role,
                        "text": text,
                        "ingested_at": ingested,
                    }
        except sqlite3.Error:
            return
        finally:
            conn.close()

    return append_records("zed-threads", records(), out_dir=out_dir)
