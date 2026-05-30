"""Ingest Kiro CLI sessions (JSON/JSONL under ~/.kiro/sessions/cli + optional SQLite)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

CLI_DIR = Path.home() / ".kiro/sessions/cli"
SQLITE_PATHS = (
    Path.home() / ".local/share/kiro-cli/data.sqlite3",
    Path.home() / "Library/Application Support/kiro-cli/data.sqlite3",
)


def _text_from_message(msg: dict[str, Any]) -> str:
    content = msg.get("content") or msg.get("text")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
        return "\n".join(str(p) for p in parts).strip()
    return ""


def _from_json_file(fpath: Path, ingested: str) -> Iterator[dict[str, Any]]:
    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    session_id = data.get("id") or data.get("sessionId") or fpath.stem
    messages = data.get("messages") or data.get("history") or []
    if not isinstance(messages, list):
        return
    for i, msg in enumerate(messages, start=1):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _text_from_message(msg)
        if not text or len(text) > 200_000:
            continue
        yield {
            "source": "kiro",
            "harness": "kiro",
            "session_id": str(session_id),
            "source_path": str(fpath),
            "line_no": i,
            "role": role,
            "text": text,
            "ingested_at": ingested,
        }


def _from_jsonl_file(fpath: Path, ingested: str) -> Iterator[dict[str, Any]]:
    session_id = fpath.stem
    with fpath.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            text = _text_from_message(msg)
            if not text or len(text) > 200_000:
                continue
            yield {
                "source": "kiro",
                "harness": "kiro",
                "session_id": session_id,
                "source_path": str(fpath),
                "line_no": i,
                "role": role,
                "text": text,
                "ingested_at": ingested,
            }


def _from_sqlite(db: Path, ingested: str, limit_rows: int | None) -> Iterator[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "messages" in tables:
            sql = "SELECT session_id, role, content FROM messages WHERE role IN ('user','assistant')"
            if limit_rows:
                sql += f" LIMIT {int(limit_rows)}"
            for session_id, role, content in conn.execute(sql):
                text = content if isinstance(content, str) else str(content or "")
                text = text.strip()
                if text.startswith("{"):
                    try:
                        obj = json.loads(text)
                        text = _text_from_message(obj) if isinstance(obj, dict) else text
                    except json.JSONDecodeError:
                        pass
                if not text or len(text) > 200_000:
                    continue
                yield {
                    "source": "kiro",
                    "harness": "kiro",
                    "session_id": str(session_id),
                    "role": role,
                    "text": text,
                    "ingested_at": ingested,
                }
            return
        if "conversations" in tables:
            sql = "SELECT id, data FROM conversations"
            if limit_rows:
                sql += f" LIMIT {int(limit_rows)}"
            for conv_id, blob in conn.execute(sql):
                if not blob:
                    continue
                try:
                    data = json.loads(blob if isinstance(blob, str) else blob.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                    continue
                if not isinstance(data, dict):
                    continue
                messages = data.get("messages") or []
                for i, msg in enumerate(messages, start=1):
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role")
                    if role not in ("user", "assistant"):
                        continue
                    text = _text_from_message(msg)
                    if not text:
                        continue
                    yield {
                        "source": "kiro",
                        "harness": "kiro",
                        "session_id": str(conv_id),
                        "line_no": i,
                        "role": role,
                        "text": text,
                        "ingested_at": ingested,
                    }
    except sqlite3.Error:
        return
    finally:
        conn.close()


def ingest(
    cli_dir: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    root = cli_dir or CLI_DIR
    ingested = datetime.now(timezone.utc).isoformat()
    files: list[Path] = []
    if root.is_dir():
        files.extend(sorted(root.glob("*.json")))
        files.extend(sorted(root.glob("*.jsonl")))
    if limit_files:
        files = files[:limit_files]

    def records() -> Iterator[dict[str, Any]]:
        for fpath in files:
            if fpath.suffix == ".jsonl":
                yield from _from_jsonl_file(fpath, ingested)
            else:
                yield from _from_json_file(fpath, ingested)
        for db in SQLITE_PATHS:
            if db.is_file():
                yield from _from_sqlite(db, ingested, limit_files)
                break

    if not files and not any(p.is_file() for p in SQLITE_PATHS):
        return None, 0
    return append_records("kiro-sessions", records(), out_dir=out_dir)
