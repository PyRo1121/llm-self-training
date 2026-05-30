"""Partial Windsurf ingest from state.vscdb (NOT .pb AES decrypt).

Reads JSON chat bubbles from VS Code ItemTable keys. See AGENT_HARNESSES.md.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

CHAT_KEYS = (
    "workbench.panel.aichat.view.aichat.chatdata",
    "aiChat.chatdata",
    "chat.data",
    "cascade.chatdata",
)


def find_vscdb_files() -> list[Path]:
    home = Path.home()
    patterns = [
        home / ".config/Windsurf",
        home / ".config/windsurf",
    ]
    dbs: list[Path] = []
    for base in patterns:
        if not base.is_dir():
            continue
        dbs.extend(base.rglob("state.vscdb"))
    return sorted(set(dbs))


def _conversations_from_db(db_path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        if "ItemTable" not in tables:
            conn.close()
            return out
        for key in CHAT_KEYS:
            cur.execute("SELECT value FROM ItemTable WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row:
                continue
            try:
                data = json.loads(row[0])
            except json.JSONDecodeError:
                continue
            for tab in data.get("tabs") or []:
                bubbles = tab.get("bubbles") or []
                messages = []
                for bubble in bubbles:
                    btype = bubble.get("type")
                    role = "user" if btype == "user" else "assistant"
                    text = (bubble.get("rawText") or bubble.get("text") or "").strip()
                    if text:
                        messages.append({"role": role, "text": text})
                if messages:
                    out.append(
                        {
                            "workspace_db": str(db_path),
                            "tab_id": tab.get("tabId"),
                            "title": tab.get("chatTitle"),
                            "messages": messages,
                        }
                    )
            if out:
                break
        conn.close()
    except sqlite3.Error:
        pass
    return out


def ingest(
    *,
    out_dir: Path | None = None,
    limit_dbs: int | None = None,
) -> tuple[Path | None, int]:
    dbs = find_vscdb_files()
    if limit_dbs:
        dbs = dbs[:limit_dbs]
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        for db in dbs:
            for conv in _conversations_from_db(db):
                session_id = conv.get("tab_id") or db.stem
                for msg in conv.get("messages") or []:
                    role = msg.get("role")
                    text = (msg.get("text") or "").strip()
                    if role not in ("user", "assistant") or not text:
                        continue
                    if len(text) > 200_000:
                        continue
                    yield {
                        "source": "windsurf",
                        "harness": "windsurf",
                        "session_id": str(session_id),
                        "source_path": conv.get("workspace_db", str(db)),
                        "role": role,
                        "text": text,
                        "ingested_at": ingested,
                        "ingest_note": "vscdb_only_not_pb_decrypt",
                    }

    if not dbs:
        return None, 0
    return append_records("windsurf-vscdb", records(), out_dir=out_dir)
