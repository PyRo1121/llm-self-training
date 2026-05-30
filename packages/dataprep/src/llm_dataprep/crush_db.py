"""Ingest Crush messages from per-project .crush/crush.db (via ~/.local/share/crush/projects.json)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

PROJECTS_JSON = Path.home() / ".local/share/crush/projects.json"


def _text_from_parts(parts_raw: str) -> str:
    try:
        parts = json.loads(parts_raw)
    except json.JSONDecodeError:
        return parts_raw.strip()[:200_000]
    if not isinstance(parts, list):
        return str(parts)[:200_000]
    texts = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            data = part.get("data") or {}
            if isinstance(data, dict):
                texts.append(data.get("text") or "")
            elif isinstance(data, str):
                texts.append(data)
    return "\n".join(texts).strip()


def _discover_dbs() -> list[Path]:
    dbs: list[Path] = []
    if PROJECTS_JSON.is_file():
        try:
            projects = json.loads(PROJECTS_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            projects = []
        if isinstance(projects, list):
            for entry in projects:
                if not isinstance(entry, dict):
                    continue
                path = entry.get("path") or entry.get("root")
                if not path:
                    continue
                db = Path(path).expanduser() / ".crush" / "crush.db"
                if db.is_file():
                    dbs.append(db)
    home = Path.home()
    for extra in (home / "Documents", home):
        if not extra.is_dir():
            continue
        for db in extra.rglob(".crush/crush.db"):
            if db not in dbs:
                dbs.append(db)
            if len(dbs) >= 50:
                break
    return dbs


def ingest(
    *,
    out_dir: Path | None = None,
    limit_dbs: int | None = None,
    limit_rows: int | None = None,
) -> tuple[Path | None, int]:
    dbs = _discover_dbs()
    if limit_dbs:
        dbs = dbs[:limit_dbs]
    if not dbs:
        return None, 0
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        for db in dbs:
            project = db.parent.parent.name
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                sql = "SELECT id, role, parts FROM messages WHERE role IN ('user','assistant')"
                if limit_rows:
                    sql += f" LIMIT {int(limit_rows)}"
                for msg_id, role, parts in conn.execute(sql):
                    text = _text_from_parts(parts or "")
                    if not text or len(text) > 200_000:
                        continue
                    yield {
                        "source": "crush",
                        "harness": "crush",
                        "session_id": str(msg_id),
                        "project": project,
                        "source_path": str(db),
                        "role": role,
                        "text": text,
                        "ingested_at": ingested,
                    }
            except sqlite3.Error:
                continue
            finally:
                conn.close()

    return append_records("crush-db", records(), out_dir=out_dir)
