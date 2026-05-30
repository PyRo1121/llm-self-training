"""Ingest T3 Code threads from ~/.t3/userdata/state.sqlite."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.harnesses import _t3_home
from llm_dataprep.raw_io import append_records


def _sqlite_path(root: Path | None) -> Path:
    home = Path(os.environ.get("T3CODE_HOME", os.environ.get("T3_HOME", _t3_home())))
    base = root or (home / "userdata")
    return base / "state.sqlite"


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_rows: int | None = None,
) -> tuple[Path | None, int]:
    db = _sqlite_path(root)
    if not db.is_file():
        return None, 0

    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            sql = """
                SELECT message_id, thread_id, turn_id, role, text, created_at
                FROM projection_thread_messages
                WHERE role IN ('user', 'assistant') AND length(text) > 0
                ORDER BY created_at
            """
            if limit_rows:
                sql += f" LIMIT {int(limit_rows)}"
            for row in conn.execute(sql):
                text = (row[4] or "").strip()
                if len(text) > 200_000:
                    continue
                yield {
                    "source": "t3code",
                    "harness": "t3code",
                    "session_id": row[1],
                    "message_id": row[0],
                    "turn_id": row[2],
                    "role": row[3],
                    "text": text,
                    "created_at": row[5],
                    "ingested_at": ingested,
                }
        finally:
            conn.close()

    return append_records("t3code-threads", records(), out_dir=out_dir)
