"""Ingest run + raw file catalog for warehouse (Phase 1.5)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_core.paths import data_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_ingest_run(
    conn,
    *,
    pipeline: str,
    run_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> str:
    rid = run_id or f"{pipeline}-{uuid.uuid4().hex[:12]}"
    conn.execute(
        """
        INSERT OR REPLACE INTO ingest_runs (
            run_id, pipeline, started_at, status, total_rows, details_json
        ) VALUES (?, ?, ?, 'running', 0, ?)
        """,
        (rid, pipeline, _utc_now(), json.dumps(details or {})),
    )
    conn.commit()
    return rid


def finish_ingest_run(
    conn,
    run_id: str,
    *,
    status: str = "completed",
    total_rows: int = 0,
    details: dict[str, Any] | None = None,
) -> None:
    if details is not None:
        conn.execute(
            """
            UPDATE ingest_runs
            SET finished_at = ?, status = ?, total_rows = ?, details_json = ?
            WHERE run_id = ?
            """,
            (_utc_now(), status, total_rows, json.dumps(details), run_id),
        )
    else:
        conn.execute(
            """
            UPDATE ingest_runs
            SET finished_at = ?, status = ?, total_rows = ?
            WHERE run_id = ?
            """,
            (_utc_now(), status, total_rows, run_id),
        )
    conn.commit()


def _infer_harness(path: Path) -> str | None:
    stem = path.stem
    if stem.startswith("public-"):
        body = stem.removeprefix("public-")
        # drop trailing YYYY-MM-DD
        parts = body.split("-")
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            parts = parts[:-3]
        slug = "_".join(parts) if parts else None
        return f"public_{slug}" if slug else "public"
    if stem.startswith("cursor-"):
        return "cursor"
    if stem.startswith("codex-"):
        return "codex"
    if stem.startswith("git-"):
        return "git"
    return stem.split("-")[0] if "-" in stem else stem


def _count_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def sync_raw_files(
    conn,
    *,
    raw_dir: Path | None = None,
    ingest_run_id: str | None = None,
) -> int:
    """Upsert ingest_files from data/raw/*.jsonl (top-level only)."""
    root = raw_dir or (data_dir() / "raw")
    if not root.is_dir():
        return 0
    loaded_at = _utc_now()
    updated = 0
    for path in sorted(root.glob("*.jsonl")):
        if not path.is_file():
            continue
        row_count = _count_jsonl_lines(path)
        harness = _infer_harness(path)
        conn.execute(
            """
            INSERT INTO ingest_files (path, harness, ingest_run_id, row_count, loaded_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                harness = excluded.harness,
                ingest_run_id = COALESCE(excluded.ingest_run_id, ingest_files.ingest_run_id),
                row_count = excluded.row_count,
                loaded_at = excluded.loaded_at
            """,
            (str(path.resolve()), harness, ingest_run_id, row_count, loaded_at),
        )
        updated += 1
    conn.commit()
    return updated


def list_ingest_files(conn, *, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT path, harness, row_count, loaded_at, ingest_run_id
        FROM ingest_files
        ORDER BY loaded_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def latest_ingest_run(conn) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT run_id, pipeline, started_at, finished_at, status, total_rows
        FROM ingest_runs
        ORDER BY started_at DESC
        LIMIT 1
        """
    ).fetchone()
    return dict(row) if row else None
