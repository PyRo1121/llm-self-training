"""Warehouse queries for API / dashboard (metadata only — not JSONL bodies)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_core.paths import runs_dir
from llm_core.warehouse import connect, init_schema
from llm_core.warehouse_config import load_warehouse_config
from llm_core.warehouse_driver import driver_label
from llm_core.ingest_tracking import latest_ingest_run, list_ingest_files


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_warehouse() -> sqlite3.Connection:
    conn = connect()
    init_schema(conn)
    return conn


def overview_stats(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    own = conn is None
    if own:
        conn = ensure_warehouse()
    try:
        tier_rows = conn.execute(
            """
            SELECT train_tier, COUNT(*) AS n
            FROM curated_examples
            GROUP BY train_tier
            ORDER BY train_tier
            """
        ).fetchall()
        by_tier = {int(r["train_tier"]): int(r["n"]) for r in tier_rows}
        ingest_files = conn.execute(
            "SELECT COUNT(*) AS n FROM ingest_files"
        ).fetchone()["n"]
        manifests = conn.execute(
            "SELECT COUNT(*) AS n FROM training_manifests"
        ).fetchone()["n"]
        quarantine_n = conn.execute(
            "SELECT COUNT(*) AS n FROM quarantine_events"
        ).fetchone()["n"]
        rag_sources = conn.execute(
            "SELECT COUNT(*) AS n FROM rag_sources WHERE status = 'indexed'"
        ).fetchone()["n"]
        last_rag = conn.execute(
            """
            SELECT run_id, status, finished_at, chunks_added
            FROM rag_index_runs
            ORDER BY started_at DESC LIMIT 1
            """
        ).fetchone()
        train_runs_db = conn.execute(
            "SELECT COUNT(*) AS n FROM training_runs"
        ).fetchone()["n"]
        registry_n = conn.execute(
            "SELECT COUNT(*) AS n FROM source_registry"
        ).fetchone()["n"]
        wh = load_warehouse_config()
        return {
            "warehouse": str(wh.path),
            "warehouse_exists": wh.path.exists(),
            "warehouse_driver": driver_label(wh),
            "curated_by_tier": by_tier,
            "curated_total": sum(by_tier.values()),
            "ingest_files": int(ingest_files),
            "source_registry_entries": int(registry_n),
            "latest_ingest_run": latest_ingest_run(conn),
            "training_manifests": int(manifests),
            "quarantine_events": int(quarantine_n),
            "rag_sources_indexed": int(rag_sources),
            "last_rag_index_run": dict(last_rag) if last_rag else None,
            "training_runs_in_db": int(train_runs_db),
            "runs_on_disk": _scan_runs_disk(),
        }
    finally:
        if own:
            conn.close()


def datalake_summary(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    own = conn is None
    if own:
        conn = ensure_warehouse()
    try:
        by_source = conn.execute(
            """
            SELECT data_source, COUNT(*) AS n
            FROM curated_examples
            GROUP BY data_source
            """
        ).fetchall()
        by_harness = conn.execute(
            """
            SELECT harness, COUNT(*) AS n
            FROM curated_examples
            GROUP BY harness
            ORDER BY n DESC
            LIMIT 20
            """
        ).fetchall()
        public_ds = conn.execute(
            """
            SELECT public_dataset, COUNT(*) AS n
            FROM curated_examples
            WHERE data_source = 'public'
            GROUP BY public_dataset
            ORDER BY n DESC
            LIMIT 15
            """
        ).fetchall()
        return {
            "by_data_source": {r["data_source"]: int(r["n"]) for r in by_source},
            "top_harnesses": [
                {"harness": r["harness"], "count": int(r["n"])} for r in by_harness
            ],
            "public_datasets": [
                {"dataset": r["public_dataset"], "count": int(r["n"])}
                for r in public_ds
            ],
            "raw_files": list_ingest_files(conn, limit=30),
        }
    finally:
        if own:
            conn.close()


def list_quarantine(
    conn: sqlite3.Connection | None = None, *, limit: int = 50
) -> list[dict[str, Any]]:
    own = conn is None
    if own:
        conn = ensure_warehouse()
    try:
        rows = conn.execute(
            """
            SELECT q.id, q.curated_id, q.reason, q.operator,
                   q.train_tier_before, q.train_tier_after, q.created_at,
                   c.harness, c.project, c.label
            FROM quarantine_events q
            LEFT JOIN curated_examples c ON c.curated_id = q.curated_id
            ORDER BY q.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        if own:
            conn.close()


def quarantine_row(
    curated_id: str,
    reason: str,
    *,
    operator: str = "operator",
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    own = conn is None
    if own:
        conn = ensure_warehouse()
    try:
        row = conn.execute(
            "SELECT train_tier FROM curated_examples WHERE curated_id = ?",
            (curated_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"unknown curated_id: {curated_id}")
        before = int(row["train_tier"])
        conn.execute(
            "UPDATE curated_examples SET train_tier = 0 WHERE curated_id = ?",
            (curated_id,),
        )
        conn.execute(
            """
            INSERT INTO quarantine_events (
                curated_id, reason, operator,
                train_tier_before, train_tier_after, created_at
            ) VALUES (?, ?, ?, ?, 0, ?)
            """,
            (curated_id, reason, operator, before, _utc_now()),
        )
        conn.commit()
        return {"curated_id": curated_id, "train_tier_before": before, "reason": reason}
    finally:
        if own:
            conn.close()


def rag_status(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    own = conn is None
    if own:
        conn = ensure_warehouse()
    try:
        sources = conn.execute(
            """
            SELECT source_id, url, kind, status, chunk_count, last_indexed_at,
                   context7_library_id
            FROM rag_sources
            ORDER BY source_id
            """
        ).fetchall()
        runs = conn.execute(
            """
            SELECT run_id, status, started_at, finished_at,
                   sources_ok, sources_failed, chunks_added
            FROM rag_index_runs
            ORDER BY started_at DESC
            LIMIT 5
            """
        ).fetchall()
        return {
            "sources": [dict(r) for r in sources],
            "recent_index_runs": [dict(r) for r in runs],
        }
    finally:
        if own:
            conn.close()


def upsert_rag_source(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    url: str,
    kind: str,
    tier: int = 0,
    context7_library_id: str | None = None,
    status: str = "indexed",
    chunk_count: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO rag_sources (
            source_id, url, kind, tier, context7_library_id,
            last_indexed_at, chunk_count, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            url = excluded.url,
            kind = excluded.kind,
            tier = excluded.tier,
            context7_library_id = excluded.context7_library_id,
            last_indexed_at = excluded.last_indexed_at,
            chunk_count = excluded.chunk_count,
            status = excluded.status
        """,
        (
            source_id,
            url,
            kind,
            tier,
            context7_library_id,
            _utc_now(),
            chunk_count,
            status,
        ),
    )


def start_rag_index_run(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute(
        """
        INSERT INTO rag_index_runs (run_id, started_at, status)
        VALUES (?, ?, 'running')
        """,
        (run_id, _utc_now()),
    )
    conn.commit()


def finish_rag_index_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    status: str,
    sources_ok: int,
    sources_failed: int,
    chunks_added: int,
    details: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """
        UPDATE rag_index_runs
        SET finished_at = ?, status = ?,
            sources_ok = ?, sources_failed = ?, chunks_added = ?,
            details_json = ?
        WHERE run_id = ?
        """,
        (
            _utc_now(),
            status,
            sources_ok,
            sources_failed,
            chunks_added,
            json.dumps(details or {}),
            run_id,
        ),
    )
    conn.commit()


def list_training_runs(
    conn: sqlite3.Connection | None = None, *, limit: int = 20
) -> dict[str, Any]:
    own = conn is None
    if own:
        conn = ensure_warehouse()
    try:
        rows = conn.execute(
            """
            SELECT run_id, run_name, base_model, adapter_path, status,
                   started_at, finished_at, train_rows, metrics_json
            FROM training_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        disk = _scan_runs_disk()
        return {"database": [dict(r) for r in rows], "disk": disk}
    finally:
        if own:
            conn.close()


def _resolve_adapter_dir(run_dir: Path) -> Path | None:
    """PEFT adapter dir: runs/<name>/adapter or latest checkpoint-*/adapter."""
    direct = run_dir / "adapter"
    if (direct / "adapter_config.json").is_file():
        return direct
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-", 1)[-1]) if p.name.split("-", 1)[-1].isdigit() else 0,
        reverse=True,
    )
    for ckpt in checkpoints:
        for candidate in (ckpt / "adapter", ckpt):
            if (candidate / "adapter_config.json").is_file():
                return candidate
    return None


def _scan_runs_disk() -> list[dict[str, Any]]:
    root = runs_dir()
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        adapter = _resolve_adapter_dir(child)
        out.append(
            {
                "run_name": child.name,
                "path": str(child),
                "has_adapter": adapter is not None,
                "adapter_path": str(adapter) if adapter else None,
            }
        )
        if len(out) >= 20:
            break
    return out


def register_benchmark_run(
    conn: sqlite3.Connection,
    *,
    suite: str,
    train_run_name: str | None,
    status: str,
    scores: dict[str, Any],
) -> str:
    train_id = f"train-{train_run_name}" if train_run_name else None
    run_id = f"bench-{suite}-{train_run_name or 'standalone'}"
    now = _utc_now()
    conn.execute(
        """
        INSERT INTO benchmark_runs (
            run_id, suite, started_at, finished_at, status,
            scores_json, linked_train_run
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            finished_at = excluded.finished_at,
            status = excluded.status,
            scores_json = excluded.scores_json,
            linked_train_run = excluded.linked_train_run
        """,
        (
            run_id,
            suite,
            now,
            now,
            status,
            json.dumps(scores),
            train_id,
        ),
    )
    conn.commit()
    return run_id


def register_training_run(
    run_name: str,
    *,
    base_model: str | None = None,
    adapter_path: str | None = None,
    status: str = "completed",
    train_rows: int | None = None,
    metrics: dict[str, Any] | None = None,
    started_at: str | None = None,
) -> str:
    conn = ensure_warehouse()
    try:
        run_id = f"train-{run_name}"
        now = _utc_now()
        start_ts = started_at or now
        finished_ts = None if status == "running" else now
        conn.execute(
            """
            INSERT INTO training_runs (
                run_id, run_name, base_model, adapter_path, status,
                started_at, finished_at, train_rows, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status = excluded.status,
                finished_at = excluded.finished_at,
                adapter_path = COALESCE(excluded.adapter_path, training_runs.adapter_path),
                base_model = COALESCE(excluded.base_model, training_runs.base_model),
                train_rows = COALESCE(excluded.train_rows, training_runs.train_rows),
                metrics_json = excluded.metrics_json
            """,
            (
                run_id,
                run_name,
                base_model,
                adapter_path,
                status,
                start_ts,
                finished_ts,
                train_rows,
                json.dumps(metrics or {}),
            ),
        )
        conn.commit()
        return run_id
    finally:
        conn.close()
