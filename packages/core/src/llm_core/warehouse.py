"""Control-plane SQLite (Turso-compatible schema; swap driver in Phase 1.5)."""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_VERSION = 4

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS _schema_version (
    version INTEGER NOT NULL PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS source_registry (
    source_key TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    display_name TEXT,
    hf_repo TEXT,
    ingest_tier TEXT,
    loader_name TEXT,
    mapping_version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',
    gated INTEGER NOT NULL DEFAULT 0,
    default_max_rows INTEGER,
    license_note TEXT,
    released TEXT,
    config_json TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_schema_probe (
    source_key TEXT PRIMARY KEY,
    hf_repo TEXT NOT NULL,
    features_json TEXT,
    row_count_estimate INTEGER,
    probed_at TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id TEXT PRIMARY KEY,
    pipeline TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    total_rows INTEGER NOT NULL DEFAULT 0,
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS ingest_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    harness TEXT,
    ingest_run_id TEXT,
    row_count INTEGER NOT NULL DEFAULT 0,
    flagged_count INTEGER NOT NULL DEFAULT 0,
    loaded_at TEXT NOT NULL,
    FOREIGN KEY (ingest_run_id) REFERENCES ingest_runs(run_id)
);

CREATE TABLE IF NOT EXISTS curated_examples (
    curated_id TEXT PRIMARY KEY,
    source_file TEXT NOT NULL,
    source_line INTEGER NOT NULL,
    harness TEXT,
    data_source TEXT NOT NULL DEFAULT 'personal',
    public_dataset TEXT,
    project TEXT,
    session_id TEXT,
    chunk_index INTEGER,
    chunk_count INTEGER,
    train_tier INTEGER NOT NULL,
    label TEXT,
    exec_status TEXT,
    verify TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    char_count INTEGER NOT NULL DEFAULT 0,
    safety_ok INTEGER NOT NULL DEFAULT 1,
    style_tags TEXT,
    tone TEXT,
    loaded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_curated_tier ON curated_examples(train_tier);
CREATE INDEX IF NOT EXISTS idx_curated_data_source ON curated_examples(data_source);
CREATE INDEX IF NOT EXISTS idx_curated_harness ON curated_examples(harness);
CREATE INDEX IF NOT EXISTS idx_curated_project ON curated_examples(project);

CREATE TABLE IF NOT EXISTS training_manifests (
    manifest_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    criteria_json TEXT NOT NULL,
    row_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS training_manifest_rows (
    manifest_id TEXT NOT NULL,
    curated_id TEXT NOT NULL,
    sort_order INTEGER NOT NULL,
    sample_weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (manifest_id, curated_id),
    FOREIGN KEY (curated_id) REFERENCES curated_examples(curated_id)
);

CREATE TABLE IF NOT EXISTS training_runs (
    run_id TEXT PRIMARY KEY,
    run_name TEXT NOT NULL,
    base_model TEXT,
    adapter_path TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    finished_at TEXT,
    train_rows INTEGER,
    config_json TEXT,
    metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS quarantine_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    curated_id TEXT,
    reason TEXT NOT NULL,
    operator TEXT NOT NULL DEFAULT 'system',
    train_tier_before INTEGER,
    train_tier_after INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (curated_id) REFERENCES curated_examples(curated_id)
);

CREATE INDEX IF NOT EXISTS idx_quarantine_curated ON quarantine_events(curated_id);

CREATE TABLE IF NOT EXISTS rag_sources (
    source_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    kind TEXT NOT NULL,
    tier INTEGER NOT NULL DEFAULT 0,
    context7_library_id TEXT,
    last_indexed_at TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    config_json TEXT
);

CREATE TABLE IF NOT EXISTS rag_index_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    sources_ok INTEGER NOT NULL DEFAULT 0,
    sources_failed INTEGER NOT NULL DEFAULT 0,
    chunks_added INTEGER NOT NULL DEFAULT 0,
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    run_id TEXT PRIMARY KEY,
    suite TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    scores_json TEXT,
    linked_train_run TEXT,
    FOREIGN KEY (linked_train_run) REFERENCES training_runs(run_id)
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Connect to warehouse (sqlite3 or pyturso via warehouse_driver)."""
    from llm_core.warehouse_driver import connect as driver_connect

    return driver_connect(db_path)


def migrate_schema(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT version FROM _schema_version ORDER BY version DESC LIMIT 1"
    ).fetchone()
    version = int(row[0]) if row else 0
    if version < 2:
        try:
            conn.execute(
                "ALTER TABLE training_manifest_rows "
                "ADD COLUMN sample_weight REAL NOT NULL DEFAULT 1.0"
            )
        except sqlite3.OperationalError:
            pass
        version = 2
    if version < 3:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS source_registry (
                source_key TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                display_name TEXT,
                hf_repo TEXT,
                ingest_tier TEXT,
                loader_name TEXT,
                mapping_version INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                gated INTEGER NOT NULL DEFAULT 0,
                default_max_rows INTEGER,
                license_note TEXT,
                released TEXT,
                config_json TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS source_schema_probe (
                source_key TEXT PRIMARY KEY,
                hf_repo TEXT NOT NULL,
                features_json TEXT,
                row_count_estimate INTEGER,
                probed_at TEXT NOT NULL,
                notes TEXT
            );
            CREATE TABLE IF NOT EXISTS ingest_runs (
                run_id TEXT PRIMARY KEY,
                pipeline TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                total_rows INTEGER NOT NULL DEFAULT 0,
                details_json TEXT
            );
            """
        )
        try:
            conn.execute(
                "ALTER TABLE ingest_files ADD COLUMN ingest_run_id TEXT"
            )
        except sqlite3.OperationalError:
            pass
        version = 3
    if version < 4:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                run_name TEXT NOT NULL,
                base_model TEXT,
                adapter_path TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                started_at TEXT NOT NULL,
                finished_at TEXT,
                train_rows INTEGER,
                config_json TEXT,
                metrics_json TEXT
            );
            CREATE TABLE IF NOT EXISTS quarantine_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                curated_id TEXT,
                reason TEXT NOT NULL,
                operator TEXT NOT NULL DEFAULT 'system',
                train_tier_before INTEGER,
                train_tier_after INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (curated_id) REFERENCES curated_examples(curated_id)
            );
            CREATE INDEX IF NOT EXISTS idx_quarantine_curated
                ON quarantine_events(curated_id);
            CREATE TABLE IF NOT EXISTS rag_sources (
                source_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                kind TEXT NOT NULL,
                tier INTEGER NOT NULL DEFAULT 0,
                context7_library_id TEXT,
                last_indexed_at TEXT,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                config_json TEXT
            );
            CREATE TABLE IF NOT EXISTS rag_index_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                sources_ok INTEGER NOT NULL DEFAULT 0,
                sources_failed INTEGER NOT NULL DEFAULT 0,
                chunks_added INTEGER NOT NULL DEFAULT 0,
                details_json TEXT
            );
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT PRIMARY KEY,
                suite TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                scores_json TEXT,
                linked_train_run TEXT,
                FOREIGN KEY (linked_train_run) REFERENCES training_runs(run_id)
            );
            """
        )
        version = 4
    conn.execute("DELETE FROM _schema_version")
    conn.execute(
        "INSERT INTO _schema_version (version) VALUES (?)",
        (version,),
    )
    conn.commit()


def fix_data_sources(conn: sqlite3.Connection) -> int:
    """Classify public_* harness rows; backfill missing public_dataset from harness."""
    cur = conn.execute(
        """
        UPDATE curated_examples
        SET data_source = 'public',
            public_dataset = COALESCE(
                NULLIF(public_dataset, ''),
                REPLACE(harness, 'public_', '')
            )
        WHERE harness LIKE 'public_%'
          AND (
            data_source != 'public'
            OR public_dataset IS NULL
            OR public_dataset = ''
          )
        """
    )
    conn.commit()
    return cur.rowcount


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    migrate_schema(conn)
