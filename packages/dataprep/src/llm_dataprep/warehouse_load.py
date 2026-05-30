"""Load curated JSONL metadata into control_plane.db (streaming; bodies stay on disk)."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_core import data_dir, warehouse_connect, warehouse_db, warehouse_fix_data_sources, warehouse_init_schema
from llm_core.ingest_tracking import finish_ingest_run, start_ingest_run, sync_raw_files


def curated_id_for(source_file: str, source_line: int) -> str:
    payload = f"{source_file}:{source_line}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _meta_row(
    *,
    source_file: str,
    source_line: int,
    row: dict[str, Any],
    loaded_at: str,
) -> tuple[str, tuple[Any, ...]]:
    meta = row.get("meta") or {}
    messages = row.get("messages") or []
    char_count = sum(
        len(m.get("content") or "") for m in messages if isinstance(m, dict)
    )
    harness = meta.get("harness") or ""
    data_source = meta.get("data_source") or (
        "public" if str(harness).startswith("public_") else "personal"
    )
    style_tags = meta.get("style_tags")
    tags_json = json.dumps(style_tags) if style_tags else None
    cid = curated_id_for(source_file, source_line)
    values = (
        cid,
        source_file,
        source_line,
        harness,
        data_source,
        meta.get("public_dataset"),
        meta.get("project"),
        meta.get("session_id"),
        meta.get("chunk_index"),
        meta.get("chunk_count"),
        int(meta.get("train_tier", 0)),
        meta.get("label"),
        meta.get("exec"),
        meta.get("verify"),
        len(messages),
        char_count,
        1 if meta.get("safety_ok", True) else 0,
        tags_json,
        meta.get("tone"),
        loaded_at,
    )
    return cid, values


INSERT_SQL = """
INSERT OR REPLACE INTO curated_examples (
    curated_id, source_file, source_line, harness, data_source, public_dataset,
    project, session_id, chunk_index, chunk_count, train_tier, label, exec_status,
    verify, message_count, char_count, safety_ok, style_tags, tone, loaded_at
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


def load_curated_file(
    conn,
    path: Path,
    *,
    tier: int | None,
    batch_size: int = 500,
) -> tuple[int, int]:
    loaded_at = datetime.now(timezone.utc).isoformat()
    source_file = str(path.resolve())
    inserted = 0
    skipped = 0
    batch: list[tuple[Any, ...]] = []

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(row, dict):
                skipped += 1
                continue
            tier_val = int((row.get("meta") or {}).get("train_tier", 0))
            if tier is not None and tier_val != tier:
                skipped += 1
                continue
            _cid, values = _meta_row(
                source_file=source_file, source_line=line_no, row=row, loaded_at=loaded_at
            )
            batch.append(values)
            if len(batch) >= batch_size:
                conn.executemany(INSERT_SQL, batch)
                inserted += len(batch)
                batch.clear()

    if batch:
        conn.executemany(INSERT_SQL, batch)
        inserted += len(batch)
    conn.commit()
    return inserted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index curated JSONL into warehouse (metadata only)"
    )
    parser.add_argument("--curated", type=Path, nargs="*", default=None)
    parser.add_argument("--latest", action="store_true", help="Use newest curated-*.jsonl")
    parser.add_argument("--tier", type=int, default=1)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--clear", action="store_true", help="DELETE all curated_examples first")
    args = parser.parse_args()

    paths: list[Path] = list(args.curated or [])
    if args.latest:
        candidates = sorted((data_dir() / "curated").glob("curated*.jsonl"))
        if candidates:
            paths = [candidates[-1]]

    if not paths:
        print("No curated files specified")
        return

    conn = warehouse_connect(args.db)
    warehouse_init_schema(conn)

    run_id = start_ingest_run(conn, pipeline="warehouse-load")

    if args.clear:
        conn.execute("DELETE FROM training_manifest_rows")
        conn.execute("DELETE FROM curated_examples")
        conn.commit()
        print("Cleared curated_examples")

    total_in = 0
    total_skip = 0
    for path in paths:
        if not path.is_file():
            print(f"Skip missing {path}")
            continue
        n, skip = load_curated_file(conn, path, tier=args.tier)
        total_in += n
        total_skip += skip
        print(f"{path.name}: indexed {n} tier-{args.tier} rows (skipped {skip})")

    fixed = warehouse_fix_data_sources(conn)
    if fixed:
        print(f"Backfilled data_source/public_dataset on {fixed} rows")

    raw_n = sync_raw_files(conn, ingest_run_id=run_id)
    print(f"Synced {raw_n} raw JSONL files → ingest_files")

    row = conn.execute("SELECT COUNT(*) AS c FROM curated_examples").fetchone()
    finish_ingest_run(
        conn,
        run_id,
        total_rows=int(row["c"]),
        details={"curated_indexed": total_in, "curated_skipped": total_skip, "raw_files": raw_n},
    )
    print(f"Warehouse {warehouse_db() if not args.db else args.db}: {row['c']} curated_examples")


if __name__ == "__main__":
    main()
