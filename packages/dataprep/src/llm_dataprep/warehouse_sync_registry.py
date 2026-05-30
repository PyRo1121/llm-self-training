"""Sync code registries (harnesses + public HF) into warehouse source_registry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from llm_core import warehouse_connect, warehouse_init_schema
from llm_dataprep.harnesses import harnesses_for_ingest
from llm_dataprep.public.registry import list_specs


def sync_registry(conn) -> int:
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for spec in harnesses_for_ingest():
        conn.execute(
            """
            INSERT OR REPLACE INTO source_registry (
                source_key, source_type, display_name, hf_repo, ingest_tier,
                loader_name, mapping_version, status, gated, default_max_rows,
                license_note, released, config_json, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                spec.harness_id,
                "local_agent",
                spec.harness_id,
                None,
                spec.ingest_tier,
                spec.harness_id,
                1,
                "active" if spec.ingest_tier in ("full", "partial") else "detect",
                0,
                None,
                spec.notes,
                None,
                None,
                now,
            ),
        )
        n += 1

    for spec in list_specs():
        conn.execute(
            """
            INSERT OR REPLACE INTO source_registry (
                source_key, source_type, display_name, hf_repo, ingest_tier,
                loader_name, mapping_version, status, gated, default_max_rows,
                license_note, released, config_json, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                spec.dataset_id,
                "hf_public",
                spec.dataset_id,
                spec.hf_repo,
                "full",
                spec.loader_name or f"load_{spec.dataset_id}",
                1,
                "active",
                1 if spec.gated else 0,
                spec.default_max_rows,
                spec.license_note,
                spec.released,
                json.dumps({"tier": spec.tier}),
                now,
            ),
        )
        n += 1
    conn.commit()
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync source_registry from code catalogs")
    parser.parse_args()
    conn = warehouse_connect()
    warehouse_init_schema(conn)
    count = sync_registry(conn)
    print(f"Synced {count} sources → source_registry")


if __name__ == "__main__":
    main()
