"""Turso Step 0/2 smoke — verify warehouse driver (docs/TURSO.md)."""

from __future__ import annotations

import argparse
import json
import sys

from llm_core.warehouse import init_schema
from llm_core.warehouse_config import load_warehouse_config
from llm_core.warehouse_driver import connect, driver_label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test warehouse DB (sqlite or pyturso)"
    )
    parser.add_argument(
        "--insert-probe",
        action="store_true",
        help="Insert/delete a probe row in ingest_runs",
    )
    args = parser.parse_args()

    cfg = load_warehouse_config()
    print(driver_label(cfg))

    conn = connect(config=cfg)
    try:
        init_schema(conn)
        one = conn.execute("SELECT 1 AS ok").fetchone()
        curated = conn.execute(
            "SELECT COUNT(*) AS n FROM curated_examples"
        ).fetchone()["n"]
        manifests = conn.execute(
            "SELECT COUNT(*) AS n FROM training_manifests"
        ).fetchone()["n"]
        print(json.dumps({"select_1": one["ok"], "curated_examples": curated, "training_manifests": manifests}, indent=2))

        if args.insert_probe:
            conn.execute(
                """
                INSERT OR REPLACE INTO ingest_runs (
                    run_id, pipeline, started_at, status, total_rows
                ) VALUES ('smoke-probe', 'warehouse-smoke', datetime('now'), 'ok', 0)
                """
            )
            conn.commit()
            conn.execute("DELETE FROM ingest_runs WHERE run_id = 'smoke-probe'")
            conn.commit()
            print("insert_probe: ok")
    finally:
        conn.close()

    print("warehouse-smoke: pass")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"warehouse-smoke: FAIL — {exc}", file=sys.stderr)
        sys.exit(1)
