"""Probe Hugging Face dataset Features → warehouse source_schema_probe (no ingest)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from llm_core import warehouse_connect, warehouse_init_schema


def probe_repo(repo: str, *, split: str = "train", max_rows_sample: int = 1) -> dict:
    from datasets import load_dataset

    ds = load_dataset(repo, split=split, streaming=True)
    row = next(iter(ds))
    features = ds.features
    feat_json = json.dumps(str(features), default=str)[:8000]
    return {
        "hf_repo": repo,
        "features_json": feat_json,
        "sample_keys": sorted(row.keys()) if isinstance(row, dict) else [],
        "row_count_estimate": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe HF schema and store in warehouse (for new registry work)"
    )
    parser.add_argument("repo", help="e.g. SALT-NLP/SWE-chat")
    parser.add_argument("--source-key", default=None, help="Registry key (default: repo slug)")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    key = args.source_key or args.repo.split("/")[-1].lower().replace("-", "_")
    info = probe_repo(args.repo)
    now = datetime.now(timezone.utc).isoformat()

    conn = warehouse_connect()
    warehouse_init_schema(conn)
    conn.execute(
        """
        INSERT OR REPLACE INTO source_schema_probe (
            source_key, hf_repo, features_json, row_count_estimate, probed_at, notes
        ) VALUES (?,?,?,?,?,?)
        """,
        (
            key,
            args.repo,
            json.dumps(
                {"features": info["features_json"], "sample_keys": info["sample_keys"]},
                ensure_ascii=False,
            ),
            info["row_count_estimate"],
            now,
            args.notes,
        ),
    )
    conn.commit()
    print(f"Probed {args.repo} → source_schema_probe[{key}]")
    print("Sample keys:", info["sample_keys"])


if __name__ == "__main__":
    main()
