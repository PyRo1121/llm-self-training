"""Extract training JSONL from manifest pointers (streaming; for Phase 2 train)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_core import warehouse_connect, warehouse_init_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract curated rows by manifest pointers")
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args()

    conn = warehouse_connect(args.db)
    warehouse_init_schema(conn)

    rows = conn.execute(
        """
        SELECT c.source_file, c.source_line, c.data_source, c.public_dataset,
               c.harness, m.sample_weight
        FROM training_manifest_rows m
        JOIN curated_examples c ON c.curated_id = m.curated_id
        WHERE m.manifest_id = ?
        ORDER BY m.sort_order
        """,
        (args.manifest_id,),
    ).fetchall()

    expected = len(rows)
    line_cache: dict[str, list[str] | None] = {}
    written = 0
    skipped = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as out_fh:
        for rec in rows:
            path = Path(rec["source_file"])
            line_no = int(rec["source_line"])
            key = str(path)
            if key not in line_cache:
                if not path.is_file():
                    line_cache[key] = None
                    skipped += 1
                    continue
                with path.open(encoding="utf-8", errors="replace") as fh:
                    line_cache[key] = fh.readlines()
            lines = line_cache[key]
            if not lines or line_no < 1 or line_no > len(lines):
                skipped += 1
                continue
            line = lines[line_no - 1].strip()
            if not line:
                skipped += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                out_fh.write(line + "\n")
                written += 1
                continue
            if isinstance(obj, dict):
                meta = obj.get("meta")
                if meta is None or not isinstance(meta, dict):
                    meta = {}
                    obj["meta"] = meta
                meta["sample_weight"] = rec["sample_weight"]
                meta.setdefault("data_source", rec["data_source"])
                if rec["public_dataset"]:
                    meta.setdefault("public_dataset", rec["public_dataset"])
                meta.setdefault("harness", rec["harness"])
                out_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                out_fh.write(line + "\n")
            written += 1

    print(
        f"Extracted {written}/{expected} examples → {args.out}"
        + (f" (skipped {skipped})" if skipped else "")
    )
    if written < expected:
        print(
            f"training-extract: {expected - written} manifest rows missing on disk",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
