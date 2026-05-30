"""Build a training manifest from warehouse SQL (personal-first mix policy)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_core import warehouse_connect, warehouse_fix_data_sources, warehouse_init_schema
from llm_dataprep.mix_policy import MixPolicy, apply_mix, load_mix_policy


def build_manifest(
    conn,
    *,
    manifest_id: str,
    tier: int = 1,
    policy: MixPolicy | None = None,
    personal_ratio: float | None = None,
    limit: int | None = None,
    exclude_public: bool = False,
    public_cap: int | None = None,
) -> dict[str, Any]:
    policy = policy or load_mix_policy()
    if personal_ratio is not None:
        policy = MixPolicy(
            prioritize_personal=policy.prioritize_personal,
            personal_ratio=personal_ratio,
            public_cap=public_cap if public_cap is not None else policy.public_cap,
            public_dataset_priority=policy.public_dataset_priority,
            personal_sample_weight=policy.personal_sample_weight,
            public_sample_weight=policy.public_sample_weight,
        )

    warehouse_fix_data_sources(conn)

    clauses = ["train_tier = ?", "safety_ok = 1"]
    params: list[Any] = [tier]
    if exclude_public:
        clauses.append("data_source = 'personal'")
    where = " AND ".join(clauses)

    rows = conn.execute(
        f"""
        SELECT curated_id, source_file, source_line, harness, data_source,
               public_dataset, project, char_count
        FROM curated_examples
        WHERE {where}
        ORDER BY CASE data_source WHEN 'personal' THEN 0 ELSE 1 END,
                 harness, curated_id
        """,
        params,
    ).fetchall()

    personal = [r for r in rows if r["data_source"] == "personal"]
    public = [r for r in rows if r["data_source"] == "public"]
    mixed = apply_mix(
        personal,
        public,
        policy,
        exclude_public=exclude_public,
        limit=limit,
    )

    personal_n = sum(1 for r, _ in mixed if r["data_source"] == "personal")
    public_n = len(mixed) - personal_n
    actual_ratio = personal_n / len(mixed) if mixed else 0.0

    loaded_at = datetime.now(timezone.utc).isoformat()
    criteria = {
        "tier": tier,
        "prioritize_personal": policy.prioritize_personal,
        "personal_ratio_target": policy.personal_ratio,
        "personal_ratio_actual": round(actual_ratio, 4),
        "public_cap": policy.public_cap,
        "limit": limit,
        "exclude_public": exclude_public,
        "personal_count": personal_n,
        "public_count": public_n,
        "row_count": len(mixed),
        "public_dataset_priority": list(policy.public_dataset_priority),
        "personal_sample_weight": policy.personal_sample_weight,
        "public_sample_weight": policy.public_sample_weight,
    }
    conn.execute(
        "INSERT OR REPLACE INTO training_manifests (manifest_id, created_at, criteria_json, row_count) VALUES (?,?,?,?)",
        (manifest_id, loaded_at, json.dumps(criteria), len(mixed)),
    )
    conn.execute(
        "DELETE FROM training_manifest_rows WHERE manifest_id = ?",
        (manifest_id,),
    )
    conn.executemany(
        "INSERT INTO training_manifest_rows (manifest_id, curated_id, sort_order, sample_weight) VALUES (?,?,?,?)",
        [(manifest_id, r["curated_id"], i, w) for i, (r, w) in enumerate(mixed)],
    )
    conn.commit()
    return criteria


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create training manifest (personal rows always kept first)"
    )
    parser.add_argument("--manifest-id", default="personal-first")
    parser.add_argument("--tier", type=int, default=1)
    parser.add_argument(
        "--personal-ratio",
        type=float,
        default=None,
        help="Target fraction personal (default: config training_mix.personal_ratio)",
    )
    parser.add_argument("--public-cap", type=int, default=None, help="Hard max public rows")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--personal-only", action="store_true")
    parser.add_argument(
        "--no-prioritize-personal",
        action="store_true",
        help="Allow subsampling personal when --limit is set",
    )
    parser.add_argument("--out", type=Path, default=None, help="Also write manifest JSONL pointers")
    args = parser.parse_args()

    policy = load_mix_policy()
    if args.no_prioritize_personal:
        policy = MixPolicy(
            prioritize_personal=False,
            personal_ratio=policy.personal_ratio,
            public_cap=policy.public_cap,
            public_dataset_priority=policy.public_dataset_priority,
            personal_sample_weight=policy.personal_sample_weight,
            public_sample_weight=policy.public_sample_weight,
        )
    if args.public_cap is not None:
        policy = MixPolicy(
            prioritize_personal=policy.prioritize_personal,
            personal_ratio=policy.personal_ratio,
            public_cap=args.public_cap,
            public_dataset_priority=policy.public_dataset_priority,
            personal_sample_weight=policy.personal_sample_weight,
            public_sample_weight=policy.public_sample_weight,
        )

    conn = warehouse_connect()
    warehouse_init_schema(conn)

    criteria = build_manifest(
        conn,
        manifest_id=args.manifest_id,
        tier=args.tier,
        policy=policy,
        personal_ratio=args.personal_ratio,
        limit=args.limit,
        exclude_public=args.personal_only,
        public_cap=args.public_cap,
    )
    print(
        f"Manifest {args.manifest_id}: {criteria['row_count']} rows "
        f"(personal {criteria['personal_count']}, public {criteria['public_count']}, "
        f"actual personal ratio {criteria['personal_ratio_actual']:.1%})"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for row in conn.execute(
                """
                SELECT c.curated_id, c.source_file, c.source_line, c.harness,
                       c.data_source, m.sample_weight
                FROM training_manifest_rows m
                JOIN curated_examples c ON c.curated_id = m.curated_id
                WHERE m.manifest_id = ?
                ORDER BY m.sort_order
                """,
                (args.manifest_id,),
            ):
                fh.write(
                    json.dumps(
                        {
                            "curated_id": row["curated_id"],
                            "source_file": row["source_file"],
                            "source_line": row["source_line"],
                            "harness": row["harness"],
                            "data_source": row["data_source"],
                            "sample_weight": row["sample_weight"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"Pointers → {args.out}")


if __name__ == "__main__":
    main()
