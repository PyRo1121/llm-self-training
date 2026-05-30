# Warehouse index (metadata only)

Large JSONL stays on disk. **Do not** load full curated files into the IDE or agent context.

Architecture (registry, auto-schema limits, Turso): [`WAREHOUSE-ARCHITECTURE.md`](WAREHOUSE-ARCHITECTURE.md)

## Flow

```text
data/raw/*.jsonl          →  (ingest, no DB required)
data/curated/*.jsonl      →  warehouse-load  →  control_plane.db
control_plane.db        →  training-manifest  →  manifest id
manifest id             →  training-extract  →  train/*.jsonl (Phase 2)
```

## Commands

```bash
# Sync harness + public dataset catalog into DB (for dashboard / new datasets)
uv run --package llm-dataprep warehouse-sync-registry

# Probe unknown HF repo schema before writing a loader
uv run --package llm-dataprep probe-hf-schema org/name --source-key my_dataset

# Stats only (stdout JSON, small)
uv run --package llm-dataprep lake-stats --latest-curated

# Index tier-1 metadata (no message bodies in DB)
uv run --package llm-dataprep warehouse-load --latest --tier 1 --clear

# Personal-first mix (default from config/default.yaml → training_mix)
# — ALL your tier-1 personal rows, then public capped to ~20% + priority order
uv run --package llm-dataprep training-manifest \
  --manifest-id personal-first \
  --out data/manifests/personal-first.jsonl

# Personal only (no public HF rows)
uv run --package llm-dataprep training-manifest --personal-only --manifest-id personal-only

# Materialize subset for Unsloth (reads JSONL by line pointer)
uv run --package llm-dataprep training-extract \
  --manifest-id 20260530-120000 \
  --out data/train/bootstrap-80-20.jsonl
```

## Schema

SQLite file: `data/warehouse/control_plane.db` (stdlib `sqlite3` now; Turso driver in Phase 1.5 per `docs/TURSO.md`).

Table `curated_examples`: `curated_id`, `source_file`, `source_line`, `train_tier`, `harness`, `data_source`, counts — **not** full `messages[]`.

## SWE-chat

Gated dataset: `export HF_TOKEN=hf_...` then `public-ingest --datasets swe_chat`. Re-run `warehouse-load` after curate.
