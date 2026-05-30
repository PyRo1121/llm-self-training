# Warehouse architecture — registry, schema, Turso path

**Short answer:** Yes, improve the pipeline — but the database should **not** mirror every Hugging Face column. It should index **canonical training metadata** plus a **source registry** that knows how each dataset maps into our fixed raw JSONL shape.

---

## Three layers (keep these separate)

| Layer | Storage | What it holds |
|-------|---------|----------------|
| **Audit** | `data/raw/*.jsonl` | Append-only message rows (fixed schema below) |
| **Training** | `data/curated/*.jsonl` | Session-grouped `messages[]` + `meta` |
| **Control plane** | `data/warehouse/control_plane.db` (→ Turso) | Pointers, counts, manifests, registry — **not** full chat bodies |

Turso replaces the **driver** (SQLite → libSQL), not this split. See [`TURSO.md`](TURSO.md) Steps 0–2.

---

## Canonical raw record (never auto-invent per dataset)

Every ingest path — Cursor, Codex, HF public — must normalize to:

```json
{
  "source": "public|cursor|codex|…",
  "harness": "public_swe_next|cursor|…",
  "dataset_id": "optional for HF",
  "session_id": "…",
  "line_no": 1,
  "role": "user|assistant",
  "text": "…",
  "label": "accepted",
  "exec": "pass|unknown",
  "verify": "…",
  "source_path": "hf://… or filesystem path"
}
```

**The warehouse never needs to “guess” this from arbitrary HF columns.** A **registered loader** (Python) or **declared mapping** in the registry does that once per dataset.

---

## What to add in the database (schema v3+)

| Table | Purpose |
|-------|---------|
| `source_registry` | One row per harness / `dataset_id`: type, HF repo, tier, gated, loader name, mapping version |
| `source_schema_probe` | Optional HF `datasets.Features` JSON snapshot + probe time (for new/unknown sets) |
| `ingest_runs` | Each `agent-ingest` / `public-ingest` / `phase1` run: started_at, rows, paths |
| `ingest_files` | (existing) Per JSONL file stats |
| `curated_examples` | (existing) Pointer index for training |
| `training_manifests` | (existing) Personal-first mixes |

**Do not** add wide tables like `opencode_input`, `opencode_output` — that duplicates HF and breaks every new dataset.

---

## How “automatic schema” should work (realistic)

```text
New HF dataset appears
    → probe-hf-schema (datasets.Features → JSON in source_schema_probe)
    → human or agent adds loader + registry row (or YAML mapping)
    → public-ingest / agent-ingest writes canonical raw JSONL
    → curate-raw → warehouse-load
```

Fully automatic ingest without a loader is **unsafe** (wrong roles, PII fields, huge blobs). Safe automation:

1. **Probe** — detect columns, types, row count, gated flag  
2. **Suggest** — template mapping `input`→user, `output`→assistant, or `messages[]`  
3. **Register** — `source_registry.loader_name` + version bump  
4. **Ingest** — only after registry says `status=active`

Your other AI working on the registry should target **`source_registry` + loaders**, not Turso tables per column.

---

## SQLite today → Turso tomorrow

| Concern | Now | Phase 1.5 |
|---------|-----|-----------|
| API | `sqlite3` in `llm_core.warehouse` | `turso` Python SDK, same SQL |
| File | `control_plane.db` | Same path or Turso Cloud sync |
| Writers | Single process OK | `PRAGMA journal_mode=mvcc` + `BEGIN CONCURRENT` for parallel ingest |
| Dashboard | SQL queries via `apps/api` | Same queries; optional CDC (`turso_cdc`) for reactive UI |

Migration steps:

1. Freeze schema in `packages/core/src/llm_core/warehouse.py` (versioned migrations).  
2. Implement Turso per `TURSO.md` — swap `connect()` only.  
3. Keep JSONL audit trail unchanged (Turso is not a replacement for `data/raw`).

---

## Pipeline improvements (recommended order)

1. **`warehouse-sync-registry`** — sync `harnesses.py` + `public/registry.py` → `source_registry`  
2. **`probe-hf-schema`** — optional CLI for unknown repos (stores Features JSON)  
3. **`warehouse-load`** — attach `ingest_run_id`, upsert `ingest_files`  
4. **Phase 1.5** — Turso driver + API routes reading same tables  
5. **Registry PRs** — new dataset = new loader + registry row + config cap, then re-run ingest  

---

## Personal-first (unchanged)

Warehouse mix policy stays in SQL manifests (`training_mix` in config). New public datasets only affect training after you **raise `public_cap`** or lower `personal_ratio` — registry growth does not auto-flood SFT.

---

## Related docs

- [`WAREHOUSE-INDEX.md`](WAREHOUSE-INDEX.md) — operator commands  
- [`PUBLIC-DATASETS.md`](PUBLIC-DATASETS.md) — HF catalog  
- [`TURSO.md`](TURSO.md) — Turso implementation playbook  
- [`PHASE2-TRAIN.md`](PHASE2-TRAIN.md) — train from manifest, not raw DB blobs
