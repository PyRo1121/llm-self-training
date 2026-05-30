# Turso warehouse — implementation playbook

**Decision (May 2026):** Bet on **Turso** for the control plane and metadata layer. Do **not** implement until Phase 1.5; follow this doc **step by step** against [official Turso docs](https://docs.turso.tech/llms.txt) at that time.

**Out of scope for Turso (unchanged):** `data/raw/*.jsonl` remains append-only audit; agent-native DBs (OpenCode, T3, etc.) are read-only ingest sources; bulk HF analytics may still use DuckDB later if needed.

---

## Product choice (read first)

| When | Read | Use |
|------|------|-----|
| 1 | [libSQL](https://docs.turso.tech/libsql) | Production-stable SQLite fork; Turso Cloud today |
| 2 | [Turso intro](https://docs.turso.tech/introduction) | Three paths: embedded DB, AgentFS, Cloud |
| 3 | [SDK introduction](https://docs.turso.tech/sdk/introduction) | Pick client |
| 4 | [TS SDK reference](https://docs.turso.tech/sdk/ts/reference) | **New code:** `@tursodatabase/database` (local), `@tursodatabase/sync` (sync), `@libsql/client` (ORM/legacy) |

**Our default for this repo:** Python — [Turso Quickstart (Python)](https://docs.turso.tech/sdk/python/quickstart.md) + [Python reference](https://docs.turso.tech/sdk/python/reference.md). Enable **Turso Database** (rewrite) features per step below, not vanilla SQLite only.

---

## Implementation order (Phase 1.5)

Check off each step only after reading the linked page and noting flags/PRAGMAs in `config/default.yaml` or `packages/core`.

### Step 0 — Local dev baseline

- [x] [Local Development](https://docs.turso.tech/local-development.md) — `turso dev` / pyturso file path
- [x] [CLI installation](https://docs.turso.tech/cli/installation.md) — `~/.turso/turso` (May 2026 install script)
- [ ] [SQLite compatibility](https://docs.turso.tech/sql-reference/compatibility.md) — gaps vs vanilla SQLite
- [ ] [Experimental features](https://docs.turso.tech/sql-reference/experimental-features.md) — enable per flag in `config/default.yaml`

**Exit:** `uv run --package llm-core warehouse-smoke` → `SELECT 1` pass (sqlite or `WAREHOUSE_DRIVER=turso`).

### Step 1 — Schema (control plane)

Design tables (runs, ingest_rows, benchmarks, quarantine) in SQL; apply via:

- [x] [CREATE TABLE](https://docs.turso.tech/sql-reference/statements/create-table.md) — `packages/core/.../warehouse.py` schema v4
- [x] [Transactions](https://docs.turso.tech/sql-reference/statements/transactions.md) — baseline `BEGIN`/`COMMIT`
- [x] [INSERT / UPSERT](https://docs.turso.tech/sql-reference/statements/upsert.md) — manifests, registry, ingest_files

**Exit:** Migrations applied; `_schema_version` table; `warehouse-smoke` pass.

### Step 2 — Python package wiring

- [x] [Python quickstart](https://docs.turso.tech/sdk/python/quickstart.md) — `pyturso` (`import turso`)
- [x] [Python reference](https://docs.turso.tech/sdk/python/reference.md)
- [x] `pyturso` in `packages/core` deps
- [x] `llm_core.warehouse_driver` + `warehouse_config` + `warehouse-smoke` CLI

**Exit:** `uv run --package llm-core warehouse-smoke --insert-probe`

### Step 3 — Concurrent ingest writers (optional, Phase 1+)

If multiple processes write metadata during ingest:

- [ ] [Concurrent writes](https://docs.turso.tech/tursodb/concurrent-writes.md) — `PRAGMA journal_mode = 'mvcc'`, `BEGIN CONCURRENT`, retry on conflict
- [ ] [Blog: MVCC](https://turso.tech/blog/beyond-the-single-writer-limitation-with-tursos-concurrent-writes) — preview limitations

**Note:** CDC and MVCC are **mutually exclusive per connection** — use MVCC on ingest workers, CDC on a dedicated connection (Step 4).

**Exit:** Two workers insert non-overlapping rows without `SQLITE_BUSY`.

### Step 4 — Change capture (reactive curation)

- [ ] [Change Data Capture](https://docs.turso.tech/tursodb/cdc.md) — `PRAGMA capture_data_changes_conn('full')`, query `turso_cdc`

**Exit:** Insert into `ingest_rows` produces queryable CDC rows for dashboard/jobs.

### Step 5 — Sync to Cloud (optional)

Only if off-machine backup or second machine dashboard:

- [ ] [Turso Cloud quickstart](https://docs.turso.tech/quickstart.md) — create DB + token
- [ ] [Sync usage](https://docs.turso.tech/sync/usage.md) — `turso.sync.connect`, `push()` / `pull()`, `checkpoint()`
- [ ] [Conflict resolution](https://docs.turso.tech/sync/conflict-resolution.md)
- [ ] [Checkpoint](https://docs.turso.tech/sync/checkpoint.md)

**Exit:** Local `control_plane.db` pushes to Cloud; pull on second host works.

### Step 6 — Vector + FTS (RAG metadata, Phase 3+)

Defer until RAG phase unless needed for DataLake search:

- [ ] [Vector search guide](https://docs.turso.tech/guides/vector-search.md)
- [ ] [Vector functions](https://docs.turso.tech/sql-reference/functions/vector.md)
- [ ] [AI & embeddings](https://docs.turso.tech/features/ai-and-embeddings.md)
- [ ] [Code indexing](https://docs.turso.tech/guides/code-indexing.md) — FTS `index_method` + `vector8()` pattern
- [ ] [FTS functions](https://docs.turso.tech/sql-reference/functions/fts.md)

**Exit:** Chunk table + hybrid search prototype query documented in `packages/rag`.

### Step 7 — API + dashboard

- [x] `apps/api` routes read/write warehouse via Python SDK
- [x] Dashboard Overview + DataLake tabs consume API
- [ ] Operator verify: API + `bun dev` in browser
- [ ] [Platform API](https://docs.turso.tech/api-reference/introduction.md) only if automating Cloud DBs

**Exit:** Phase 1.5 ROADMAP checklist complete (except optional Turso 3–5).

### Step 8 — Advanced (defer)

Read when needed; not blocking 1.5:

- [ ] [Materialized views](https://docs.turso.tech/sql-reference/statements/create-materialized-view.md) + [DBSP blog](https://turso.tech/blog/introducing-real-time-data-with-materialized-views-in-turso) — **experimental, not production-ready**
- [ ] [AgentFS](https://docs.turso.tech/agentfs/introduction.md) — agent audit / sandbox
- [ ] [Branching](https://docs.turso.tech/features/branching.md) — CI DB branches
- [ ] [Encryption](https://docs.turso.tech/tursodb/encryption.md)
- [ ] [Multiprocess WAL](https://docs.turso.tech/sql-reference/multiprocess-access.md)

---

## File layout

```text
data/warehouse/
  control_plane.db    # Turso Database file (local SSOT)
config/default.yaml   # warehouse.path, sync.*, experimental_features
docs/TURSO.md         # this playbook
```

---

## Doc index (bookmark)

Full sitemap: https://docs.turso.tech/llms.txt

Changelog (engine): https://docs.turso.tech/tursodb/changelog.md

---

## Risks to re-check at implementation time

1. **Beta engine** — Turso Database vs libSQL maturity per [libSQL page](https://docs.turso.tech/libsql.md).
2. **MVCC / CDC / MV** — preview or experimental; read changelog before enabling in production timers.
3. **Large analytics** — full scans over millions of chat rows may still need DuckDB batch jobs; Turso holds metadata + curated pointers, not necessarily every HF shard byte.
