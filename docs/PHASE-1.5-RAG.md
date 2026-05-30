# Phase 1.5 + RAG — integrated control plane

**Status:** Phase 1.5 core complete (May 2026) — API + dashboard tabs + ingest tracking. Turso Steps 3–5 optional.  
**Spec:** [`PLAN.md`](../PLAN.md) · [`docs/TURSO.md`](TURSO.md) · [`ROADMAP.md`](../ROADMAP.md)

---

## Research synthesis (not deferred)

| Source | Finding | Decision |
|--------|---------|----------|
| **Turso / pyturso** | `pyturso` (`import turso`) is the 2026 local Python SDK; sqlite3-shaped API; `turso.sync` for push/pull. Still **beta**. | **Now:** stdlib `sqlite3` + schema designed for Turso. **Step 2:** optional `pyturso` in `llm_core.warehouse_driver` when env `WAREHOUSE_DRIVER=turso`. |
| **Turso Step 6** | Vector + FTS on Turso for code/docs metadata. | **Defer vectors in SQL** until gold set needs hybrid SQL search. **Now:** Chroma holds embeddings; warehouse holds `rag_sources` + index run stats only. |
| **Chroma ≥1.5** | `PersistentClient`, `.add` / `.query`, precomputed embeddings from Ollama. | `data/chroma_db/` SSOT for vectors; collection `allowlist_v1`. |
| **Context7** | Public library API truth at inference. | Allowlist entries may set `context7_library_id`; crawl skips those URLs (no duplicate in Chroma). MCP stays **read-only**; agents use Context7 plugin for `/org/project` libs. |
| **Exa** | Live web research (rate-limited on shared MCP key). | Use for ad-hoc research; not in ingest path. |
| **GitNexus** | Repo not indexed (`LLM Self Training` missing). | Index repo when ready; IDE-time code graph, not bulk train. |
| **FastMCP** | Thin stdio server, 2 tools max, shared `query.py`. | `rag-mcp` → `search_allowlist_docs`, `rag_status`. |

---

## Hybrid architecture

```text
data/raw/*.jsonl          ← audit lake (unchanged)
data/train/*.jsonl        ← SFT manifests (unchanged)
data/chroma_db/           ← embeddings + chunk text (RAG)
data/warehouse/control_plane.db  ← catalog, runs, quarantine, rag metadata
apps/api                  ← FastAPI (dashboard only talks here)
apps/dashboard            ← Bun/Vite operator UI
packages/rag              ← index, query, MCP
```

**Division of labor**

- **Train** — personal-first behavior in weights.
- **RAG (allowlist)** — private/operator docs not in Context7.
- **Context7** — versioned public framework docs at tool time.
- **GitNexus** — structural code search when repo is indexed.

---

## Implementation map

| Component | Path | Exit |
|-----------|------|------|
| Schema v4 | `packages/core/.../warehouse.py` | `training_runs`, `quarantine_events`, `rag_*`, `benchmark_runs` |
| Control queries | `packages/core/.../control_plane.py` | Overview + datalake stats |
| Allowlist | `config/doc_allowlist.yaml` | Tier-0 llms.txt seeds |
| RAG package | `packages/rag` | `rag-index`, `rag-mcp` |
| API | `apps/api/src/llm_api/routes/` | `/api/v1/*` |
| Dashboard | `apps/dashboard` | `bun dev` → overview + datalake + RAG cards |

---

## Commands

```bash
# Warehouse + API
uv sync
uv run --package llm-api llm-api

# RAG (Ollama embed model must be pulled)
ollama pull qwen3-embedding:4b   # or nomic-embed-text per config
uv run --package llm-rag rag-index
uv run --package llm-rag rag-mcp   # stdio MCP for Cursor

# Dashboard (separate terminal)
cd apps/dashboard && bun install && bun dev
```

---

## Turso playbook alignment

Follow [`TURSO.md`](TURSO.md) checkboxes in order. This slice completes **Step 1 (schema)** and **Step 7 (API stub + dashboard scaffold)** locally on sqlite3. Steps 3–5 (MVCC, CDC, sync) remain optional. Step 6 stays Chroma-first until Turso vector gold path is validated.

---

## GitNexus

Run `gitnexus analyze` on this repo when you want impact/call-graph in IDE. Not required for RAG ingest.
