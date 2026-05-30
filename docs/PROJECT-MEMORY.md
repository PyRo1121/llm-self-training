# Project memory — LLM Self Training (operator + agent)

Updated: 2026-05-30. **Ordered work only** — defer doc/npm bulk ingest and extra public HF until Phase 2 exit.

## Architecture (decided)

| Layer | Role |
|-------|------|
| `data/raw/`, `data/curated/`, `data/train/*.jsonl` | **Lake** — source of truth, keep all JSONL |
| Turso / `control_plane.db` | **Catalog** — pointers, manifests, registry (not full chats) |
| QLoRA adapter | **Behavior** — personal style, tools, judgment (80%+ personal mix) |
| Chroma + Context7 (Phase 4) | **Facts** — APIs, libs, docs; refresh without retrain |

**Policy:** Facts in RAG; judgment in the adapter. Do not bulk-train npm/top packages or official language docs — retrieve them later.

**GPU:** `train-qlora` stops `hyprwhspr.service` + Ollama by default (`GpuMutex`).

## Phase status

| Phase | Status |
|-------|--------|
| 0 Ollama baseline | Done |
| 1 Data lake + personal-first extract | Done (`data/train/personal-first.jsonl`) |
| 1.5 Turso dashboard | Deferred |
| 2 QLoRA v0 | **In progress** — smoke OK (`runs/smoke-test/adapter/`); bootstrap started, not finished (`runs/pyro-coder-bootstrap/` has config only) |
| 1.5 Control plane + RAG | **In progress** — schema v4, API routes, `packages/rag`, dashboard scaffold (`docs/PHASE-1.5-RAG.md`) |
| 3 Logger + RAG MCP | RAG MCP stub (`python -m llm_rag.mcp_server` + `--extra mcp`) |
| 4 RAG v1 | Index path live (`rag-index`); gold eval not run |

## Ordered next steps (do in order)

1. **Finish Phase 2 bootstrap train** — `docs/PHASE2-TRAIN.md` Step 5  
2. **Export GGUF** — Step 6 → `ollama create pyro-coder:7b`  
3. **Eval gate** — implement/run `packages/eval/run_eval.py` when ready (Step 7)  
4. **Promote** only if gates pass  
5. Phase 1.5 + RAG integrated with train (user priority May 2026) — see `docs/PHASE-1.5-RAG.md`  

## Deferred (do not block Phase 2)

- Extra HF datasets / GitHub JSONL hunt / `swe_hero` registry  
- npm top-2500 / Go Rust TS doc **training** ingest  
- Full warehouse load of all curated rows  

## Key paths

- Train: `data/train/personal-first.jsonl`  
- Smoke adapter: `runs/smoke-test/adapter/`  
- Bootstrap out: `runs/pyro-coder-bootstrap/adapter/` (target)  
- Runbook: `docs/PHASE2-TRAIN.md`, `ROADMAP.md`
