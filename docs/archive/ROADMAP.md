# ROADMAP — LLM Self Training

**Spec:** [`PLAN.md`](PLAN.md) (what to build)  
**This file:** when to build it, exit criteria, checklists  
**Tracking:** Linear team COM — filter `[LLM-ST]`

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| Linux + NVIDIA driver + CUDA | 4070 Ti 12 GB VRAM |
| Ollama 0.23.2+ | Daily inference |
| uv + Python 3.11+ | Monorepo |
| Bun | Dashboard only |
| Cursor | Optional BYOK → `http://127.0.0.1:11434/v1` (Ollama native API) |
| Disk | **100 GB min** / **250 GB comfortable** on data volume |

```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
ollama pull deepseek-r1:7b    # debug eval swap only
```

**Paths (ingest):**

| Harness | Path / format |
|---------|----------------|
| Cursor | `~/.cursor/projects/*/agent-transcripts/**/*.jsonl` |
| Codex | `~/.codex/sessions/**/rollout-*.jsonl` |
| Claude Code | `~/.claude/projects/<hash>/*.jsonl` |
| Pi | `~/.pi/agent/sessions/--<cwd>--/*.jsonl` |
| OpenCode | `~/.local/share/opencode/opencode.db` (message+part) |
| T3 Code | `~/.t3/userdata/state.sqlite` (`projection_thread_messages`) |
| Aider | `.aider.chat.history.md` (per-repo markdown) |
| Cline | VS Code `globalStorage/saoudrizwan.claude-dev/tasks` or `~/.cline/data` |
| Continue | `~/.continue/sessions/*.json` |
| Gemini CLI | `~/.gemini/tmp/<hash>/chats/session-*.jsonl` |
| Windsurf | `~/.codeium/windsurf/cascade` — **encrypted; export only** |
| Git | PyDriller `--repo` |

Probe: `uv run --package llm-dataprep agent-ingest --list-harnesses`

**GPU rule:** one heavy job — train **or** chat **or** embed batch **or** vLLM synth. `train-qlora` stops **hyprwhspr** (~5 GiB) + **Ollama** automatically; restart voice after train.

---

## Fast track (recommended)

| Phase | Week | Milestone | Exit |
|-------|------|-----------|------|
| **0** | 0 | Baseline + Ollama | Coder pulled; local API works; eval JSONL stubs; ingest scripts run |
| **1** | 1 | Data lake | ≥200 tier-1 rows; 50-row secrets+PII audit |
| **1.5** | 1 | Control plane | Turso `control_plane.db` — `docs/oss/TURSO.md` step-by-step + API + dashboard |
| **2** | 2 | QLoRA v0 | `pyro-coder-*` passes promote gates; Training page shows run |
| **2.5** | 2 | Benchmarks UI | Internal suites charted; manual re-run from UI |
| **3** | 3 | Logger + RAG MCP | JSONL logging; thin FastMCP over `packages/rag` |
| **4** | 4 | RAG v1 | Allowlist crawl; retrieval gold ≥70% (80% → hybrid) |
| **4.5** | 4 | External bench | `swe_micro` once; trends in warehouse |
| **5** | 5 | Weekly loop | Dual systemd timers; eval → train → promote; quarantine |
| **6+** | — | Experiments | LoRA+, prefs, Studio — `experiments/` only |

**Default track:** same phases but do **Phase 4 (RAG)** before **Phase 2 (train)** if doc-truth matters more than fast adapter.

---

## Build order (repo scaffold)

Do this **before** Phase 0 feature work:

1. [x] **COM-98 / COM-100** — `pyproject.toml` uv workspace (`uv` 0.11.x), `packages/*`, `apps/api`, `services/logger`
2. [x] **`packages/core`** — `llm_core.paths` (repo/data/config roots)
3. [x] **`apps/api`** stub — `uv run --package llm-api` → `:8080/health`
4. [x] **`apps/dashboard`** — Bun + Vite scaffold (shadcn/TanStack in Phase 1.5 polish)
5. [x] **`eval/internal/`** JSONL templates
6. [x] **`config/default.yaml`**, `.gitignore`

```bash
cd "/path/to/llm-self-training"
uv sync --group dev              # core + RAG deps (no GPU train yet)
uv sync --package llm-api        # control plane API
uv sync --package llm-train --package llm-eval   # Phase 2 — Chronicals + TRL + torch (no Unsloth extra)
```

---

## Phase checklists

### Phase 0 — Baseline + Ollama (COM-91, COM-102–105) — **complete**


- [x] `ollama pull qwen2.5-coder:7b` — pulled (~4.7 GB); ship model per [Ollama library](https://ollama.com/library/qwen2.5-coder:7b) (not the same as `qwen2.5:7b` general)
- [x] Confirm native API — Ollama **0.23.2**; `GET /v1/models` and `GET /api/version` OK on `127.0.0.1:11434` ([OpenAI compat](https://docs.ollama.com/api/openai-compatibility), Context7 May 2026)
- [x] Smoke chat: `POST /v1/chat/completions` → `ollama-ok` (web: [LM Studio/Ollama 2026 local `/v1`](https://www.promptquorum.com/local-llms/local-llm-openai-compatible-api))
- [ ] Optional Modelfile alias only if a client needs a different display name
- [x] **Cloudflare Tunnel / Cursor Verify — skipped (complete by decision)**  
  **Reason:** Ollama already exposes OpenAI-compatible API locally (`http://127.0.0.1:11434/v1`). No public tunnel unless a future client cannot reach localhost. Research: [Ollama API intro](https://docs.ollama.com/api/introduction), web (LM Studio/Ollama 2026 local `/v1` guides). Exa rate-limited; Context7 + web used.
- [x] `eval/internal/*.jsonl` — placeholder templates committed (replace with 15–25 real tasks before Phase 2 promote)
- [x] `cursor_transcripts` v0 — `uv run --package llm-dataprep cursor-transcripts` (parent sessions only; excludes `subagents/`)
- [x] `agent-ingest` — unified harness CLI: cursor, codex, claude_code (if present), git
- [x] `git_diffs` (PyDriller) — `uv sync --extra dataprep` then `agent-ingest --harness git`

### Phase 1 — Data lake (COM-92, COM-106–111)

**Research (May 2026):** Cursor stores `agent-transcripts/<uuid>/*.jsonl` (user/assistant + `tool_use`; tool *outputs* often missing). [cursor-history](https://github.com/S2thend/cursor-history) for SQLite export; [AgentProbe](https://github.com/vtemian/agentprobe) for passive parse patterns. Exa rate-limited — Context7 + web. Dual SQLite ingest (AI-Data-Extraction) deferred.

- [x] Cursor JSONL ingest v0 (`packages/dataprep`)
- [x] Ingest: Codex reducer (`output_text`), git diffs (`--repo`); Cursor SQLite deferred
- [x] `filters.py` + `scan-raw` (regex; gitleaks per-file opt-in) — `make sanitize`
- [x] Tier-1 gate (`tier1.py`) + `curate-raw` + `uv run --package llm-dataprep phase1`
- [x] ≥200 tier-1 rows in `data/curated/` (re-run `phase1` after ingest)
- [x] Audit sample CLI — `audit-sample` → `docs/audits/` (operator review before train)
- [x] Replay buffer seed — `replay-seed` → `data/replay/`

### Phase 1.5 — Control plane (COM-122)

**Warehouse:** Turso — implement by following [docs/oss/TURSO.md](docs/oss/TURSO.md) in order (official docs linked per step). Local file uses **pyturso** when `warehouse.driver: turso`.

- [x] Step 0–2 in `docs/oss/TURSO.md`: CLI + schema + Python SDK (`warehouse-smoke`)
- [x] `data/warehouse/control_plane.db` — curated, manifests, ingest_runs/files, quarantine, rag/benchmark tables
- [ ] Optional Step 3–5: MVCC ingest / CDC / Cloud sync (read docs before enabling flags)
- [x] `apps/api`: overview, datalake summary, quarantine POST/GET, training runs, RAG status
- [x] Dashboard: **Overview** + **DataLake** + **Training** tabs
- [x] `scripts/verify-phase15.sh` — warehouse-smoke + API curl + dashboard build
- [ ] Operator: `uv run --package llm-api llm-api` + `bun dev` while developing (optional live sign-off)

### Phase 2 — QLoRA v0 (COM-93, COM-112–113)

**Train:** `make train` / `README.md` — Chronicals + TRL + PEFT (Context7 `/huggingface/trl` when auditing).

- [x] GPU mutex — `train-qlora` stops/restarts `hyprwhspr.service` + `ollama stop` (use `--no-gpu-mutex` to skip)
- [x] `packages/train` — `train-qlora`, `train-export` per PLAN defaults
- [ ] Optional NEFTune A/B (`neftune_noise_alpha=5` vs off)
- [ ] Replay-only consolidation slice
- [x] Bootstrap train `pyro-coder-bootstrap` (150 steps, adapter on disk)
- [x] `train-register` + warehouse `training_runs` + dashboard Training tab
- [x] `run-eval` — placeholder suites pass (`--strict` for real tasks later)
- [ ] Export: `train-export` → GGUF → `ollama create pyro-coder:7b` (needs CUDA + llama.cpp)
- [ ] Promote to daily driver after real eval tasks + `--strict` pass

### Phase 2.5 — Benchmarks UI (COM-121)

- [ ] Recharts trends per suite
- [ ] Trigger benchmark job from UI (`ollama stop` first)
- [ ] Store results in warehouse

### Phase 3 — Logger + RAG MCP (COM-94)

- [ ] `services/logger` FastAPI → `data/raw/logs-*.jsonl`
- [ ] Optional: point Cursor at `http://127.0.0.1:8080/v1` (logger → Ollama)
- [ ] `packages/rag/mcp_server.py` FastMCP stdio, read-only tools

### Phase 4 — RAG v1 (COM-95)

- [ ] `llms.txt` tier-0 before Crawl4AI
- [ ] Chroma ingest; nomic embed
- [ ] `retrieval_gold.jsonl` on dashboard
- [ ] If gold &lt;80%: BM25+RRF + rerank; then embed upgrade path

### Phase 4.5 — External benchmarks (COM-121)

- [ ] `swe_micro` manifest + worktree runner (no Docker)
- [ ] Optional: LCB lite, Aider micro (~30 tasks)
- [ ] Monthly cadence; regression floor vs last promoted model

### Phase 5 — Weekly loop (COM-96)

- [ ] `orchestrator/loop.py` + `program.md`
- [ ] `deploy/llm-self-train-activity.service` (15–30 min poll)
- [ ] `deploy/llm-self-train-weekly.service` (crawl + full eval)
- [ ] Activity: debounce 2–4h, GPU mutex, train often / promote rarely
- [ ] Quarantine rows linked to benchmark failures (Phase 5 schema)
- [ ] `eval_score` for autoresearch keep/discard; hard gates for promote

### Phase 6+ — Experiments

- [ ] `experiments/train_adv_peft.py` only
- [ ] LoRA+ → DoRA → ORPO (after two failed style gates)
- [ ] Unsloth Studio manual sandbox — not CI

---

## Spikes (before / during Phase 2)

| Spike | Pass criteria |
|-------|----------------|
| Local Ollama API smoke | `GET /v1/models` done; chat after `qwen2.5-coder:7b` pull |
| Tunnel / Verify | **N/A** — localhost API sufficient |
| 50-row PII audit | No leaks in curated sample |
| QLoRA smoke ~200 rows | No OOM @ 2048; GGUF loads in Ollama |
| NEFTune A/B | Style **and** debug gates vs baseline |
| Judge calibration (20 samples) | RISE vs Prometheus2 — pick one |

---

## Definition of done (global)

- Behavior matches **PLAN.md**; no secrets in committed samples
- GPU steps document `ollama stop` where required
- **Promote** only via `run_eval.py` per-suite `"verdict": "pass"`
- Train success alone is **not** promote

---

## Linear index

| ID | Epic |
|----|------|
| [COM-97](https://linear.app/competitor-intel/issue/COM-97) | Scaffold + docs |
| [COM-98+](https://linear.app/competitor-intel/issue/COM-98) | Monorepo scaffold |
| [COM-91](https://linear.app/competitor-intel/issue/COM-91) | Phase 0 |
| [COM-92](https://linear.app/competitor-intel/issue/COM-92) | Phase 1 |
| [COM-122](https://linear.app/competitor-intel/issue/COM-122) | Phase 1.5 Control plane |
| [COM-93](https://linear.app/competitor-intel/issue/COM-93) | Phase 2 QLoRA |
| [COM-121](https://linear.app/competitor-intel/issue/COM-121) | Benchmarks hub |
| [COM-94](https://linear.app/competitor-intel/issue/COM-94) | Phase 3 Logger + MCP |
| [COM-95](https://linear.app/competitor-intel/issue/COM-95) | Phase 4 RAG |
| [COM-96](https://linear.app/competitor-intel/issue/COM-96) | Phase 5 Loop |

**Start order:** COM-98 → COM-100 → COM-102–105 → COM-106–111 (parallel COM-122 late Phase 1) → COM-112–113 → COM-121 children → COM-94+.

---

## Open decisions (you)

1. Languages + doc crawl allowlist  
2. Repos + Cursor project boundaries  
3. DeepSeek / qwen3.5 inference bake-off (after Phase 0 baselines)  
4. Fast vs default track (fast track recommended in table above)

---

## Status

- [x] PLAN.md + ROADMAP.md (docs tree removed)
- [x] Linear epics COM-91–122
- [x] 165-agent research merged into PLAN
- [x] Monorepo scaffold (uv workspace)
- [x] **Phase 0** — Ollama native API + `qwen2.5-coder:7b`; tunnel skipped
- [x] **Phase 1** — data lake (208k tier-1, audit, replay, manifest)
- [x] **Phase 1.5** — control plane (API, warehouse, dashboard Training tab; Turso 3–5 optional)
- [ ] **Phase 2 promote** — export + real eval tasks + `run-eval --strict`
