# LLM Self Training

Local, private coding copilot: **RAG + Chronicals QLoRA + eval-gated loop + dashboard** on RTX 4070 Ti (12 GB), bare metal only.

| Doc | Purpose |
|-----|---------|
| **[PLAN.md](PLAN.md)** | Architecture, stack, schemas, training/eval spec |
| **[ROADMAP.md](ROADMAP.md)** | Phases, checklists, Linear order, exit criteria |
| **[docs/TURSO.md](docs/TURSO.md)** | Turso warehouse — doc-driven implementation (Phase 1.5) |

**Python:** [uv](https://docs.astral.sh/uv/) workspace — `uv sync --group dev`, `uv run --package llm-api llm-api`

**Ingest (local agents):** `uv sync --package llm-dataprep --extra git` → `agent-ingest --list-harnesses` → `agent-ingest`. Full catalog: `packages/dataprep/AGENT_HARNESSES.md` (Windsurf `.pb` not decryptable; `state.vscdb` partial only).

**Safety scan (Phase 1):** `uv run --package llm-dataprep scan-raw` — regex + gitleaks (if installed) + Presidio (if `uv sync --extra safety`). See `docs/PHASE1-FILTERS.md`.

**Curate (Phase 1):** `uv run --package llm-dataprep curate-raw` — group by session, chunk long threads, tier-1 gate (`tier1.py`, bootstrap exec/verify).

**Phase 1 (all steps):** `uv run --package llm-dataprep phase1 --fresh-raw --include-subagents --repo /path/to/git/repo` — see [`docs/PHASE1-RUNBOOK.md`](docs/PHASE1-RUNBOOK.md).

**Public HF data:** `uv run --package llm-dataprep public-ingest` — SWE-Next, OpenCode broad, Nemotron, etc. See [`docs/PUBLIC-DATASETS.md`](docs/PUBLIC-DATASETS.md). Add `--public` to `phase1` to ingest before personal logs.

**Phase 2 train:** [`docs/PHASE2-TRAIN.md`](docs/PHASE2-TRAIN.md) — `uv sync --package llm-train` → `train-qlora` (personal-first JSONL + sample weights).

**Bulk ingest tip:** `scan-raw` defaults gitleaks off; use `--gitleaks --gitleaks-per-file` when the CLI is installed.

**Linear:** filter `[LLM-ST]` on team COM — [COM-91](https://linear.app/competitor-intel/issue/COM-91) Phase 0 next (scaffold done).
