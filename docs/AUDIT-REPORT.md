# Repository audit report

**Date:** 2026-05-30  
**Scope:** Docs cleanup, `.gitignore`, Makefile, non-GPU verification  
**Models:** Not started (no `train-qlora`, Ollama, or export)

## Overall verdict: **PASS (with follow-ups)**

| Area | Verdict | Notes |
|------|---------|-------|
| `.gitignore` | **PASS** | Data, runs, weights, secrets, caches covered |
| Secrets in tree | **PASS** | No `hf_*`, `sk-*`, or `AKIA*` in tracked files |
| Tests | **PASS** | 16 passed (`core` + `train` unit tests) |
| Warehouse | **PASS** | `warehouse-smoke` exit 0 |
| API import | **PASS** | `llm_api.main.app` loads |
| Dashboard build | **PASS** | `bun run build` exit 0 |
| Documentation | **PASS** | Canonical docs in `docs/oss/`; archive preserved |
| Lint | **FAIL** | `ruff check` — 35 issues (mostly unused imports) |
| Promote readiness | **FAIL** | Eval suites still placeholders |
| Full AGENTS audit | **INCOMPLETE** | Context7/Exa subagent batch interrupted |

## Shell evidence

```text
make help                          → exit 0
uv run pytest packages/core/tests packages/train/tests -q → 16 passed, exit 0
uv run --package llm-core warehouse-smoke → pass, exit 0
uv run ruff check packages apps/api → 35 errors, exit 1
cd apps/dashboard && bun run build → exit 0
git ls-files '*.gguf' '*.db' etc.  → (none)
git check-ignore data/ runs/ .env  → matched
```

## Documentation structure (post-cleanup)

| Keep | Role |
|------|------|
| `docs/oss/*` | **Canonical** operator + contributor library |
| `docs/archive/PLAN.md`, `ROADMAP.md` | Historical specs (intentionally kept) |
| `docs/AUDIT-PROTOCOL.md` | Agent audit checklist |
| `docs/AUDIT-REPORT.md` | This report |
| `AGENTS.md`, `CONTRIBUTING.md`, `LICENSE` | Governance |
| `packages/dataprep/AGENT_HARNESSES.md` | Harness catalog (code-adjacent) |

Removed duplicates: legacy `docs/PUBLIC-DATASETS.md`, `docs/PHASE2-TRAIN.md`, phase runbooks, warehouse index duplicates.

## `.gitignore` coverage

Ignores: `data/`, `runs/`, `exports/`, `adapters/`, model weights (`*.gguf`, `*.safetensors`, …), HF/torch caches, `wandb/`, secrets (`.env*`), `.cursor/mcp.json`, audit JSONL previews, `.claude/`.

Tracked on purpose: `eval/internal/*.jsonl` (suite definitions), `uv.lock`, `docs/oss/`, archive.

Template: [`.env.example`](../.env.example) — copy to `.env` locally.

## Must-fix (priority)

1. **Ruff** — run `uv run ruff check packages apps/api --fix` and clean remaining 35 diagnostics (`packages/train/src/llm_train/unsloth_runtime.py` unused imports, etc.).
2. **Eval suites** — replace placeholder tasks in `eval/internal/*.jsonl` before `run-eval --strict`.
3. **Train integration** — run `make train-preflight` + smoke when GPU free (not run this audit).

## Nice-to-have (world-class)

- Makefile: `lint`, `warehouse-sync-registry`, `probe-hf-schema` targets
- CI workflow: `ruff`, `pytest`, `warehouse-smoke`, dashboard build (no GPU job)
- `docs/oss/`: SECURITY.md, CHANGELOG.md, architecture diagram export
- Archive: optional banner at top of PLAN/ROADMAP pointing to `docs/oss/`

## Archive hygiene

Cross-links updated: `docs/TURSO.md` → `docs/oss/TURSO.md` in archive files. Personal path `/home/pyro1121/...` removed from `docs/archive/ROADMAP.md`.
