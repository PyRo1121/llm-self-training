# Phase 1.5 + Phase 2 sign-off

## Phase 1.5 — Control plane

```bash
uv sync --package llm-api --package llm-core
uv run --package llm-core warehouse-smoke
uv run --package llm-api llm-api   # :8080
./scripts/verify-phase15.sh
cd apps/dashboard && bun dev       # :5173 → API proxy
```

| Check | Command |
|-------|---------|
| Health | `curl http://127.0.0.1:8080/health` |
| Overview | `curl http://127.0.0.1:8080/api/v1/overview` |
| Training runs | `curl http://127.0.0.1:8080/api/v1/training/runs` |
| Quarantine POST | `curl -X POST http://127.0.0.1:8080/api/v1/datalake/quarantine -H 'Content-Type: application/json' -d '{"curated_id":"…","reason":"test"}'` |

## Phase 2 — QLoRA bootstrap

```bash
uv sync --package llm-train --package llm-eval
uv run --package llm-core clear-gpu-vram    # if ghost VRAM
uv run --package llm-train train-qlora --run-name pyro-coder-bootstrap
./scripts/phase2-complete.sh pyro-coder-bootstrap
```

| Step | Status |
|------|--------|
| Train 150 steps @ seq≤768 | Done if `runs/pyro-coder-bootstrap/adapter` exists |
| `train-register` → warehouse | `phase2-complete.sh` |
| `run-eval` bootstrap gate | Passes placeholder suites; use `--strict` when tasks are real |
| `train-export` → Ollama | Manual when GPU free |

**Promote rule (PLAN):** `run-eval --strict` + real tasks in `eval/internal/*.jsonl` — not loss alone.
