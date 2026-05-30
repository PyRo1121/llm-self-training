# Tool-mandatory audits — 2026-05-30

Parent + five `Task` subagents per [`AGENTS.md`](../AGENTS.md) AUDIT MODE.

**2026-05-30 (retry):** API keys in `~/.cursor/mcp.json` → MCP servers `user-context7` + `user-exa` **working**. Subagent batch still **INCOMPLETE** (ran before keys). Parent re-check below.

**Original batch:** INCOMPLETE — Context7 monthly quota and Exa HTTP 429 on plugin MCP.

## Summary

| Scope | Agent | Verdict | Context7 | Exa | Shell | Repo |
|-------|-------|---------|----------|-----|-------|------|
| API / Phase 1.5 | subagent `32135311` | **INCOMPLETE** | BLOCKED | BLOCKED | warehouse-smoke fail (DB lock); API down; gpu_mutex 5/5 | Done |
| Train / Phase 2 | subagent `6b7babc5` | **INCOMPLETE** | BLOCKED | BLOCKED | checkpoint-150, train-register OK | 768 + GpuMutex PASS |
| Eval | subagent `3a5924aa` | **INCOMPLETE** | BLOCKED | BLOCKED | non-strict pass; `--strict` fail | Bootstrap honest; promote FAIL |
| Dashboard / gpu_mutex | subagent `e39d08d6` | **INCOMPLETE** | skipped | BLOCKED | clear-gpu-vram 0; bash -n 0; bun build 0 | PASS sub-areas |
| Docs / ROADMAP | subagent `e8c870dc` | **INCOMPLETE** | BLOCKED | BLOCKED | rg + train_settings OK | Doc accuracy **FAIL** (stale refs; Phase 2 OK) |

**Parent shell (same session):** `warehouse-smoke` pass, `gpu_mutex` 5/5, `train-register` OK, `run-eval` placeholder pass, `run-eval --strict` fail (expected).

## Parent evidence

```text
warehouse-smoke: pass (curated_examples=208613, training_manifests=2)
pytest packages/core/tests/test_gpu_mutex.py: 5 passed
train-register: checkpoint-150 registered
run-eval (no --strict): verdict pass
run-eval --strict: placeholder_tasks_only → fail
nvidia-smi: ~10781 MiB free / 12282 MiB
```

## 1. API (subagent)

- **Context7 / Exa:** BLOCKED (quota / 429).
- **Shell:** Turso file lock on `control_plane.db` during subagent run; `/health` connection refused (API not started).
- **Repo must-fix:** `main.py:35` legacy `on_event("startup")`; Turso exclusive lock vs concurrent smoke (`warehouse_driver.py`, `docs/TURSO.md` MVCC).

## 2. Train (parent; subagent stalled)

- **Repo:** `vram_budget.py:8` `HARD_MAX_SEQ_12GB = 768`; `config.py` defaults 768; `train_qlora.py` uses `resolve_vram_train_params` + GpuMutex.
- **Shell:** `runs/pyro-coder-bootstrap/checkpoint-150` exists; `train-register` succeeded.
- **Promote:** not done — export/Ollama pending.

## 3. Eval (parent)

- **Repo:** `run_eval.py:58-72` — placeholder suites **pass** without `--strict`, **fail** with `--strict` (`placeholder_tasks_only`). Honest for bootstrap; not for PLAN promote.
- **Shell:** `--strict` fails as designed.

## 4. Dashboard / gpu_mutex (subagent)

- **Shell:** reclaimed 9762 MiB stale train via `clear-gpu-vram`; dashboard build OK.
- **Repo:** full ghost pipeline in `gpu_mutex.py` (`ensure_gpu_ready_for_train`, `resolve_gpu_ghost_vram`); Training tab in `App.tsx`.
- **Exa:** BLOCKED — verdict INCOMPLETE per protocol.

## 5. Docs (parent)

- **ROADMAP fixes applied:** sync line no longer says Unsloth extra; Phase 2 runbook points to TRL/Chronicals.
- **Accurate:** bootstrap train, train-register, placeholder eval checked [x]; promote/export still [ ].
- **Lie removed:** `uv sync --extra train` / “Unsloth May 2026” as primary stack.

## Parent re-check (keys fixed)

| Tool | Status | Evidence |
|------|--------|----------|
| Context7 | **OK** | `user-context7`: FastAPI `lifespan` vs deprecated `on_event`; TRL `peft_config` + `SFTConfig.max_length` + `assistant_only_loss` |
| Exa | **OK** | `user-exa`: 7B QLoRA 12GB — 768 conservative vs 2048 with Unsloth; our 768 cap justified for Chronicals/non-Unsloth |
| Shell | **partial** | `warehouse-smoke` pass; API `/health` ok when server up; re-run `scripts/verify-phase15.sh` for full Phase 1.5 |
| Repo | **unchanged** | API audit must-fix: `main.py:35` lifespan migration; Turso single-writer |

## To reach full audit PASS

1. ~~Restore Context7/Exa~~ done via `~/.cursor/mcp.json`.
2. Re-dispatch five subagents using MCP servers **`user-context7`** / **`user-exa`** (not `plugin-*`).
3. Run `scripts/verify-phase15.sh` with API up.
