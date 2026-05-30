# AGENTS.md — LLM Self Training

Instructions for **all** Cursor agents, subagents (`Task` tool), and automations in this repo.

## Non-negotiable: every task uses tools, not memory

Applies to **all** work: audits, implementation, debugging, research, phase sign-off, promote gates, and code review.

**Never** claim done, PASS, or “verified” from model recall, prior chat, or ROADMAP text alone.

| Task type | Context7 | Exa | Shell | Repo |
|-----------|----------|-----|-------|------|
| Audit / review | Required | Required | Required (listed cmds) | Required (`path:line`) |
| Implement feature | Required if touching unfamiliar library API | If external behavior/env claim | Run tests/smoke you add or touch | Read before edit |
| Debug / fix | If error mentions library | If GPU/OS/Ollama claim | Reproduce + show exit/output | Grep trace to root |
| Research doc | Required for API contracts | Required for “May 2026” facts | Optional | Read existing docs first |

If Exa or Context7 is rate-limited, report **BLOCKED: &lt;tool&gt;** and set audit verdict to **INCOMPLETE** — not PASS. For non-audit tasks, state what is unverified and do not mark external claims as confirmed.

Every audit **must** produce evidence from:

| # | Tool | MCP server / command |
|---|------|----------------------|
| 1 | **Context7** | `user-context7` (from `~/.cursor/mcp.json` + `CONTEXT7_API_KEY`) → `resolve-library-id` then `query-docs` |
| 2 | **Exa** | `user-exa` (from `~/.cursor/mcp.json` + `x-api-key`) → `web_search_exa` and/or `web_fetch_exa` |
| 3 | **Shell** | `rtk` / `uv run` / `curl` / `pytest` / `nvidia-smi` (exact commands in task) |
| 4 | **Repo** | `Grep`, `Read`, `git diff`, `git log` |

If Exa or Context7 is rate-limited, report **BLOCKED: &lt;tool&gt;** and set overall verdict to **INCOMPLETE** — not PASS.

Details: [`docs/AUDIT-PROTOCOL.md`](docs/AUDIT-PROTOCOL.md)

**MCP API keys (local only):** Copy [`.cursor/mcp.json.example`](.cursor/mcp.json.example) → `~/.cursor/mcp.json` headers (`CONTEXT7_API_KEY`, `x-api-key` for Exa). Reload Cursor MCP after change. Never commit real keys.

## Task / subagent dispatch (mandatory prompt block)

When spawning **any** review or audit subagent, include verbatim:

```
AUDIT MODE — MANDATORY TOOLS (skip = audit INVALID)

1. Context7: Use MCP server **`user-context7`** (not `plugin-context7-*`). Call `resolve-library-id` + `query-docs` for each library in scope. Paste 1–2 snippets.

2. Exa: Use MCP server **`user-exa`** (not `plugin-exa-*`). Call `web_search_exa` for each external claim. If rate-limited, say BLOCKED.

3. Shell: Run every command in your scope. Paste exit code + last 15 lines. No PASS without running them.

4. Repo: Grep/Read cited paths. Every must-fix needs file:line from Read/Grep, not paraphrase.

5. Verdict: PASS | FAIL | INCOMPLETE (if any mandatory tool blocked/skipped)

6. Do NOT use training data or parent chat as proof.
```

Subagent settings:

- `readonly: false` (shell + MCP required)
- `subagent_type: generalPurpose` for full tool surface
- Scope **one** area per agent (API, train, eval, dashboard, docs)

## Project map

| Path | Role |
|------|------|
| `packages/core` | Warehouse, `gpu_mutex`, `control_plane`, `clear-gpu-vram` |
| `packages/train` | `train-qlora`, Chronicals runtime, VRAM budget |
| `packages/eval` | `run-eval` promote gate |
| `packages/rag` | Chroma index, MCP server |
| `apps/api` | Control plane FastAPI `:8080` |
| `apps/dashboard` | Bun/Vite UI `:5173` |
| `config/default.yaml` | Train + gpu_mutex + chronicals |
| `eval/internal/*.jsonl` | Eval suites (placeholders until real tasks) |
| `docs/PHASE2-TRAIN.md` | Train runbook |
| `docs/PHASE15-PHASE2-SIGNOFF.md` | Sign-off commands |

## Commands (prefer `rtk` in agent shell per user rules)

```bash
uv sync --package llm-core --package llm-train --package llm-eval --package llm-api
uv run --package llm-core warehouse-smoke
uv run --package llm-core clear-gpu-vram
uv run --package llm-train train-register --run-name pyro-coder-bootstrap
uv run --package llm-eval run-eval --train-run pyro-coder-bootstrap --no-smoke-chat
uv run pytest packages/core/tests/test_gpu_mutex.py -q
```

API (separate terminal): `uv run --package llm-api llm-api`

## Phase completion criteria

- **1.5:** API routes + warehouse + dashboard Training tab + `scripts/verify-phase15.sh` evidence
- **2 train:** `runs/pyro-coder-bootstrap/adapter` or `checkpoint-*` + warehouse row
- **2 promote:** `run-eval --strict` + real eval JSONL + `train-export` + Ollama — not placeholder pass alone

## GPU

12 GB 4070 Ti. Before train: `uv run --package llm-core clear-gpu-vram`. Ghost VRAM → logout/reboot; primary GPU blocks `nvidia-smi --gpu-reset`.
