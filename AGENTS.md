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
| 1 | **Context7** | Cursor plugin `plugin-context7-plugin-context7` → `resolve-library-id` then `query-docs` |
| 2 | **Exa** | Cursor plugin `plugin-exa-exa` → `web_search_exa` and/or `web_fetch_exa` |
| 3 | **Shell** | `rtk` / `uv run` / `curl` / `pytest` / `nvidia-smi` (exact commands in task) |
| 4 | **Repo** | `Grep`, `Read`, `git diff`, `git log` |

If Exa or Context7 is rate-limited, report **BLOCKED: &lt;tool&gt;** and set overall verdict to **INCOMPLETE** — not PASS.

Details: [`docs/AUDIT-PROTOCOL.md`](docs/AUDIT-PROTOCOL.md)

**MCP (Cursor built-in):** Use marketplace **Context7** + **Exa** plugins — do **not** duplicate them in `~/.cursor/mcp.json` unless you need a paid API key. Free-tier plugin quotas often require the workspace linked to a **public** GitHub repo; private repos may hit rate limits until you pay or make the repo public.

## Task / subagent dispatch (mandatory prompt block)

When spawning **any** review or audit subagent, include verbatim:

```
AUDIT MODE — MANDATORY TOOLS (skip = audit INVALID)

1. Context7: MCP server **`plugin-context7-plugin-context7`**. Call `resolve-library-id` + `query-docs` for each library in scope. Paste 1–2 snippets.

2. Exa: MCP server **`plugin-exa-exa`**. Call `web_search_exa` for each external claim. If rate-limited, say BLOCKED (check repo visibility on GitHub).

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
| `Makefile` | Operator shortcuts — `make help` |
| `packages/core` | Warehouse, `gpu_mutex`, `control_plane`, `clear-gpu-vram` |
| `packages/train` | `train-qlora`, Chronicals runtime, VRAM budget |
| `packages/eval` | `run-eval` promote gate |
| `packages/rag` | Chroma index, MCP server |
| `apps/api` | Control plane FastAPI `:8080` |
| `apps/dashboard` | Bun/Vite UI `:5173` |
| `config/default.yaml` | Train + gpu_mutex + chronicals |
| `eval/internal/*.jsonl` | Eval suite definitions |
| `docs/oss/` | Canonical documentation — start at `docs/oss/README.md` |
| `docs/AUDIT-PROTOCOL.md` | Audit tool checklist |

## Commands (prefer `rtk` in agent shell per user rules)

**Operator Makefile** (root `Makefile`): `make help` — wraps uv entrypoints for train, data, sanitize, GPU, API.

```bash
make sync-all
make warehouse-smoke          # or: uv run --package llm-core warehouse-smoke
make gpu-clear                # or: uv run --package llm-core clear-gpu-vram
make prepare-mixed            # manifest + extract → data/train/personal-first.jsonl
make train-smoke
make train                    # mixed 80/20; make train-personal for personal-only
make phase2-done RUN=pyro-coder-bootstrap
make test                     # pytest gpu_mutex
```

Equivalent uv (when Makefile flags are insufficient):

```bash
uv sync --package llm-core --package llm-train --package llm-eval --package llm-api
uv run --package llm-core warehouse-smoke
uv run --package llm-core clear-gpu-vram
uv run --package llm-train train-register --run-name pyro-coder-bootstrap
uv run --package llm-eval run-eval --train-run pyro-coder-bootstrap --no-smoke-chat
uv run pytest packages/core/tests/test_gpu_mutex.py -q
```

API (separate terminal): `make api` or `uv run --package llm-api llm-api`

## Promote gate

- **Train artifact:** `runs/<run>/adapter` or `checkpoint-*` + warehouse row via `make train-register`
- **Eval:** `make eval RUN=…` — use `--strict` with real tasks in `eval/internal/*.jsonl`
- **Export:** `make export RUN=…` + Ollama — not placeholder eval pass alone
- **Verify stack:** `make verify-phase15` (API + dashboard build)

## GPU

12 GB 4070 Ti. Before train: `make gpu-clear` or `uv run --package llm-core clear-gpu-vram`. Ghost VRAM → logout/reboot; primary GPU blocks `nvidia-smi --gpu-reset`.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **llm-self-training** (2770 symbols, 4534 relationships, 218 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/llm-self-training/context` | Codebase overview, check index freshness |
| `gitnexus://repo/llm-self-training/clusters` | All functional areas |
| `gitnexus://repo/llm-self-training/processes` | All execution flows |
| `gitnexus://repo/llm-self-training/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
