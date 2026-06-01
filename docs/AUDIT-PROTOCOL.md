# Audit protocol — mandatory tools (all agents)

See also root [`AGENTS.md`](../AGENTS.md).

## Rule

**All agent tasks** (see `AGENTS.md` tool matrix): use Context7 / Exa / Shell / Repo as required for the task type. Memory-only completion is **invalid**.

**PASS / FAIL / promote** requires evidence from Context7, Exa (or BLOCKED), Shell, and Repo tools. Memory-only audits are **invalid**.

## Tool checklist (every audit)

| Step | Action | Pass criterion |
|------|--------|----------------|
| C7-1 | `resolve-library-id` + `query-docs` per dependency in scope | Quote snippet + URL/source |
| EX-1 | `web_search_exa` per external/environment claim | Cite title/URL or BLOCKED |
| SH-1 | Run listed shell commands | Show `$?` and output tail |
| RP-1 | `Grep`/`Read`/`git diff` for changed files | `path:line` on every must-fix |

## Subagent policy

- **`readonly: false`** — required for Shell and MCP
- Prompt must include the **AUDIT MODE** block from `AGENTS.md`
- If agent returns PASS without tool evidence → parent rejects audit and re-runs
- Parent may pre-run shell and attach logs, but subagent must still run Context7 + Exa + verify repo paths itself

## Sign-off commands

```bash
make verify-phase15
make phase2-done RUN=pyro-coder-bootstrap
make test
```

Equivalent uv:

```bash
uv run --package llm-core warehouse-smoke
curl -sf http://127.0.0.1:8080/health
curl -sf http://127.0.0.1:8080/api/v1/overview | head -c 500
uv run --package llm-train train-register --run-name pyro-coder-bootstrap
uv run --package llm-eval run-eval --train-run pyro-coder-bootstrap --no-smoke-chat
uv run pytest packages/core/tests/test_gpu_mutex.py -q
```

## Report template

```markdown
## Verdict: PASS | FAIL | INCOMPLETE

### Context7
- Library: /huggingface/trl — [finding] — [snippet]

### Exa
- Query: … — [finding] or BLOCKED: …

### Shell
- `command` → exit N — [summary]

### Repo
- path:line — issue

### Must-fix
1. …

### Unknowns / blocked tools
- …
```
