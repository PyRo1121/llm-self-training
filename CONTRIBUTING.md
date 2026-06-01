# Contributing

Thank you for your interest in LLM Self Training. Please read this file and [LICENSE](LICENSE) before participating.

## License reminder

This project uses the **LLM Self Training Source Available License v1.0** — not MIT/Apache.

| Allowed without permission | Requires written permission |
|----------------------------|----------------------------|
| Clone and run for **personal / non-commercial** use | **Commercial** use or SaaS |
| Open issues (bugs, docs, questions) | **Forks** intended for public redistribution |
| Send **patch proposals** (see below) | **Modified** public mirrors or derivative products |
| Security reports (private disclosure welcome) | Removing copyright notices |

Commercial licensing: **olen@latham.cloud**

## Ways to contribute

### 1. Issues (preferred first step)

Open a GitHub issue for:

- Bug reports (include command, exit code, last log lines)
- Documentation gaps in [docs/oss/](docs/oss/)
- Feature ideas (may be deferred — this is a personal operator stack)

Do **not** paste secrets, API keys, or full agent transcripts into issues.

### 2. Documentation

Documentation fixes are welcome. Target:

- [docs/oss/](docs/oss/) — public-facing guides (canonical)
- [docs/oss/USER-GUIDE.md](docs/oss/USER-GUIDE.md) — operator workflows

Keep changes factual; link to file paths and commands that exist in the repo.

### 3. Code changes

Because the license restricts public derivative works, code contribution flow is **maintainer-merge only**:

1. **Open an issue** describing the problem before large PRs.
2. Fork **only** if you accept that your fork must stay **private** unless you have a commercial license, or you intend a PR back to upstream.
3. Keep PRs **small and focused** — one logical change per PR.
4. Match existing style (ruff, uv workspace, Makefile patterns).
5. Do not commit `data/`, `runs/`, `.env`, or audit JSONL with local paths.

**We do not merge:**

- Drive-by refactors unrelated to the issue
- New dependencies without justification
- Changes that weaken safety filters (secrets/PII) without discussion
- License header removal or permissive re-licensing

### 4. Security

Report vulnerabilities to **olen@latham.cloud** with:

- Steps to reproduce
- Impact assessment
- Suggested fix (optional)

Do not open public issues for exploitable secrets-in-training-data scenarios until coordinated.

## Development setup

```bash
git clone https://github.com/PyRo1121/llm-self-training.git
cd llm-self-training
make sync-all
# Unsloth runtime (if train-qlora import fails):
uv pip install "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo.git"
make help
```

| Task | Command |
|------|---------|
| Lint | `make lint` or `uv run ruff check packages apps/api` |
| Tests | `make test` |
| Warehouse smoke | `make warehouse-smoke` |
| Train dry-run | `make train-dry-run` |
| Phase 1.5 verify | `make verify-phase15` (API + dashboard build) |

Python **3.11–3.13**. GPU optional for dataprep/docs work; required for train integration tests you run locally.

## Project structure (where to edit)

| Area | Path |
|------|------|
| Core / warehouse / GPU mutex | `packages/core/` |
| Ingest / curate | `packages/dataprep/` |
| Train / export | `packages/train/` |
| Eval gate | `packages/eval/` |
| RAG | `packages/rag/` |
| API | `apps/api/` |
| Dashboard | `apps/dashboard/` |
| Operator config | `config/default.yaml` |
| Public docs | `docs/oss/` |

Architecture overview: [docs/oss/ARCHITECTURE.md](docs/oss/ARCHITECTURE.md)

## Data and privacy

- **`data/`, `runs/`, `exports/`** are gitignored — never commit them.
- **`docs/audits/*.jsonl`** are gitignored — generated locally by `audit-sample`; may contain path previews.
- Safety policy: filter **secrets + PII only**, not topic/refusal content ([docs/oss/OSS-RELEASE.md](docs/oss/OSS-RELEASE.md)).

## Pull request checklist

- [ ] Issue linked or rationale stated in PR description
- [ ] Scope is minimal (Karpathy-style: every changed line traces to the goal)
- [ ] `ruff check` clean on touched packages
- [ ] No secrets, `.env`, or personal JSONL in the diff
- [ ] Docs updated if CLI, config keys, or workflows changed
- [ ] You agree your contribution may be included under the project LICENSE (copyright remains with the project; you retain authorship credit in commit history)

## Code style

- Follow patterns in surrounding files — naming, imports, yaml keys.
- Prefer `make` targets over documenting long raw `uv run` chains unless flags are non-obvious.
- Comments only for non-obvious business logic.
- Do not add tests that assert the obvious unless they catch a real regression.

## Governance

This is a **maintainer-led** project. The owner decides merge priority, roadmap ([docs/archive/ROADMAP.md](docs/archive/ROADMAP.md)), and commercial licensing.

Questions: open a GitHub issue or email **olen@latham.cloud**.
