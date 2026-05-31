# Documentation index

Canonical documentation for **LLM Self Training**. Everything operators and contributors need lives in this folder.

| Document | Audience | Contents |
|----------|----------|----------|
| **[PRODUCT.md](PRODUCT.md)** | Evaluators, contributors | What the system does, eval-gated loop, differentiation |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Contributors, integrators | Monorepo, data flow, warehouse, train, RAG, API |
| **[USER-GUIDE.md](USER-GUIDE.md)** | Operators | Workflows, Makefile, train, troubleshooting |
| **[DATA-FORMATS.md](DATA-FORMATS.md)** | Dataprep + train authors | JSONL schemas raw → curated → train |
| **[CONFIG-REFERENCE.md](CONFIG-REFERENCE.md)** | Operators tuning hardware | `config/default.yaml` keys and profiles |
| **[CLOUD-TRAIN.md](CLOUD-TRAIN.md)** | Cloud operators | Jarvis H100 (legacy managed) |
| **[../cloud/README.md](../cloud/README.md)** | Cloud operators | **Vast.ai one-command** (`make cloud-vast`) |
| **[PUBLIC-DATASETS.md](PUBLIC-DATASETS.md)** | Operators | Hugging Face public dataset registry and ingest |
| **[CODING-SAFEGUARDS.md](CODING-SAFEGUARDS.md)** | Operators | Decensor / inform training profile |
| **[OLLAMA-INVENTORY.md](OLLAMA-INVENTORY.md)** | Operators | Local Ollama models for inference / embed / rerank |
| **[TURSO.md](TURSO.md)** | Contributors | Optional Turso warehouse migration playbook |
| **[OSS-RELEASE.md](OSS-RELEASE.md)** | Maintainers | Pre-public release safety checklist |
| **[LICENSE](../../LICENSE)** | All | Source Available License v1.0 |
| **[CONTRIBUTING.md](../../CONTRIBUTING.md)** | Contributors | Issues, PRs, dev setup |

## Package-local references

| Path | Contents |
|------|----------|
| [packages/dataprep/AGENT_HARNESSES.md](../../packages/dataprep/AGENT_HARNESSES.md) | Supported agent log formats |

## Quick paths

**Personal copilot only:** [USER-GUIDE § Personal-only](USER-GUIDE.md#personal-only-copilot)

**Mixed 80/20 train:** [USER-GUIDE § Mixed + public](USER-GUIDE.md#mixed-personal--public-hf)

**Promote gates:** [PRODUCT § Eval-gated loop](PRODUCT.md#eval-gated-loop)

**Hack on train stack:** [ARCHITECTURE § Training](ARCHITECTURE.md#training-stack)

**Eval suite definitions:** `eval/internal/*.jsonl` — see [USER-GUIDE § Post-train](USER-GUIDE.md#post-train--register--eval)

## Historical specs

Superseded by this folder — kept for reference only:

- [docs/archive/PLAN.md](../archive/PLAN.md)
- [docs/archive/ROADMAP.md](../archive/ROADMAP.md)

## Audit

- [docs/AUDIT-REPORT.md](../AUDIT-REPORT.md) — latest repo audit (non-GPU checks)
- [docs/AUDIT-PROTOCOL.md](../AUDIT-PROTOCOL.md) — agent tool requirements

## Repo entry points

- [README.md](../../README.md) — clone landing page
- [Makefile](../../Makefile) — `make help`
- [AGENTS.md](../../AGENTS.md) — agent tool requirements
