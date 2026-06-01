# LLM Self Training

Local, private coding copilot: **RAG + QLoRA + eval-gated loop + dashboard** on RTX 4070 Ti (12 GB), bare metal only.

**License:** [Source Available License v1.0](LICENSE) — personal/non-commercial use; no commercial use, public forks, or derivatives without permission. [CONTRIBUTING.md](CONTRIBUTING.md)

## Documentation

**Start here:** **[docs/oss/README.md](docs/oss/README.md)**

| Audience | Document |
|----------|----------|
| Operators | [USER-GUIDE.md](docs/oss/USER-GUIDE.md) |
| Product / evaluators | [PRODUCT.md](docs/oss/PRODUCT.md) |
| Contributors | [ARCHITECTURE.md](docs/oss/ARCHITECTURE.md) |

Day-to-day commands: **`make help`** (see [Makefile](Makefile)).

## Quick start

```bash
cd "/path/to/llm-self-training"
make sync-all

# 1) All data prep (personal + public HF → train JSONL). HF_TOKEN in .env
make prep

# 2) Local GPU train (4070 Ti)
make train

# Or rent H100 on Vast.ai (HF_TOKEN + VAST_API_KEY in .env)
make train-cloud
```

Optional: `make prep REPO=/path/to/git/repo` · `make prep-bg` (background) · `make train-smoke`

## Repository layout

| Path | Role |
|------|------|
| `Makefile` | Operator shortcuts |
| `config/default.yaml` | Train, ingest, warehouse, RAG settings |
| `packages/*` | Python workspace (core, dataprep, train, eval, rag) |
| `apps/api`, `apps/dashboard` | Control plane + UI |
| `docs/oss/` | **Canonical documentation** |
| `docs/archive/` | Historical PLAN + ROADMAP (superseded by docs/oss) |
| `AGENTS.md` | Cursor agent + audit tool requirements |

## Agents & audits

Contributors and Cursor agents: read **[AGENTS.md](AGENTS.md)** and **[docs/AUDIT-PROTOCOL.md](docs/AUDIT-PROTOCOL.md)** before claiming PASS or promote.
