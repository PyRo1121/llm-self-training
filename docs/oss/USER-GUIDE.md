# User guide

Operator documentation for running LLM Self Training on your machine. For architecture see [ARCHITECTURE.md](ARCHITECTURE.md).

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Linux + **NVIDIA GPU** | Designed for **RTX 4070 Ti 12 GB** |
| [uv](https://docs.astral.sh/uv/) | Python 3.11–3.13 workspace |
| CUDA driver | For train + export |
| [Bun](https://bun.sh) | Dashboard only |
| Ollama | `qwen2.5-coder:7b` for eval; embed model for RAG |
| Optional: gitleaks on PATH | Secrets scan in Phase 1 |
| Optional: `HF_TOKEN` | Gated Hugging Face datasets |
| Optional: `GITHUB_TOKEN` | GitHub code-search harvest (`make github-harvest`) |

```bash
cd "/path/to/llm-self-training"
make help
make sync-all
```

## Environment variables

| Variable | Effect |
|----------|--------|
| `LLM_SELF_TRAINING_ROOT` | Override repo root |
| `LLM_DATA_DIR` | Relocate entire `data/` tree |
| `HF_TOKEN` | Hugging Face auth for gated datasets |
| `GITHUB_TOKEN` | GitHub API for public session harvest |
| `WAREHOUSE_DRIVER` | `sqlite` or `turso` |

## Quick start

### Personal-only copilot

```bash
make phase1 REPO="/path/to/your/git/repo"
make train-personal
```

Produces `data/train/personal-only.jsonl` and trains to `runs/pyro-coder-bootstrap/adapter/`.

### Mixed personal + public HF (recommended)

```bash
export HF_TOKEN=hf_...   # optional, for gated sets
make data-public         # ingest → curate → warehouse
make train               # 80/20 mix from config
```

## Workflows

### Phase 1 — data lake

**Goal:** ≥200 tier-1 curated rows, 50-row safety audit, replay buffer.

```bash
make phase1 REPO="/path/to/repo"
# or with public HF first:
make phase1-public REPO="/path/to/repo"
```

**Steps (automatic):** ingest → scan-raw → curate → link-logs-to-diffs → replay-seed → audit-sample → warehouse-sync-registry → warehouse-load

**Verify:**

```bash
make lake-stats
ls data/curated/curated-*.jsonl
cat docs/audits/phase1-audit-*.md   # review flagged rows
```

**Common failures:**

| Symptom | Fix |
|---------|-----|
| Empty curated | Run ingest; check agent logs exist (`make ingest-list`) |
| &lt;200 tier-1 | Add repos or `make data-public` |
| Duplicates in raw | `phase1 --fresh-raw` archives prior raw |

See [USER-GUIDE § Sanitize](USER-GUIDE.md#sanitize--secrets--pii) and [OSS-RELEASE.md](OSS-RELEASE.md) for safety policy.

### Sanitize — secrets + PII

```bash
make sync-safety          # once — Presidio
python -m spacy download en_core_web_sm
make sanitize             # scan-raw
make curate               # honors safety-failures
make audit-sample         # 50-row policy audit → docs/audits/
make safety-eval          # fixture precision/recall (regex-only)
make warehouse-load
make prepare-mixed
```

Policy: **secrets + PII only** — not topic/refusal filtering. See [OSS-RELEASE.md](OSS-RELEASE.md).

**Block vs warn.** Each finding is classified `block` or `warn` (regex kind, Presidio entity, or `gitleaks_severity`). `config/default.yaml` → `safety.quarantine_severity`:

| Setting | Quarantine when | Output |
|---------|-----------------|--------|
| `block` (default) | ≥1 **block** finding | `data/raw/safety-failures-*.jsonl` |
| `warn` | ≥1 block **or** warn finding | failures file only |

With `quarantine_severity: block`, warn-only rows land in `data/raw/safety-warn-*.jsonl` (review only — curate still ingests them). Entire session dropped at curate if any raw line appears in failures.

**Allowlist.** `config/safety-allowlist.yaml` — `exact` strings and `regex` patterns matched against finding detail or text span. Merged into `safety` policy at load. Add doc/test placeholders (e.g. `user@example.com`, git SHAs) so examples do not quarantine real data.

**Diff mode.** Rows from harnesses in `safety.diff_harnesses` (`git`, `git-diffs`, or `source_path` containing `git-diffs`) scan **added lines only** (`+` in unified diff, not `+++` headers). Presidio skipped on diff rows; gitleaks sidecar still runs. Tweak harness list in `config/default.yaml`.

**Safety eval.** `make safety-eval` runs labeled fixtures (`packages/dataprep/fixtures/safety_eval.jsonl`) through `scan_text` / `scan_diff_text` (regex-only, no gitleaks/Presidio) and prints precision/recall/F1 overall and per label. Use after changing allowlist or severity rules.

Config: `safety` block in `config/default.yaml` + `config/safety-allowlist.yaml`. Finding schema: [DATA-FORMATS.md](DATA-FORMATS.md#safety-failure-row).

### Prepare train file

```bash
make prepare-mixed        # → data/train/personal-first.jsonl (80/20)
make prepare-personal     # → data/train/personal-only.jsonl
```

Override mix: `make manifest-mixed PERSONAL_RATIO=0.75 PUBLIC_CAP=5000`

### Train

**Always clear GPU first on 12 GB:**

```bash
make gpu-clear
make gpu-status
```

| Goal | Command |
|------|---------|
| Preflight | `make train-preflight` or `make train-preflight-promote` |
| Dry run (no GPU) | `make train-dry-run` |
| Smoke (~5 steps) | `make train-smoke` |
| Bootstrap | `make train` |
| Personal only | `make train-personal` |
| Promote profile | `make train-promote RUN=pyro-coder-unsloth-v1` |

**Direct uv (extra flags):**

```bash
uv sync --package llm-train --extra unsloth
uv pip install "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo.git"  # if import fails

uv run --package llm-train train-preflight --promote
uv run --package llm-train train-qlora --promote --run-name my-run \
  --train-file data/train/personal-first.jsonl
```

**Useful flags:** `--smoke`, `--dry-run`, `--max-steps N`, `--no-gpu-mutex`, `--chronicals` (legacy backend)

**Artifacts:**

```
runs/<run-name>/
  adapter/              # LoRA + tokenizer
  train_config.json     # frozen hyperparams + VRAM plan
  eval_report.json      # after eval
```

**Common failures:**

| Symptom | Fix |
|---------|-----|
| Missing train JSONL | `make prepare-mixed` |
| CUDA OOM | Confirm seq≤768 without FA2; `make gpu-clear`; stop competing trains |
| Unsloth import error | `uv sync --package llm-train --extra unsloth`; install unsloth-zoo via pip |
| Ghost VRAM | Logout/reboot; `make gpu-clear` |

See [CONFIG-REFERENCE.md](CONFIG-REFERENCE.md) for promote profiles and VRAM ceilings.

### Flash-attn unlock (promote v2 @ longer seq)

```bash
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
# Set unsloth.disable_auto_padding_free: false in config when ready
make train-promote RUN=pyro-coder-unsloth-v2
```

### Post-train — register + eval

```bash
make phase2-done RUN=pyro-coder-bootstrap
# = train-register + run-eval (bootstrap placeholders)
```

**Real promote gate:**

```bash
uv run --package llm-eval run-eval --train-run my-run --strict
```

Replace placeholder tasks in `eval/internal/*.jsonl` first (15–25 real tasks per suite).

### Export to Ollama

```bash
make gpu-clear
make export RUN=pyro-coder-bootstrap

# Unsloth GGUF (recommended):
uv run --package llm-train train-export \
  --adapter-dir runs/pyro-coder-bootstrap/adapter \
  --out exports/pyro-coder-bootstrap --unsloth

ollama create pyro-coder:7b -f exports/pyro-coder-bootstrap/Modelfile
```

### RAG (optional)

```bash
ollama pull qwen3-embedding:4b   # or nomic-embed-text
uv run --package llm-rag rag-index
uv sync --package llm-rag --extra mcp
# MCP: python -m llm_rag.mcp_server — see ARCHITECTURE.md § RAG package
```

### Control plane UI

**Terminal 1:**

```bash
make api    # :8080
```

**Terminal 2:**

```bash
make dashboard    # :5173
```

Open http://127.0.0.1:5173 — Overview, Training, Data Lake tabs.

**Register a finished run in UI/DB:**

```bash
make train-register RUN=my-run
# or POST /api/v1/training/runs/register
```

## Makefile reference

Run `make help` for full list. Key targets:

| Group | Targets |
|-------|---------|
| Setup | `sync`, `sync-all`, `sync-train`, `sync-dataprep`, `sync-safety` |
| Ingest | `ingest`, `public-ingest`, `phase1`, `phase1-public` |
| Safety | `sanitize`, `curate`, `curate-fast`, `audit-sample`, `safety-eval` |
| Warehouse | `warehouse-load`, `warehouse-smoke`, `lake-stats` |
| Train prep | `prepare-mixed`, `prepare-personal` |
| GPU | `gpu-clear`, `gpu-status` |
| Train | `train-preflight`, `train-smoke`, `train`, `train-promote` |
| Post-train | `train-register`, `eval`, `export`, `phase2-done` |
| Dev | `api`, `dashboard`, `verify-phase15`, `test` |

## End-to-end sequences

**First bootstrap:**

```
sync-all → phase1 REPO=… → prepare-mixed → gpu-clear
→ train-preflight → train-smoke → train → phase2-done → export
```

**Promote after real eval tasks:**

```
train-preflight-promote → train-promote RUN=v1
→ run-eval --strict → export → ollama create
```

**Public data boost:**

```
HF_TOKEN=… → data-public → prepare-mixed → train
```

## Troubleshooting

| Issue | Command / action |
|-------|------------------|
| GPU memory stuck | `make gpu-clear`; reboot if ghost VRAM |
| Warehouse empty | `make warehouse-load` after curate |
| API dashboard empty | Start `make api` before dashboard |
| Config drift | See [CONFIG-REFERENCE.md](CONFIG-REFERENCE.md) yaml vs code notes |

## Getting help

- [README.md](README.md) — doc index
- [ARCHITECTURE.md](ARCHITECTURE.md) — implementation detail
- [AGENTS.md](../../AGENTS.md) — contributor tool requirements
