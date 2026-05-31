# Cloud training on Jarvis H100

Run a **75/25 personal/public** QLoRA promote on rented H100 with full public ingest support (multi-million-row HF datasets).

See also: [USER-GUIDE.md](USER-GUIDE.md), [PUBLIC-DATASETS.md](PUBLIC-DATASETS.md), [CONFIG-REFERENCE.md](CONFIG-REFERENCE.md).

## Profile

Set on cloud instances (automatic with `--cloud` or `make cloud-train`):

```bash
export LLM_CONFIG_PROFILE=cloud-h100
```

Overlay file: `config/cloud-h100.yaml` — H100 tuning (seq up to 8192, r=64, batch 8, FA2 padding-free, ingest workers 20/12).

## Before you rent (local)

```bash
make phase1 REPO="/path/to/your/repo"
make sanitize && make curate && make warehouse-load
make lake-stats

# Personal data for cloud (tier-1 only, ~100MB — not full 93GB raw/)
make cloud-export-personal
```

### Personal data on GitHub

**Your main repo is public** — do **not** push `personal-tier1.jsonl` there (Cursor/Codex transcripts).

| Approach | Steps |
|----------|--------|
| **A. Same repo (private only)** | GitHub → Settings → change repo to **Private**, then `git add -f data/cloud/personal/personal-tier1.jsonl` and push |
| **B. Private data repo (recommended)** | Create `PyRo1121/llm-self-training-data` (private), push only `personal-tier1.jsonl`, set `CLOUD_DATA_REPO_URL` on Jarvis |
| **C. Private HF dataset** | `make cloud-pack` |

After `git clone`, Jarvis uses `data/cloud/personal/personal-tier1.jsonl` automatically if present.

Public HF datasets still ingest via `HF_TOKEN` (gated sets).

### Hugging Face token

**Never commit tokens.** Put in local `.env` (gitignored):

```bash
cp .env.example .env
# edit: HF_TOKEN=hf_...
```

On Jarvis: `jl upload <id> ~/.cache/huggingface/token /home/hf_token` or copy `.env` to instance home (not into git).

If a token was pasted in chat or logs, **revoke it** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a new one.

## Jarvis one-time setup

```bash
uv tool install jarvislabs
jl setup

# Optional: persistent HF cache across instance destroys (IN2)
jl filesystem create --name pyro-coder-data --storage 500 --region IN2
export JARVIS_FS_ID=<fs_id>
```

Upload HF token to instance once (do not commit):

```bash
jl create --gpu H100 --region IN2 --storage 500 --name pyro-upload --yes
jl upload <machine_id> ~/.cache/huggingface/token /home/hf_token
jl pause <machine_id>
```

## Launch full run from laptop

```bash
export HF_DATASET=PyRo1121/pyro-coder-personal-bundle
export RUN=pyro-coder-h100-v1
export JARVIS_FS_ID=...   # optional

make cloud-jarvis
# smoke first:
make cloud-jarvis-smoke
```

Equivalent:

```bash
./scripts/cloud/run-jarvis.sh
./scripts/cloud/run-jarvis.sh --smoke-only
./scripts/cloud/run-jarvis.sh --skip-ingest   # re-train on cached ingest
```

## On-instance manual run

```bash
export LLM_CONFIG_PROFILE=cloud-h100
bash scripts/cloud/setup-jarvis.sh
bash scripts/cloud/train-cloud.sh \
  --run pyro-coder-h100-v1 \
  --personal-dataset PyRo1121/pyro-coder-personal-bundle \
  --personal-ratio 0.75
```

## Pull artifacts

```bash
jl download <machine_id> runs/pyro-coder-h100-v1 ./runs/pyro-coder-h100-v1 -r
jl download <machine_id> exports/pyro-coder-h100-v1 ./exports/pyro-coder-h100-v1 -r
jl pause <machine_id>
```

Local Ollama:

```bash
make gpu-clear
ollama create pyro-coder:7b -f exports/pyro-coder-h100-v1/Modelfile
```

## Mega datasets (20M+ raw rows)

| Stage | What happens |
|-------|----------------|
| **Raw ingest** | `swe_zero_12m`, `codex_7m`, etc. can be **hundreds of GB** — your 2.8 TB Jarvis disk fits it |
| **Curate** | Tier-1 + safety filters shrink train rows dramatically |
| **Train manifest** | 75% personal / 25% public from warehouse |
| **One epoch steps** | `train_rows / 16` — can be **days** if millions of tier-1 rows |

**Recommendations:**

1. Run **`make cloud-jarvis-smoke`** before full ingest+train.
2. First time: `--ingest-mode bootstrap` in `train-cloud.sh` to validate pipeline (~1 h).
3. Full ingest: default `INGEST_MODE=full` (all enabled datasets in config).
4. Cap training if needed: `CLOUD_TRAIN_MAX_STEPS=50000 ./scripts/cloud/run-jarvis.sh` or `--max-steps N`.
5. Re-runs: `--skip-ingest` when HF cache on filesystem/instance is warm.

Personal hundreds-of-GB: use `pack-personal.sh` for tier-1 JSONL only (much smaller), or `jl upload` raw tarball to filesystem + extract to `data/raw/` before train.

## Makefile targets

| Target | Action |
|--------|--------|
| `make cloud-export-personal` | Export tier-1 personal → `data/cloud/personal/` |
| `make cloud-pack` | Alternative: private HF dataset upload |
| `make cloud-jarvis-smoke` | Jarvis smoke (5 train steps) |
| `make cloud-jarvis` | Full Jarvis managed run |
| `make cloud-train-local` | Same pipeline on current machine (needs GPU) |

## Environment

| Variable | Purpose |
|----------|---------|
| `LLM_CONFIG_PROFILE=cloud-h100` | H100 yaml overlay |
| `LLM_DATA_DIR` | Put `data/` on Jarvis filesystem mount |
| `LLM_HF_CACHE_DIR` | HF parquet cache (default `data/hf_cache`) |
| `HF_TOKEN` | Gated public datasets |
| `CLOUD_TRAIN_MAX_STEPS` | Optional step cap for mega runs |
| `JARVIS_GPU` | Default `H100` |
| `JARVIS_REGION` | Default `IN2` |
| `JARVIS_STORAGE` | Default `500` (GB) |
| `JARVIS_FS_ID` | Persistent filesystem |
| `HF_DATASET` | Private personal bundle repo |
