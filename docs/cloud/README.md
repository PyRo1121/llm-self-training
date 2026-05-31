# Cloud training — operator docs

One-command **Vast.ai** H100 training. Jarvis alternative: [../oss/CLOUD-TRAIN.md](../oss/CLOUD-TRAIN.md).

## Quick start (Vast)

```bash
# 1. One-time local
pip install vastai
vastai set api-key YOUR_KEY          # https://cloud.vast.ai/manage-keys/
vastai create ssh-key ~/.ssh/id_ed25519.pub

cp config/cloud.env.example config/cloud.env
# Fill: VAST_API_KEY, HF_TOKEN, GITHUB_TOKEN (private repo PAT)

git push   # instance clones main on boot — push before launch

# 2. Smoke (~cheap — validates ingest + 5 train steps)
make cloud-vast-smoke

# 3. Full run
make cloud-vast

# 4. Pull adapter + GGUF export
make cloud-vast-pull

# 5. Stop billing
make cloud-vast-destroy
```

**Billing:** Vast charges from instance create. `vast-bootstrap.sh` clones and starts training immediately — no manual SSH required.

## What runs on boot

```
vast-bootstrap.sh  →  git clone (private PAT)  →  vast-onstart.sh
  →  setup-cloud.sh (uv sync, HF login, flash-attn if needed)
  →  train-cloud.sh (ingest → curate → train → export)
```

Personal data: `data/cloud/personal/personal-tier1.jsonl` in repo (private).

Public HF: fast ingest + **hf_xet** (`HF_XET_HIGH_PERFORMANCE=1`).

## Docs in this folder

| File | Purpose |
|------|---------|
| [VAST.md](VAST.md) | Full Vast runbook, volumes, troubleshooting |
| [HARNESSES.md](HARNESSES.md) | Local agent harness catalog (Cursor, Codex, …) |
| [DATA-FORMATS.md](DATA-FORMATS.md) | JSONL row shapes for personal + public |

## Re-runs (save money)

Mount a Vast volume for HF cache:

```bash
# Create volume once in Vast console, then:
export VAST_VOLUME=12345:/workspace/data
export LLM_DATA_DIR=/workspace/data
make cloud-vast VAST_SKIP_INGEST=1   # or --skip-ingest on script
```

Second run skips multi-hour HF download if cache is warm.

## Config profile

`LLM_CONFIG_PROFILE=cloud-h100` — see `config/cloud-h100.yaml` and [CONFIG-REFERENCE](../oss/CONFIG-REFERENCE.md).
