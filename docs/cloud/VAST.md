# Vast.ai production runbook

## Prerequisites

| Item | Where |
|------|--------|
| Vast API key | [cloud.vast.ai/manage-keys](https://cloud.vast.ai/manage-keys/) → `VAST_API_KEY` |
| HF token | Vast template secret **or** `HF_TOKEN` in `config/cloud.env` |
| GitHub PAT | `GITHUB_TOKEN` with **Contents: read** on private repo |
| SSH key | `vastai create ssh-key ~/.ssh/id_ed25519.pub` (before first rent) |

## One command

```bash
make cloud-vast-smoke   # validate first
make cloud-vast        # full pipeline
```

Equivalent:

```bash
./scripts/cloud/run-vast.sh --smoke-only
./scripts/cloud/run-vast.sh
```

## Instance selection

Default search (cheapest verified H100 with ≥600 GB disk, ≥500 Mbps down):

```
gpu_name=H100_SXM num_gpus=1 disk_space>=600 verified=true rentable=true direct_port_count>=1 inet_down>=500 reliability>=0.98
```

Override:

```bash
export VAST_OFFER_ID=12345678
make cloud-vast
```

Or relax search:

```bash
export VAST_SEARCH_QUERY='gpu_name=H100_PCIE num_gpus=1 disk_space>=500 verified=true rentable=true'
make cloud-vast
```

**Avoid interruptible** offers for multi-day training.

## Docker image

Default: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` (has `nvcc` for flash-attn build).

Override: `export VAST_IMAGE=...`

First run: flash-attn compile ~10–20 min. Re-runs on same image/volume skip if already installed.

## Secrets on Vast website

You can set `HF_TOKEN` as a Vast **template secret** instead of `config/cloud.env`. The bootstrap script reads `HF_TOKEN` from container env either way.

For **private repo**, `GITHUB_TOKEN` must reach the container (env or Vast secret). Without it, clone fails on boot.

## Artifacts

```bash
make cloud-vast-pull INSTANCE=12345678
# uses .vast/last-instance-id if INSTANCE omitted
```

Local paths: `runs/<RUN>/adapter`, `exports/<RUN>/`

```bash
make gpu-clear
ollama create pyro-coder:7b -f exports/pyro-coder-h100-v1/Modelfile
```

## Destroy (stop billing)

```bash
make cloud-vast-destroy
# or: INSTANCE=12345678 bash scripts/cloud/vast-destroy.sh
```

## Persistent HF cache (recommended for re-runs)

1. Create a Vast **volume** (500+ GB) in console.
2. Mount on create:

```bash
export VAST_VOLUME=98765:/workspace/data
export LLM_DATA_DIR=/workspace/data
make cloud-vast
```

3. Re-train without re-download:

```bash
make cloud-vast-smoke --skip-ingest   # via run-vast: ./scripts/cloud/run-vast.sh --skip-ingest
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Clone failed | Check `GITHUB_TOKEN`, repo private + PAT scope |
| Gated datasets skipped | Set `HF_TOKEN` in env / Vast secret |
| flash-attn failed | SSH in, check `nvcc`; or `SKIP_FLASH_ATTN=1` (lower seq) |
| Disk full | Raise `VAST_DISK_GB` or mount volume |
| Slow ingest | Filter offers: `inet_down>=1000`; use volume cache |
| OOM on train | Smoke logs show seq/batch; set `CLOUD_TRAIN_MAX_STEPS` to cap cost |

## SSH (optional)

```bash
vastai ssh-url $(cat .vast/last-instance-id)
# tail training: ps aux | grep train
```

## Jarvis alternative

Managed launcher: `make cloud-jarvis` — see [CLOUD-TRAIN.md](../oss/CLOUD-TRAIN.md).
