# Phase 2 — Chronicals QLoRA (strict order)

**Prerequisites (do not skip):**

| Phase | Status | Required artifact |
|-------|--------|-------------------|
| **1** Data lake | Done | `data/curated/*.jsonl`, `warehouse-load` |
| **1** Personal-first mix | Done | `training-manifest --manifest-id personal-first` |
| **1** Train extract | Done | `data/train/personal-first.jsonl` |
| **1.5** Turso dashboard | Deferred | SQLite warehouse OK for now (`docs/TURSO.md` later) |
| **0** Ollama base | You | `qwen2.5-coder:7b` pulled |

**Stack (May 2026):** [Chronicals](https://github.com/Ajwebdevs/Chronicals) + TRL 1.5 + Liger kernels + LoRA+ + BFD sequence packing. Assistant-only loss via TRL `assistant_only_loss=True` on ChatML `messages` datasets.

---

## Step 0 — Free GPU (automatic)

`train-qlora` runs **GPU VRAM reclaim** before training (`gpu_mutex` in `config/default.yaml`): stops `hyprwhspr.service`, `ollama stop`, SIGKILLs alive GPU hogs, waits briefly on ghost VRAM, tries `nvidia-smi --gpu-reset` (often blocked on primary display GPU). If ~9 GiB **ghost** VRAM persists after a crashed train, run `uv run --package llm-core clear-gpu-vram` or log out/reboot. hyprwhspr restarts when the run finishes.

```bash
uv run --package llm-train train-qlora --smoke --run-name smoke-test
uv run --package llm-train train-qlora --smoke --no-gpu-mutex   # skip mutex
```

Manual check:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
systemctl --user stop hyprwhspr.service
ollama stop
```

---

## Step 1 — Install train stack

```bash
cd "/home/pyro1121/Documents/LLM Self Training"
uv sync --package llm-train
```

**Note:** Do not `uv sync --extra train --extra dataprep` in one shot unless both lockfiles agree — use train-only sync for Phase 2; dataprep sync when ingesting.

Pins: `chronicals[kernels,training,8bit]`, `liger-kernel`, TRL `>=1.5`, `bitsandbytes`, base `Qwen/Qwen2.5-Coder-7B-Instruct` + on-the-fly 4-bit.

---

## Step 2 — Prepare train file (personal-first)

```bash
uv run --package llm-dataprep training-manifest --manifest-id personal-first
uv run --package llm-dataprep training-extract \
  --manifest-id personal-first \
  --out data/train/personal-first.jsonl
```

---

## Step 3 — Preflight (required)

Doc-aligned checks (TRL chat template, assistant-only filter, deps, VRAM). See `docs/TRAIN-DOC-AUDIT.md`.

```bash
uv sync --package llm-train
uv run --package llm-train train-preflight              # bootstrap profile
uv run --package llm-train train-preflight --promote    # promote profile
```

Exit 0 before any GPU train.

---

## Step 4 — Dry-run (no GPU)

```bash
uv run --package llm-train train-qlora --dry-run
```

---

## Step 5 — Smoke train (~5 steps)

First run downloads `Qwen/Qwen2.5-Coder-7B-Instruct`. Resume interrupted HF cache:

```bash
uv run --package llm-train python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Coder-7B-Instruct', resume_download=True)"
```

```bash
uv sync --package llm-train   # always before train
uv run --package llm-train train-qlora --smoke --run-name smoke-test
```

---

## Step 6 — Full bootstrap train

```bash
uv sync --package llm-train
uv run --package llm-train train-qlora --run-name pyro-coder-bootstrap
```

Defaults: RSLoRA r=16, seq 768, LoRA+ 16× B LR, cosine 2e-4, max_steps 150.

---

## Step 6b — Promote train (Chronicals-tuned quality)

After bootstrap completes, run the **promote profile** — maps online Chronicals guidance to what fits a 4070 Ti 12GB:

```bash
uv sync --package llm-train
uv run --package llm-train train-qlora --promote --run-name pyro-coder-promote
```

| Setting | Bootstrap | Promote (`--promote`) |
|---------|-----------|------------------------|
| `max_seq_length` | 768 | **1024** (activation offload) |
| LoRA rank / alpha | 16 / 32 | **32 / 64** |
| Learning rate | 2e-4 | **1.5e-4** |
| Warmup | 5% | **10%** |
| Max steps | 150 | **400** |
| Grad accum | 8 | **12** |
| Weight decay | 0 | **0.01** |
| LoRA dropout | 0 | **0.05** |
| Activation offload | off | **on** (uses 32GB RAM) |
| Liger kernels | off | **class-level** (RMSNorm/SwiGLU) |
| DataLoader workers | 0 | **8** (i9 prefetch) |

Overrides live in `config/default.yaml` under `train.promote` and `chronicals.promote`.

### Research vs this rig (what we adopt / skip)

| Chronicals recommendation | 4070 Ti verdict |
|---------------------------|-----------------|
| QLoRA 4-bit NF4 + bf16 | **On** (default) |
| LoRA+ ratio 16 | **On** |
| All linear targets | **On** (q/k/v/o + MLP) |
| BFD sequence packing | **After** `flash-attn` install |
| FlashAttention | **Opt-in** (`use_flash_attention: true`) |
| Activation offload + 8-bit optim | **On in promote** |
| sqrt(n) gradient checkpointing | **On** |
| Liger fused LCE | **Off** (PEFT + assistant_only_loss unstable) |
| Liger RMSNorm/SwiGLU | **On in promote** |
| `max_seq_length` 2048–4096 | **Skip** — OOM without FA + offload; try 1536 only after flash-attn |
| `torch.compile` max-autotune | **Opt-in** — step-0 VRAM spike on 12GB |
| FP8 | **Skip** — unstable on 7B consumer GPUs |
| ChronicalsTrainer native | **Skip** — TRL SFTTrainer + Chronicals patches |

**Ablation order (after promote stable):**

1. `pip install flash-attn --no-build-isolation` → `use_flash_attention: true`, `use_sequence_packing: true`
2. Raise `max_seq_length_12gb_cap` to 1536 if no OOM
3. `chronicals.torch_compile: true` with `torch_compile_mode: reduce-overhead`
4. `lora_r` 64 only if VRAM headroom confirmed

---

## Step 7 — Export (merge + GGUF)

```bash
uv run --package llm-train train-export \
  --adapter-dir runs/pyro-coder-bootstrap/adapter \
  --out exports/pyro-coder
```

Writes `exports/pyro-coder/merged-hf/`. If `convert_hf_to_gguf.py` is on PATH (llama.cpp), emits GGUF automatically.

---

## Chronicals tuning (`config/default.yaml`)

| Key | Bootstrap | Promote |
|-----|-----------|---------|
| `chronicals.use_liger_kernel` | false | **true** (class-level only) |
| `chronicals.use_lora_plus` | true | true |
| `chronicals.use_sequence_packing` | false | false until flash-attn |
| `chronicals.use_activation_offload` | false | **true** |
| `chronicals.use_chronicals_gradient_checkpointing` | true | true |
| `train.max_seq_length` | 768 | **1024** |
| `train.per_device_train_batch_size` | 1 | 1 |
| `chronicals.use_flash_attention` | false | opt-in |
| `chronicals.torch_compile` | false | opt-in |
| `chronicals.use_8bit_optimizer` | true | true |
| `train.base_model` | `Qwen/Qwen2.5-Coder-7B-Instruct` | same |

**Speed experiments (after stable promote):** install flash-attn, then enable FA + packing, optional torch_compile.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FlashAttentionConfig` / `_attn_implementation` | Pull latest `llm-train`; TRL gets `peft_config` + `ensure_transformers_config` — no `--extra unsloth` |
| CUDA OOM | Confirm log shows `seq<=768` not 1024 (old run). `uv sync --package llm-train` then retry. Ghost VRAM after crash: wait 30s or reboot GPU. Tune `gpu_mutex.kill_process_substrings`. If still tight: `use_activation_offload: true`. `--smoke` |
| BFD packing disabled warning | Install flash-attn or `use_sequence_packing: false` |
| Liger `rope` error | `use_liger_kernel: false` |

---

## Config knobs

| Key | Default |
|-----|---------|
| `train.base_model` | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `training_mix.personal_ratio` | 0.80 |
| `training_mix.personal_sample_weight` | 1.0 |
| `training_mix.public_sample_weight` | 0.35 |
