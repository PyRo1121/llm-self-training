# Train stack — documentation compliance audit

**Purpose:** Stop train-debug loops. Every promote run must pass `train-preflight` first.

**Sources (May 2026):**

| Library | Doc |
|---------|-----|
| TRL SFT | [sft_trainer.md](https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md) |
| TRL chat templates | [chat_templates.md](https://github.com/huggingface/trl/blob/main/docs/source/chat_templates.md) |
| TRL memory | [reducing_memory_usage.md](https://github.com/huggingface/trl/blob/main/docs/source/reducing_memory_usage.md) |
| PEFT QLoRA | [peft examples / BitsAndBytesConfig](https://github.com/huggingface/peft) |
| Chronicals | [GitHub README](https://github.com/Ajwebdevs/Chronicals), [PyPI 0.1.4](https://pypi.org/project/chronicals/) |

---

## Architecture choice (documented deviation)

| Chronicals docs | Our stack | Verdict |
|-----------------|-----------|---------|
| `ChronicalsTrainer` + `ChronicalsConfig` end-to-end | TRL `SFTTrainer` + Chronicals patches (GC, LoRA+, optional Liger/FA) | **Intentional** — TRL owns dataset tokenization, `assistant_only_loss`, PEFT wrap. Chronicals supplies optimizers + memory kernels only. |

Do **not** mix native `ChronicalsTrainer` and our TRL path in one run.

---

## TRL compliance matrix

| Requirement | Doc reference | Implementation | Status |
|---------------|-----------------|----------------|--------|
| Dataset: `{"messages": [{role, content}, …]}` | sft_trainer.md | `dataset.py` | ✅ |
| `assistant_only_loss=True` | sft_trainer.md | `build_sft_config` | ✅ |
| Training chat template via `get_training_chat_template()` | chat_templates.md | `dataset_filter.py` + SFTTrainer auto-swap | ✅ |
| Qwen2.5 in supported families | chat_templates.md | `Qwen2.5-Coder-7B-Instruct` | ✅ |
| Drop rows with zero assistant tokens after truncate | sft_trainer.md (RuntimeError) | `filter_dataset_to_max_tokens` + `keep_end` slice | ✅ |
| `peft_config` passed to `SFTTrainer` | reducing_memory_usage.md | `train_qlora.py` | ✅ |
| `packing=True` only with flash-attn | sft_trainer.md | Disabled when FA missing | ✅ |
| `truncation_mode` | SFTConfig defaults `keep_start` | We use **`keep_end`** for assistant-only + pre-filter | ⚠️ Deprecated in TRL 1.5 but required for coding SFT at seq 1024; filter mirrors collator |
| Save adapter after train | PEFT/TRL | `trainer.model.save_pretrained` | ✅ (fixed) |

---

## PEFT / QLoRA compliance matrix

| Requirement | Doc reference | Implementation | Status |
|---------------|-----------------|----------------|--------|
| 4-bit NF4 + double quant | PEFT BitsAndBytes examples | `chronicals_runtime.load_qlora_model` | ✅ |
| `bnb_4bit_compute_dtype=bfloat16` | PEFT | ✅ | ✅ |
| `device_map={"": 0}` single GPU | PEFT QLoRA examples | ✅ | ✅ |
| `prepare_model_for_kbit_training` | PEFT | Before LoRA wrap | ✅ |
| Target modules q/k/v/o + MLP | Coding best practice | `LoraConfig` | ✅ |
| `use_rslora=True` | — | Our choice (RSLoRA) | ⚠️ Not in Chronicals default; stable |

---

## Chronicals compliance matrix (partial integration)

| Feature | Chronicals docs | Our wiring | Status |
|---------|-----------------|----------|--------|
| LoRA+ `lr_ratio=16` | README / paper | `create_lora_plus_optimizer` | ✅ |
| sqrt(n) gradient checkpointing | `apply_gradient_checkpointing` | `model.model.layers` | ✅ |
| Activation offload | `activation_offload=True` | promote profile | ✅ |
| FlashAttention + varlen packing | README | Opt-in after `flash-attn` install | ⏸️ |
| Liger fused LCE / CCE | README | **Off** — TRL `assistant_only_loss` + PEFT breaks LCE | ⚠️ Documented skip |
| Liger class kernels (RMSNorm/SwiGLU) | liger-kernel | promote, `model=None` patch | ✅ |
| `torch.compile` | README default on | **Off** on 12GB | ⚠️ Opt-in after stable promote |
| Native `ChronicalsTrainer` | README quick start | Not used | N/A |

---

## 4070 Ti constraints (hardware overlay)

These override Chronicals paper defaults:

| Setting | Chronicals default | 4070 Ti |
|---------|-------------------|---------|
| `max_seq_length` | 2048–4096 | 768 bootstrap / **1024 promote** (offload) |
| `batch_size` | 4+ | **1** |
| `use_flash_attention` | true | false until `flash-attn` built |
| `use_sequence_packing` | true | false until flash-attn |
| FP8 | paper feature | **skip** |

---

## Known data effects (not bugs)

| Effect | Cause |
|--------|-------|
| ~538 rows dropped @ seq 1024 | Prompt fills budget; `keep_end` leaves no assistant tokens — filtered per TRL rules |
| `max_steps=322` with cap 400 | One epoch = `3874 / (1×12)` steps; cap only binds above epoch length |
| Noisy step loss | Weighted sampler + variable-length assistant spans |

---

## Preflight gate (required before train)

```bash
uv sync --package llm-train
uv run --package llm-train train-preflight --promote   # or omit --promote for bootstrap
```

Exit code 0 = safe to train. Non-zero = fix listed failures first.

---

## Train order (no shortcuts)

1. `train-preflight` (bootstrap or `--promote`)
2. `train-qlora --smoke`
3. `train-qlora` bootstrap OR `--promote`
4. `train-export`

---

## Future ablation (after promote stable)

1. Install `flash-attn` → enable FA + BFD packing
2. Try seq 1536 with offload
3. Optional `torch_compile`
4. Evaluate native `ChronicalsTrainer` in isolated experiment (separate branch)
