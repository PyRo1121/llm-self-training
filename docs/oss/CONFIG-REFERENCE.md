# Configuration reference

Central file: `config/default.yaml`. Loaded by package-specific helpers — not one global loader.

**Profiles:** `--promote` and `--decensor` shallow-merge `*.promote` blocks onto bootstrap defaults.

## Environment overrides

| Variable | Overrides |
|----------|-----------|
| `LLM_SELF_TRAINING_ROOT` | Repo root |
| `LLM_DATA_DIR` | `data/` tree |
| `WAREHOUSE_DRIVER` | `warehouse.driver` (`sqlite` \| `turso`) |
| `HF_TOKEN` | Hugging Face auth (ingest) |

## `train` — QLoRA hyperparameters

| Key | Bootstrap | Promote (`train.promote`) |
|-----|-----------|---------------------------|
| `backend` | `unsloth` | same |
| `base_model` | `Qwen/Qwen2.5-Coder-7B-Instruct` | same |
| `max_seq_length` | 768 | 2048 (VRAM may cap lower) |
| `max_seq_length_12gb_cap` | 768 | 2048 |
| `max_chars_per_message` | null (derived) | 12288 |
| `lora_r` | 16 (`lora_r_bootstrap`) | 32 |
| `lora_alpha` | 32 | 64 |
| `learning_rate` | 2.0e-4 | 1.5e-4 |
| `per_device_train_batch_size` | 2 | 1 |
| `gradient_accumulation_steps` | 8 | 16 |
| `max_steps_cap` | 150 | 400 |
| `warmup_ratio` | 0.05 | 0.10 |
| `num_epochs` | 1 | 1 |
| `seed` | 3407 | 3407 |

**From `training_mix` (injected into train_settings):** `personal_sample_weight: 1.0`, `public_sample_weight: 0.35`

### Decensor (`--decensor`)

Promote hyperparams + `train.decensor.base_model` → abliterated Qwen2.5-Coder-7B.

## `unsloth` — runtime

| Key | Bootstrap | Promote | Runtime gate |
|-----|-----------|---------|--------------|
| `use_prequantized` | true | — | skipped if decensor |
| `prequantized_model` | `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` | — | |
| `use_rslora` | true | — | |
| `use_lora_plus` | true | — | B-matrix LR ×16 |
| `lora_plus_lr_ratio` | 16.0 | — | |
| `clamp_to_one_epoch` | true | — | unless `--max-steps` |
| `token_audit` | true | — | skipped on `--smoke` |
| `token_audit_percentile` | 95 | — | |
| `eval_holdout_ratio` | 0.10 | — | skipped on smoke |
| `stratified_eval_holdout` | true | — | by `_data_source` |
| `eval_steps` | 32 | — | cap |
| `eval_step_divisor` | 6 | — | |
| `load_best_model_at_end` | true | — | metric eval_loss |
| `effective_batch_target` | 16 | — | VRAM adjusts grad_accum |
| `disable_torch_compile` | true | — | UNSLOTH_COMPILE_DISABLE |
| `disable_auto_padding_free` | **true** | — | set **false** to enable with FA2 |
| `auto_padding_free` | true | — | only if FA2 + disable flag false |
| `auto_pack_with_fa2` | false | — | BFD pre-pack alternative |
| `pack_by_data_source` | true | — | BFD strata |
| `dataset_num_proc` | 1 | **8** | tokenize prefetch |
| `dataloader_num_workers` | 0 | 4 | **forced 0 without FA2** |
| `activation_offloading` | false | **true** | **FA2 only** at runtime |
| `step0_headroom_mib` | 1200 | — | post-load seq downgrade |
| `max_grad_norm` | 1.0 | — | |
| `optim` | adamw_8bit | — | |
| `lr_scheduler_type` | cosine | cosine | |

### VRAM seq ceilings (no FA2)

| LoRA rank | Max seq |
|-----------|---------|
| r=16 | 1024 |
| r=32 | **768** |
| r=64 | 512 |

With FA2: up to 2048 (rank-dependent). Free VRAM &lt;9.5 GiB → cap 768; &lt;10.5 GiB → cap 1024.

## `chronicals` — legacy backend

| Key | Bootstrap | Promote |
|-----|-----------|---------|
| `use_chronicals_gradient_checkpointing` | true | true |
| `use_activation_offload` | false | **true** |
| `use_sequence_packing` | false | false |
| `use_liger_kernel` | false | true |
| `use_lora_plus` | true | true |
| `use_flash_attention` | false | false |
| `torch_compile` | false | false |
| `dataloader_num_workers` | 0 | 8 |

Hard max seq: 768 bootstrap; 1024 promote + offload.

## `gpu_mutex`

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | true | Master switch |
| `min_free_vram_mib` | 8000 | Target free VRAM |
| `min_competitor_mib` | 250 | Ignore tiny GPU users |
| `stop_ollama` | true | `ollama stop` before train |
| `stop_systemd_units` | `[hyprwhspr.service]` | user units |
| `reclaim_unknown_hogs` | false | Kill unknown large PIDs |
| `restore_hyprwhspr` | true | Restart after train |

CLI: `--no-gpu-mutex`, `--gpu-reclaim-warn-only`, `--gpu-reclaim-conservative`

## `training_mix`

| Key | Default |
|-----|---------|
| `prioritize_personal` | true |
| `personal_ratio` | 0.80 |
| `public_cap` | null |
| `personal_sample_weight` | 1.0 |
| `public_sample_weight` | 0.35 |
| `public_dataset_priority` | ordered list — see yaml |

## `curation`

| Key | Default |
|-----|---------|
| `filter_secrets_and_pii` | true |
| `filter_topics_or_refusals` | **false** (not implemented) |
| `bootstrap_mode` | true |
| `skip_roles` | `[developer, system]` |
| `min_messages` | 2 |
| `min_message_chars` | 40 |
| `min_total_chars` | 200 |
| `max_chars_per_message` | 16000 |
| `max_messages_per_example` | 24 |
| `chunk_overlap_messages` | 4 |

## `data`

| Key | Default | Enforced in code? |
|-----|---------|-----------------|
| `min_tier1_smoke` | 200 | **No** — operator threshold |
| `min_tier1_promote` | 500 | **No** — PLAN intent |
| `personal_ratio_mature` | 0.80 | fallback for mix |

## `warehouse`

```yaml
warehouse:
  path: data/warehouse/control_plane.db
  driver: turso   # or sqlite
  sync:
    enabled: false
```

## `rag`

```yaml
rag:
  chroma_path: data/chroma_db
  collection: allowlist_v1
  allowlist: config/doc_allowlist.yaml
  top_k: 8
  force_index_context7: false
```

## `ollama`

```yaml
ollama:
  host: http://127.0.0.1:11434
  inference_model: qwen2.5-coder:7b
  embed_model: qwen3-embedding:4b
  embed_model_fallback: nomic-embed-text
```

## `eval`

```yaml
eval:
  style_win_rate_delta: 0.05
  regression_floor: 0.05
```

Used by PLAN/loop design; not yet wired in `run_eval.py`.

## `public_datasets`

Per-dataset `enabled`, `max_rows`, loader extras. See [PUBLIC-DATASETS.md](PUBLIC-DATASETS.md).

## Known yaml ↔ code drift

Verify before tuning:

1. `unsloth.auto_pack_with_fa2` — yaml `false`, code default `true` if key absent
2. `unsloth.eval_holdout_ratio` — yaml 0.10, code fallback 0.05
3. `data.min_tier1_*` — documented only, not enforced in preflight
4. `disable_auto_padding_free: true` — must set false manually to unlock padding-free with FA2

## Related

- [USER-GUIDE.md](USER-GUIDE.md) — commands using these settings
- [ARCHITECTURE.md](ARCHITECTURE.md) — how config flows into train
