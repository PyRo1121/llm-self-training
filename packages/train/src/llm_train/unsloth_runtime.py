"""Unsloth + TRL SFT runtime — world-class 4070 Ti QLoRA profile."""

from __future__ import annotations

import os
import warnings
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

QWEN25_CHAT_MARKERS = {
    "instruction_part": "<|im_start|>user\n",
    "response_part": "<|im_start|>assistant\n",
}

TOKENIZER_ENCODE_HEADROOM = 1_000_000


def ensure_unsloth_imported() -> None:
    import unsloth  # noqa: F401


def resolve_unsloth_runtime_flags(unsloth: dict[str, Any]) -> dict[str, Any]:
    """Derive packing / padding_free / FA2 flags from config + installed deps."""
    from llm_train.flash_attn import flash_attn_available

    fa = flash_attn_available()
    # FA2 + padding-free keeps weighted sampler; BFD pack disables per-row weights.
    prefer_padding_free = fa and bool(unsloth.get("auto_padding_free", True))
    if bool(unsloth.get("disable_auto_padding_free", True)):
        prefer_padding_free = False
    manual_pack = bool(unsloth.get("packing", False))
    auto_pack = bool(unsloth.get("auto_pack_with_fa2", True)) and fa and not prefer_padding_free
    use_packing = manual_pack or auto_pack
    if use_packing:
        prefer_padding_free = False
    use_padding_free = fa and prefer_padding_free and not use_packing
    return {
        "flash_attn": fa,
        "use_packing": use_packing,
        "use_padding_free": use_padding_free,
        "disable_auto_padding_free": not use_padding_free,
        "use_liger_kernel": bool(unsloth.get("use_liger_kernel", False)),
    }


def apply_unsloth_env(unsloth: dict[str, Any]) -> None:
    from llm_train.quiet import apply_train_quiet

    apply_train_quiet(before_unsloth=False)
    flags = resolve_unsloth_runtime_flags(unsloth)
    if flags["disable_auto_padding_free"]:
        os.environ.setdefault("UNSLOTH_DISABLE_AUTO_PADDING_FREE", "1")
    else:
        os.environ.pop("UNSLOTH_DISABLE_AUTO_PADDING_FREE", None)
    ensure_unsloth_imported()
    if unsloth.get("disable_torch_compile", True):
        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def _resolve_target_modules(unsloth: dict[str, Any]) -> list[str]:
    raw = unsloth.get("target_modules", "all-linear")
    if raw in ("all-linear", "all_linear", None):
        return list(DEFAULT_TARGET_MODULES)
    if isinstance(raw, str):
        parts = [m.strip() for m in raw.split(",") if m.strip()]
        return parts or list(DEFAULT_TARGET_MODULES)
    return list(raw)


def resolve_unsloth_model_id(cfg: dict[str, Any], unsloth: dict[str, Any]) -> str:
    base = str(cfg["base_model"])
    if cfg.get("decensor") or "abliterated" in base.lower():
        return base
    if unsloth.get("use_prequantized", True):
        pre = unsloth.get("prequantized_model")
        if pre:
            return str(pre)
    return base


def load_unsloth_tokenizer(cfg: dict[str, Any], unsloth: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    model_id = resolve_unsloth_model_id(cfg, unsloth)
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def load_unsloth_model(
    cfg: dict[str, Any],
    unsloth: dict[str, Any],
    max_seq: int,
    *,
    reclaim_vram: bool = True,
) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    from llm_train.flash_attn import set_model_flash_attn

    if reclaim_vram:
        from llm_core.gpu_mutex import reclaim_gpu_before_load

        if not reclaim_gpu_before_load():
            print(
                "Warning: VRAM still tight before Unsloth load — stop hyprwhspr/Ollama",
                flush=True,
            )

    model_id = resolve_unsloth_model_id(cfg, unsloth)
    print(f"Loading {model_id} via Unsloth (max_seq={max_seq})…", flush=True)

    load_kwargs: dict[str, Any] = dict(
        model_name=model_id,
        max_seq_length=max_seq,
        load_in_4bit=bool(unsloth.get("load_in_4bit", True)),
        dtype=None,
    )
    tiled_min = int(unsloth.get("tiled_mlp_min_seq", 1536))
    if bool(unsloth.get("unsloth_tiled_mlp", False)) and max_seq >= tiled_min:
        load_kwargs["unsloth_tiled_mlp"] = True
        print(f"Unsloth: tiled MLP enabled (seq={max_seq} >= {tiled_min})", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    gc = unsloth.get("use_gradient_checkpointing", "unsloth")
    target = _resolve_target_modules(unsloth)
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        bias="none",
        use_gradient_checkpointing=gc,
        random_state=int(cfg["seed"]),
        use_rslora=bool(unsloth.get("use_rslora", True)),
        target_modules=target,
    )
    if set_model_flash_attn(model):
        print("Unsloth: flash_attention_2 enabled on model config", flush=True)
    return model, tokenizer


def split_train_eval_dataset(
    dataset: Any,
    *,
    holdout_ratio: float,
    seed: int,
    stratify_col: str | None = "_data_source",
) -> tuple[Any, Any | None]:
    if holdout_ratio <= 0 or len(dataset) < 20:
        return dataset, None

    if not stratify_col or stratify_col not in dataset.column_names:
        split = dataset.train_test_split(test_size=holdout_ratio, seed=seed)
        return split["train"], split["test"]

    from datasets import concatenate_datasets

    sources = sorted({row[stratify_col] for row in dataset})
    train_parts: list[Any] = []
    eval_parts: list[Any] = []
    for src in sources:
        sub = dataset.filter(lambda row, s=src: row.get(stratify_col) == s)
        n = len(sub)
        if n < 8:
            train_parts.append(sub)
            continue
        split = sub.train_test_split(test_size=holdout_ratio, seed=seed)
        train_parts.append(split["train"])
        eval_parts.append(split["test"])

    train_out = (
        concatenate_datasets(train_parts)
        if len(train_parts) > 1
        else train_parts[0]
    )
    if not eval_parts:
        return train_out, None
    eval_out = (
        concatenate_datasets(eval_parts) if len(eval_parts) > 1 else eval_parts[0]
    )
    print(
        f"Stratified holdout by {stratify_col}: {len(train_out)} train / {len(eval_out)} eval",
        flush=True,
    )
    return train_out, eval_out


def prepare_unsloth_messages_dataset(
    dataset: Any,
    tokenizer: Any,
    *,
    max_seq: int,
    num_proc: int = 1,
) -> Any:
    from datasets import Dataset
    from trl.chat_template_utils import get_training_chat_template

    if not isinstance(dataset, Dataset):
        raise TypeError("prepare_unsloth_messages_dataset expects a HuggingFace Dataset")

    training_template = get_training_chat_template(tokenizer)
    saved_max = int(getattr(tokenizer, "model_max_length", max_seq))
    meta_cols = [c for c in ("_sample_weight", "_data_source") if c in dataset.column_names]

    def _tokenize_row(example: dict[str, Any]) -> dict[str, Any]:
        messages = example.get("messages") or []
        try:
            tokenizer.model_max_length = TOKENIZER_ENCODE_HEADROOM
            processed = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                chat_template=training_template,
                return_assistant_tokens_mask=True,
            )
        finally:
            tokenizer.model_max_length = saved_max

        ids = list(processed.get("input_ids") or [])
        masks = list(processed.get("assistant_masks") or [])
        if len(ids) > max_seq:
            ids = ids[-max_seq:]
            masks = masks[-max_seq:]
        if not ids or 1 not in masks:
            return {"input_ids": [], "assistant_masks": []}
        out: dict[str, Any] = {"input_ids": ids, "assistant_masks": masks}
        for col in meta_cols:
            out[col] = example[col]
        return out

    mapped = dataset.map(
        _tokenize_row,
        desc=f"Unsloth: tokenize messages (keep_end @ {max_seq})",
        num_proc=num_proc if num_proc > 1 else None,
    )
    keep = [i for i in range(len(mapped)) if mapped[i]["input_ids"]]
    if not keep:
        raise ValueError(
            f"No examples with assistant tokens after keep_end truncate @ max_seq={max_seq}"
        )
    dropped = len(mapped) - len(keep)
    if dropped:
        print(
            f"Dropped {dropped} examples (no assistant tokens after keep_end @ {max_seq})",
            flush=True,
        )
    out = mapped.select(keep)
    drop_cols = [
        c for c in out.column_names if c not in {"input_ids", "assistant_masks", *meta_cols}
    ]
    if drop_cols:
        out = out.remove_columns(drop_cols)
    return out


def _pack_one_split(
    dataset: Any,
    *,
    max_seq: int,
    strategy: str,
    desc: str,
) -> Any:
    from trl.data_utils import pack_dataset

    cols = [c for c in ("input_ids", "assistant_masks") if c in dataset.column_names]
    return pack_dataset(
        dataset.select_columns(cols),
        seq_length=max_seq,
        strategy=strategy,
        map_kwargs={"desc": desc, "num_proc": 1},
    )


def maybe_pack_unsloth_dataset(
    dataset: Any,
    unsloth: dict[str, Any],
    max_seq: int,
) -> tuple[Any, bool]:
    """BFD-pack pre-tokenized rows when FA2 is available (VRAM / throughput)."""
    flags = resolve_unsloth_runtime_flags(unsloth)
    if flags["use_padding_free"]:
        print("Unsloth: padding-free (FA2) — skipping BFD pre-pack", flush=True)
        return dataset, False
    if not flags["use_packing"]:
        return dataset, False

    from datasets import concatenate_datasets

    strategy = str(unsloth.get("packing_strategy", "bfd"))
    stratify = str(unsloth.get("pack_stratify_column", "_data_source"))
    print(f"Unsloth: BFD packing @ {max_seq} (FA2, strategy={strategy})…", flush=True)

    if (
        bool(unsloth.get("pack_by_data_source", True))
        and stratify in dataset.column_names
    ):
        sources = sorted({row[stratify] for row in dataset})
        packed_parts = []
        for src in sources:
            sub = dataset.filter(lambda row, s=src: row.get(stratify) == s)
            if len(sub) == 0:
                continue
            packed_parts.append(
                _pack_one_split(
                    sub,
                    max_seq=max_seq,
                    strategy=strategy,
                    desc=f"Unsloth: pack {src} ({strategy})",
                )
            )
        packed = (
            concatenate_datasets(packed_parts)
            if len(packed_parts) > 1
            else packed_parts[0]
        )
        print(
            f"Packed by {stratify} ({len(sources)} strata): "
            f"{len(dataset)} → {len(packed)} sequences",
            flush=True,
        )
    else:
        packed = _pack_one_split(
            dataset,
            max_seq=max_seq,
            strategy=strategy,
            desc=f"Unsloth: pack ({strategy})",
        )
        print(f"Packed {len(dataset)} → {len(packed)} sequences", flush=True)

    if "_sample_weight" in dataset.column_names:
        print(
            "Note: per-row sample weights off after BFD pack (use auto_padding_free to keep weights)",
            flush=True,
        )
    return packed, True


def compute_eval_steps(unsloth: dict[str, Any], max_steps: int) -> int | None:
    if max_steps <= 0:
        return None
    divisor = max(int(unsloth.get("eval_step_divisor", 6)), 2)
    cap = int(unsloth.get("eval_steps", 50))
    return max(10, min(cap, max(max_steps // divisor, 10)))


def build_unsloth_sft_config(
    *,
    cfg: dict[str, Any],
    unsloth: dict[str, Any],
    out_dir: str,
    max_seq: int,
    batch_size: int,
    grad_accum: int,
    max_steps: int,
    warmup_ratio: float,
    eval_steps: int | None = None,
    has_eval: bool = False,
) -> Any:
    from trl import SFTConfig

    warmup_steps = max(int(max_steps * warmup_ratio), 1)
    max_grad_norm = float(unsloth.get("max_grad_norm", 1.0))
    optim = str(unsloth.get("optim", "adamw_8bit"))
    scheduler = str(unsloth.get("lr_scheduler_type", "linear"))
    flags = resolve_unsloth_runtime_flags(unsloth)

    use_activation_offload = bool(unsloth.get("activation_offloading", False))
    if use_activation_offload and not flags["flash_attn"]:
        use_activation_offload = bool(unsloth.get("allow_activation_offload_without_fa", False))
        if use_activation_offload:
            print(
                "Unsloth: activation offload without FA2 (CPU RAM tradeoff — install flash-attn for 2048)",
                flush=True,
            )

    dataloader_workers = int(unsloth.get("dataloader_num_workers", 0))
    if not flags["flash_attn"]:
        # Forked workers after CUDA init risk VRAM spikes on 12GB no-FA2.
        dataloader_workers = 0

    save_steps = max(max_steps // 4, 25)
    if has_eval and eval_steps:
        save_steps = eval_steps

    kwargs: dict[str, Any] = dict(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        learning_rate=float(cfg["learning_rate"]),
        warmup_steps=warmup_steps,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=scheduler,
        max_grad_norm=max_grad_norm,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,
        seed=int(cfg["seed"]),
        dataset_num_proc=int(unsloth.get("dataset_num_proc", 1)),
        max_length=max_seq,
        truncation_mode="keep_end",
        # Pre-tokenized input_ids + assistant_masks (unsloth-zoo#323); collator masks labels.
        assistant_only_loss=False,
        completion_only_loss=False,
        packing=False,
        padding_free=flags["use_padding_free"],
        gradient_checkpointing=False,
        optim=optim,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=bool(unsloth.get("dataloader_pin_memory", True)),
    )
    pf = unsloth.get("dataloader_prefetch_factor")
    if pf is not None and dataloader_workers > 0:
        kwargs["dataloader_prefetch_factor"] = int(pf)

    if use_activation_offload:
        kwargs["activation_offloading"] = True

    if cfg.get("neftune_noise_alpha") is not None:
        kwargs["neftune_noise_alpha"] = float(cfg["neftune_noise_alpha"])
    if unsloth.get("loss_type"):
        kwargs["loss_type"] = str(unsloth["loss_type"])

    if has_eval and eval_steps:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = eval_steps
        if unsloth.get("load_best_model_at_end", True):
            kwargs["load_best_model_at_end"] = True
            kwargs["metric_for_best_model"] = "eval_loss"
            kwargs["greater_is_better"] = False

    if flags["use_liger_kernel"] and flags["flash_attn"]:
        kwargs["use_liger_kernel"] = True

    if flags["use_packing"]:
        kwargs["packing"] = False  # manual pre-pack in maybe_pack_unsloth_dataset

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*keep_end.*truncation mode is deprecated",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*warmup_ratio is deprecated",
            category=FutureWarning,
        )
        return SFTConfig(**kwargs)


def attach_unsloth_lora_plus(trainer: Any, cfg: dict[str, Any], unsloth: dict[str, Any]) -> Any:
    if not unsloth.get("use_lora_plus", True):
        return trainer
    from llm_train.lora_plus import create_lora_plus_optimizer

    opt = create_lora_plus_optimizer(
        trainer.model,
        cfg,
        lr_ratio=float(unsloth.get("lora_plus_lr_ratio", 16.0)),
        use_8bit=str(unsloth.get("optim", "adamw_8bit")).endswith("8bit"),
    )
    if opt is not None:
        trainer.optimizer = opt
        print(
            f"LoRA+ optimizer (B-matrix LR ×{unsloth.get('lora_plus_lr_ratio', 16.0)})",
            flush=True,
        )
    return trainer


def apply_train_on_responses_only(trainer: Any, unsloth: dict[str, Any]) -> Any:
    if not unsloth.get("use_train_on_responses_only", False):
        return trainer
    from unsloth.chat_templates import train_on_responses_only

    markers = {**QWEN25_CHAT_MARKERS, **(unsloth.get("chat_markers") or {})}
    return train_on_responses_only(
        trainer,
        instruction_part=markers["instruction_part"],
        response_part=markers["response_part"],
    )
