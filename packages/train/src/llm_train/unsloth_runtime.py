"""Unsloth + TRL SFT runtime — world-class 4070 Ti QLoRA profile.

Tuning basis (May 2026):
- https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
- Effective batch 16 (batch=2 × grad_accum=8) on 12 GB when VRAM allows
- use_gradient_checkpointing=\"unsloth\", lora_dropout=0, use_rslora=True
- assistant_only_loss=True; packing=False (Unsloth auto padding-free)
- max_grad_norm=1.0 (NOT 0.3 — promote Chronicals path diverged with LoRA+)
"""

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


def ensure_unsloth_imported() -> None:
    """Import Unsloth before trl/transformers (required for kernel patches)."""
    import unsloth  # noqa: F401


def _resolve_target_modules(unsloth: dict[str, Any]) -> Any:
    """Unsloth accepts 'all-linear' or explicit module list."""
    raw = unsloth.get("target_modules", "all-linear")
    if raw == "all-linear" or raw is None:
        return "all-linear"
    if isinstance(raw, str):
        return [m.strip() for m in raw.split(",") if m.strip()]
    return list(raw)


def apply_unsloth_env(unsloth: dict[str, Any]) -> None:
    """Process env for stable 12 GB Unsloth runs (avoid step-0 compile VRAM spike)."""
    ensure_unsloth_imported()
    if unsloth.get("disable_torch_compile", True):
        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def resolve_unsloth_model_id(cfg: dict[str, Any], unsloth: dict[str, Any]) -> str:
    """Prefer Unsloth pre-quantized checkpoint for faster load when configured."""
    base = str(cfg["base_model"])
    if cfg.get("decensor") or "abliterated" in base.lower():
        return base
    if unsloth.get("use_prequantized", True):
        pre = unsloth.get("prequantized_model")
        if pre:
            return str(pre)
    return base


def load_unsloth_model(
    cfg: dict[str, Any],
    unsloth: dict[str, Any],
    max_seq: int,
    *,
    reclaim_vram: bool = True,
) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    if reclaim_vram:
        from llm_core.gpu_mutex import reclaim_gpu_before_load

        if not reclaim_gpu_before_load():
            print(
                "Warning: VRAM still tight before Unsloth load — stop hyprwhspr/Ollama",
                flush=True,
            )

    model_id = resolve_unsloth_model_id(cfg, unsloth)
    print(f"Loading {model_id} via Unsloth (max_seq={max_seq})…", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq,
        load_in_4bit=bool(unsloth.get("load_in_4bit", True)),
        dtype=None,
    )

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
    return model, tokenizer


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
) -> Any:
    from trl import SFTConfig

    warmup_steps = max(int(max_steps * warmup_ratio), 1)
    max_grad_norm = float(unsloth.get("max_grad_norm", 1.0))
    optim = str(unsloth.get("optim", "adamw_8bit"))
    scheduler = str(unsloth.get("lr_scheduler_type", "linear"))

    kwargs: dict[str, Any] = dict(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        learning_rate=float(cfg["learning_rate"]),
        warmup_steps=warmup_steps,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=scheduler,
        max_grad_norm=max_grad_norm,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(max_steps // 2, 1),
        save_total_limit=2,
        bf16=True,
        seed=int(cfg["seed"]),
        dataset_num_proc=int(unsloth.get("dataset_num_proc", 1)),
        max_length=max_seq,
        truncation_mode="keep_end",
        assistant_only_loss=True,
        packing=bool(unsloth.get("packing", False)),
        gradient_checkpointing=False,
        optim=optim,
    )
    if cfg.get("neftune_noise_alpha") is not None:
        kwargs["neftune_noise_alpha"] = float(cfg["neftune_noise_alpha"])

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


def apply_train_on_responses_only(trainer: Any, unsloth: dict[str, Any]) -> Any:
    """Optional Unsloth chat masking (Qwen2.5 ChatML). Usually redundant with assistant_only_loss."""
    if not unsloth.get("use_train_on_responses_only", False):
        return trainer
    from unsloth.chat_templates import train_on_responses_only

    markers = {**QWEN25_CHAT_MARKERS, **(unsloth.get("chat_markers") or {})}
    return train_on_responses_only(
        trainer,
        instruction_part=markers["instruction_part"],
        response_part=markers["response_part"],
    )
