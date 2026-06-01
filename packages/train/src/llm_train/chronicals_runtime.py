"""Chronicals + TRL training runtime (4070 Ti QLoRA profile)."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

# TRL collator truncates at SFTConfig.max_length; allow full encode without transformers warnings.
TOKENIZER_ENCODE_HEADROOM = 1_000_000

# Before torch import when this module loads first from train_qlora
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def apply_chronicals_env(chronicals: dict[str, Any]) -> None:
    """Set process env flags recommended by Chronicals / TRL on 12 GB GPUs."""
    if not chronicals.get("torch_compile", False):
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    triton_cache = chronicals.get("triton_cache_dir")
    if triton_cache:
        os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))

    # Liger still reads config.use_return_dict (deprecated property → logger noise).
    logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=".*_check_is_size will be removed",
        category=FutureWarning,
    )


def _4070_ti_defaults(chronicals: dict[str, Any]) -> dict[str, Any]:
    """Conservative 12 GB defaults; yaml overrides win."""
    base = {
        "use_liger_kernel": False,
        "use_lora_plus": True,
        "lora_plus_lr_ratio": 16.0,
        "use_sequence_packing": False,
        "use_chronicals_gradient_checkpointing": False,
        "use_activation_offload": False,
        "packing_strategy": "bfd",
        "use_8bit_optimizer": True,
        "torch_compile": False,
        "use_gradient_checkpointing": True,
        "use_flash_attention": False,
        "loss_type": "nll",
        "gpu_peak_tflops": 330.0,
    }
    base.update(chronicals)
    return base


def _flash_attn_available() -> bool:
    try:
        from chronicals.kernels.flash_attention_optimizer import FLASH_ATTN_AVAILABLE

        return bool(FLASH_ATTN_AVAILABLE)
    except ImportError:
        return False


def _resolve_attn_implementation(chronicals: dict[str, Any]) -> str:
    c = _4070_ti_defaults(chronicals)
    if c.get("use_flash_attention") and _flash_attn_available():
        return "flash_attention_2"
    return "sdpa"


def _unwrap_hf_causal_lm(model: Any) -> Any:
    """Inner transformers causal LM (handles PeftModel / LoraModel wrappers)."""
    from peft import PeftModel

    if isinstance(model, PeftModel):
        return model.base_model.model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def ensure_transformers_config(
    model: Any, model_id: str, chronicals: dict[str, Any]
) -> None:
    """TRL reads model.config._attn_implementation (not Chronicals FlashAttentionConfig)."""
    from transformers import AutoConfig

    inner = _unwrap_hf_causal_lm(model)
    cfg = inner.config
    if not hasattr(cfg, "_attn_implementation") or type(cfg).__name__ == "FlashAttentionConfig":
        hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        inner.config = hf_cfg
        cfg = hf_cfg
    cfg._attn_implementation = _resolve_attn_implementation(chronicals)
    if hasattr(cfg, "return_dict"):
        cfg.return_dict = True


def relax_tokenizer_encode_limit(tokenizer: Any) -> None:
    """Let TRL encode full chats; SFTConfig.max_length + collator truncate."""
    tokenizer.model_max_length = TOKENIZER_ENCODE_HEADROOM


def apply_kernel_patches(model: Any, chronicals: dict[str, Any]) -> tuple[Any, bool]:
    """Liger / Chronicals FA patches on the **base** HF model (never on PeftModel)."""
    cfg = _4070_ti_defaults(chronicals)
    liger_applied = False

    if cfg.get("use_liger_kernel"):
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2

            apply_liger_kernel_to_qwen2(
                model=None,
                rope=True,
                rms_norm=True,
                swiglu=True,
                cross_entropy=False,
                fused_linear_cross_entropy=False,
            )
            liger_applied = True
            print("Chronicals: Liger Qwen2 kernels applied on base model", flush=True)
        except ImportError:
            print(
                "Warning: liger-kernel not installed; skipping fused kernels",
                flush=True,
            )
        except Exception as exc:
            print(f"Warning: Liger patch failed ({exc}); continuing", flush=True)

    if cfg.get("use_flash_attention"):
        if not _flash_attn_available():
            print(
                "Warning: use_flash_attention=true but flash-attn not installed; "
                "using sdpa (BFD packing disabled in build_sft_config)",
                flush=True,
            )
        else:
            try:
                from chronicals.kernels.flash_attention_optimizer import (
                    optimize_model_for_speed,
                )

                result = optimize_model_for_speed(
                    model,
                    enable_varlen=bool(cfg.get("use_sequence_packing")),
                    enable_fp8=False,
                    enable_compile=bool(cfg.get("torch_compile")),
                    enable_ring_attention=False,
                    verbose=False,
                )
                model = result.model
                print("Chronicals: FlashAttention speed patch applied", flush=True)
            except Exception as exc:
                print(f"Warning: FlashAttention patch skipped ({exc})", flush=True)

    return model, liger_applied


def load_train_tokenizer(model_id: str, *, max_seq: int) -> Any:
    """Tokenizer only — for dataset filtering before GPU model load."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq
    return tokenizer


def load_qlora_model(
    cfg: dict[str, Any],
    chronicals: dict[str, Any],
    max_seq: int,
    *,
    tokenizer: Any | None = None,
) -> tuple[Any, Any, Any, bool]:
    """Load 4-bit base model + LoRA config. PEFT wrap happens inside SFTTrainer."""
    import torch
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from llm_core.gpu_mutex import load_gpu_mutex_settings, reclaim_gpu_before_load

    gpu_cfg = load_gpu_mutex_settings()
    if not reclaim_gpu_before_load(settings=gpu_cfg):
        free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        raise RuntimeError(
            f"Need ~8 GiB free VRAM for QLoRA load; only {free_gb:.1f} GiB free. "
            "Stop hyprwhspr/Ollama or other GPU apps and retry."
        )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_id = cfg["base_model"]
    attn_impl = _resolve_attn_implementation(chronicals)
    print(
        f"Loading {model_id} (attn={attn_impl}; HF cache may resume ~15 GB download)...",
        flush=True,
    )
    if tokenizer is None:
        tokenizer = load_train_tokenizer(model_id, max_seq=max_seq)
    relax_tokenizer_encode_limit(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False
    ensure_transformers_config(model, model_id, chronicals)

    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gb <= 12.5 and not chronicals.get("use_chronicals_gradient_checkpointing"):
        chronicals = {
            **chronicals,
            "use_chronicals_gradient_checkpointing": True,
        }
        print(
            f"12GB GPU ({total_gb:.1f} GiB): enabling Chronicals gradient checkpointing",
            flush=True,
        )

    if chronicals.get("use_gradient_checkpointing", True):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    if chronicals.get("use_chronicals_gradient_checkpointing"):
        from chronicals.training.gradient_checkpointing import (
            apply_gradient_checkpointing,
        )

        applied = False
        gc_targets: list[tuple[Any, str]] = []
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            gc_targets.append((model.model, "layers"))
        gc_targets.append((model, "layers"))
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            gc_targets.append((model.base_model.model, "layers"))
        for target, layers_attr in gc_targets:
            try:
                apply_gradient_checkpointing(
                    target,
                    layers_attr=layers_attr,
                    use_activation_offloading=bool(
                        chronicals.get("use_activation_offload")
                    ),
                )
                print(
                    f"Chronicals: gradient checkpointing on {type(target).__name__}.{layers_attr}",
                    flush=True,
                )
                applied = True
                break
            except (ValueError, AttributeError):
                continue
        if not applied:
            print(
                "Warning: Chronicals gradient checkpointing skipped (no layer stack found)",
                flush=True,
            )

    model, liger_applied = apply_kernel_patches(model, chronicals)
    ensure_transformers_config(model, model_id, chronicals)

    peft_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    )
    return model, tokenizer, peft_cfg, liger_applied


def build_sft_config(
    *,
    cfg: dict[str, Any],
    chronicals: dict[str, Any],
    out_dir: str,
    max_seq: int,
    batch_size: int,
    grad_accum: int,
    max_steps: int,
    warmup_ratio: float,
    liger_applied: bool = False,
) -> Any:
    from trl import SFTConfig

    c = _4070_ti_defaults(chronicals)
    use_liger = bool(c.get("use_liger_kernel")) and liger_applied
    loss_type = str(c.get("loss_type", "nll"))
    if use_liger and loss_type == "chunked_nll":
        loss_type = "nll"

    use_packing = bool(c.get("use_sequence_packing", True))
    if use_packing and not _flash_attn_available():
        print(
            "Warning: BFD packing needs flash-attn; using unpacked batches",
            flush=True,
        )
        use_packing = False

    optim = "paged_adamw_8bit" if c.get("use_8bit_optimizer", True) else "adamw_torch"
    num_workers = int(c.get("dataloader_num_workers", 0))
    pin_memory = bool(c.get("dataloader_pin_memory", True))
    prefetch = int(c.get("dataloader_prefetch_factor", 2))
    warmup_steps = max(int(max_steps * warmup_ratio), 1)

    kwargs: dict[str, Any] = dict(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        learning_rate=cfg["learning_rate"],
        warmup_steps=warmup_steps,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(max_steps // 2, 1),
        bf16=True,
        seed=cfg["seed"],
        dataset_num_proc=1,
        max_length=max_seq,
        # keep_end preserves assistant tail on long prompts; pre-filter mirrors collator slice.
        truncation_mode="keep_end",
        assistant_only_loss=True,
        packing=use_packing,
        packing_strategy=str(c.get("packing_strategy", "bfd")),
        gradient_checkpointing=bool(c.get("use_gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        optim=optim,
        use_liger_kernel=use_liger,
        loss_type=loss_type,
    )
    if num_workers > 0:
        kwargs["dataloader_prefetch_factor"] = prefetch
    if c.get("torch_compile"):
        kwargs["torch_compile"] = True
        kwargs["torch_compile_mode"] = str(c.get("torch_compile_mode", "reduce-overhead"))
    if cfg.get("neftune_noise_alpha") is not None:
        kwargs["neftune_noise_alpha"] = float(cfg["neftune_noise_alpha"])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*keep_end.*truncation mode is deprecated",
            category=FutureWarning,
        )
        return SFTConfig(**kwargs)


def create_lora_plus_optimizer(model, cfg: dict[str, Any], chronicals: dict[str, Any]):
    """LoRA+ optimizer from Chronicals (B matrices get higher LR)."""
    c = _4070_ti_defaults(chronicals)
    if not c.get("use_lora_plus", True):
        return None

    try:
        from chronicals.optimizers.lora_plus_optimizer import (
            create_lora_plus_optimizer as _chronicals_lora_plus,
        )

        import torch

        ratio = float(c.get("lora_plus_lr_ratio", 16.0))
        optim_cls = torch.optim.AdamW
        if c.get("use_8bit_optimizer"):
            try:
                from bitsandbytes.optim import AdamW8bit

                optim_cls = AdamW8bit
            except ImportError:
                pass
        return _chronicals_lora_plus(
            model,
            base_lr=float(cfg["learning_rate"]),
            lr_ratio=ratio,
            betas=(0.9, 0.95),
            optimizer_class=optim_cls,
        )
    except ImportError:
        from llm_train.lora_plus import create_lora_plus_optimizer as _fallback

        return _fallback(
            model,
            cfg,
            lr_ratio=float(c.get("lora_plus_lr_ratio", 16.0)),
            use_8bit=bool(c.get("use_8bit_optimizer", True)),
        )
