"""Load train.* settings from config/default.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_core import data_dir, repo_root, runs_dir
from llm_core.yaml_config import load_yaml_config
def _merge_profile(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    if not overrides:
        return base
    return {**base, **overrides}


def decensor_settings() -> dict[str, Any]:
    """Paths and model ids for inform-don't-refuse / abliterated-base track."""
    doc = load_yaml_config()
    d = doc.get("train", {}).get("decensor") or {}
    root = repo_root()
    modelfile = d.get("modelfile", "config/modelfiles/pyro-coder-inform.modelfile")
    inform = d.get("inform_slice", "data/train/inform-dont-refuse.jsonl")
    return {
        "base_model": d.get(
            "base_model", "huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated"
        ),
        "upstream_aligned": d.get(
            "upstream_aligned", "Qwen/Qwen2.5-Coder-7B-Instruct"
        ),
        "ollama_reference": d.get(
            "ollama_reference", "huihui_ai/qwen2.5-coder-abliterate:7b"
        ),
        "modelfile": root / modelfile if not str(modelfile).startswith("/") else Path(modelfile),
        "inform_slice": root / inform if not str(inform).startswith("/") else Path(inform),
    }


def default_train_backend() -> str:
    doc = load_yaml_config()
    return str((doc.get("train") or {}).get("backend", "unsloth")).lower()


def unsloth_settings(*, promote: bool = False, decensor: bool = False) -> dict[str, Any]:
    doc = load_yaml_config()
    u = doc.get("unsloth") or {}
    cfg: dict[str, Any] = {
        "use_prequantized": bool(u.get("use_prequantized", True)),
        "prequantized_model": u.get(
            "prequantized_model", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
        ),
        "load_in_4bit": bool(u.get("load_in_4bit", True)),
        "use_gradient_checkpointing": u.get("use_gradient_checkpointing", "unsloth"),
        "use_rslora": bool(u.get("use_rslora", True)),
        "disable_torch_compile": bool(u.get("disable_torch_compile", True)),
        "packing": bool(u.get("packing", False)),
        "packing_strategy": str(u.get("packing_strategy", "bfd")),
        "auto_pack_with_fa2": bool(u.get("auto_pack_with_fa2", True)),
        "auto_padding_free": bool(u.get("auto_padding_free", True)),
        "padding_free": bool(u.get("padding_free", False)),
        "disable_auto_padding_free": bool(u.get("disable_auto_padding_free", True)),
        "pack_by_data_source": bool(u.get("pack_by_data_source", True)),
        "pack_stratify_column": str(u.get("pack_stratify_column", "_data_source")),
        "stratified_eval_holdout": bool(u.get("stratified_eval_holdout", True)),
        "use_lora_plus": bool(u.get("use_lora_plus", True)),
        "lora_plus_lr_ratio": float(u.get("lora_plus_lr_ratio", 16.0)),
        "activation_offloading": bool(u.get("activation_offloading", False)),
        "allow_activation_offload_without_fa": bool(
            u.get("allow_activation_offload_without_fa", False)
        ),
        "use_liger_kernel": bool(u.get("use_liger_kernel", False)),
        "unsloth_tiled_mlp": bool(u.get("unsloth_tiled_mlp", False)),
        "tiled_mlp_min_seq": int(u.get("tiled_mlp_min_seq", 1536)),
        "dataloader_num_workers": int(u.get("dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(u.get("dataloader_pin_memory", True)),
        "dataloader_prefetch_factor": u.get("dataloader_prefetch_factor"),
        "eval_step_divisor": int(u.get("eval_step_divisor", 6)),
        "vram_probe_max_seq": int(u.get("vram_probe_max_seq", 2048)),
        "token_audit_headroom_ratio": float(u.get("token_audit_headroom_ratio", 1.05)),
        "optim": str(u.get("optim", "adamw_8bit")),
        "max_grad_norm": float(u.get("max_grad_norm", 1.0)),
        "lr_scheduler_type": str(u.get("lr_scheduler_type", "linear")),
        "dataset_num_proc": int(u.get("dataset_num_proc", 1)),
        "use_train_on_responses_only": bool(u.get("use_train_on_responses_only", False)),
        "max_seq_12gb_cap": int(u.get("max_seq_12gb_cap", 2048)),
        "max_seq_12gb_no_fa": int(u.get("max_seq_12gb_no_fa", 768)),
        "max_seq_12gb_no_fa_r16": int(u.get("max_seq_12gb_no_fa_r16", 1024)),
        "max_seq_12gb_no_fa_r32": int(u.get("max_seq_12gb_no_fa_r32", 768)),
        "max_seq_12gb_no_fa_r32_reclaimed": int(
            u.get("max_seq_12gb_no_fa_r32_reclaimed", 1024)
        ),
        "no_fa_stretch_min_free_gb": float(u.get("no_fa_stretch_min_free_gb", 10.8)),
        "max_seq_12gb_no_fa_r64": int(u.get("max_seq_12gb_no_fa_r64", 512)),
        "max_seq_12gb_with_fa": int(u.get("max_seq_12gb_with_fa", 2048)),
        "max_seq_12gb_with_fa_r16": int(u.get("max_seq_12gb_with_fa_r16", 2048)),
        "max_seq_12gb_with_fa_r32": int(u.get("max_seq_12gb_with_fa_r32", 2048)),
        "max_seq_12gb_with_fa_r64": int(u.get("max_seq_12gb_with_fa_r64", 1536)),
        "max_seq_tight_vram": int(u.get("max_seq_tight_vram", 768)),
        "max_seq_moderate_vram": int(u.get("max_seq_moderate_vram", 1024)),
        "max_seq_tight_vram_fa": int(u.get("max_seq_tight_vram_fa", 1536)),
        "max_seq_moderate_vram_fa": int(u.get("max_seq_moderate_vram_fa", 2048)),
        "vram_probe_batch_size": int(u.get("vram_probe_batch_size", 2)),
        "effective_batch_target": int(u.get("effective_batch_target", 16)),
        "token_audit": bool(u.get("token_audit", True)),
        "token_audit_percentile": float(u.get("token_audit_percentile", 95)),
        "token_audit_round_to": int(u.get("token_audit_round_to", 256)),
        "token_audit_min_seq": int(u.get("token_audit_min_seq", 512)),
        "clamp_to_one_epoch": bool(u.get("clamp_to_one_epoch", True)),
        "eval_holdout_ratio": float(u.get("eval_holdout_ratio", 0.05)),
        "eval_steps": int(u.get("eval_steps", 50)),
        "load_best_model_at_end": bool(u.get("load_best_model_at_end", True)),
        "step0_headroom_mib": int(u.get("step0_headroom_mib", 1200)),
        "loss_type": u.get("loss_type"),
        "target_modules": u.get("target_modules", "all-linear"),
        "chat_markers": u.get("chat_markers"),
    }
    if promote or decensor:
        cfg = _merge_profile(cfg, u.get("promote") or {})
    return cfg


def train_settings(*, promote: bool = False, decensor: bool = False) -> dict[str, Any]:
    doc = load_yaml_config()
    train = doc.get("train") or {}
    mix = doc.get("training_mix") or {}
    cfg: dict[str, Any] = {
        "base_model": train.get(
            "base_model", "Qwen/Qwen2.5-Coder-7B-Instruct"
        ),
        "max_seq_length": int(train.get("max_seq_length", 768)),
        "max_seq_length_12gb_cap": int(train.get("max_seq_length_12gb_cap", 768)),
        "max_chars_per_message": train.get("max_chars_per_message"),
        "lora_r": int(train.get("lora_r_bootstrap", 16)),
        "lora_alpha": int(train.get("lora_alpha", 32)),
        "lora_dropout": float(train.get("lora_dropout", 0.0)),
        "learning_rate": float(train.get("learning_rate", 2e-4)),
        "per_device_train_batch_size": int(train.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(
            train.get("gradient_accumulation_steps", 8)
        ),
        "num_epochs": float(train.get("num_epochs", 1)),
        "max_steps_cap": train.get("max_steps_cap", 150),
        "warmup_ratio": float(train.get("warmup_ratio", 0.05)),
        "weight_decay": float(train.get("weight_decay", 0.0)),
        "neftune_noise_alpha": train.get("neftune_noise_alpha"),
        "seed": int(train.get("seed", 3407)),
        "personal_sample_weight": float(mix.get("personal_sample_weight", 1.0)),
        "public_sample_weight": float(mix.get("public_sample_weight", 0.35)),
    }
    if promote:
        cfg = _merge_profile(cfg, train.get("promote") or {})
        if "lora_r" not in (train.get("promote") or {}):
            cfg["lora_r"] = int(train.get("lora_r_promote", 32))
        if "lora_alpha" not in (train.get("promote") or {}):
            cfg["lora_alpha"] = int(train.get("lora_alpha_promote", 64))
    if decensor:
        cfg = train_settings(promote=True)
        dec = decensor_settings()
        cfg["base_model"] = dec["base_model"]
        cfg["decensor"] = True
    return cfg


def chronicals_settings(*, promote: bool = False, decensor: bool = False) -> dict[str, Any]:
    doc = load_yaml_config()
    c = doc.get("chronicals") or {}
    cfg: dict[str, Any] = {
        "use_liger_kernel": bool(c.get("use_liger_kernel", False)),
        "use_lora_plus": bool(c.get("use_lora_plus", True)),
        "lora_plus_lr_ratio": float(c.get("lora_plus_lr_ratio", 16.0)),
        "use_sequence_packing": bool(c.get("use_sequence_packing", False)),
        "packing_strategy": str(c.get("packing_strategy", "bfd")),
        "use_8bit_optimizer": bool(c.get("use_8bit_optimizer", True)),
        "torch_compile": bool(c.get("torch_compile", False)),
        "torch_compile_mode": str(c.get("torch_compile_mode", "reduce-overhead")),
        "use_gradient_checkpointing": bool(c.get("use_gradient_checkpointing", True)),
        "use_flash_attention": bool(c.get("use_flash_attention", False)),
        "use_chronicals_gradient_checkpointing": bool(
            c.get("use_chronicals_gradient_checkpointing", True)
        ),
        "use_activation_offload": bool(c.get("use_activation_offload", False)),
        "loss_type": str(c.get("loss_type", "nll")),
        "gpu_peak_tflops": float(c.get("gpu_peak_tflops", 330.0)),
        "dataloader_num_workers": int(c.get("dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(c.get("dataloader_pin_memory", True)),
        "dataloader_prefetch_factor": int(c.get("dataloader_prefetch_factor", 2)),
    }
    if promote or decensor:
        cfg = _merge_profile(cfg, c.get("promote") or {})
    return cfg


def default_train_file() -> Path:
    path = data_dir() / "train" / "personal-first.jsonl"
    return path


def default_output_dir(run_name: str | None = None) -> Path:
    from datetime import datetime, timezone

    stamp = run_name or datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
    out = runs_dir() / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def adapters_dir() -> Path:
    d = repo_root() / "adapters"
    d.mkdir(parents=True, exist_ok=True)
    return d


def exports_dir() -> Path:
    d = repo_root() / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d
