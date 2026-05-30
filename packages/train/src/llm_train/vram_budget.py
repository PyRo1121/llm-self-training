"""VRAM-aware train hyperparameters for 12 GB cards (4070 Ti class)."""

from __future__ import annotations

from typing import Any

# Default hard ceiling for 12 GiB cards without activation offload.
HARD_MAX_SEQ_12GB = 768
# With Chronicals sqrt GC + CPU activation offload (32GB RAM).
HARD_MAX_SEQ_12GB_OFFLOAD = 1024
# With offload + flash-attn + BFD packing (experimental on 4070 Ti).
HARD_MAX_SEQ_12GB_OFFLOAD_FA = 1536


def _flash_attn_available() -> bool:
    try:
        from chronicals.kernels.flash_attention_optimizer import FLASH_ATTN_AVAILABLE

        return bool(FLASH_ATTN_AVAILABLE)
    except ImportError:
        return False


def _hard_max_seq(chronicals: dict[str, Any] | None) -> int:
    if not chronicals:
        return HARD_MAX_SEQ_12GB
    if chronicals.get("use_activation_offload"):
        if _flash_attn_available() and chronicals.get("use_sequence_packing"):
            return HARD_MAX_SEQ_12GB_OFFLOAD_FA
        return HARD_MAX_SEQ_12GB_OFFLOAD
    return HARD_MAX_SEQ_12GB


def resolve_vram_train_params(
    cfg: dict[str, Any],
    *,
    smoke: bool,
    free_gb: float,
    total_gb: float,
    chronicals: dict[str, Any] | None = None,
    backend: str = "chronicals",
    unsloth: dict[str, Any] | None = None,
) -> tuple[int, int, int, str]:
    """Return (max_seq, batch_size, grad_accum, reason) capped for available VRAM."""
    max_seq = int(cfg["max_seq_length"])
    batch = int(cfg["per_device_train_batch_size"])
    grad_accum = int(cfg["gradient_accumulation_steps"])
    cap_12 = int(cfg.get("max_seq_length_12gb_cap", 768))
    hard_max = (
        int(unsloth.get("max_seq_12gb_cap", 2048))
        if backend == "unsloth" and unsloth
        else _hard_max_seq(chronicals)
    )

    if smoke:
        return min(max_seq, 1024), 1, max(grad_accum, 2), "smoke run"

    reasons: list[str] = []

    if backend == "unsloth":
        max_seq = min(max_seq, cap_12, hard_max)
        grad_accum = max(grad_accum, 8)
        batch = max(int(batch), 1)
        if total_gb <= 12.5:
            reasons.append(
                f"Unsloth 12GB (total={total_gb:.1f} GiB, eff_batch={batch * grad_accum})"
            )
        if free_gb < 9.5:
            max_seq = min(max_seq, 1024, hard_max)
            batch = 1
            grad_accum = max(grad_accum, 12)
            reasons.append(f"Unsloth VRAM tight ({free_gb:.1f} GiB free → seq≤1024 batch=1")
    elif total_gb <= 12.5:
        max_seq = min(max_seq, cap_12, hard_max)
        batch = 1
        grad_accum = max(grad_accum, 8)
        suffix = ""
        if chronicals and chronicals.get("use_activation_offload"):
            suffix = ", activation offload"
        reasons.append(f"12GB profile (total={total_gb:.1f} GiB{suffix})")

    if backend != "unsloth" and free_gb < 10.0:
        max_seq = min(max_seq, cap_12, hard_max)
        batch = 1
        grad_accum = max(grad_accum, 8)
        reasons.append(f"low free VRAM ({free_gb:.1f} GiB)")
    elif free_gb < 11.0 and total_gb > 12.5:
        max_seq = min(max_seq, min(cap_12 + 512, 1536))
        batch = min(batch, 1)
        reasons.append(f"moderate free VRAM ({free_gb:.1f} GiB)")

    reason = "; ".join(reasons) if reasons else "yaml defaults"
    return max_seq, batch, grad_accum, reason


def log_vram_snapshot() -> None:
    import sys

    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return

    from llm_core.gpu_mutex import format_gpu_competitors

    free, total = torch.cuda.mem_get_info()
    print(
        f"GPU VRAM: {free / (1024**3):.1f} GiB free / {total / (1024**3):.1f} GiB total",
        file=sys.stderr,
        flush=True,
    )
    comps = format_gpu_competitors()
    if comps != "none":
        print(f"GPU compute: {comps}", file=sys.stderr, flush=True)
