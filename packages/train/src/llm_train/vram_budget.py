"""VRAM-aware train hyperparameters — 4070 Ti 12GB and datacenter (H100+) profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_train.flash_attn import flash_attn_available

# Default hard ceiling for 12 GiB cards without activation offload.
HARD_MAX_SEQ_12GB = 768
# With Chronicals sqrt GC + CPU activation offload (32GB RAM).
HARD_MAX_SEQ_12GB_OFFLOAD = 1024
# With offload + flash-attn + BFD packing (experimental on 4070 Ti).
HARD_MAX_SEQ_12GB_OFFLOAD_FA = 1536
DATACENTER_GPU_MIN_GB = 70.0


def _is_datacenter_gpu(total_gb: float) -> bool:
    return total_gb >= DATACENTER_GPU_MIN_GB


def _rank_base_seq_cap_h100(cfg: dict[str, Any], unsloth: dict[str, Any], *, fa: bool) -> int:
    lora_r = int(cfg.get("lora_r", 16))
    hard = int(unsloth.get("max_seq_h100_cap", 8192))
    if fa:
        if lora_r >= 64:
            return int(unsloth.get("max_seq_h100_with_fa_r64", hard))
        if lora_r >= 32:
            return int(unsloth.get("max_seq_h100_with_fa_r32", hard))
        return int(unsloth.get("max_seq_h100_with_fa_r16", hard))
    return int(unsloth.get("max_seq_h100_no_fa", min(hard, 4096)))


def _rank_base_seq_cap(cfg: dict[str, Any], unsloth: dict[str, Any], *, fa: bool) -> int:
    """Rank-aware seq ceiling from yaml (Unsloth docs: higher r → more VRAM)."""
    lora_r = int(cfg.get("lora_r", 16))
    if fa:
        if lora_r >= 64:
            return int(unsloth.get("max_seq_12gb_with_fa_r64", 1536))
        if lora_r >= 32:
            return int(
                unsloth.get(
                    "max_seq_12gb_with_fa_r32",
                    unsloth.get("max_seq_12gb_with_fa", 2048),
                )
            )
        return int(
            unsloth.get(
                "max_seq_12gb_with_fa_r16",
                unsloth.get("max_seq_12gb_with_fa", 2048),
            )
        )
    if lora_r >= 64:
        return int(unsloth.get("max_seq_12gb_no_fa_r64", 512))
    if lora_r >= 32:
        return int(unsloth.get("max_seq_12gb_no_fa_r32", unsloth.get("max_seq_12gb_no_fa", 768)))
    return int(unsloth.get("max_seq_12gb_no_fa_r16", unsloth.get("max_seq_12gb_no_fa", 1024)))


@dataclass
class UnslothTrainPlan:
    max_seq: int
    batch_size: int
    grad_accum: int
    vram_ceiling: int
    reason: str


def _hard_max_seq(chronicals: dict[str, Any] | None) -> int:
    if not chronicals:
        return HARD_MAX_SEQ_12GB
    if chronicals.get("use_activation_offload"):
        if flash_attn_available() and chronicals.get("use_sequence_packing"):
            return HARD_MAX_SEQ_12GB_OFFLOAD_FA
        return HARD_MAX_SEQ_12GB_OFFLOAD
    return HARD_MAX_SEQ_12GB


def unsloth_vram_seq_ceiling(
    cfg: dict[str, Any],
    unsloth: dict[str, Any] | None,
    *,
    free_gb: float,
    total_gb: float,
) -> tuple[int, str]:
    """Upper seq bound from FA2, LoRA rank, and free VRAM."""
    if not unsloth:
        return HARD_MAX_SEQ_12GB, "no unsloth settings"
    fa = flash_attn_available()
    if _is_datacenter_gpu(total_gb):
        cap = _rank_base_seq_cap_h100(cfg, unsloth, fa=fa)
        hard = int(unsloth.get("max_seq_h100_cap", unsloth.get("max_seq_12gb_cap", cap)))
        cap = min(cap, hard)
        return cap, f"ceiling={cap} (H100 {'FA2' if fa else 'no-FA2'} r={int(cfg.get('lora_r', 16))})"

    cap = _rank_base_seq_cap(cfg, unsloth, fa=fa)
    hard = int(unsloth.get("max_seq_12gb_cap", cap))
    cap = min(cap, hard)
    notes: list[str] = [f"{'FA2' if fa else 'no-FA2'} r={int(cfg.get('lora_r', 16))}"]

    if total_gb <= 12.5:
        if fa:
            # FA2 + padding-free: moderate free VRAM still allows 2048 after gpu_mutex reclaim.
            tight = int(unsloth.get("max_seq_tight_vram_fa", 1536))
            moderate = int(unsloth.get("max_seq_moderate_vram_fa", cap))
            if free_gb < 9.0:
                cap = min(cap, tight)
                notes.append(f"tight free {free_gb:.1f} GiB (FA2)")
            elif free_gb < 9.5:
                cap = min(cap, moderate)
                notes.append(f"moderate free {free_gb:.1f} GiB (FA2)")
        else:
            if free_gb < 9.5:
                cap = min(cap, int(unsloth.get("max_seq_tight_vram", 768)))
                notes.append(f"tight free {free_gb:.1f} GiB")
            elif free_gb < 10.5:
                cap = min(cap, int(unsloth.get("max_seq_moderate_vram", 1024)))
                notes.append(f"moderate free {free_gb:.1f} GiB")
            # Stretch no-FA2 when browser closed + activation offload (still below 2048).
            stretch_free = float(unsloth.get("no_fa_stretch_min_free_gb", 10.8))
            if (
                bool(unsloth.get("allow_activation_offload_without_fa", False))
                and free_gb >= stretch_free
                and int(cfg.get("lora_r", 16)) >= 32
            ):
                stretch = int(unsloth.get("max_seq_12gb_no_fa_r32_reclaimed", 1024))
                cap = min(hard, max(cap, stretch))
                notes.append(f"reclaimed stretch → {cap}")

    return cap, f"ceiling={cap} ({', '.join(notes)})"


def post_load_step_headroom_mib(unsloth: dict[str, Any] | None) -> int:
    """Min free MiB after model load for step-0 fused CE (observed ~594 MiB spike @ seq 1024)."""
    return int((unsloth or {}).get("step0_headroom_mib", 1200))


def post_load_vram_ok(unsloth: dict[str, Any] | None) -> tuple[bool, float]:
    """Return (has_headroom, free_gib) after model is on GPU."""
    import torch

    if not torch.cuda.is_available():
        return True, 0.0
    free, _ = torch.cuda.mem_get_info()
    free_gib = free / (1024**3)
    need_mib = post_load_step_headroom_mib(unsloth)
    return free >= need_mib * (1024**2), free_gib


def downgrade_seq_for_post_load(
    max_seq: int,
    *,
    round_to: int = 256,
    floor: int = 512,
) -> int | None:
    """Next lower seq for post-load OOM retry, or None if already at floor."""
    if max_seq <= floor:
        return None
    next_seq = max(floor, max_seq - round_to)
    return next_seq if next_seq < max_seq else None


def resolve_vram_train_params(
    cfg: dict[str, Any],
    *,
    smoke: bool,
    free_gb: float,
    total_gb: float,
    chronicals: dict[str, Any] | None = None,
    backend: str = "chronicals",
    unsloth: dict[str, Any] | None = None,
    token_recommended_seq: int | None = None,
    vram_ceiling: int | None = None,
) -> tuple[int, int, int, str]:
    """Return (max_seq, batch_size, grad_accum, reason) capped for available VRAM."""
    max_seq = int(cfg["max_seq_length"])
    batch = int(cfg["per_device_train_batch_size"])
    grad_accum = int(cfg["gradient_accumulation_steps"])
    cap_12 = int(cfg.get("max_seq_length_12gb_cap", 768))

    reasons: list[str] = []

    if backend == "unsloth":
        seq_ceiling = vram_ceiling
        if seq_ceiling is None:
            seq_ceiling, ceiling_note = unsloth_vram_seq_ceiling(
                cfg, unsloth, free_gb=free_gb, total_gb=total_gb
            )
            reasons.append(ceiling_note)
        max_seq = min(max_seq, cap_12, seq_ceiling)
        if token_recommended_seq is not None:
            max_seq = min(max_seq, token_recommended_seq)
            reasons.append(f"token audit seq={token_recommended_seq}")
        target_eff = int((unsloth or {}).get("effective_batch_target", 16))
        batch = max(int(batch), 1)
        grad_accum = max(grad_accum, (target_eff + batch - 1) // batch)
        fa = flash_attn_available()
        if fa:
            label = "H100+FA2" if _is_datacenter_gpu(total_gb) else "Unsloth 12GB+FA2"
            reasons.append(f"{label} (eff_batch={batch * grad_accum})")
        else:
            label = "H100 no-FA2" if _is_datacenter_gpu(total_gb) else "Unsloth 12GB no-FA2"
            reasons.append(f"{label} (eff_batch={batch * grad_accum})")
        probe_min_free = 40.0 if _is_datacenter_gpu(total_gb) else 10.0
        if (
            not smoke
            and fa
            and free_gb >= probe_min_free
            and batch <= int(cfg.get("per_device_train_batch_size", 1))
            and max_seq <= int((unsloth or {}).get("vram_probe_max_seq", 2048))
        ):
            trial_batch = int((unsloth or {}).get("vram_probe_batch_size", 2))
            yaml_batch = int(cfg.get("per_device_train_batch_size", trial_batch))
            trial_batch = max(trial_batch, yaml_batch)
            if trial_batch > batch:
                batch = trial_batch
                grad_accum = max((target_eff + batch - 1) // batch, grad_accum)
                reasons.append(f"VRAM headroom → batch={batch}")
        if _is_datacenter_gpu(total_gb) and not smoke and batch < int(
            cfg.get("per_device_train_batch_size", batch)
        ):
            batch = int(cfg.get("per_device_train_batch_size", batch))
            grad_accum = max((target_eff + batch - 1) // batch, grad_accum)
            reasons.append(f"H100 yaml batch={batch}")
        if smoke:
            max_seq = min(max_seq, seq_ceiling)
            batch = 1
            grad_accum = max(grad_accum, 2)
            reasons.append("smoke run")
    elif total_gb <= 12.5:
        hard_max = _hard_max_seq(chronicals)
        max_seq = min(max_seq, cap_12, hard_max)
        batch = 1
        grad_accum = max(grad_accum, 8)
        suffix = ""
        if chronicals and chronicals.get("use_activation_offload"):
            suffix = ", activation offload"
        reasons.append(f"12GB profile (total={total_gb:.1f} GiB{suffix})")

    if backend != "unsloth" and free_gb < 10.0:
        hard_max = _hard_max_seq(chronicals)
        max_seq = min(max_seq, cap_12, hard_max)
        batch = 1
        grad_accum = max(grad_accum, 8)
        reasons.append(f"low free VRAM ({free_gb:.1f} GiB)")
    elif free_gb < 11.0 and total_gb > 12.5 and backend != "unsloth":
        max_seq = min(max_seq, min(cap_12 + 512, 1536))
        batch = min(batch, 1)
        reasons.append(f"moderate free VRAM ({free_gb:.1f} GiB)")

    reason = "; ".join(reasons) if reasons else "yaml defaults"
    return max_seq, batch, grad_accum, reason


def plan_unsloth_training(
    cfg: dict[str, Any],
    unsloth: dict[str, Any],
    *,
    smoke: bool,
    free_gb: float,
    total_gb: float,
    token_recommended_seq: int | None = None,
) -> UnslothTrainPlan:
    """Single entry point for preflight + train-qlora VRAM planning."""
    vram_ceiling, _ = unsloth_vram_seq_ceiling(cfg, unsloth, free_gb=free_gb, total_gb=total_gb)
    max_seq, batch, grad_accum, reason = resolve_vram_train_params(
        cfg,
        smoke=smoke,
        free_gb=free_gb,
        total_gb=total_gb,
        backend="unsloth",
        unsloth=unsloth,
        token_recommended_seq=token_recommended_seq,
        vram_ceiling=vram_ceiling,
    )
    return UnslothTrainPlan(
        max_seq=max_seq,
        batch_size=batch,
        grad_accum=grad_accum,
        vram_ceiling=vram_ceiling,
        reason=reason,
    )


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
