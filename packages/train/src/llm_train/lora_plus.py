"""LoRA+ optimizer — higher LR on LoRA B matrices (ICML 2024). Backend-agnostic."""

from __future__ import annotations

from typing import Any


def create_lora_plus_optimizer(
    model: Any,
    cfg: dict[str, Any],
    *,
    lr_ratio: float = 16.0,
    use_8bit: bool = True,
) -> Any | None:
    """Return AdamW (or 8-bit) with separate LR for lora_B vs other trainable params."""
    import torch

    base_lr = float(cfg["learning_rate"])
    lora_b: list[torch.nn.Parameter] = []
    other: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_B" in name:
            lora_b.append(param)
        else:
            other.append(param)

    if not lora_b:
        return None

    optim_cls = torch.optim.AdamW
    if use_8bit:
        try:
            from bitsandbytes.optim import AdamW8bit

            optim_cls = AdamW8bit
        except ImportError:
            pass

    param_groups = [
        {"params": other, "lr": base_lr},
        {"params": lora_b, "lr": base_lr * lr_ratio},
    ]
    return optim_cls(param_groups, betas=(0.9, 0.95))
