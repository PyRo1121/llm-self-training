"""Flash Attention 2 availability (Unsloth + Chronicals)."""

from __future__ import annotations


def flash_attn_available() -> bool:
    """True when flash-attn imports or Chronicals reports FA2."""
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        from chronicals.kernels.flash_attention_optimizer import FLASH_ATTN_AVAILABLE

        return bool(FLASH_ATTN_AVAILABLE)
    except ImportError:
        return False


def set_model_flash_attn(model: object) -> bool:
    """Set ``_attn_implementation`` on the inner HF config when FA2 is available."""
    if not flash_attn_available():
        return False
    inner = model
    if hasattr(model, "get_base_model"):
        try:
            inner = model.get_base_model()
        except Exception:
            inner = model
    if hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
        inner = inner.base_model.model
    cfg = getattr(inner, "config", None)
    if cfg is None:
        return False
    cfg._attn_implementation = "flash_attention_2"
    return True
