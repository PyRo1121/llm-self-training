"""VRAM plan unit tests (no GPU)."""

from __future__ import annotations

from llm_train.config import train_settings, unsloth_settings
from llm_train.vram_budget import (
    plan_unsloth_training,
    resolve_vram_train_params,
    unsloth_vram_seq_ceiling,
)


def test_no_fa_r32_ceiling_768():
    cfg = train_settings(promote=True)
    u = unsloth_settings(promote=True)
    ceiling, _ = unsloth_vram_seq_ceiling(cfg, u, free_gb=10.4, total_gb=11.6)
    assert ceiling == 768


def test_no_fa_r16_ceiling_1024():
    cfg = train_settings(promote=False)
    u = unsloth_settings(promote=False)
    ceiling, _ = unsloth_vram_seq_ceiling(cfg, u, free_gb=10.4, total_gb=11.6)
    assert ceiling == 1024


def test_tight_vram_lowers_ceiling():
    cfg = train_settings(promote=False)
    u = unsloth_settings(promote=False)
    ceiling, note = unsloth_vram_seq_ceiling(cfg, u, free_gb=9.0, total_gb=11.6)
    assert ceiling == 768
    assert "tight" in note


def test_effective_batch_16_on_promote():
    cfg = train_settings(promote=True)
    u = unsloth_settings(promote=True)
    plan = plan_unsloth_training(
        cfg, u, smoke=False, free_gb=10.4, total_gb=11.6, token_recommended_seq=768
    )
    assert plan.batch_size * plan.grad_accum == 16
    assert plan.max_seq == 768


def test_fa2_r32_ceiling_2048(monkeypatch):
    cfg = train_settings(promote=True)
    u = unsloth_settings(promote=True)
    monkeypatch.setattr(
        "llm_train.vram_budget.flash_attn_available",
        lambda: True,
    )
    ceiling, note = unsloth_vram_seq_ceiling(cfg, u, free_gb=10.4, total_gb=11.6)
    assert ceiling == 2048
    assert "FA2" in note


def test_fa2_moderate_free_still_2048(monkeypatch):
    cfg = train_settings(promote=True)
    u = unsloth_settings(promote=True)
    monkeypatch.setattr(
        "llm_train.vram_budget.flash_attn_available",
        lambda: True,
    )
    ceiling, _ = unsloth_vram_seq_ceiling(cfg, u, free_gb=9.2, total_gb=11.6)
    assert ceiling == 2048


def test_no_fa_reclaimed_stretch_1024():
    cfg = train_settings(promote=True)
    u = unsloth_settings(promote=True)
    ceiling, note = unsloth_vram_seq_ceiling(cfg, u, free_gb=10.9, total_gb=11.6)
    assert ceiling == 1024
    assert "stretch" in note


def test_token_audit_lowers_final_seq():
    cfg = train_settings(promote=True)
    u = unsloth_settings(promote=True)
    ceiling, _ = unsloth_vram_seq_ceiling(cfg, u, free_gb=10.4, total_gb=11.6)
    max_seq, batch, grad, _ = resolve_vram_train_params(
        cfg,
        smoke=False,
        free_gb=10.4,
        total_gb=11.6,
        backend="unsloth",
        unsloth=u,
        token_recommended_seq=512,
        vram_ceiling=ceiling,
    )
    assert max_seq == 512
    assert batch == 1
    assert grad == 16


def test_h100_ceiling_8192(monkeypatch):
    monkeypatch.setenv("LLM_CONFIG_PROFILE", "cloud-h100")
    cfg = train_settings(promote=True)
    u = unsloth_settings(promote=True)
    monkeypatch.setattr(
        "llm_train.vram_budget.flash_attn_available",
        lambda: True,
    )
    ceiling, note = unsloth_vram_seq_ceiling(cfg, u, free_gb=70.0, total_gb=80.0)
    assert ceiling == 8192
    assert "H100" in note


def test_resolve_max_steps_null_cap():
    from llm_train.train_qlora import _resolve_max_steps

    assert _resolve_max_steps(
        args_max_steps=None,
        steps_per_epoch=10000,
        epochs=1.0,
        max_steps_cap=None,
        clamp_one_epoch=True,
        smoke=False,
    ) == 10000
