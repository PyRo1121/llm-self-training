"""Token audit unit tests (no GPU / tokenizer)."""

from __future__ import annotations

from llm_train.token_audit import (
    effective_audit_cap,
    recommend_seq_from_lengths,
)


def test_effective_cap_respects_vram_ceiling():
    assert effective_audit_cap(yaml_cap=2048, vram_ceiling=768) == 768


def test_long_tail_does_not_recommend_yaml_cap():
    lengths = [4000] * 50 + [15000] * 50
    rec = recommend_seq_from_lengths(
        lengths,
        cap_limit=768,
        percentile=95.0,
        round_to=256,
        min_seq=512,
    )
    assert rec == 768


def test_short_data_uses_percentile():
    lengths = [400, 500, 600, 700, 800]
    rec = recommend_seq_from_lengths(
        lengths,
        cap_limit=2048,
        percentile=95.0,
        round_to=256,
        min_seq=512,
    )
    assert rec == 1024
