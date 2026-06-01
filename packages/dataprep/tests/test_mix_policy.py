"""Training mix policy tests."""

from __future__ import annotations

from llm_dataprep.mix_policy import MixPolicy, apply_mix


def _row(source: str, curated_id: str = "x") -> dict[str, str]:
    return {"data_source": source, "curated_id": curated_id, "public_dataset": "swe_chat"}


def test_apply_mix_zero_personal_ratio_excludes_public() -> None:
    """personal_ratio <= 0 must not divide by zero; treat as personal-only."""
    personal = [_row("personal", "p1"), _row("personal", "p2")]
    public = [_row("public", f"u{i}") for i in range(5)]
    policy = MixPolicy(personal_ratio=0.0, prioritize_personal=True)

    out = apply_mix(personal, public, policy)

    assert len(out) == 2
    assert all(r["data_source"] == "personal" for r, _ in out)


def test_apply_mix_negative_personal_ratio_excludes_public() -> None:
    personal = [_row("personal")]
    public = [_row("public")]
    policy = MixPolicy(personal_ratio=-0.5, prioritize_personal=True)

    out = apply_mix(personal, public, policy)

    assert len(out) == 1
    assert out[0][0]["data_source"] == "personal"


def test_apply_mix_ratio_caps_public() -> None:
    personal = [_row("personal", f"p{i}") for i in range(4)]
    public = [_row("public", f"u{i}") for i in range(10)]
    policy = MixPolicy(personal_ratio=0.5, prioritize_personal=True)

    out = apply_mix(personal, public, policy)

    pub_count = sum(1 for r, _ in out if r["data_source"] != "personal")
    assert pub_count == 4
