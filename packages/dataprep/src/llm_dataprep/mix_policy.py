"""Training mix policy: always keep personal rows; cap public by ratio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_core.yaml_config import load_yaml_config


@dataclass(frozen=True)
class MixPolicy:
    prioritize_personal: bool = True
    personal_ratio: float = 1.0
    public_cap: int | None = None
    public_dataset_priority: tuple[str, ...] = (
        "swe_chat",
        "zen_agentic",
    )
    personal_sample_weight: float = 1.0
    public_sample_weight: float = 0.25


def load_mix_policy() -> MixPolicy:
    doc = load_yaml_config()
    raw = doc.get("training_mix") or {}
    pr = float(raw.get("personal_ratio", doc.get("data", {}).get("personal_ratio_mature", 1.0)))
    cap = raw.get("public_cap")
    return MixPolicy(
        prioritize_personal=bool(raw.get("prioritize_personal", True)),
        personal_ratio=pr,
        public_cap=int(cap) if cap is not None else None,
        public_dataset_priority=tuple(raw.get("public_dataset_priority") or MixPolicy().public_dataset_priority),
        personal_sample_weight=float(raw.get("personal_sample_weight", 1.0)),
        public_sample_weight=float(raw.get("public_sample_weight", 0.25)),
    )


def _public_sort_key(row: Any, priority: tuple[str, ...]) -> tuple[int, str]:
    ds = row["public_dataset"] or ""
    if ds in priority:
        return (priority.index(ds), row["curated_id"])
    return (len(priority), row["curated_id"])


def apply_mix(
    personal: list[Any],
    public: list[Any],
    policy: MixPolicy,
    *,
    exclude_public: bool = False,
    limit: int | None = None,
) -> list[tuple[Any, float]]:
    """Return (row, sample_weight) with personal rows first."""
    if exclude_public or not public:
        out = [(r, policy.personal_sample_weight) for r in personal]
    elif not policy.prioritize_personal:
        combined = personal + public
        out = [
            (r, policy.personal_sample_weight if r["data_source"] == "personal" else policy.public_sample_weight)
            for r in combined
        ]
    else:
        pub_sorted = sorted(
            public,
            key=lambda r: _public_sort_key(r, policy.public_dataset_priority),
        )
        p_count = len(personal)
        if p_count == 0:
            max_pub = policy.public_cap or len(pub_sorted)
        elif policy.personal_ratio >= 0.999 or policy.personal_ratio <= 0:
            max_pub = 0
        else:
            max_pub = int(p_count * (1.0 - policy.personal_ratio) / policy.personal_ratio)
            if policy.public_cap is not None:
                max_pub = min(max_pub, policy.public_cap)
        out = [(r, policy.personal_sample_weight) for r in personal]
        out.extend((r, policy.public_sample_weight) for r in pub_sorted[:max_pub])

    if limit is not None and len(out) > limit:
        # Never drop personal below limit: cap public only
        if policy.prioritize_personal and personal:
            personal_part = [(r, w) for r, w in out if r["data_source"] == "personal"]
            public_part = [(r, w) for r, w in out if r["data_source"] != "personal"]
            slots = max(0, limit - len(personal_part))
            out = personal_part + public_part[:slots]
        else:
            out = out[:limit]
    return out
