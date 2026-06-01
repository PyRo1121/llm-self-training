"""Token-length audit for VRAM-aware seq selection (Unsloth / TRL)."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from typing import Any

TOKENIZER_ENCODE_HEADROOM = 1_000_000


@dataclass
class TokenAuditReport:
    samples: int
    p50: int
    p95: int
    p99: int
    max_len: int
    yaml_cap: int
    vram_ceiling: int
    effective_cap: int
    recommended_seq: int
    would_drop_at_recommended: int
    would_drop_assistant_at_cap: int

    def summary(self) -> str:
        return (
            f"token audit n={self.samples}: p50={self.p50} p95={self.p95} "
            f"p99={self.p99} max={self.max_len} → seq={self.recommended_seq} "
            f"(effective cap {self.effective_cap}, yaml {self.yaml_cap}, "
            f"VRAM ceiling {self.vram_ceiling}, ~{self.would_drop_at_recommended} truncated, "
            f"~{self.would_drop_assistant_at_cap} lose assistant @ cap)"
        )


def _round_up(value: int, step: int) -> int:
    if step <= 1:
        return value
    return int(math.ceil(value / step) * step)


def effective_audit_cap(yaml_cap: int, vram_ceiling: int) -> int:
    """Cap token audit by hardware ceiling — not yaml aspiration alone."""
    return max(512, min(int(yaml_cap), int(vram_ceiling)))


def recommend_seq_from_lengths(
    lengths: list[int],
    cap_limit: int,
    *,
    percentile: float = 99.0,
    round_to: int = 256,
    min_seq: int = 512,
    headroom_ratio: float = 1.05,
) -> int:
    """Pick seq from length distribution, never above ``cap_limit``."""
    if not lengths:
        return cap_limit
    sorted_lens = sorted(lengths)
    idx = min(
        len(sorted_lens) - 1,
        max(0, int(math.ceil(percentile / 100.0 * len(sorted_lens))) - 1),
    )
    p = sorted_lens[idx]
    target = int(p * headroom_ratio)
    target = _round_up(target, round_to)
    target = max(min_seq, min(cap_limit, target))
    return target


def _would_lose_assistant_at_cap(length: int, cap: int) -> bool:
    """Rough keep_end proxy: assistant often in last ~40% of coding chats."""
    if length <= cap:
        return False
    kept = cap
    assistant_start_est = int(length * 0.55)
    return kept <= assistant_start_est


def audit_messages_lengths(
    dataset: Any,
    tokenizer: Any,
    *,
    yaml_cap: int,
    vram_ceiling: int,
    percentile: float = 99.0,
    round_to: int = 256,
    min_seq: int = 512,
    headroom_ratio: float = 1.05,
    max_samples: int | None = None,
) -> tuple[TokenAuditReport, list[int]]:
    """Return audit report and per-row token lengths (keep_end not applied)."""
    from trl.chat_template_utils import get_training_chat_template

    effective_cap = effective_audit_cap(yaml_cap, vram_ceiling)
    training_template = get_training_chat_template(tokenizer)
    saved_max = int(getattr(tokenizer, "model_max_length", effective_cap))
    lengths: list[int] = []
    n = len(dataset)
    limit = n if max_samples is None else min(n, max_samples)

    for i in range(limit):
        messages = dataset[i].get("messages") or []
        if not messages:
            continue
        try:
            tokenizer.model_max_length = TOKENIZER_ENCODE_HEADROOM
            processed = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                chat_template=training_template,
                return_assistant_tokens_mask=True,
            )
        except Exception:
            continue
        finally:
            tokenizer.model_max_length = saved_max
        ids = processed.get("input_ids") or []
        masks = processed.get("assistant_masks") or []
        if ids and masks and 1 in masks:
            lengths.append(len(ids))

    if not lengths:
        rec = min(effective_cap, 1024)
        return (
            TokenAuditReport(
                samples=0,
                p50=0,
                p95=0,
                p99=0,
                max_len=0,
                yaml_cap=yaml_cap,
                vram_ceiling=vram_ceiling,
                effective_cap=effective_cap,
                recommended_seq=rec,
                would_drop_at_recommended=0,
                would_drop_assistant_at_cap=0,
            ),
            [],
        )

    sorted_lens = sorted(lengths)

    def _pct(p: float) -> int:
        j = min(len(sorted_lens) - 1, max(0, int(math.ceil(p / 100.0 * len(sorted_lens))) - 1))
        return sorted_lens[j]

    rec = recommend_seq_from_lengths(
        lengths,
        effective_cap,
        percentile=percentile,
        round_to=round_to,
        min_seq=min_seq,
        headroom_ratio=headroom_ratio,
    )
    dropped_est = sum(1 for length in lengths if length > rec)
    assistant_loss_est = sum(
        1 for length in lengths if _would_lose_assistant_at_cap(length, effective_cap)
    )

    report = TokenAuditReport(
        samples=len(lengths),
        p50=_pct(50),
        p95=_pct(95),
        p99=_pct(99),
        max_len=max(sorted_lens),
        yaml_cap=yaml_cap,
        vram_ceiling=vram_ceiling,
        effective_cap=effective_cap,
        recommended_seq=rec,
        would_drop_at_recommended=dropped_est,
        would_drop_assistant_at_cap=assistant_loss_est,
    )
    return report, lengths


def print_token_audit(report: TokenAuditReport, *, stream: Any = sys.stderr) -> None:
    print(report.summary(), file=stream, flush=True)
    print(json.dumps({"token_audit": asdict(report)}, indent=2), flush=True)
