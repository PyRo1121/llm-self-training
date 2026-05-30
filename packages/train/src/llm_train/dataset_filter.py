"""Drop examples that lose assistant tokens under TRL assistant-only SFT.

See TRL docs:
- https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md (assistant_only_loss)
- https://github.com/huggingface/trl/blob/main/trl/chat_templates/README.md (generation markers)
"""

from __future__ import annotations

import sys
from typing import Any

from datasets import Dataset
from trl.chat_template_utils import get_training_chat_template

from llm_train.chronicals_runtime import TOKENIZER_ENCODE_HEADROOM


def max_chars_for_seq(max_seq: int) -> int:
    """Rough chars/token budget per message (ChatML overhead excluded)."""
    return max(2000, int(max_seq) * 4)


def _tokenize_for_assistant_loss(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_seq: int,
    training_template: str | None,
) -> tuple[list[int], list[int]] | None:
    """Mirror TRL SFTTrainer tokenize_fn + keep_end collator slice."""
    saved_max = int(getattr(tokenizer, "model_max_length", max_seq))
    try:
        # Encode full chat first (TRL tokenize_fn), then keep_end slice in collator.
        # Raise limit so transformers does not warn before we truncate.
        tokenizer.model_max_length = TOKENIZER_ENCODE_HEADROOM
        processed = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            chat_template=training_template,
            return_assistant_tokens_mask=True,
        )
    except Exception:
        return None
    finally:
        tokenizer.model_max_length = saved_max

    ids = processed.get("input_ids") or []
    masks = processed.get("assistant_masks") or []
    if not ids or not masks:
        return None

    # SFTConfig.truncation_mode="keep_end" for assistant_only_loss (TRL SFTTrainer collator).
    if len(ids) > max_seq:
        ids = ids[-max_seq:]
        masks = masks[-max_seq:]

    if 1 not in masks:
        return None
    return ids, masks


def filter_dataset_to_max_tokens(
    dataset: Dataset,
    tokenizer: Any,
    *,
    max_seq: int,
    sample_weights: list[float] | None = None,
) -> tuple[Dataset, list[float] | None]:
    """Keep rows valid for TRL `assistant_only_loss=True` after keep_end truncation."""
    training_template = get_training_chat_template(tokenizer)
    n = len(dataset)
    print(
        f"Assistant-only filter: scanning {n} examples (CPU, ~1 min)…",
        file=sys.stderr,
        flush=True,
    )
    keep: list[int] = []
    for i in range(n):
        if i > 0 and i % 500 == 0:
            print(f"  …{i}/{n}", file=sys.stderr, flush=True)
        messages = dataset[i].get("messages") or []
        if not any(m.get("role") == "assistant" for m in messages):
            continue
        if _tokenize_for_assistant_loss(
            tokenizer, messages, max_seq=max_seq, training_template=training_template
        ) is None:
            continue
        keep.append(i)

    dropped = len(dataset) - len(keep)
    if dropped:
        print(
            f"Filtered {dropped} examples (no assistant tokens after "
            f"keep_end truncate @ max_seq={max_seq}; TRL training template)",
            file=sys.stderr,
            flush=True,
        )
    out_ds = dataset.select(keep)
    out_w = None
    if sample_weights is not None:
        if len(sample_weights) != len(dataset):
            out_w = sample_weights
        else:
            out_w = [sample_weights[i] for i in keep]
    return out_ds, out_w
