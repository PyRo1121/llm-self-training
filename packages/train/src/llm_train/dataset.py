"""Load curated/train JSONL for Chronicals/TRL (streaming stats + HF Dataset)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset


def stream_train_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("messages"):
                yield obj


def train_file_stats(path: Path) -> dict[str, Any]:
    personal = 0
    public = 0
    total = 0
    weight_sum = 0.0
    for row in stream_train_jsonl(path):
        total += 1
        meta = row.get("meta") or {}
        ds = meta.get("data_source") or (
            "public" if str(meta.get("harness", "")).startswith("public_") else "personal"
        )
        if ds == "public":
            public += 1
        else:
            personal += 1
        weight_sum += float(meta.get("sample_weight", 1.0))
    return {
        "path": str(path),
        "total": total,
        "personal": personal,
        "public": public,
        "personal_ratio": round(personal / total, 4) if total else 0.0,
        "mean_sample_weight": round(weight_sum / total, 4) if total else 0.0,
    }


def _cap_message_chars(convo: list[dict[str, str]], *, max_chars: int) -> list[dict[str, str]]:
    if max_chars <= 0:
        return convo
    capped: list[dict[str, str]] = []
    for msg in convo:
        content = msg.get("content") or ""
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...[truncated for VRAM]"
        capped.append({**msg, "content": content})
    return capped


def load_messages_dataset(
    path: Path,
    *,
    max_examples: int | None = None,
    max_chars_per_message: int | None = None,
) -> tuple[Dataset, list[float]]:
    rows: list[dict[str, Any]] = []
    weights: list[float] = []
    for row in stream_train_jsonl(path):
        if max_examples is not None and len(rows) >= max_examples:
            break
        messages = row.get("messages") or []
        convo = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in ("user", "assistant", "system") and content:
                convo.append({"role": role, "content": content})
        if len(convo) < 2:
            continue
        if not any(m.get("role") == "assistant" for m in convo):
            continue
        if max_chars_per_message:
            convo = _cap_message_chars(convo, max_chars=max_chars_per_message)
        meta = row.get("meta") or {}
        ds = meta.get("data_source") or (
            "public" if str(meta.get("harness", "")).startswith("public_") else "personal"
        )
        w = float(meta.get("sample_weight", 1.0))
        w = max(w, 0.05)
        rows.append(
            {
                "messages": convo,
                "_sample_weight": w,
                "_data_source": ds,
            }
        )
        weights.append(w)
    if not rows:
        raise ValueError(f"No valid examples in {path}")
    return Dataset.from_list(rows), weights


def sample_weights_from_dataset(dataset: Dataset, cfg: dict[str, Any]) -> list[float]:
    """Per-row weights after any filter — reads _sample_weight / _data_source columns."""
    personal_w = float(cfg.get("personal_sample_weight", 1.0))
    public_w = float(cfg.get("public_sample_weight", 0.25))
    cols = dataset.column_names if hasattr(dataset, "column_names") else []
    out: list[float] = []
    for i in range(len(dataset)):
        row = dataset[i]
        if "_sample_weight" in cols:
            out.append(max(float(row["_sample_weight"]), 0.05))
        elif "_data_source" in cols:
            out.append(public_w if row["_data_source"] == "public" else personal_w)
        else:
            out.append(personal_w)
    return out
