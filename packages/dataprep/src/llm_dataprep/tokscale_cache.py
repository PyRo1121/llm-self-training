"""Ingest tokscale sync caches (Antigravity JSONL, Trae JSON)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records


def _role_text(obj: dict[str, Any]) -> tuple[str | None, str]:
    role = obj.get("role") or obj.get("type")
    if role in ("human", "Human"):
        role = "user"
    if role in ("model", "Model", "ai"):
        role = "assistant"
    if role not in ("user", "assistant"):
        return None, ""
    content = obj.get("content") or obj.get("text") or obj.get("message")
    if isinstance(content, str):
        return role, content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
        return role, "\n".join(str(p) for p in parts).strip()
    if isinstance(content, dict):
        return role, (content.get("text") or content.get("content") or "")[:200_000]
    return role, ""


def _records_from_jsonl(
    files: list[Path],
    *,
    harness: str,
    source: str,
) -> Iterator[dict[str, Any]]:
    ingested = datetime.now(timezone.utc).isoformat()
    for fpath in files:
        session_id = fpath.stem
        with fpath.open(encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                role, text = _role_text(obj)
                if not role or not text or len(text) > 200_000:
                    continue
                yield {
                    "source": source,
                    "harness": harness,
                    "session_id": session_id,
                    "source_path": str(fpath),
                    "line_no": i,
                    "role": role,
                    "text": text,
                    "ingested_at": ingested,
                }


def _records_from_json_files(
    files: list[Path],
    *,
    harness: str,
    source: str,
) -> Iterator[dict[str, Any]]:
    ingested = datetime.now(timezone.utc).isoformat()
    for fpath in files:
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        session_id = fpath.stem
        messages: list[Any] = []
        if isinstance(data, list):
            messages = data
        elif isinstance(data, dict):
            messages = data.get("messages") or data.get("turns") or []
        for i, obj in enumerate(messages, start=1):
            if not isinstance(obj, dict):
                continue
            role, text = _role_text(obj)
            if not role or not text or len(text) > 200_000:
                continue
            yield {
                "source": source,
                "harness": harness,
                "session_id": session_id,
                "source_path": str(fpath),
                "line_no": i,
                "role": role,
                "text": text,
                "ingested_at": ingested,
            }


def ingest_antigravity(
    cache_root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    root = cache_root or Path.home() / ".config/tokscale/antigravity-cache/sessions"
    files = sorted(root.glob("*.jsonl")) if root.is_dir() else []
    if limit_files:
        files = files[:limit_files]
    if not files:
        return None, 0
    return append_records(
        "antigravity-tokscale",
        _records_from_jsonl(files, harness="antigravity", source="antigravity"),
        out_dir=out_dir,
    )


def ingest_trae(
    cache_root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    root = cache_root or Path.home() / ".config/tokscale/trae-cache/sessions"
    files = sorted(root.glob("*.json")) if root.is_dir() else []
    if limit_files:
        files = files[:limit_files]
    if not files:
        return None, 0
    return append_records(
        "trae-tokscale",
        _records_from_json_files(files, harness="trae", source="trae"),
        out_dir=out_dir,
    )
