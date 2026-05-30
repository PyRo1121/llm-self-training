"""Ingest Codex rollout JSONL from ~/.codex/sessions (May 2026 format)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

DEFAULT_ROOT = Path.home() / ".codex/sessions"
SKIP_TYPES = frozenset(
    {
        "session_meta",
        "turn_context",
        "event_msg",
        "token_count",
        "compacted",
    }
)
USER_BLOCK_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)


def _text_from_codex_content(blocks: list[Any]) -> str:
    """Codex rollouts: user uses input_text; assistant uses output_text (May 2026)."""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind not in ("input_text", "output_text"):
            continue
        t = block.get("text") or ""
        m = USER_BLOCK_RE.search(t)
        parts.append(m.group(1).strip() if m else t)
    return "\n".join(p for p in parts if p).strip()


def iter_rollout_files(root: Path, *, max_file_mb: float | None = 50.0) -> Iterator[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("rollout-*.jsonl")):
        if max_file_mb and path.stat().st_size > max_file_mb * 1024 * 1024:
            continue
        yield path


def _records_from_file(path: Path, *, limit_lines: int | None = None) -> Iterator[dict[str, Any]]:
    session_id = path.stem.removeprefix("rollout-")[:36] if "rollout-" in path.name else path.stem
    ingested = datetime.now(timezone.utc).isoformat()
    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, start=1):
            if limit_lines and i > limit_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            top = obj.get("type")
            if top in SKIP_TYPES:
                continue
            if top != "response_item":
                continue
            payload = obj.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            if payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in ("user", "developer", "assistant"):
                continue
            content = payload.get("content")
            if not isinstance(content, list):
                continue
            text = _text_from_codex_content(content)
            if not text or len(text) > 200_000:
                continue
            yield {
                "source": "codex",
                "harness": "codex",
                "session_id": session_id,
                "source_path": str(path),
                "line_no": i,
                "record_type": obj.get("type"),
                "role": role,
                "text": text,
                "ingested_at": ingested,
            }


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    max_file_mb: float | None = 50.0,
    limit_files: int | None = None,
    limit_lines_per_file: int | None = None,
) -> tuple[Path | None, int]:
    root = root or DEFAULT_ROOT
    files = list(iter_rollout_files(root, max_file_mb=max_file_mb))
    if limit_files:
        files = files[:limit_files]

    def all_records() -> Iterator[dict[str, Any]]:
        for fpath in files:
            yield from _records_from_file(fpath, limit_lines=limit_lines_per_file)

    if not files:
        return None, 0
    return append_records("codex-sessions", all_records(), out_dir=out_dir)
