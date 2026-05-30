"""Mux ingest: transcript JSONL if present; otherwise usage metadata only (partial)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records
from llm_dataprep.tokscale_cache import _records_from_jsonl

DEFAULT_ROOT = Path.home() / ".mux/sessions"


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    root = root or DEFAULT_ROOT
    if not root.is_dir():
        return None, 0

    jsonl_files = sorted(root.rglob("*.jsonl"))
    if limit_files:
        jsonl_files = jsonl_files[:limit_files]
    if jsonl_files:
        return append_records(
            "mux-sessions",
            _records_from_jsonl(jsonl_files, harness="mux", source="mux"),
            out_dir=out_dir,
        )

    usage_files = sorted(root.rglob("session-usage.json"))
    if limit_files:
        usage_files = usage_files[:limit_files]
    if not usage_files:
        return None, 0

    ingested = datetime.now(timezone.utc).isoformat()

    def usage_records() -> Iterator[dict[str, Any]]:
        for fpath in usage_files:
            try:
                obj = json.loads(fpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            text = json.dumps(obj, ensure_ascii=False)[:50_000]
            yield {
                "source": "mux",
                "harness": "mux",
                "session_id": fpath.parent.name,
                "source_path": str(fpath),
                "role": "metadata",
                "text": text,
                "ingested_at": ingested,
                "note": "usage-only — no chat transcript path known",
            }

    return append_records("mux-usage", usage_records(), out_dir=out_dir)
