"""Ingest Aider markdown chat histories (.aider.chat.history.md)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

DEFAULT_SCAN_ROOTS = (
    Path.home() / "Documents",
    Path.home(),
)
USER_HEADING = re.compile(r"^####\s+", re.MULTILINE)
TOOL_BLOCK = re.compile(r"^>\s+", re.MULTILINE)


def iter_history_files(scan_roots: tuple[Path, ...] | None = None) -> Iterator[Path]:
    roots = scan_roots or DEFAULT_SCAN_ROOTS
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.name == ".aider.chat.history.md":
            seen.add(root)
            yield root
            continue
        for path in root.rglob(".aider.chat.history.md"):
            if path not in seen:
                seen.add(path)
                yield path
        for path in root.rglob("*.chat.md"):
            if ".aider.history" in path.parts and path not in seen:
                seen.add(path)
                yield path


def _parse_markdown(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (role, text) chunks from Aider history markdown."""
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    # Split on #### user blocks; assistant blocks have no prefix per Aider docs
    parts = re.split(r"\n(?=#### )", body)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("#### "):
            text = USER_HEADING.sub("", part, count=1).strip()
            if text:
                yield "user", text
        elif part.startswith(">"):
            text = TOOL_BLOCK.sub("", part).strip()
            if text:
                yield "tool", text
        else:
            if len(part) > 20:
                yield "assistant", part


def ingest(
    scan_roots: tuple[Path, ...] | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
    include_tool: bool = False,
) -> tuple[Path | None, int]:
    files = list(iter_history_files(scan_roots))
    if limit_files:
        files = files[:limit_files]
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        for fpath in files:
            for role, text in _parse_markdown(fpath):
                if role == "tool" and not include_tool:
                    continue
                if len(text) > 200_000:
                    continue
                yield {
                    "source": "aider",
                    "harness": "aider",
                    "session_id": fpath.stem,
                    "source_path": str(fpath),
                    "role": role,
                    "text": text,
                    "ingested_at": ingested,
                }

    if not files:
        return None, 0
    return append_records("aider-history", records(), out_dir=out_dir)
