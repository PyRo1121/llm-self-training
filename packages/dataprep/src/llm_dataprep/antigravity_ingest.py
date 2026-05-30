"""Antigravity: tokscale cache (preferred); .pb conversations are encrypted (skipped)."""

from __future__ import annotations

from pathlib import Path

from llm_dataprep import tokscale_cache


def ingest(
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    path, n = tokscale_cache.ingest_antigravity(out_dir=out_dir, limit_files=limit_files)
    if n > 0:
        return path, n

    pb_dir = Path.home() / ".gemini/antigravity/conversations"
    if pb_dir.is_dir() and any(pb_dir.glob("*.pb")):
        # No decrypt path — document via zero records + caller message
        return None, 0
    return None, 0
