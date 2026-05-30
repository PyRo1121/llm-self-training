"""Ingest Roo Code task JSON (Cline-compatible VS Code globalStorage layout)."""

from __future__ import annotations

from pathlib import Path

from llm_dataprep import cline_tasks

ROO_EXT = "rooveterinaryinc.roo-cline"


def ingest(
    *,
    out_dir: Path | None = None,
    limit_tasks: int | None = None,
) -> tuple[Path | None, int]:
    return cline_tasks.ingest(
        out_dir=out_dir,
        limit_tasks=limit_tasks,
        extension_id=ROO_EXT,
        harness_id="roo_code",
        source="roo_code",
        include_cli_data=False,
    )
