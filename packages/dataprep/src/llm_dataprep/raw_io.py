"""Append JSONL records to data/raw with dated filenames."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_core import data_dir


def dated_raw_path(prefix: str, out_dir: Path | None = None) -> Path:
    root = out_dir or (data_dir() / "raw")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return root / f"{prefix}-{stamp}.jsonl"


def append_records(
    prefix: str,
    records: Iterator[dict[str, Any]],
    *,
    out_dir: Path | None = None,
    replace: bool = False,
) -> tuple[Path, int]:
    path = dated_raw_path(prefix, out_dir)
    write_fresh = replace or not path.is_file()
    count = append_records_buffered(
        path, records, buffer_rows=500, replace=write_fresh
    )
    return path, count


def append_records_buffered(
    path: Path,
    records: Iterator[dict[str, Any]],
    *,
    buffer_rows: int = 500,
    replace: bool = True,
) -> int:
    """Write JSONL with batched buffer flushes."""
    mode = "w" if replace else "a"
    count = 0
    buf: list[str] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as fh:
        for rec in records:
            buf.append(json.dumps(rec, ensure_ascii=False))
            count += 1
            if len(buf) >= buffer_rows:
                fh.write("\n".join(buf) + "\n")
                buf.clear()
        if buf:
            fh.write("\n".join(buf) + "\n")
    return count


def merge_jsonl_parts(parts: list[Path], out_path: Path) -> None:
    """Concatenate worker part files into one JSONL (streaming, low RAM)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out_fh:
        for part in parts:
            with part.open("rb") as in_fh:
                shutil.copyfileobj(in_fh, out_fh)
