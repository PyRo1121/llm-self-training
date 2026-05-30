"""Append JSONL records to data/raw with dated filenames."""

from __future__ import annotations

import json
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
    count = 0
    mode = "w" if replace else "a"
    with path.open(mode, encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return path, count
