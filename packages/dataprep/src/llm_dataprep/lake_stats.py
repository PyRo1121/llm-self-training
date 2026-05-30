"""Streaming stats over data/raw and data/curated — never loads full files into memory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm_core import data_dir


def _stream_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                yield line_no, None


def stats_raw(raw_dir: Path) -> dict[str, Any]:
    by_harness: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted(raw_dir.glob("*.jsonl")):
        if path.name.startswith("safety-failures"):
            continue
        fcount = 0
        for _line_no, rec in _stream_jsonl(path):
            if not rec:
                continue
            fcount += 1
            total += 1
            by_harness[rec.get("harness") or rec.get("source") or "?"] += 1
            by_source[rec.get("source") or "?"] += 1
        files.append({"path": str(path), "rows": fcount})
    return {
        "total_rows": total,
        "by_harness": dict(by_harness.most_common(30)),
        "by_source": dict(by_source.most_common(10)),
        "files": files,
    }


def stats_curated(curated_path: Path) -> dict[str, Any]:
    by_harness: Counter[str] = Counter()
    by_tier: Counter[int] = Counter()
    by_data_source: Counter[str] = Counter()
    by_public: Counter[str] = Counter()
    char_total = 0
    total = 0
    for _line_no, row in _stream_jsonl(curated_path):
        if not row:
            continue
        total += 1
        meta = row.get("meta") or {}
        by_tier[int(meta.get("train_tier", 0))] += 1
        ds = meta.get("data_source") or (
            "public" if str(meta.get("harness", "")).startswith("public_") else "personal"
        )
        by_data_source[ds] += 1
        by_harness[str(meta.get("harness") or "?")] += 1
        if meta.get("public_dataset"):
            by_public[str(meta["public_dataset"])] += 1
        msgs = row.get("messages") or []
        char_total += sum(len(m.get("content") or "") for m in msgs if isinstance(m, dict))
    return {
        "path": str(curated_path),
        "total_examples": total,
        "tier_counts": {str(k): v for k, v in sorted(by_tier.items())},
        "by_data_source": dict(by_data_source.most_common()),
        "by_harness": dict(by_harness.most_common(25)),
        "by_public_dataset": dict(by_public.most_common()),
        "total_chars": char_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming lake statistics (stdout JSON)")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--curated", type=Path, default=None)
    parser.add_argument("--latest-curated", action="store_true")
    args = parser.parse_args()

    out: dict[str, Any] = {}
    raw_dir = args.raw_dir or (data_dir() / "raw")
    if raw_dir.is_dir():
        out["raw"] = stats_raw(raw_dir)

    curated = args.curated
    if args.latest_curated and not curated:
        candidates = sorted((data_dir() / "curated").glob("curated*.jsonl"))
        curated = candidates[-1] if candidates else None
    if curated and curated.is_file():
        out["curated"] = stats_curated(curated)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
