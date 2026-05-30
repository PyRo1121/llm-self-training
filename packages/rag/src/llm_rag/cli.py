"""CLI entrypoints: rag-index."""

from __future__ import annotations

import argparse
import json
import sys

from llm_rag.index import run_index as _run_index


def run_index(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index doc allowlist into Chroma")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate collection")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Only index these source ids (repeatable)",
    )
    args = parser.parse_args(argv)
    result = _run_index(reset=args.reset, source_ids=args.sources)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
