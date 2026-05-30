"""Parse Cursor agent-transcripts JSONL into raw training-oriented records.

Format (May 2026): one JSON object per line with `role` + `message.content[]`
blocks (`type`: text | tool_use). Tool outputs are often absent in JSONL;
see PLAN.md dual-ingest with SQLite (AI-Data-Extraction) later.

Refs: cursor-history (S2thend), AgentProbe (vtemian) — web research pass 2.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_core import data_dir, repo_root

DEFAULT_PROJECTS_ROOT = Path.home() / ".cursor/projects"
USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)


@dataclass
class TranscriptRecord:
    session_id: str
    source_path: str
    line_no: int
    role: str
    text: str
    has_tool_use: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "cursor",
            "harness": "cursor",
            "session_id": self.session_id,
            "source_path": self.source_path,
            "line_no": self.line_no,
            "role": self.role,
            "text": self.text,
            "has_tool_use": self.has_tool_use,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }


def _extract_text(content_blocks: list[dict[str, Any]]) -> tuple[str, bool]:
    parts: list[str] = []
    has_tool = False
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            t = block.get("text") or ""
            m = USER_QUERY_RE.search(t)
            parts.append(m.group(1).strip() if m else t)
        elif kind == "tool_use":
            has_tool = True
    return "\n".join(p for p in parts if p).strip(), has_tool


def parse_line(line: str, *, session_id: str, source_path: str, line_no: int) -> TranscriptRecord | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    role = obj.get("role")
    if role not in ("user", "assistant"):
        return None
    message = obj.get("message") or {}
    content = message.get("content")
    if not isinstance(content, list):
        return None
    text, has_tool = _extract_text(content)
    if not text:
        return None
    return TranscriptRecord(
        session_id=session_id,
        source_path=source_path,
        line_no=line_no,
        role=role,
        text=text,
        has_tool_use=has_tool,
    )


def iter_transcript_files(
    projects_root: Path | None = None,
    *,
    include_subagents: bool = False,
) -> Iterator[Path]:
    """Walk ~/.cursor/projects/*/agent-transcripts/**/*.jsonl (parent transcripts only by default)."""
    root = projects_root or (Path.home() / ".cursor/projects")
    if not root.exists():
        return
    for path in sorted(root.rglob("*.jsonl")):
        if "agent-transcripts" not in path.parts:
            continue
        if not include_subagents and "subagents" in path.parts:
            continue
        yield path


def session_id_from_path(path: Path) -> str:
    # .../agent-transcripts/<uuid>/<uuid>.jsonl → uuid
    parent = path.parent.name
    if path.stem == parent:
        return parent
    return path.stem


def ingest(
    projects_root: Path | None = None,
    *,
    out_dir: Path | None = None,
    include_subagents: bool = False,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    from llm_dataprep.raw_io import append_records

    files = list(iter_transcript_files(projects_root, include_subagents=include_subagents))
    if limit_files:
        files = files[:limit_files]
    if not files:
        return None, 0

    def from_files() -> Iterator[dict[str, Any]]:
        for fpath in files:
            sid = session_id_from_path(fpath)
            rel = str(fpath)
            with fpath.open(encoding="utf-8", errors="replace") as src:
                for i, line in enumerate(src, start=1):
                    rec = parse_line(line, session_id=sid, source_path=rel, line_no=i)
                    if rec is not None:
                        yield rec.to_dict()

    return append_records("cursor-transcripts", from_files(), out_dir=out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Cursor agent-transcripts JSONL")
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=DEFAULT_PROJECTS_ROOT,
        help="Cursor projects root (default: ~/.cursor/projects)",
    )
    parser.add_argument("--include-subagents", action="store_true")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    path, n = ingest(
        args.projects_root,
        out_dir=args.out_dir,
        include_subagents=args.include_subagents,
        limit_files=args.limit_files,
    )
    dest = path or args.out_dir or data_dir() / "raw"
    print(f"Wrote {n} records → {dest} (repo={repo_root()})")


if __name__ == "__main__":
    main()
