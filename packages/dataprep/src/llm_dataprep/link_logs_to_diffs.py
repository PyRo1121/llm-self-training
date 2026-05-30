"""Link curated sessions to git commits for exec/verify labels (Phase 1 v0)."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_core import data_dir

_REPO_RE = re.compile(r"(/home/[^/]+/Documents/[^/]+|/home/[^/]+/[^/]+/[^/]+)")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_git_index(raw_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """repo path -> commits sorted by committed_at."""
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(raw_dir.glob("git-diffs-*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                repo = rec.get("repo") or ""
                if repo:
                    by_repo[repo].append(rec)
    for repo in by_repo:
        by_repo[repo].sort(key=lambda r: r.get("committed_at") or "")
    return dict(by_repo)


def infer_repo_from_source(source_path: str | None) -> str | None:
    if not source_path:
        return None
    m = _REPO_RE.search(source_path.replace("\\", "/"))
    if m:
        return m.group(1)
    return None


def nearest_commits(
    commits: list[dict[str, Any]],
    *,
    window_hours: int = 72,
) -> list[str]:
    """v0: return recent commit hashes (no session timestamp in raw yet)."""
    if not commits:
        return []
    return [c["session_id"] for c in commits[-5:] if c.get("session_id")]


def enrich_curated_row(
    row: dict[str, Any],
    git_index: dict[str, list[dict[str, Any]]],
    *,
    mark_exec_on_link: bool,
) -> dict[str, Any]:
    meta = row.setdefault("meta", {})
    source_path = meta.get("source_path") or ""
    repo = infer_repo_from_source(source_path)
    if repo:
        meta["project"] = meta.get("project") or Path(repo).name
    commits = git_index.get(repo or "", [])
    linked = nearest_commits(commits)
    if linked:
        meta["linked_commits"] = linked
        if mark_exec_on_link and meta.get("exec") == "unknown":
            meta["exec"] = "pass"
            meta["verify"] = "git_linked"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach git commit links to curated JSONL")
    parser.add_argument("--curated", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--mark-exec",
        action="store_true",
        help="Set meta.exec=pass when repo has linked commits (bootstrap labeling)",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir or (data_dir() / "raw")
    git_index = load_git_index(raw_dir)
    out_path = args.out or args.curated

    rows: list[str] = []
    linked = 0
    with args.curated.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            before = row.get("meta", {}).get("linked_commits")
            enrich_curated_row(row, git_index, mark_exec_on_link=args.mark_exec)
            if row.get("meta", {}).get("linked_commits") and not before:
                linked += 1
            rows.append(json.dumps(row, ensure_ascii=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_fh:
        for line in rows:
            out_fh.write(line + "\n")
    print(f"Wrote {len(rows)} rows → {out_path} ({linked} newly linked)")


if __name__ == "__main__":
    main()
