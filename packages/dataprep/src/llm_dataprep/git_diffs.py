"""Extract small git commit patches via PyDriller for personal SFT."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.raw_io import append_records

MAX_DIFF_CHARS = 12_000
MAX_FILES_PER_COMMIT = 8
CODE_EXTS = {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".go", ".md"}


def _git_config_email(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "config", "user.email"],
            check=True,
            capture_output=True,
            text=True,
        )
        email = out.stdout.strip()
        return email or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _iter_commits(
    repo: Path,
    *,
    max_count: int = 500,
    only_email: str | None = None,
) -> Iterator[Any]:
    from pydriller import Repository  # lazy — optional until uv sync --extra dataprep

    kwargs: dict[str, Any] = {
        "path_to_repo": str(repo),
        "only_no_merge": True,
        "order": "reverse",
    }
    if only_email:
        kwargs["only_authors"] = [only_email]

    repo_obj = Repository(**kwargs)
    for i, commit in enumerate(repo_obj.traverse_commits()):
        if i >= max_count:
            break
        yield commit


def _commit_record(repo: Path, commit: Any) -> dict[str, Any] | None:
    files = []
    for mod in commit.modified_files:
        if mod.change_type.name not in ("ADD", "MODIFY"):
            continue
        if mod.filename and Path(mod.filename).suffix.lower() not in CODE_EXTS:
            continue
        diff = (mod.diff or "")[:MAX_DIFF_CHARS]
        if not diff.strip():
            continue
        files.append({"path": mod.filename, "change": mod.change_type.name, "diff": diff})
        if len(files) >= MAX_FILES_PER_COMMIT:
            break
    if not files:
        return None
    msg = (commit.msg or "").strip()[:2000]
    return {
        "source": "git",
        "harness": "git",
        "session_id": commit.hash,
        "repo": str(repo.resolve()),
        "author_email": getattr(commit.author, "email", None),
        "committed_at": commit.committer_date.isoformat(),
        "message": msg,
        "files": files,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def ingest(
    repo: Path,
    *,
    out_dir: Path | None = None,
    max_commits: int = 500,
    author_email: str | None = None,
) -> tuple[Any, int]:
    repo = repo.resolve()
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"Not a git repo: {repo}")

    email = author_email or _git_config_email(repo)

    def records() -> Iterator[dict[str, Any]]:
        for commit in _iter_commits(repo, max_count=max_commits, only_email=email):
            rec = _commit_record(repo, commit)
            if rec:
                yield rec

    return append_records("git-diffs", records(), out_dir=out_dir)
