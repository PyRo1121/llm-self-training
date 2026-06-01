"""Load eval/internal JSONL suites."""

from __future__ import annotations

import json
from typing import Any

from llm_core.paths import eval_dir

SUITE_FILES = {
    "diff_apply": "internal/tasks_diff_apply.jsonl",
    "style": "internal/tasks_style.jsonl",
    "debug": "internal/tasks_debug.jsonl",
    "retrieval_gold": "internal/retrieval_gold.jsonl",
}


def load_suite(name: str) -> list[dict[str, Any]]:
    rel = SUITE_FILES.get(name)
    if not rel:
        raise ValueError(f"unknown suite: {name}")
    path = eval_dir() / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_placeholder_task(task: dict[str, Any]) -> bool:
    meta = task.get("meta") or {}
    note = str(meta.get("note", "")).lower()
    if "replace" in note:
        return True
    if str(task.get("repo", "")).upper() == "REPLACE_ME":
        return True
    if str(task.get("id", "")).endswith("-example-001"):
        return True
    return False


def suite_names() -> list[str]:
    return list(SUITE_FILES)


def suite_is_placeholder_only(tasks: list[dict[str, Any]]) -> bool:
    if not tasks:
        return False
    return all(is_placeholder_task(t) for t in tasks)
