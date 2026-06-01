"""Ingest Cline task JSON from VS Code globalStorage and ~/.cline/data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.paths_util import vscode_global_storage
from llm_dataprep.raw_io import append_records

CLINE_EXT = "saoudrizwan.claude-dev"


def _messages_from_api_history(data: list[Any]) -> Iterator[tuple[str, str]]:
    for item in data:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant"):
            continue
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text") or "")
                    elif "text" in block:
                        parts.append(str(block["text"]))
            text = "\n".join(parts).strip()
        else:
            continue
        if text:
            yield role, text


_USER_SAY = frozenset({"task", "user_feedback", "user_feedback_diff"})
_ASSIST_SAY = frozenset({"text", "reasoning", "completion_result"})


def _ui_role(item: dict[str, Any]) -> str | None:
    role = item.get("role")
    if isinstance(role, str):
        role_key = role.lower()
        if role_key in ("user", "assistant"):
            return role_key

    msg_type = item.get("type")
    if not isinstance(msg_type, str):
        return None
    msg_type = msg_type.lower()
    if msg_type in ("user", "assistant"):
        return msg_type
    if msg_type == "say":
        say = item.get("say")
        if say in _USER_SAY:
            return "user"
        if say in _ASSIST_SAY:
            return "assistant"
        return None
    if msg_type == "ask" and item.get("ask") is not None:
        return "assistant"
    return None


def _ui_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    text = item.get("text") or item.get("content") or ""
    if isinstance(text, list):
        text = "\n".join(str(x) for x in text)
    text = str(text).strip()
    if text:
        parts.append(text)
    reasoning = item.get("reasoning")
    if isinstance(reasoning, str):
        reasoning = reasoning.strip()
        if reasoning:
            parts.append(reasoning)
    return "\n".join(parts).strip()


def _messages_from_ui(data: list[Any]) -> Iterator[tuple[str, str]]:
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("partial"):
            continue
        role = _ui_role(item)
        if role is None:
            continue
        text = _ui_text(item)
        if text:
            yield role, text


def iter_task_dirs(
    extension_id: str = CLINE_EXT,
    *,
    include_cli_data: bool = True,
) -> Iterator[tuple[Path, str]]:
    for root in vscode_global_storage(extension_id):
        tasks = root / "tasks"
        if tasks.is_dir():
            for task_dir in sorted(tasks.iterdir()):
                if task_dir.is_dir():
                    yield task_dir, str(task_dir.name)
    if include_cli_data and extension_id == CLINE_EXT:
        cli = Path.home() / ".cline/data"
        if cli.is_dir():
            for task_dir in sorted(cli.iterdir()):
                if task_dir.is_dir():
                    yield task_dir, f"cli-{task_dir.name}"


def ingest(
    *,
    out_dir: Path | None = None,
    limit_tasks: int | None = None,
    extension_id: str = CLINE_EXT,
    harness_id: str = "cline",
    source: str = "cline",
    include_cli_data: bool = True,
) -> tuple[Path | None, int]:
    ingested = datetime.now(timezone.utc).isoformat()
    tasks = list(
        iter_task_dirs(extension_id, include_cli_data=include_cli_data)
    )
    if limit_tasks:
        tasks = tasks[:limit_tasks]

    def records() -> Iterator[dict[str, Any]]:
        for task_dir, session_id in tasks:
            api_path = task_dir / "api_conversation_history.json"
            ui_path = task_dir / "ui_messages.json"
            for path, parser in (
                (api_path, _messages_from_api_history),
                (ui_path, _messages_from_ui),
            ):
                if not path.is_file():
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(data, list):
                    continue
                for role, text in parser(data):
                    if len(text) > 200_000:
                        continue
                    yield {
                        "source": source,
                        "harness": harness_id,
                        "session_id": session_id,
                        "source_path": str(path),
                        "role": role,
                        "text": text,
                        "ingested_at": ingested,
                    }

    if not tasks:
        return None, 0
    slug = harness_id.replace("_", "-")
    return append_records(f"{slug}-tasks", records(), out_dir=out_dir)
