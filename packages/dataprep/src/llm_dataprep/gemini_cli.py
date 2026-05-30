"""Ingest Google Gemini CLI sessions from ~/.gemini/tmp/.../chats/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from llm_dataprep.harnesses import _gemini_root
from llm_dataprep.raw_io import append_records


def iter_session_files(root: Path | None = None) -> Iterator[Path]:
    root = root or _gemini_root()
    if not root.exists():
        return
    for chats in root.rglob("chats"):
        if not chats.is_dir():
            continue
        for path in sorted(chats.glob("session-*.jsonl")):
            yield path
        for path in sorted(chats.glob("session-*.json")):
            yield path


def _messages_from_obj(obj: dict[str, Any]) -> Iterator[tuple[str, str]]:
    messages = obj.get("messages") or []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        mtype = msg.get("type")
        role = "user" if mtype == "user" else "assistant" if mtype in ("gemini", "assistant", "model") else None
        if not role:
            continue
        content = msg.get("content") or msg.get("text") or ""
        if isinstance(content, list):
            content = "\n".join(str(x) for x in content)
        text = str(content).strip()
        if text:
            yield role, text


def _parse_jsonl(path: Path) -> Iterator[tuple[str, str]]:
    header: dict[str, Any] | None = None
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('{"$set"'):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "messages" in obj and isinstance(obj["messages"], list):
                yield from _messages_from_obj(obj)
                continue
            mtype = obj.get("type")
            if mtype == "user":
                text = (obj.get("content") or "").strip()
                if text:
                    yield "user", text
            elif mtype in ("gemini", "assistant", "model"):
                text = (obj.get("content") or obj.get("text") or "").strip()
                if text:
                    yield "assistant", text
            elif header is None and "sessionId" in obj:
                header = obj


def ingest(
    root: Path | None = None,
    *,
    out_dir: Path | None = None,
    limit_files: int | None = None,
) -> tuple[Path | None, int]:
    files = list(iter_session_files(root))
    if limit_files:
        files = files[:limit_files]
    ingested = datetime.now(timezone.utc).isoformat()

    def records() -> Iterator[dict[str, Any]]:
        for fpath in files:
            session_id = fpath.stem
            if fpath.suffix == ".json":
                try:
                    obj = json.loads(fpath.read_text(encoding="utf-8"))
                    pairs = list(_messages_from_obj(obj))
                except (json.JSONDecodeError, OSError):
                    continue
            else:
                pairs = list(_parse_jsonl(fpath))
            for role, text in pairs:
                if len(text) > 200_000:
                    continue
                yield {
                    "source": "gemini_cli",
                    "harness": "gemini_cli",
                    "session_id": session_id,
                    "source_path": str(fpath),
                    "role": role,
                    "text": text,
                    "ingested_at": ingested,
                }

    if not files:
        return None, 0
    return append_records("gemini-cli-sessions", records(), out_dir=out_dir)
