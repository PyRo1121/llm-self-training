"""Normalize Hugging Face rows into data/raw JSONL records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator


def _message_text(content: Any) -> str:
    """Normalize HF message content (str, list of parts, or dict)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or part.get("value")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text.strip()
    return str(content).strip()


def make_record(
    *,
    dataset_id: str,
    session_id: str,
    line_no: int,
    role: str,
    text: str,
    hf_repo: str,
    label: str = "accepted",
    exec_status: str = "pass",
    verify: str = "public_verified",
    map_tool_to_user: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "source": "public",
        "harness": f"public_{dataset_id}",
        "dataset_id": dataset_id,
        "session_id": session_id,
        "source_path": f"hf://{hf_repo}",
        "line_no": line_no,
        "role": role,
        "text": text,
        "label": label,
        "exec": exec_status,
        "verify": verify,
        "map_tool_to_user": map_tool_to_user,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        rec.update(extra)
    return rec


def iter_messages_records(
    *,
    dataset_id: str,
    session_id: str,
    messages: list[dict[str, Any]],
    hf_repo: str,
    map_tool_to_user: bool,
    skip_system: bool = True,
    label: str = "accepted",
    exec_status: str = "pass",
    verify: str = "public_verified",
    extra: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    line_no = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "").lower()
        content = _message_text(msg.get("content") or msg.get("text"))
        if not content:
            continue
        if skip_system and role == "system":
            continue
        out_role = role
        prefix = ""
        if role == "tool":
            if map_tool_to_user:
                out_role = "user"
                prefix = "[tool]\n"
            else:
                continue
        if role not in ("user", "assistant"):
            continue
        line_no += 1
        yield make_record(
            dataset_id=dataset_id,
            session_id=session_id,
            line_no=line_no,
            role=out_role,
            text=prefix + content,
            hf_repo=hf_repo,
            label=label,
            exec_status=exec_status,
            verify=verify,
            map_tool_to_user=map_tool_to_user,
            extra=extra,
        )
