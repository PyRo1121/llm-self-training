"""Shared role normalization and content-block text extraction for session JSONL."""

from __future__ import annotations

import json
import re
from typing import Any

USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)

_TOOL_INPUT_KEYS = (
    "command",
    "description",
    "path",
    "pattern",
    "query",
    "url",
    "glob_pattern",
    "target_directory",
    "working_directory",
)


def _summarize_tool_input(inp: Any, *, max_len: int = 480) -> str:
    if isinstance(inp, dict):
        bits: list[str] = []
        for key in _TOOL_INPUT_KEYS:
            val = inp.get(key)
            if val is None or val == "":
                continue
            text = str(val).strip()
            if len(text) > 120:
                text = text[:117] + "…"
            bits.append(f"{key}={text}")
        if not bits:
            raw = json.dumps(inp, ensure_ascii=False, separators=(",", ":"))
            text = raw if len(raw) <= max_len else raw[: max_len - 1] + "…"
        else:
            text = " ".join(bits)
    else:
        text = str(inp).strip()
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _strip_user_query(text: str) -> str:
    m = USER_QUERY_RE.search(text)
    return m.group(1).strip() if m else text


def normalize_role(obj: dict[str, Any]) -> str | None:
    """Map human/Human/type/role fields (and message.role) to user|assistant."""
    role = obj.get("role")
    if isinstance(role, str):
        role_key = role.lower()
        if role_key == "human":
            return "user"
        if role_key in ("user", "assistant"):
            return role_key

    kind = obj.get("type")
    if isinstance(kind, str):
        kind_key = kind.lower()
        if kind_key == "human":
            return "user"
        if kind_key in ("user", "assistant"):
            return kind_key

    message = obj.get("message")
    if isinstance(message, dict):
        msg_role = message.get("role")
        if isinstance(msg_role, str):
            msg_key = msg_role.lower()
            if msg_key == "human":
                return "user"
            if msg_key in ("user", "assistant"):
                return msg_key

    return None


def text_from_content_blocks(
    content: list[Any],
    *,
    include_tool_use: bool = False,
) -> tuple[str, bool]:
    """Extract joined text from a message content block array."""
    parts: list[str] = []
    has_tool = False
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            t = block.get("text") or ""
            parts.append(_strip_user_query(t))
        elif kind == "tool_use":
            has_tool = True
            if include_tool_use:
                name = block.get("name") or "tool"
                summary = _summarize_tool_input(block.get("input"))
                parts.append(f"[tool {name}] {summary}")
    return "\n".join(p for p in parts if p).strip(), has_tool


def role_and_text_from_opencode(obj: dict[str, Any]) -> tuple[str | None, str]:
    """OpenCode legacy message JSON (role + parts[])."""
    info = obj.get("info")
    if isinstance(info, dict):
        role = info.get("role")
        parts = obj.get("parts") or info.get("parts")
    else:
        role = obj.get("role")
        parts = obj.get("parts") or obj.get("content")
    text = ""
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                text += (part.get("text") or "") + "\n"
    elif isinstance(obj.get("text"), str):
        text = obj["text"]
    return role, text.strip()
