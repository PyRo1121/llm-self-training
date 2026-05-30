"""Infer style_tags / tone from transcript text (Phase 1 metadata)."""

from __future__ import annotations

import re
from typing import Any

_TAG_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("security", re.compile(r"\b(CVE|exploit|burp|pentest|vuln|OWASP|XSS|SQLi)\b", re.I)),
    ("refactor", re.compile(r"\b(refactor|extract|rename|move module)\b", re.I)),
    ("debug", re.compile(r"\b(traceback|stack trace|segfault|pytest|failing test)\b", re.I)),
    ("infra", re.compile(r"\b(docker|kubernetes|terraform|systemd|nginx)\b", re.I)),
    ("docs", re.compile(r"\b(README|CHANGELOG|ADR|architecture doc)\b", re.I)),
    ("planning", re.compile(r"\b(roadmap|milestone|phase \d|implementation plan)\b", re.I)),
)


def infer_style_tags(text: str, *, max_tags: int = 6) -> list[str]:
    tags: list[str] = []
    for name, pattern in _TAG_RULES:
        if pattern.search(text):
            tags.append(name)
        if len(tags) >= max_tags:
            break
    return tags


def infer_tone(text: str) -> str:
    if re.search(r"\b(urgent|asap|blocking|production down)\b", text, re.I):
        return "urgent"
    if re.search(r"\b(please review|PR|pull request)\b", text, re.I):
        return "collaborative"
    return "neutral"


def enrich_meta(meta: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    combined = "\n".join(m.get("content", "") for m in messages)
    meta.setdefault("style_tags", infer_style_tags(combined))
    meta.setdefault("tone", infer_tone(combined))
    return meta
