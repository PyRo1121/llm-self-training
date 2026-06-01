"""Shared message block helpers."""

from __future__ import annotations

from llm_dataprep.message_blocks import (
    USER_QUERY_RE,
    normalize_role,
    text_from_content_blocks,
)


def test_normalize_role_from_role_field() -> None:
    assert normalize_role({"role": "Human"}) == "user"
    assert normalize_role({"role": "assistant"}) == "assistant"
    assert normalize_role({"role": "user"}) == "user"


def test_normalize_role_from_type_field() -> None:
    assert normalize_role({"type": "human"}) == "user"
    assert normalize_role({"type": "assistant"}) == "assistant"


def test_normalize_role_from_message_role() -> None:
    assert normalize_role({"message": {"role": "user", "content": "hi"}}) == "user"
    assert normalize_role({"message": {"role": "Human", "content": "hi"}}) == "user"


def test_normalize_role_unknown_returns_none() -> None:
    assert normalize_role({"role": "system"}) is None
    assert normalize_role({}) is None


def test_user_query_re_strips_tags_in_blocks() -> None:
    blocks = [{"type": "text", "text": "<user_query>\nfix auth\n</user_query>"}]
    text, has_tool = text_from_content_blocks(blocks)
    assert text == "fix auth"
    assert has_tool is False
    assert USER_QUERY_RE.search("<user_query>x</user_query>") is not None


def test_text_from_content_blocks_text_only() -> None:
    blocks = [
        {"type": "text", "text": "line one"},
        {"type": "text", "text": "line two"},
    ]
    text, has_tool = text_from_content_blocks(blocks)
    assert text == "line one\nline two"
    assert has_tool is False


def test_text_from_content_blocks_skips_tool_by_default() -> None:
    blocks = [
        {"type": "text", "text": "Checking repo."},
        {
            "type": "tool_use",
            "name": "Shell",
            "input": {"command": "rtk git status"},
        },
    ]
    text, has_tool = text_from_content_blocks(blocks)
    assert text == "Checking repo."
    assert has_tool is True


def test_text_from_content_blocks_include_tool_use() -> None:
    blocks = [
        {"type": "text", "text": "Checking repo."},
        {
            "type": "tool_use",
            "name": "Shell",
            "input": {
                "command": "rtk git status",
                "description": "Check git status",
            },
        },
    ]
    text, has_tool = text_from_content_blocks(blocks, include_tool_use=True)
    assert "Checking repo." in text
    assert "[tool Shell]" in text
    assert "rtk git status" in text
    assert has_tool is True


def test_tool_only_blocks_with_include_tool_use() -> None:
    blocks = [
        {
            "type": "tool_use",
            "name": "Read",
            "input": {"path": "/tmp/foo.py"},
        }
    ]
    text, has_tool = text_from_content_blocks(blocks, include_tool_use=True)
    assert text.startswith("[tool Read]")
    assert "foo.py" in text
    assert has_tool is True
