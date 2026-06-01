"""Unit tests for SWE-Zero public loaders (no network)."""

from __future__ import annotations

from llm_dataprep.public.loaders import _append_tool_calls, _normalize_openhands_messages


def test_append_tool_calls_serializes_function_calls() -> None:
    text = _append_tool_calls(
        "Planning fix.",
        [
            {
                "function": {
                    "name": "str_replace_editor",
                    "arguments": '{"command":"view","path":"/workspace"}',
                }
            }
        ],
    )
    assert "Planning fix." in text
    assert "[tool str_replace_editor]" in text
    assert "/workspace" in text


def test_normalize_openhands_maps_tool_role() -> None:
    messages = _normalize_openhands_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "fix bug"},
            {
                "role": "assistant",
                "content": "I'll inspect.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"ls"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "file.py"},
        ]
    )
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool"]
    assert "[tool bash]" in messages[2]["content"]
    assert messages[3]["content"] == "file.py"
