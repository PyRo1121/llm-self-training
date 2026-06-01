"""OpenHands event JSON — MessageEvent/llm_message and V0 shapes."""

from __future__ import annotations

import json

from llm_dataprep.github_harvest import CodeHit, looks_like_chat_blob, parse_blob_text
from llm_dataprep.openhands_events import _text_from_event


def _hit(path: str) -> CodeHit:
    return CodeHit(
        repo_full_name="someone/agent-logs",
        path=path,
        sha="deadbeef",
        html_url="https://github.com/someone/agent-logs/blob/main/" + path,
        query_id="test",
    )


def test_message_event_llm_message_agent() -> None:
    ev = {
        "kind": "MessageEvent",
        "source": "agent",
        "llm_message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "I'll fix the failing test."}],
        },
    }
    role, text = _text_from_event(ev)
    assert role == "assistant"
    assert text == "I'll fix the failing test."


def test_message_event_source_assistant() -> None:
    ev = {
        "kind": "MessageEvent",
        "source": "assistant",
        "llm_message": {
            "role": "assistant",
            "content": "Patch applied.",
        },
    }
    role, text = _text_from_event(ev)
    assert role == "assistant"
    assert text == "Patch applied."


def test_message_event_user() -> None:
    ev = {
        "kind": "MessageEvent",
        "source": "user",
        "llm_message": {
            "role": "user",
            "content": [{"type": "text", "text": "Run pytest on this repo."}],
        },
    }
    role, text = _text_from_event(ev)
    assert role == "user"
    assert "pytest" in text


def test_v0_action_message_still_works() -> None:
    ev = {
        "source": "user",
        "action": "message",
        "args": {"content": "Fix the failing test please"},
    }
    role, text = _text_from_event(ev)
    assert role == "user"
    assert "Fix the failing test" in text


def test_looks_like_rejects_source_without_text() -> None:
    blob = json.dumps({"source": "agent", "kind": "ActionEvent", "tool_name": "bash"})
    assert not looks_like_chat_blob(blob, "openhands", min_hits=2)


def test_parse_message_event_from_github_harvest() -> None:
    hit = _hit(".openhands-state/sessions/sess-xyz/events/7.json")
    blob = json.dumps(
        {
            "kind": "MessageEvent",
            "source": "agent",
            "llm_message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Running pytest now."}],
            },
        }
    )
    recs = list(parse_blob_text(hit, blob, harness_hint="openhands", max_lines=100))
    assert len(recs) == 1
    assert recs[0]["role"] == "assistant"
    assert recs[0]["session_id"] == "sess-xyz"
    assert "pytest" in recs[0]["text"]
