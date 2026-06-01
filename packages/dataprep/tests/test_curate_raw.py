"""curate_raw session chunking."""

from __future__ import annotations

from llm_dataprep.curate_raw import chunk_messages


def _session(n: int) -> list[dict[str, str]]:
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"message {i} " + "x" * 20,
        }
        for i in range(n)
    ]


def test_chunk_messages_never_exceeds_max_messages() -> None:
    """35-msg session with max_messages=10 must not fall back to one giant chunk."""
    messages = _session(35)
    chunks = chunk_messages(
        messages,
        max_messages=10,
        overlap=4,
        min_messages=2,
        min_total_chars=500,
    )
    for chunk in chunks:
        assert len(chunk) <= 10
    assert not (len(chunks) == 1 and len(chunks[0]) == 35)
