"""curate_raw session chunking and curation config."""

from __future__ import annotations

from pathlib import Path

from llm_core.yaml_config import load_yaml_config

from llm_dataprep.curate_raw import _curation_for_session, chunk_messages


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


def test_curation_config_from_load_yaml_config() -> None:
    cfg = load_yaml_config().get("curation") or {}
    assert cfg.get("min_messages") == 2
    assert cfg.get("personal", {}).get("min_messages") == 4


def test_curation_for_session_merges_personal_overrides() -> None:
    cfg = load_yaml_config().get("curation") or {}
    rows = [{"source": "cursor", "harness": "cursor", "data_source": "personal"}]
    merged = _curation_for_session(rows, cfg)
    assert merged["min_messages"] == 4
    assert merged["min_message_chars"] == 8


def test_iter_raw_records_counts_parse_errors(tmp_path: Path) -> None:
    from llm_dataprep.curate_raw import iter_raw_records

    raw = tmp_path / "bad.jsonl"
    raw.write_text('{"ok":true}\nnot-json\n{"also":true}\n', encoding="utf-8")
    stats: dict[str, int] = {}
    rows = list(iter_raw_records([raw], stats=stats))
    assert len(rows) == 2
    assert stats.get("parse_errors") == 1
