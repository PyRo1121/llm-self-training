"""Hub revision / lastModified skip logic for public ingest."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from llm_dataprep.public.hf_cache import (
    loader_cfg_fingerprint,
    try_skip_ingest,
    write_ingest_state,
)
from llm_dataprep.public.registry import get_spec


def test_try_skip_ingest_with_state(tmp_path: Path) -> None:
    spec = get_spec("swe_zero_openhands")
    cache = tmp_path / "cache"
    cache.mkdir()
    raw = tmp_path / "raw" / "public-swe-zero-openhands-2026-05-31.jsonl"
    raw.parent.mkdir()
    raw.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")

    hub_meta = {
        "sha": "abc123def456",
        "last_modified": datetime(2026, 5, 5, 19, 34, 5, tzinfo=timezone.utc),
    }
    loader_cfg: dict = {}
    write_ingest_state(
        cache,
        {
            "hf_repo": spec.hf_repo,
            "hub_sha": hub_meta["sha"],
            "hub_last_modified": hub_meta["last_modified"].isoformat(),
            "ingested_at": datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
            "raw_path": str(raw),
            "record_count": 2,
            "loader_fingerprint": loader_cfg_fingerprint(loader_cfg),
        },
    )

    skipped, path, count, msg = try_skip_ingest(
        cache,
        spec,
        hub_meta,
        loader_cfg,
        raw_prefix="public-swe-zero-openhands",
        out_dir=tmp_path / "raw",
    )
    assert skipped is True
    assert path == raw
    assert count == 2
    assert "up to date" in msg


def test_try_skip_ingest_hub_newer(tmp_path: Path) -> None:
    spec = get_spec("swe_zero_openhands")
    cache = tmp_path / "cache"
    cache.mkdir()
    raw = tmp_path / "raw" / "public-swe-zero-openhands-2026-05-31.jsonl"
    raw.parent.mkdir()
    raw.write_text("{}\n", encoding="utf-8")

    old_hub = {
        "sha": "old_sha",
        "last_modified": datetime(2026, 5, 5, tzinfo=timezone.utc),
    }
    write_ingest_state(
        cache,
        {
            "hub_sha": old_hub["sha"],
            "loader_fingerprint": "fp",
            "raw_path": str(raw),
            "record_count": 1,
        },
    )
    new_hub = {
        "sha": "new_sha",
        "last_modified": datetime(2026, 6, 1, tzinfo=timezone.utc),
    }
    skipped, _, _, _ = try_skip_ingest(
        cache,
        spec,
        new_hub,
        {},
        raw_prefix="public-swe-zero-openhands",
        out_dir=tmp_path / "raw",
    )
    assert skipped is False
