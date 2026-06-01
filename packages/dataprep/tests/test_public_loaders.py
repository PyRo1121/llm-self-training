"""Smoke tests for public HF loaders (no network)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from llm_dataprep.public import loaders
from llm_dataprep.public.hf_cache import list_parquet_shards
from llm_dataprep.public.loaders import _active_shard_files, _cooper_traj_path


def test_active_shard_files_uses_list_parquet_shards_filters(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "train.parquet").write_bytes(b"")
    (cache / "data").mkdir()
    (cache / "data" / "shard.parquet").write_bytes(b"")
    skip_dir = cache / ".cache"
    skip_dir.mkdir()
    (skip_dir / "hidden.parquet").write_bytes(b"")
    (cache / ".ingest_part_99.parquet").write_bytes(b"")

    assert _active_shard_files(None, cache) == list_parquet_shards(cache)
    assert [p.name for p in _active_shard_files(None, cache)] == [
        "shard.parquet",
        "train.parquet",
    ]
    explicit = _active_shard_files([cache / ".ingest_part_99.parquet"], cache)
    assert explicit == [cache / ".ingest_part_99.parquet"]


def test_cooper_traj_path() -> None:
    assert _cooper_traj_path("coop/anyhow_task/390/f1_f2", "agent1_traj.json") == (
        "coop/anyhow_task/390/f1_f2/agent1_traj.json"
    )
    assert _cooper_traj_path("anyhow_task/390/f1_f2", "agent2_traj.json") == (
        "coop/anyhow_task/390/f1_f2/agent2_traj.json"
    )


def test_stream_dataset_local_cache_failure_reraises(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    loaders.set_loader_context(local_dir=cache)
    try:
        with patch("datasets.load_dataset", side_effect=OSError("corrupt cache")):
            with pytest.raises(OSError, match="corrupt cache"):
                list(loaders._stream_dataset("repo/example"))
    finally:
        loaders.set_loader_context(local_dir=None)


def test_load_swe_chat_flushes_session_on_session_change() -> None:
    rows = [
        {"session_id": "s1", "role": "user", "content": "hi", "turn_number": 1},
        {"session_id": "s1", "role": "assistant", "content": "hello", "turn_number": 2},
        {"session_id": "s2", "role": "user", "content": "bug", "turn_number": 1},
    ]
    with patch.object(loaders, "_hf_token", return_value="tok"):
        with patch.object(loaders, "_stream_dataset", return_value=iter(rows)):
            records = list(loaders.load_swe_chat())
    s1 = [r for r in records if r["session_id"] == "s1"]
    s2 = [r for r in records if r["session_id"] == "s2"]
    assert len(s1) == 2
    assert len(s2) == 1
    assert s1[0]["text"] == "hi"
    assert s1[1]["text"] == "hello"
    assert records.index(s1[0]) < records.index(s2[0])
