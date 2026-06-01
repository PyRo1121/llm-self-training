"""Fast HF cache + local parquet loader tests."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from llm_dataprep.public.loaders import (
    _iter_local_parquet_shards,
    load_swe_zero_openhands,
    set_loader_context,
)


def test_iter_local_parquet_shards_stops_early(tmp_path: Path) -> None:
    rows_a = [{"instance_id": f"a-{i}", "trajectory": []} for i in range(3)]
    rows_b = [{"instance_id": f"b-{i}", "trajectory": []} for i in range(3)]
    pq.write_table(pa.Table.from_pylist(rows_a), tmp_path / "a.parquet")
    pq.write_table(pa.Table.from_pylist(rows_b), tmp_path / "b.parquet")

    out = []
    for row in _iter_local_parquet_shards([tmp_path / "a.parquet", tmp_path / "b.parquet"]):
        out.append(row["instance_id"])
        if len(out) >= 2:
            break

    assert out == ["a-0", "a-1"]


def test_local_parquet_loader_integration(tmp_path: Path) -> None:
    row = {
        "instance_id": "org__repo-1",
        "trajectory_id": "abc-123",
        "repo": "org/repo",
        "dataset": "test",
        "license": "MIT",
        "model_patch": "diff --git a/foo b/foo",
        "trajectory": [
            {"role": "user", "content": "Fix the bug in the test suite please."},
            {
                "role": "assistant",
                "content": "I'll run the tests first.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"pytest -q"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "1 passed"},
        ],
    }
    shard = tmp_path / "train-00000.parquet"
    pq.write_table(pa.Table.from_pylist([row]), shard)

    set_loader_context(local_dir=str(tmp_path))
    try:
        records = list(
            load_swe_zero_openhands(max_rows=1, hf_repo="nvidia/SWE-Zero-openhands-trajectories")
        )
    finally:
        set_loader_context(local_dir=None, shard_files=None)

    assert len(records) >= 2
    assert records[0]["harness"] == "public_swe_zero_openhands"
    assert any("[tool bash]" in (r.get("text") or "") for r in records)
