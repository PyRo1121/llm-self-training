"""fast_ingest helpers — dated raw path append/replace."""

from __future__ import annotations

from llm_dataprep.public.fast_ingest import append_records_buffered_to_dated


def test_append_records_buffered_to_dated_replace(tmp_path) -> None:
    def records():
        yield {"text": "one"}
        yield {"text": "two"}

    path1, n1 = append_records_buffered_to_dated(
        "public-test-dataset",
        records(),
        out_dir=tmp_path,
        replace=True,
        buffer_rows=10,
    )
    assert n1 == 2
    assert path1.is_file()
    assert path1.parent == tmp_path

    def more():
        yield {"text": "three"}

    path2, n2 = append_records_buffered_to_dated(
        "public-test-dataset",
        more(),
        out_dir=tmp_path,
        replace=True,
        buffer_rows=10,
    )
    assert path2 == path1
    assert n2 == 1
    lines = path2.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
