"""raw_io dated JSONL append helpers."""

from __future__ import annotations

from llm_dataprep.raw_io import append_records, append_records_buffered, dated_raw_path, merge_jsonl_parts


def test_dated_raw_path_uses_prefix_and_dir(tmp_path) -> None:
    path = dated_raw_path("codex-sessions", tmp_path)
    assert path.parent == tmp_path
    assert path.name.startswith("codex-sessions-")
    assert path.name.endswith(".jsonl")


def test_append_records_replace_overwrites_same_day(tmp_path) -> None:
    def batch_a():
        yield {"n": 1}

    path1, n1 = append_records("test-prefix", batch_a(), out_dir=tmp_path, replace=True)
    assert n1 == 1

    def batch_b():
        yield {"n": 2}
        yield {"n": 3}

    path2, n2 = append_records("test-prefix", batch_b(), out_dir=tmp_path, replace=True)
    assert path2 == path1
    assert n2 == 2
    lines = path2.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_append_records_buffered_append_mode(tmp_path) -> None:
    path = tmp_path / "manual.jsonl"

    def first():
        yield {"a": 1}

    n1 = append_records_buffered(path, first(), buffer_rows=10, replace=True)
    assert n1 == 1

    def second():
        yield {"a": 2}

    n2 = append_records_buffered(path, second(), buffer_rows=10, replace=False)
    assert n2 == 1
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_merge_jsonl_parts(tmp_path) -> None:
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text('{"x":1}\n', encoding="utf-8")
    p2.write_text('{"x":2}\n', encoding="utf-8")
    out = tmp_path / "merged.jsonl"
    merge_jsonl_parts([p1, p2], out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
