"""public_ingest CLI — explicit dataset list overrides config disabled flag."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_dataprep.public_ingest import ingest_one


def test_ingest_one_respects_disabled_without_force(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_dataprep.public_ingest._load_config",
        lambda: {"datasets": {"swe_next": {"enabled": False}}},
    )
    monkeypatch.setattr(
        "llm_dataprep.public_ingest.ingest_one_fast",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not ingest")),
    )
    did, path, n = ingest_one("swe_next", out_dir=None, max_rows=None, skip_gated=True)
    assert did == "swe_next"
    assert path is None
    assert n == 0


def test_ingest_one_force_bypasses_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_dataprep.public_ingest._load_config",
        lambda: {"datasets": {"swe_next": {"enabled": False}}},
    )
    out = Path("/tmp/public-swe-next.jsonl")
    fast = MagicMock(return_value=("swe_next", out, 42))
    monkeypatch.setattr("llm_dataprep.public_ingest.ingest_one_fast", fast)
    did, path, n = ingest_one(
        "swe_next",
        out_dir=None,
        max_rows=None,
        skip_gated=True,
        force=True,
    )
    assert did == "swe_next"
    assert path == out
    assert n == 42
    fast.assert_called_once()
