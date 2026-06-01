"""Warehouse path resolution tests."""

from __future__ import annotations

from llm_core.warehouse_config import _resolve_db_path


def test_resolve_db_path_default(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_DATA_DIR", raising=False)
    path = _resolve_db_path(None)
    assert path.name == "control_plane.db"
    assert path.parent.name == "warehouse"
    assert path.parent.parent.name == "data"


def test_resolve_db_path_relative_under_data_dir(monkeypatch, tmp_path):
    data_root = tmp_path / "volume"
    monkeypatch.setenv("LLM_DATA_DIR", str(data_root))
    assert _resolve_db_path("warehouse/custom.db") == data_root / "warehouse" / "custom.db"


def test_resolve_db_path_strips_legacy_data_prefix(monkeypatch, tmp_path):
    data_root = tmp_path / "volume"
    monkeypatch.setenv("LLM_DATA_DIR", str(data_root))
    assert (
        _resolve_db_path("data/warehouse/control_plane.db")
        == data_root / "warehouse" / "control_plane.db"
    )


def test_resolve_db_path_absolute_unchanged(monkeypatch, tmp_path):
    absolute = tmp_path / "abs" / "control_plane.db"
    monkeypatch.setenv("LLM_DATA_DIR", str(tmp_path / "ignored"))
    assert _resolve_db_path(str(absolute)) == absolute
