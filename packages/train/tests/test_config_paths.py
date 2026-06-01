"""Smoke tests for train config path helpers."""

from __future__ import annotations

from llm_train.config import (
    adapters_dir,
    default_train_file,
    decensor_settings,
    default_output_dir,
    exports_dir,
)


def test_decensor_settings_paths_exist_as_paths() -> None:
    dec = decensor_settings()
    assert dec["modelfile"].suffix == ".modelfile"
    assert dec["inform_slice"].suffix == ".jsonl"


def test_default_train_file_under_data() -> None:
    p = default_train_file()
    assert "train" in p.parts
    assert p.name == "personal-first.jsonl"


def test_output_and_adapter_dirs() -> None:
    out = default_output_dir("test-run-smoke")
    assert out.name == "test-run-smoke"
    assert adapters_dir().name == "adapters"
    assert exports_dir().name == "exports"
