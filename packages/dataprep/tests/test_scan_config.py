"""Presidio scan/curate mode resolution."""

from __future__ import annotations

from llm_dataprep.scan_config import resolve_curate_presidio_mode, resolve_scan_presidio_mode


def test_scan_defaults_to_pattern() -> None:
    assert resolve_scan_presidio_mode() == "pattern"


def test_scan_cli_no_presidio() -> None:
    assert resolve_scan_presidio_mode(cli_no_presidio=True) == "off"


def test_curate_defaults_off_when_honor_failures() -> None:
    assert resolve_curate_presidio_mode(honor_safety_failures=True) == "off"


def test_curate_cli_mode_overrides() -> None:
    assert resolve_curate_presidio_mode(cli_mode="pattern", honor_safety_failures=True) == "pattern"
