"""Scan / curate safety scanner mode resolution (Presidio off | pattern | full)."""

from __future__ import annotations

import os
from typing import Literal

PresidioMode = Literal["off", "pattern", "full"]

_VALID: frozenset[str] = frozenset({"off", "pattern", "full"})


def _normalize_mode(raw: str | bool | None, *, default: PresidioMode) -> PresidioMode:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return "off" if not raw else default
    if not raw:
        return default
    mode = str(raw).strip().lower()
    if mode not in _VALID:
        raise SystemExit(f"invalid presidio mode {raw!r} — use off, pattern, or full")
    return mode  # type: ignore[return-value]


def resolve_scan_presidio_mode(
    *,
    cli_no_presidio: bool = False,
    cli_mode: str | None = None,
    file_glob: str | None = None,
) -> PresidioMode:
    """Presidio mode for scan-raw."""
    if cli_no_presidio:
        return "off"
    if cli_mode:
        return _normalize_mode(cli_mode, default="pattern")
    env = os.environ.get("SCAN_PRESIDIO_MODE", "").strip()
    if env:
        return _normalize_mode(env, default="pattern")

    from llm_core.yaml_config import load_yaml_config

    safety = (load_yaml_config().get("safety") or {})
    scan_cfg = safety.get("scan") or {}
    if file_glob and scan_cfg.get("profiles"):
        profile = _profile_for_glob(str(file_glob), scan_cfg["profiles"])
        if profile and profile.get("presidio_mode"):
            return _normalize_mode(str(profile["presidio_mode"]), default="pattern")
    if scan_cfg.get("presidio_mode"):
        return _normalize_mode(str(scan_cfg["presidio_mode"]), default="pattern")
    return "pattern"


def resolve_curate_presidio_mode(
    *,
    cli_no_presidio: bool = False,
    cli_mode: str | None = None,
    honor_safety_failures: bool = True,
) -> PresidioMode:
    """Presidio mode for curate-raw (default off when scan quarantine is honored)."""
    if cli_no_presidio:
        return "off"
    if cli_mode:
        return _normalize_mode(cli_mode, default="off")
    env = os.environ.get("CURATE_PRESIDIO_MODE", "").strip()
    if env:
        return _normalize_mode(env, default="off")

    from llm_core.yaml_config import load_yaml_config

    safety = (load_yaml_config().get("safety") or {})
    curate_cfg = safety.get("curate") or {}
    if curate_cfg.get("presidio_mode") is not None:
        return _normalize_mode(str(curate_cfg["presidio_mode"]), default="off")
    if honor_safety_failures:
        return "off"
    scan_mode = (safety.get("scan") or {}).get("presidio_mode")
    if scan_mode:
        return _normalize_mode(str(scan_mode), default="pattern")
    return "off"


def _profile_for_glob(glob: str, profiles: dict) -> dict | None:
    g = glob.lower()
    for key, spec in profiles.items():
        if not isinstance(spec, dict):
            continue
        match = spec.get("glob") or spec.get("match")
        if match and _glob_matches(g, str(match).lower()):
            return spec
        if key.lower() in g:
            return spec
    return None


def _glob_matches(name: str, pattern: str) -> bool:
    from fnmatch import fnmatch

    return fnmatch(name, pattern)
