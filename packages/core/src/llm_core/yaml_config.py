"""Load config/default.yaml with optional profile overlay (LLM_CONFIG_PROFILE)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from llm_core.paths import config_dir


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def profile_path(name: str) -> Path | None:
    """Resolve overlay yaml for profile name (e.g. cloud-h100 → config/cloud-h100.yaml)."""
    stem = name.strip()
    if not stem:
        return None
    if stem.endswith(".yaml") or stem.endswith(".yml"):
        candidate = config_dir() / stem
    else:
        candidate = config_dir() / f"{stem}.yaml"
    return candidate if candidate.is_file() else None


def load_yaml_config(*, profile: str | None = None) -> dict[str, Any]:
    """Merge default.yaml with optional profile overlay."""
    base_path = config_dir() / "default.yaml"
    doc: dict[str, Any] = {}
    if base_path.is_file():
        with base_path.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}

    prof = profile if profile is not None else os.environ.get("LLM_CONFIG_PROFILE", "").strip()
    overlay_file = profile_path(prof) if prof else None
    if overlay_file is None:
        return doc

    with overlay_file.open(encoding="utf-8") as fh:
        overlay = yaml.safe_load(fh) or {}
    return _deep_merge(doc, overlay)
