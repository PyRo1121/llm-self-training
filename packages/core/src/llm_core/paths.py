"""Repository paths — single source for data volumes and config."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Project root (directory containing root pyproject.toml)."""
    env = os.environ.get("LLM_SELF_TRAINING_ROOT")
    if env:
        return Path(env).resolve()
    # packages/core/src/llm_core/paths.py -> 4 parents to repo root
    return Path(__file__).resolve().parents[4]


def data_dir() -> Path:
    """Override with LLM_DATA_DIR to put data/raw, warehouse, etc. on another volume."""
    env = os.environ.get("LLM_DATA_DIR")
    if env:
        return Path(env).resolve()
    return repo_root() / "data"


def config_dir() -> Path:
    return repo_root() / "config"


def eval_dir() -> Path:
    return repo_root() / "eval"


def runs_dir() -> Path:
    return repo_root() / "runs"


def warehouse_db() -> Path:
    return data_dir() / "warehouse" / "control_plane.db"


def chroma_dir() -> Path:
    return data_dir() / "chroma_db"


def hf_cache_dir() -> Path:
    env = os.environ.get("LLM_HF_CACHE_DIR")
    if env:
        return Path(env).resolve()
    return data_dir() / "hf_cache"
