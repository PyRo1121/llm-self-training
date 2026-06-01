"""Warehouse settings from config/default.yaml (+ env overrides)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from llm_core.paths import config_dir, data_dir


@dataclass(frozen=True)
class WarehouseConfig:
    path: Path
    driver: str  # sqlite | turso
    experimental_features: tuple[str, ...]


def _resolve_db_path(raw: str | None) -> Path:
    if not raw:
        return data_dir() / "warehouse" / "control_plane.db"
    p = Path(raw)
    if p.is_absolute():
        return p
    # Relative paths live under data_dir (respects LLM_DATA_DIR).
    parts = p.parts
    if parts and parts[0] == "data":
        p = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return data_dir() / p


def load_warehouse_config() -> WarehouseConfig:
    path = config_dir() / "default.yaml"
    raw: dict = {}
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        raw = doc.get("warehouse") or {}

    driver_env = os.environ.get("WAREHOUSE_DRIVER", "").strip().lower()
    raw_driver = (raw.get("driver") or raw.get("engine") or "sqlite").strip().lower()
    if driver_env:
        driver = driver_env
    elif raw_driver in ("turso", "turso_database", "pyturso"):
        driver = "turso"
    else:
        driver = "sqlite"

    feats = raw.get("experimental_features") or []
    if isinstance(feats, str):
        feats = [f.strip() for f in feats.split(",") if f.strip()]
    experimental = tuple(str(f).strip() for f in feats if str(f).strip())

    return WarehouseConfig(
        path=_resolve_db_path(raw.get("path")),
        driver=driver,
        experimental_features=experimental,
    )
