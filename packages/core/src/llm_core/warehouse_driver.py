"""Warehouse DB connection — sqlite3 or pyturso per docs/TURSO.md."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from llm_core.warehouse_config import WarehouseConfig, load_warehouse_config


def connect(path: Path | None = None, *, config: WarehouseConfig | None = None) -> Any:
    """Return a DB connection with dict-like rows (sqlite3.Row or turso.Row)."""
    cfg = config or load_warehouse_config()
    db_path = path or cfg.path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.driver == "turso":
        try:
            import turso  # pyturso
        except ImportError as exc:
            raise ImportError(
                "warehouse.engine=turso requires pyturso: "
                "uv add pyturso --package llm-core"
            ) from exc

        kwargs: dict[str, Any] = {}
        if cfg.experimental_features:
            kwargs["experimental_features"] = ",".join(cfg.experimental_features)
        conn = turso.connect(str(db_path), **kwargs)
        conn.row_factory = turso.Row
        return conn

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def driver_label(config: WarehouseConfig | None = None) -> str:
    cfg = config or load_warehouse_config()
    feats = ",".join(cfg.experimental_features) if cfg.experimental_features else "none"
    return f"{cfg.driver} ({cfg.path}) experimental=[{feats}]"
