"""Register on-disk training runs into warehouse (shared by CLI + API)."""

from __future__ import annotations

import json
from typing import Any

from llm_core.control_plane import _resolve_adapter_dir, register_training_run
from llm_core.paths import runs_dir


def register_run_from_disk(
    run_name: str,
    *,
    status: str = "completed",
) -> dict[str, Any]:
    run_dir = runs_dir() / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    base_model = None
    train_rows = None
    metrics: dict[str, Any] = {}
    cfg_path = run_dir / "train_config.json"
    if cfg_path.is_file():
        doc = json.loads(cfg_path.read_text(encoding="utf-8"))
        settings = doc.get("settings") or {}
        base_model = settings.get("base_model")
        train_rows = (doc.get("train_file_stats") or {}).get("total")

    adapter = _resolve_adapter_dir(run_dir)
    if status == "completed" and adapter is None:
        raise ValueError(f"No adapter in {run_dir}")

    run_id = register_training_run(
        run_name,
        base_model=base_model,
        adapter_path=str(adapter) if adapter else None,
        status=status,
        train_rows=train_rows,
        metrics=metrics,
    )
    return {
        "run_id": run_id,
        "run_name": run_name,
        "adapter_path": str(adapter) if adapter else None,
        "status": status,
    }
