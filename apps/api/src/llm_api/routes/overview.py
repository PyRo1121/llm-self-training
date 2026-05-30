from __future__ import annotations

from fastapi import APIRouter

from llm_core.control_plane import overview_stats

router = APIRouter()


@router.get("/overview")
def get_overview() -> dict[str, object]:
    return overview_stats()
