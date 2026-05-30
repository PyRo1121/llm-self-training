from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm_core.control_plane import (
    datalake_summary,
    list_quarantine,
    quarantine_row,
)

router = APIRouter(prefix="/datalake")


@router.get("/summary")
def get_summary() -> dict[str, object]:
    return datalake_summary()


@router.get("/quarantine")
def get_quarantine(limit: int = 50) -> list[dict[str, object]]:
    return list_quarantine(limit=limit)


class QuarantineRequest(BaseModel):
    curated_id: str
    reason: str
    operator: str = "dashboard"


@router.post("/quarantine")
def post_quarantine(body: QuarantineRequest) -> dict[str, object]:
    try:
        return quarantine_row(
            body.curated_id,
            body.reason,
            operator=body.operator,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
