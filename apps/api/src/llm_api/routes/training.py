from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm_core.control_plane import list_training_runs
from llm_core.paths import runs_dir
from llm_core.register_run import register_run_from_disk

router = APIRouter(prefix="/training")


@router.get("/runs")
def get_runs(limit: int = 20) -> dict[str, object]:
    return list_training_runs(limit=limit)


class RegisterRunRequest(BaseModel):
    run_name: str
    status: str = "completed"


@router.post("/runs/register")
def post_register_run(body: RegisterRunRequest) -> dict[str, object]:
    run_dir = runs_dir() / body.run_name
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"unknown run: {body.run_name}")
    try:
        return register_run_from_disk(body.run_name, status=body.status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
