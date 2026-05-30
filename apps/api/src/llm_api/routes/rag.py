from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from llm_core.control_plane import rag_status as warehouse_rag_status
from llm_rag.index import run_index
from llm_rag.query import rag_overview, search_allowlist

router = APIRouter(prefix="/rag")


@router.get("/status")
def get_status() -> dict[str, object]:
    overview = rag_overview()
    return {
        "warehouse": warehouse_rag_status(),
        "chroma": overview.get("chroma"),
        "embed_model": overview.get("embed_model"),
        "allowlist": overview.get("allowlist"),
    }


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    source_id: str | None = None


@router.post("/search")
def post_search(body: SearchRequest) -> dict[str, object]:
    hits = search_allowlist(
        body.query,
        top_k=body.top_k,
        source_id=body.source_id,
    )
    return {"query": body.query, "hits": hits}


class ReindexRequest(BaseModel):
    reset: bool = False
    source_ids: list[str] | None = None


@router.post("/reindex")
def post_reindex(
    body: ReindexRequest,
    background_tasks: BackgroundTasks,
    sync: bool = False,
) -> dict[str, object]:
    if sync:
        return run_index(reset=body.reset, source_ids=body.source_ids)

    def _job() -> None:
        run_index(reset=body.reset, source_ids=body.source_ids)

    background_tasks.add_task(_job)
    return {"status": "started", "reset": body.reset, "source_ids": body.source_ids}
