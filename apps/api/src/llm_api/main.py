"""Control plane API — warehouse + RAG + training runs."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_core import repo_root
from llm_core.control_plane import ensure_warehouse
from llm_api.routes import datalake, overview, rag, training

app = FastAPI(title="LLM Self Training API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = FastAPI()
api.include_router(overview.router)
api.include_router(datalake.router)
api.include_router(rag.router)
api.include_router(training.router)

app.mount("/api/v1", api)


@app.on_event("startup")
def _startup() -> None:
    conn = ensure_warehouse()
    conn.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "root": str(repo_root())}


def run() -> None:
    uvicorn.run("llm_api.main:app", host="127.0.0.1", port=8080, reload=False)


if __name__ == "__main__":
    run()
