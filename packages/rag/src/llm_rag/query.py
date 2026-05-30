"""Vector search over allowlist collection."""

from __future__ import annotations

from typing import Any

from llm_rag.config import rag_settings
from llm_rag.embed import embed_texts
from llm_rag.store import collection_stats, get_collection


def search_allowlist(
    query: str,
    *,
    top_k: int | None = None,
    source_id: str | None = None,
) -> list[dict[str, Any]]:
    settings = rag_settings()
    k = top_k or settings["top_k"]
    coll = get_collection()
    if coll.count() == 0:
        return []
    q_emb = embed_texts([query])[0]
    where: dict[str, Any] | None = None
    if source_id:
        where = {"source_id": source_id}
    result = coll.query(
        query_embeddings=[q_emb],
        n_results=min(k, coll.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    hits: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists, strict=True):
        hits.append(
            {
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
        )
    return hits


def rag_overview() -> dict[str, Any]:
    stats = collection_stats()
    settings = rag_settings()
    return {
        "chroma": stats,
        "embed_model": settings["embed_model"],
        "allowlist": str(settings["allowlist_path"]),
    }
