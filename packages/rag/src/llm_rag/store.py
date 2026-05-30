"""Chroma persistent collection for allowlist chunks."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from llm_rag.config import rag_settings


def get_client() -> chromadb.PersistentClient:
    settings = rag_settings()
    path = settings["chroma_path"]
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def get_collection(*, reset: bool = False) -> Collection:
    settings = rag_settings()
    client = get_client()
    name = settings["collection"]
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def collection_stats() -> dict[str, Any]:
    coll = get_collection()
    return {
        "collection": coll.name,
        "count": coll.count(),
        "path": str(rag_settings()["chroma_path"]),
    }
