"""Ollama /api/embed — matches config ollama.embed_model."""

from __future__ import annotations

import httpx

from llm_rag.config import rag_settings


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    settings = rag_settings()
    host = settings["ollama_host"]
    model = settings["embed_model"]
    fallback = settings["embed_fallback"]
    try:
        return _request_embed(host, model, texts)
    except httpx.HTTPError:
        if model != fallback:
            return _request_embed(host, fallback, texts)
        raise


def _request_embed(host: str, model: str, texts: list[str]) -> list[list[float]]:
    payload = {"model": model, "input": texts}
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{host}/api/embed", json=payload)
        resp.raise_for_status()
        data = resp.json()
    embeddings = data.get("embeddings")
    if not embeddings:
        raise RuntimeError(f"Ollama embed returned no embeddings (model={model})")
    return embeddings
