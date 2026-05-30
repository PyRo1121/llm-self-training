"""RAG settings from config/default.yaml and doc_allowlist.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_core import config_dir, repo_root


def load_yaml_config() -> dict[str, Any]:
    path = config_dir() / "default.yaml"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def rag_settings() -> dict[str, Any]:
    doc = load_yaml_config()
    rag = doc.get("rag") or {}
    ollama = doc.get("ollama") or {}
    chroma_path = rag.get("chroma_path", "data/chroma_db")
    return {
        "chroma_path": repo_root() / chroma_path,
        "collection": rag.get("collection", "allowlist_v1"),
        "allowlist_path": repo_root() / rag.get(
            "allowlist", "config/doc_allowlist.yaml"
        ),
        "top_k": int(rag.get("top_k", 8)),
        "force_index_context7": bool(rag.get("force_index_context7", False)),
        "chunk_size": 512,
        "chunk_overlap": 50,
        "max_chars_per_source": 200_000,
        "ollama_host": ollama.get("host", "http://127.0.0.1:11434").rstrip("/"),
        "embed_model": ollama.get("embed_model", "qwen3-embedding:4b"),
        "embed_fallback": ollama.get("embed_model_fallback", "nomic-embed-text"),
    }


def load_allowlist(path: Path | None = None) -> dict[str, Any]:
    p = path or rag_settings()["allowlist_path"]
    with p.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    defaults = doc.get("defaults") or {}
    sources = doc.get("sources") or []
    return {"defaults": defaults, "sources": sources}
