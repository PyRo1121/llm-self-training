"""Index allowlist llms.txt sources into Chroma + warehouse metadata."""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from llm_core.control_plane import (
    ensure_warehouse,
    finish_rag_index_run,
    start_rag_index_run,
    upsert_rag_source,
)
from llm_rag.chunking import chunk_text
from llm_rag.config import load_allowlist, rag_settings
from llm_rag.embed import embed_texts
from llm_rag.store import get_collection


def _fetch_text(url: str, *, timeout: float = 60.0) -> str:
    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "llm-self-training-rag/0.1"},
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def _expand_llms_txt(body: str, base_url: str) -> str:
    """Concatenate linked markdown from llms.txt index (best-effort)."""
    lines = body.splitlines()
    parts: list[str] = [body[:8000]]
    fetched = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^-+\s*\[([^\]]+)\]\(([^)]+)\)", line)
        if not m:
            continue
        href = m.group(2)
        if href.startswith("http"):
            doc_url = href
        elif href.startswith("/"):
            from urllib.parse import urljoin

            doc_url = urljoin(base_url, href)
        else:
            from urllib.parse import urljoin

            doc_url = urljoin(base_url.rsplit("/", 1)[0] + "/", href)
        if fetched >= 12:
            break
        try:
            parts.append(_fetch_text(doc_url, timeout=30.0)[:12000])
            fetched += 1
        except httpx.HTTPError:
            continue
    return "\n\n".join(parts)


def _chunk_id(source_id: str, idx: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{source_id}:{idx}:{digest}"


def run_index(*, reset: bool = False, source_ids: list[str] | None = None) -> dict[str, Any]:
    settings = rag_settings()
    allowlist = load_allowlist()
    defaults = allowlist["defaults"]
    chunk_size = int(defaults.get("chunk_size", settings["chunk_size"]))
    overlap = int(defaults.get("chunk_overlap", settings["chunk_overlap"]))
    max_chars = int(defaults.get("max_chars_per_source", settings["max_chars_per_source"]))

    run_id = f"rag-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    conn = ensure_warehouse()
    start_rag_index_run(conn, run_id)

    coll = get_collection(reset=reset)
    sources_ok = 0
    sources_failed = 0
    chunks_added = 0
    errors: dict[str, str] = {}

    try:
        for src in allowlist["sources"]:
            sid = src["id"]
            if source_ids and sid not in source_ids:
                continue
            c7 = src.get("context7_library_id")
            if c7 and not settings["force_index_context7"]:
                upsert_rag_source(
                    conn,
                    source_id=sid,
                    url=src["url"],
                    kind=src.get("kind", "llms_txt"),
                    tier=int(src.get("tier", 0)),
                    context7_library_id=c7,
                    status="context7_only",
                    chunk_count=0,
                )
                sources_ok += 1
                continue
            try:
                body = _fetch_text(src["url"])
                if src.get("kind") == "llms_txt":
                    body = _expand_llms_txt(body, src["url"])
                chunks = chunk_text(
                    body,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    max_chars=max_chars,
                )
                if not chunks:
                    raise ValueError("no chunks produced")
                batch_size = 16
                source_chunks = 0
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    embeddings = embed_texts(batch)
                    ids = [
                        _chunk_id(sid, i + j, batch[j])
                        for j in range(len(batch))
                    ]
                    metadatas = [
                        {
                            "source_id": sid,
                            "url": src["url"],
                            "chunk_index": i + j,
                        }
                        for j in range(len(batch))
                    ]
                    coll.upsert(
                        ids=ids,
                        documents=batch,
                        embeddings=embeddings,
                        metadatas=metadatas,
                    )
                    source_chunks += len(batch)
                upsert_rag_source(
                    conn,
                    source_id=sid,
                    url=src["url"],
                    kind=src.get("kind", "llms_txt"),
                    tier=int(src.get("tier", 0)),
                    context7_library_id=c7,
                    status="indexed",
                    chunk_count=source_chunks,
                )
                chunks_added += source_chunks
                sources_ok += 1
            except Exception as exc:
                sources_failed += 1
                errors[sid] = str(exc)
                upsert_rag_source(
                    conn,
                    source_id=sid,
                    url=src["url"],
                    kind=src.get("kind", "llms_txt"),
                    tier=int(src.get("tier", 0)),
                    context7_library_id=c7,
                    status="failed",
                    chunk_count=0,
                )
        conn.commit()
        status = "completed" if sources_failed == 0 else "completed_with_errors"
        finish_rag_index_run(
            conn,
            run_id,
            status=status,
            sources_ok=sources_ok,
            sources_failed=sources_failed,
            chunks_added=chunks_added,
            details={"errors": errors},
        )
        return {
            "run_id": run_id,
            "status": status,
            "sources_ok": sources_ok,
            "sources_failed": sources_failed,
            "chunks_added": chunks_added,
            "errors": errors,
            "collection_count": coll.count(),
        }
    except Exception:
        finish_rag_index_run(
            conn,
            run_id,
            status="failed",
            sources_ok=sources_ok,
            sources_failed=sources_failed,
            chunks_added=chunks_added,
            details={"errors": errors},
        )
        raise
    finally:
        conn.close()
