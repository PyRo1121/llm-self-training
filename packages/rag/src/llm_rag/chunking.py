"""Simple fixed-size chunking (Phase 1); Chonkie CodeChunker when --extra rag-chunk."""

from __future__ import annotations


def chunk_text(
    text: str,
    *,
    chunk_size: int = 512,
    overlap: int = 50,
    max_chars: int | None = None,
) -> list[str]:
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks
