"""Thin read-only FastMCP over allowlist RAG (stdio)."""

from __future__ import annotations

import json

try:
    from fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "Install MCP extra: uv sync --package llm-rag --extra mcp"
    ) from exc

from llm_rag.query import rag_overview, search_allowlist

mcp = FastMCP(
    "llm-self-train-rag",
    instructions=(
        "Read-only retrieval over operator allowlisted docs (Chroma). "
        "Use Context7 MCP for public library API docs when context7_library_id is set."
    ),
)


@mcp.tool()
def search_allowlist_docs(query: str, top_k: int = 8) -> str:
    """Search local allowlist documentation chunks. Returns JSON list of hits."""
    hits = search_allowlist(query, top_k=top_k)
    return json.dumps(hits, indent=2)


@mcp.tool()
def rag_status() -> str:
    """Chroma collection stats and embed model configuration."""
    return json.dumps(rag_overview(), indent=2)


def run_mcp() -> None:
    mcp.run()


if __name__ == "__main__":
    run_mcp()
