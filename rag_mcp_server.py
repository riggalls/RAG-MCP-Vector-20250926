#!/usr/bin/env python3
"""MCP server exposing the Baby RAG system."""

from __future__ import annotations

from typing import Any, Dict

from mcp.server.fastmcp import Context, FastMCP

from rag_system import BabyRAGSystem


mcp_server = FastMCP(name="Baby RAG MCP", instructions="Query 15 tech snippets via vector search")


def _ensure_rag(context: Context) -> BabyRAGSystem:
    if not hasattr(context.fastmcp, "rag_instance"):
        context.fastmcp.rag_instance = BabyRAGSystem()  # type: ignore[attr-defined]
    return context.fastmcp.rag_instance  # type: ignore[attr-defined]


@mcp_server.tool(name="rag_query", description="Query the Baby RAG system for relevant snippets")
async def rag_query(question: str, n_results: int = 3, context: Context | None = None) -> Dict[str, Any]:
    if not question.strip():
        raise ValueError("question cannot be empty")
    if not 1 <= n_results <= 10:
        raise ValueError("n_results must be between 1 and 10")

    if context is None:
        raise ValueError("context is required")

    rag = _ensure_rag(context)
    results = rag.query(question, n_results=n_results)
    return {
        "question": question,
        "results": results,
        "total_results": len(results),
    }


def main() -> None:
    mcp_server.run()


if __name__ == "__main__":
    main()

