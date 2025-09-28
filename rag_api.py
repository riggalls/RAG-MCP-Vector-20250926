#!/usr/bin/env python3
"""FastAPI server exposing the Baby RAG System."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_system import BabyRAGSystem

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


_configure_logging()


app = FastAPI(
    title="Baby RAG API",
    description="Query the Baby RAG system over HTTP",
    version="1.0.0",
)

rag_system: Optional[BabyRAGSystem] = None


class QueryRequest(BaseModel):
    question: str
    n_results: int = 3


class QueryResult(BaseModel):
    id: str
    title: str
    content: str
    similarity_score: float
    distance: float


class QueryResponse(BaseModel):
    question: str
    results: List[QueryResult]
    total_results: int


class HealthResponse(BaseModel):
    status: str
    message: str
    collection_size: int


@app.on_event("startup")
async def startup_event() -> None:
    global rag_system
    logger.info("Initializing Baby RAG system for API …")
    rag_system = BabyRAGSystem()
    logger.info("Baby RAG system ready (collection: %s)", rag_system.collection_name)


@app.get("/")
async def root() -> Dict[str, str]:
    return {
        "message": "Baby RAG API",
        "docs": "/docs",
        "health": "/health",
        "query": "/query",
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")

    try:
        collection_size = rag_system.collection_size
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Health check failed")
        raise HTTPException(status_code=500, detail=f"Health check failed: {exc}") from exc

    return HealthResponse(
        status="healthy",
        message="RAG system is ready",
        collection_size=collection_size,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if request.n_results < 1 or request.n_results > 10:
        raise HTTPException(status_code=400, detail="n_results must be between 1 and 10")

    logger.info("Processing query: %s", question)

    try:
        results = rag_system.query(question, n_results=request.n_results)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    formatted = [
        QueryResult(
            id=result["id"],
            title=result["title"],
            content=result["content"],
            similarity_score=round(result["similarity_score"], 4),
            distance=round(result["distance"], 4),
        )
        for result in results
    ]

    return QueryResponse(
        question=question,
        results=formatted,
        total_results=len(formatted),
    )


@app.get("/collection/info")
async def collection_info() -> Dict[str, object]:
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")

    return {
        "collection_name": rag_system.collection_name,
        "total_documents": rag_system.collection_size,
        "vector_dimensions": rag_system.vector_dimensions,
    }


@app.get("/collection/snippets")
async def collection_snippets() -> Dict[str, object]:
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet")

    snippets = rag_system.get_snippets()

    return {"total_snippets": len(snippets), "snippets": snippets}


if __name__ == "__main__":
    print("🚀 Starting Baby RAG API server…")
    print("📚 Swagger UI: http://localhost:8000/docs")
    print("🔍 Query endpoint: POST http://localhost:8000/query")

    import uvicorn

    uvicorn.run(
        "rag_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

