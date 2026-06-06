"""FastAPI surface for the RAG pipeline.

Run with ``askdocs serve`` or ``uvicorn askdocs.api:app``. Interactive docs at /docs.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import RagPipeline

app = FastAPI(
    title="askdocs",
    version="0.1.0",
    description="Production RAG document Q&A with retrieval + answer-quality evaluation.",
)

_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    """Lazily construct one shared pipeline (loads the embedding model once)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


class IngestRequest(BaseModel):
    path: str = Field(..., description="File or directory to index")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    p = get_pipeline()
    return {
        "indexed_chunks": p.store.count(),
        "answer_model": p.settings.answer_model,
        "embedding_model": p.settings.embedding_model,
        "top_k": p.settings.top_k,
    }


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest) -> dict:
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"path not found: {req.path}")
    added = get_pipeline().index(path)
    return {"indexed_chunks": added}


@app.post("/query")
def query_endpoint(req: QueryRequest) -> dict:
    pipeline = get_pipeline()
    if pipeline.store.count() == 0:
        raise HTTPException(status_code=409, detail="no documents indexed; POST /ingest first")
    ans = pipeline.answer(req.question, top_k=req.top_k)
    return {
        "answer": ans.text,
        "contexts": [
            {
                "source": h.source,
                "chunk_index": h.chunk_index,
                "score": round(h.score, 4),
                "text": h.text,
            }
            for h in ans.contexts
        ],
        "usage": ans.usage.as_dict(),
    }
