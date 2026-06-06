"""Thin wrapper over a persistent Chroma collection (cosine similarity)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk plus its similarity score (1.0 = identical)."""

    text: str
    source: str
    chunk_index: int
    score: float


class VectorStore:
    """Persisted vector index. IDs are ``<source>:<chunk_index>`` so re-indexing
    the same corpus upserts rather than duplicates."""

    def __init__(self, persist_dir: Path, collection: str) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[f"{c.source}:{c.index}" for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "chunk_index": c.index} for c in chunks],
        )

    def query(self, embedding: list[float], top_k: int) -> list[Hit]:
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists, strict=True):
            hits.append(
                Hit(
                    text=doc,
                    source=str(meta["source"]),
                    chunk_index=int(meta["chunk_index"]),
                    score=1.0 - float(dist),  # cosine distance -> similarity
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()
