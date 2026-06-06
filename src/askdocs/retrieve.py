"""Retrieve the most relevant chunks for a query."""

from __future__ import annotations

from .embeddings import Embedder
from .store import Hit, VectorStore


class Retriever:
    """Embeds a query with the same model used at index time, then asks the
    vector store for the nearest chunks."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        if not query.strip():
            return []
        (vector,) = self._embedder.embed([query])
        return self._store.query(vector, top_k=top_k)
