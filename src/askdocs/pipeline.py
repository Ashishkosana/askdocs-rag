"""Wire ingest → embed → store → retrieve → generate into one object."""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .config import settings as default_settings
from .embeddings import Embedder, SentenceTransformerEmbedder
from .generate import Answer, generate_answer
from .ingest import ingest
from .retrieve import Retriever
from .store import VectorStore


class RagPipeline:
    """End-to-end RAG. Dependencies are injectable so tests can run without a
    model download or an API key."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        client=None,
    ) -> None:
        self.settings = settings or default_settings
        self.embedder = embedder or SentenceTransformerEmbedder(self.settings.embedding_model)
        self.store = store or VectorStore(self.settings.persist_dir, self.settings.collection)
        self.retriever = Retriever(self.store, self.embedder)
        self._client = client

    @property
    def client(self):  # noqa: ANN201 - anthropic.Anthropic
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def index(self, corpus: Path) -> int:
        """Ingest and index a file or directory. Returns the number of chunks added."""
        chunks = ingest(
            Path(corpus),
            size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if chunks:
            embeddings = self.embedder.embed([c.text for c in chunks])
            self.store.add(chunks, embeddings)
        return len(chunks)

    def answer(self, question: str, *, top_k: int | None = None) -> Answer:
        """Retrieve context and generate a grounded answer."""
        hits = self.retriever.retrieve(question, top_k=top_k or self.settings.top_k)
        return generate_answer(
            self.client,
            question,
            hits,
            model=self.settings.answer_model,
            effort=self.settings.effort,
            max_tokens=self.settings.max_tokens,
        )
