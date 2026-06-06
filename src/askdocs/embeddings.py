"""Embeddings abstraction.

We embed locally with sentence-transformers so the pipeline runs offline and costs
nothing per query. ``Embedder`` is a Protocol so tests can inject a deterministic
fake and the rest of the system never imports torch.
"""

from __future__ import annotations

from functools import cached_property
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into fixed-length vectors."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Local embedding model. Loaded lazily so import is cheap and test-friendly."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    @cached_property
    def _model(self):  # noqa: ANN202 - third-party type
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self._model_name)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return vectors.tolist()
