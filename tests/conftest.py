"""Shared test fakes — no network, no model downloads, no API key."""

from __future__ import annotations

import pytest

from askdocs.store import Hit


class FakeEmbedder:
    """Deterministic stand-in for the sentence-transformers model."""

    dim = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]


class FakeStore:
    """In-memory stand-in for the Chroma-backed VectorStore."""

    def __init__(self, hits: list[Hit] | None = None, count: int = 0) -> None:
        self._hits = hits or []
        self._count = count
        self.added: list[tuple] = []

    def add(self, chunks, embeddings) -> None:
        self.added.append((chunks, embeddings))
        self._count += len(chunks)

    def query(self, embedding, top_k):  # noqa: ARG002 - embedding unused in fake
        return self._hits[:top_k]

    def count(self) -> int:
        return self._count


class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Message:
    def __init__(self, text: str, parsed=None) -> None:
        self.content = [_Block(text)]
        self.usage = _Usage(100, 20)
        self.parsed_output = parsed


class _Messages:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Message(self._text)

    def parse(self, *, output_format, **kwargs):
        self.calls.append(kwargs)
        # produce a minimally valid instance of the requested schema
        return _Message(self._text, parsed=output_format(faithfulness=5, correctness=5,
                                                          relevance=5, rationale="ok"))


class FakeClient:
    def __init__(self, text: str = "Answer [1].") -> None:
        self.messages = _Messages(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_store_factory():
    def _make(hits: list[Hit] | None = None, count: int = 0) -> FakeStore:
        return FakeStore(hits, count)

    return _make


@pytest.fixture
def fake_client_factory():
    def _make(text: str = "Answer [1].") -> FakeClient:
        return FakeClient(text)

    return _make
