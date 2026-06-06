from askdocs.retrieve import Retriever
from askdocs.store import Hit


def test_retrieve_passes_store_hits_through(fake_embedder, fake_store_factory):
    expected = [Hit("context", "a.md", 0, 0.8)]
    retriever = Retriever(fake_store_factory(hits=expected), fake_embedder)
    assert retriever.retrieve("a question?", top_k=3) == expected


def test_blank_query_short_circuits(fake_embedder, fake_store_factory):
    store = fake_store_factory(hits=[Hit("x", "a.md", 0, 1.0)])
    retriever = Retriever(store, fake_embedder)
    assert retriever.retrieve("   ", top_k=3) == []


def test_top_k_limits_results(fake_embedder, fake_store_factory):
    hits = [Hit(f"c{i}", "a.md", i, 0.9 - i * 0.1) for i in range(5)]
    retriever = Retriever(fake_store_factory(hits=hits), fake_embedder)
    assert len(retriever.retrieve("q?", top_k=2)) == 2
