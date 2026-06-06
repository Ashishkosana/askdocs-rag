"""Retrieval metrics — measure whether the right document was retrieved at all,
independent of what the LLM then wrote with it.

A RAG system can fail in two distinct places: retrieval (wrong context) or
generation (right context, wrong answer). Measuring them separately is what lets
you debug which half is broken.
"""

from __future__ import annotations


def hit_at_k(retrieved_sources: list[str], expected_source: str) -> bool:
    """True if the expected source appears anywhere in the top-k retrieved chunks."""
    return expected_source in retrieved_sources


def reciprocal_rank(retrieved_sources: list[str], expected_source: str) -> float:
    """1/rank of the first chunk from the expected source (0 if never retrieved)."""
    for rank, source in enumerate(retrieved_sources, start=1):
        if source == expected_source:
            return 1.0 / rank
    return 0.0


def aggregate(per_question: list[dict]) -> dict:
    """Mean hit@k and MRR over the answerable questions only."""
    scored = [q for q in per_question if q.get("expected_source")]
    if not scored:
        return {"hit_at_k": 0.0, "mrr": 0.0, "n": 0}
    n = len(scored)
    return {
        "hit_at_k": sum(q["hit"] for q in scored) / n,
        "mrr": sum(q["rr"] for q in scored) / n,
        "n": n,
    }
