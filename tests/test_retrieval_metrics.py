import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from retrieval_metrics import aggregate, hit_at_k, reciprocal_rank  # noqa: E402


def test_hit_at_k():
    assert hit_at_k(["a.md", "b.md"], "b.md") is True
    assert hit_at_k(["a.md", "b.md"], "c.md") is False


def test_reciprocal_rank_uses_first_match():
    assert reciprocal_rank(["a.md", "b.md", "c.md"], "b.md") == 0.5
    assert reciprocal_rank(["b.md"], "b.md") == 1.0
    assert reciprocal_rank(["a.md"], "z.md") == 0.0


def test_aggregate_ignores_unanswerable_rows():
    rows = [
        {"expected_source": "a.md", "hit": True, "rr": 1.0},
        {"expected_source": "b.md", "hit": False, "rr": 0.0},
        {"expected_source": None},  # unanswerable — excluded from retrieval metrics
    ]
    out = aggregate(rows)
    assert out["n"] == 2
    assert out["hit_at_k"] == 0.5
    assert out["mrr"] == 0.5
