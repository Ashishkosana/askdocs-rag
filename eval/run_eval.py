"""Run the end-to-end evaluation and print a report.

    uv run python eval/run_eval.py --corpus data/corpus

Requires ANTHROPIC_API_KEY (it makes real answer + judge calls). Embeddings and
retrieval run locally. The corpus is (idempotently) indexed before evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow `python eval/run_eval.py` without installing the eval package
sys.path.insert(0, str(Path(__file__).parent))

from judge import judge_answer  # noqa: E402
from retrieval_metrics import aggregate, hit_at_k, reciprocal_rank  # noqa: E402

from askdocs.generate import ABSTAIN  # noqa: E402
from askdocs.pipeline import RagPipeline  # noqa: E402


def load_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the askdocs RAG pipeline")
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.jsonl"))
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args(argv)

    pipeline = RagPipeline()
    added = pipeline.index(args.corpus)
    print(f"Indexed {added} chunks from {args.corpus} (collection now has "
          f"{pipeline.store.count()}).\n")

    dataset = load_dataset(args.dataset)
    per_question: list[dict] = []
    faithfulness, correctness, relevance = [], [], []
    abstention_total = abstention_correct = 0
    total_cost = 0.0

    for row in dataset:
        ans = pipeline.answer(row["question"], top_k=args.top_k)
        total_cost += ans.usage.cost_usd
        sources = [h.source for h in ans.contexts]

        record = {
            "question": row["question"],
            "expected_source": row.get("expected_source"),
            "hit": hit_at_k(sources, row["expected_source"]) if row.get("expected_source") else None,
            "rr": reciprocal_rank(sources, row["expected_source"]) if row.get("expected_source") else None,
        }

        if row["answerable"]:
            result = judge_answer(
                pipeline.client,
                question=row["question"],
                answer=ans.text,
                contexts=[h.text for h in ans.contexts],
                reference=row["reference_answer"],
                model=pipeline.settings.judge_model,
            )
            total_cost += result.usage.cost_usd
            verdict = result.judgement
            faithfulness.append(verdict.faithfulness)
            correctness.append(verdict.correctness)
            relevance.append(verdict.relevance)
            record["scores"] = verdict.model_dump()
        else:
            abstention_total += 1
            did_abstain = ABSTAIN.lower() in ans.text.lower()
            abstention_correct += int(did_abstain)
            record["abstained"] = did_abstain

        per_question.append(record)

    _print_report(per_question, faithfulness, correctness, relevance,
                   abstention_correct, abstention_total, total_cost)


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_report(per_question, faithfulness, correctness, relevance,
                  abstention_correct, abstention_total, total_cost) -> None:
    retr = aggregate(per_question)
    print("=" * 60)
    print("RETRIEVAL")
    print(f"  hit@k : {retr['hit_at_k']:.2f}   (n={retr['n']})")
    print(f"  MRR   : {retr['mrr']:.2f}")
    print("\nANSWER QUALITY (LLM-as-judge, 1-5)")
    print(f"  faithfulness : {_mean(faithfulness):.2f}")
    print(f"  correctness  : {_mean(correctness):.2f}")
    print(f"  relevance    : {_mean(relevance):.2f}")
    print("\nABSTENTION (unanswerable questions)")
    if abstention_total:
        print(f"  correct abstentions : {abstention_correct}/{abstention_total}")
    print(f"\ntotal Claude cost (answers + judge): ${total_cost:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
