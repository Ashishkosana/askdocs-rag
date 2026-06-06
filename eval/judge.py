"""LLM-as-judge — scores answer quality with a second Claude call.

Retrieval metrics can't tell you whether the *answer* is good — only whether the
right document was fetched. An LLM judge grades the generated answer on three axes
using structured output, so the scores are machine-parseable and never free-text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import BaseModel, Field

from askdocs.observability import Usage

JUDGE_SYSTEM = (
    "You are a strict evaluator of retrieval-augmented answers. You are given a "
    "question, a reference answer, the retrieved context, and a candidate answer. "
    "Grade the candidate on a 1-5 integer scale for each axis. Be critical: reserve "
    "5 for answers that are fully correct and grounded."
)


class Judgement(BaseModel):
    """Structured verdict returned by the judge model."""

    faithfulness: int = Field(
        ge=1, le=5, description="Is every claim supported by the retrieved context? Penalize hallucination."
    )
    correctness: int = Field(
        ge=1, le=5, description="Does the answer agree with the reference answer?"
    )
    relevance: int = Field(
        ge=1, le=5, description="Does the answer directly address the question asked?"
    )
    rationale: str = Field(description="One or two sentences explaining the scores.")


@dataclass(frozen=True)
class JudgeResult:
    judgement: Judgement
    usage: Usage


def judge_answer(
    client,
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
    model: str,
) -> JudgeResult:
    context_block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1)) or "(none)"
    prompt = (
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Grade the candidate answer."
    )
    start = time.perf_counter()
    resp = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_format=Judgement,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    usage = Usage(
        model=model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        latency_ms=latency_ms,
    )
    return JudgeResult(judgement=resp.parsed_output, usage=usage)
