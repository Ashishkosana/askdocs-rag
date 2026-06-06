"""Generate a grounded answer from retrieved context using Claude.

The system prompt forces the model to answer *only* from the supplied passages and
to abstain when the answer isn't there — the behaviour the evaluation harness then
measures as "faithfulness".
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .observability import Usage
from .store import Hit

SYSTEM = (
    "You are a precise question-answering assistant. Answer ONLY from the numbered "
    "context passages provided. Cite the passages you rely on with bracketed numbers "
    "like [1] or [2]. If the answer is not contained in the context, reply exactly: "
    '"I don\'t have enough information to answer that." Never use outside knowledge.'
)

ABSTAIN = "I don't have enough information to answer that."


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    if hits:
        context = "\n\n".join(
            f"[{i}] (source: {h.source})\n{h.text}" for i, h in enumerate(hits, start=1)
        )
    else:
        context = "(no context retrieved)"
    return f"Context passages:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"


@dataclass(frozen=True)
class Answer:
    text: str
    usage: Usage
    contexts: list[Hit]


def generate_answer(
    client,  # anthropic.Anthropic | compatible
    question: str,
    hits: list[Hit],
    *,
    model: str,
    effort: str,
    max_tokens: int,
) -> Answer:
    prompt = build_user_prompt(question, hits)
    start = time.perf_counter()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    text = next((b.text for b in resp.content if b.type == "text"), "").strip()
    usage = Usage(
        model=model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        latency_ms=latency_ms,
    )
    return Answer(text=text, usage=usage, contexts=hits)
