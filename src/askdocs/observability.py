"""Cost and latency accounting for Claude calls.

Every RAG answer reports the tokens it burned and the dollar cost, so the service
is observable from day one instead of being a black box.
"""

from __future__ import annotations

from dataclasses import dataclass

# USD per 1M tokens (input, output). Source: Claude API pricing.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one call. Unknown models cost 0 rather than raising."""
    in_rate, out_rate = PRICING.get(model, (0.0, 0.0))
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


@dataclass(frozen=True)
class Usage:
    """Per-call telemetry attached to every answer."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 6),
        }
