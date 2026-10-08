"""LLM cost estimation from LiteLLM's public price table.

Prices come from ``litellm.model_cost``. A model missing from that table has an
*unknown* price — callers get ``None`` rather than a silent ``0.0`` so reports
never present unknown cost as free.
"""

from __future__ import annotations

from functools import lru_cache

import litellm


@lru_cache(maxsize=64)
def model_cost_per_token(model: str) -> tuple[float, float] | None:
    """Return (input_cost_per_token, output_cost_per_token), or None if unpriced."""
    info = litellm.model_cost.get(model)
    if not info:
        return None
    input_cpt = info.get("input_cost_per_token")
    output_cpt = info.get("output_cost_per_token")
    if input_cpt is None and output_cpt is None:
        return None
    return float(input_cpt or 0), float(output_cpt or 0)


def estimate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Estimated USD cost of a completion, or None when the model is unpriced."""
    prices = model_cost_per_token(model)
    if prices is None:
        return None
    input_cpt, output_cpt = prices
    return input_cpt * prompt_tokens + output_cpt * completion_tokens
