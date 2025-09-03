#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict

PRICING_USD_PER_1K_TOKENS = {
    # Example pricing snapshot (placeholder)
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.003, "output": 0.006},
}


def estimate_cost(tokens_in: int, tokens_out: int, model: str, pricing_date: str = "2025-09-01") -> Dict[str, float | str]:
    price = PRICING_USD_PER_1K_TOKENS.get(model, PRICING_USD_PER_1K_TOKENS["gpt-4o-mini"])
    cin = (tokens_in / 1000.0) * float(price["input"])  # type: ignore
    cout = (tokens_out / 1000.0) * float(price["output"])  # type: ignore
    return {
        "model": model,
        "pricing_date": pricing_date,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_input_usd": round(cin, 6),
        "cost_output_usd": round(cout, 6),
        "cost_total_usd": round(cin + cout, 6),
    }

