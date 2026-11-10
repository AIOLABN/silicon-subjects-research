from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


def price(model: str) -> tuple[float, float] | None:
    return PRICES.get(model)


def cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    p = price(model)
    return None if p is None else (input_tokens * p[0] + output_tokens * p[1]) / 1e6
