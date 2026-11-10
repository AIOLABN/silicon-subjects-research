from __future__ import annotations

from .config import ModelSpec

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


def price_for(spec: ModelSpec) -> tuple[float, float] | None:
    return spec.price_usd_per_mtok or price(spec.model)


def cost_for(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float | None:
    p = price_for(spec)
    return None if p is None else (input_tokens * p[0] + output_tokens * p[1]) / 1e6
