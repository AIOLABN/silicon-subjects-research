from __future__ import annotations

from ..config import ModelSpec
from .base import Provider


def build_provider(spec: ModelSpec, timeout_s: float) -> Provider:
    if spec.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(spec, timeout_s)
    if spec.provider == "openai_compatible":
        from .openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(spec, timeout_s)
    from .mock import MockProvider

    return MockProvider(spec)
