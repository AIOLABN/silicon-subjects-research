from __future__ import annotations

import os
import time
from typing import Any

import httpx

from ..config import ModelSpec
from .base import ChatMessage, Completion, TransientProviderError, parse_retry_after


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(self, spec: ModelSpec, timeout_s: float) -> None:
        if not spec.base_url:
            raise ValueError(f"model {spec.id!r}: openai_compatible requires base_url")
        self._spec = spec
        headers = {}
        if key := os.environ.get("OPENAI_COMPATIBLE_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(base_url=spec.base_url, timeout=timeout_s, headers=headers)

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        seed: int,
        response_schema: dict[str, Any] | None = None,
    ) -> Completion:
        body: dict[str, Any] = {
            "model": self._spec.model,
            "max_tokens": self._spec.max_tokens,
            "seed": seed,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            **self._spec.sampling,
        }
        if response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "forecast", "schema": response_schema, "strict": True},
            }
        start = time.perf_counter()
        try:
            r = await self._client.post("/chat/completions", json=body)
        except httpx.TransportError as e:
            raise TransientProviderError(str(e)) from e
        if r.status_code in (408, 409, 429) or r.status_code >= 500:
            raise TransientProviderError(
                f"HTTP {r.status_code}: {r.text[:200]}",
                retry_after=parse_retry_after(r.headers.get("retry-after")),
            )
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return Completion(
            text=choice["message"].get("content") or "",
            stop_reason=choice.get("finish_reason"),
            served_model=data.get("model"),
            platform_refusal=choice.get("finish_reason") == "content_filter",
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_s=time.perf_counter() - start,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
