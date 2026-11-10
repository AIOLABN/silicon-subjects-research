from __future__ import annotations

import time
from typing import Any

from ..config import ModelSpec
from .base import ChatMessage, Completion, TransientProviderError, parse_retry_after


class AnthropicProvider:
    name = "anthropic"
    # The Messages API has no seed parameter (and this config pins no sampling parameters), so
    # repeated calls are independent draws and reruns cannot reproduce a reply.
    sends_seed = False

    def __init__(self, spec: ModelSpec, timeout_s: float) -> None:
        import anthropic

        self._anthropic = anthropic
        self._spec = spec
        self._client = anthropic.AsyncAnthropic(max_retries=0, timeout=timeout_s)

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        seed: int,
        response_schema: dict[str, Any] | None = None,
    ) -> Completion:
        a = self._anthropic
        kwargs: dict[str, Any] = {
            "model": self._spec.model,
            "max_tokens": self._spec.max_tokens,
            "system": system,
            "messages": [
                {"role": m.role, "content": m.raw if m.raw is not None else m.content}
                for m in messages
            ],
        }
        output_config: dict[str, Any] = {}
        if self._spec.effort:
            output_config["effort"] = self._spec.effort
        if response_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": response_schema}
        if output_config:
            kwargs["output_config"] = output_config
        start = time.perf_counter()
        try:
            resp = await self._client.messages.create(**kwargs)
        except a.APIConnectionError as e:
            raise TransientProviderError(str(e)) from e
        except a.APIStatusError as e:
            if e.status_code in (408, 409, 429) or e.status_code >= 500:
                retry_after = parse_retry_after(e.response.headers.get("retry-after"))
                raise TransientProviderError(str(e), retry_after=retry_after) from e
            raise
        text = "".join(b.text for b in resp.content if b.type == "text")
        stop_details = getattr(resp, "stop_details", None)
        return Completion(
            text=text,
            stop_reason=resp.stop_reason,
            served_model=resp.model,
            platform_refusal=resp.stop_reason == "refusal",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            request_id=getattr(resp, "_request_id", None),
            latency_s=time.perf_counter() - start,
            extra={"refusal_category": getattr(stop_details, "category", None)}
            if stop_details
            else {},
            raw_content=[b.model_dump(mode="json", exclude_none=True) for b in resp.content],
        )

    async def aclose(self) -> None:
        await self._client.close()
