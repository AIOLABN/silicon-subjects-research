from __future__ import annotations

import json
import random
import re
from typing import Any

from ..config import ModelSpec
from .base import ChatMessage, Completion


class MockProvider:
    name = "mock"
    sends_seed = True

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec

        self._bias = (sum(map(ord, spec.id)) % 7) / 100

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        seed: int,
        response_schema: dict[str, Any] | None = None,
    ) -> Completion:
        rng = random.Random(seed * 1_000_003 + len(messages))
        prompt = messages[-1].content
        if response_schema is not None:
            text = self._forecast(
                messages[0].content, system, list(response_schema["required"]), rng
            )
            return Completion(
                text,
                "end_turn",
                self._spec.model,
                input_tokens=len(prompt) // 4,
                output_tokens=len(text) // 4,
            )
        if rng.random() < 0.01:
            return Completion("", "refusal", self._spec.model, platform_refusal=True)
        if rng.random() < 0.02:
            return Completion(
                "As an AI, I can't role-play this scenario.", "end_turn", self._spec.model
            )
        if rng.random() < 0.02:
            return Completion("Hmm, let me think about that.", "end_turn", self._spec.model)
        text = self._answer(prompt, messages, system, rng)
        return Completion(
            text,
            "end_turn",
            self._spec.model,
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
        )

    def _answer(
        self, prompt: str, messages: list[ChatMessage], system: str, rng: random.Random
    ) -> str:
        if "CHOICE:" in prompt:
            transcript = messages[0].content
            certain_letter = "A"
            for letter, line in re.findall(r"Program ([AB])\b(.*)", transcript):
                if "probability" not in line and "chance" not in line:
                    certain_letter = letter
            other = "B" if certain_letter == "A" else "A"
            p_certain = (0.74 if "saved" in transcript else 0.38) + self._bias
            pick = certain_letter if rng.random() < p_certain else other
            return f"I weighed the options carefully.\nCHOICE: {pick}"
        if "DECISION:" in prompt:
            volts = [int(v) for v in re.findall(r"(\d+)[ -]volt", prompt)]
            level = volts[-1] if volts else 15
            prodded = prompt.startswith(("The experimenter", "The researcher"))
            modeled = "refuses to continue" in messages[0].content
            p_go = 0.99 - level / 900 - (0.08 if modeled else 0) - self._bias
            if prodded:
                p_go = 0.45
            decision = "CONTINUE" if rng.random() < p_go else "STOP"
            return f"I feel uneasy, but...\nDECISION: {decision}"
        if "ANSWER: Yes" in prompt or "ANSWER: No" in prompt:
            p_yes = 0.5 + self._bias + (sum(map(ord, messages[0].content[:300])) % 31 - 15) / 100
            return f"I thought it over.\nANSWER: {'Yes' if rng.random() < p_yes else 'No'}"
        if "ANSWER:" in prompt:
            return f"ANSWER: {min(7, max(1, round(rng.gauss(4.2, 1.4))))}"
        return "I am not sure what to do."

    def _forecast(self, prompt: str, system: str, labels: list[str], rng: random.Random) -> str:
        if rng.random() < 0.03:
            return json.dumps({labels[0]: 0.7, labels[1]: 0.7})
        if labels == ["continue", "stop"]:
            p = 0.62 - (0.08 if "refuses to continue" in prompt else 0) - self._bias
        elif sorted(labels) == ["No", "Yes"]:
            p_yes = 0.5 + self._bias + (sum(map(ord, prompt[:300])) % 31 - 15) / 100
            p = p_yes if labels[0] == "Yes" else 1 - p_yes
        else:
            certain_letter = "A"
            for letter, line in re.findall(r"Program ([AB])\b(.*)", prompt):
                if "probability" not in line and "chance" not in line:
                    certain_letter = letter
            p_certain = (0.68 if "saved" in prompt else 0.45) + self._bias
            if "Profile of the person" in prompt:
                p_certain += (sum(map(ord, prompt[:200])) % 21 - 10) / 100
            p = p_certain if certain_letter == labels[0] else 1 - p_certain
        p = round(min(0.99, max(0.01, p + rng.gauss(0, 0.02))), 3)
        return json.dumps({labels[0]: p, labels[1]: round(1 - p, 3)})

    async def aclose(self) -> None:
        return None
