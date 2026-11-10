from __future__ import annotations

from ..coding import Status
from ..personas import Persona
from ..providers import ChatMessage, Provider
from .base import Task, TaskResult, TrialSpec, TurnLog

ORDERS = {"yes_first": ["Yes", "No"], "no_first": ["No", "Yes"]}


class VignetteTask(Task):
    kind = "vignette"
    outcome_type = "binary"

    def factor_levels(self) -> dict[str, list[str]]:
        if self.spec.counterbalance_option_order:
            return {"answer_order": list(ORDERS)}
        return {"answer_order": ["yes_first"]}

    @staticmethod
    def labels(trial: TrialSpec) -> list[str]:
        return ORDERS[trial.factors.get("answer_order", "yes_first")]

    def outcome_answer(self, condition: str) -> str:
        # The answer that counts as the outcome (1). Yes unless the stimuli say otherwise, for
        # example "No" to "should the US forbid ...?" when the outcome is permitting the speeches.
        mapping = self.stimuli.get("outcome_answer") or {}
        answer = str(mapping.get(condition, "Yes"))
        if answer not in ("Yes", "No") or not set(mapping) <= set(self.conditions()):
            raise ValueError(f"{self.id}: outcome_answer must map conditions to Yes or No")
        return answer

    def materials(self, trial: TrialSpec) -> str:
        c = self.stimuli["variants"][trial.prompt_variant]["conditions"][trial.condition]
        # A question-only item (no scenario) is shown as just the question.
        parts = (str(c.get("scenario") or "").strip(), str(c["question"]).strip())
        text = "\n\n".join(p for p in parts if p)
        if trial.design_disclosure:
            text = f"{self.stimuli['design_disclosure'].strip()}\n\n{text}"
        return text

    def instruction(self, trial: TrialSpec, persona: Persona | None = None) -> str:
        first, second = self.labels(trial)
        return self.response_instruction(persona).format(first=first, second=second)

    async def run(
        self, trial: TrialSpec, persona: Persona | None, provider: Provider
    ) -> TaskResult:
        if trial.method == "choice":
            return await self._choice(trial, persona, provider)
        return await self._forecast(trial, persona, provider)

    async def _choice(
        self, trial: TrialSpec, persona: Persona | None, provider: Provider
    ) -> TaskResult:
        system = self.system_prompt(persona)
        instruction = self.instruction(trial, persona)
        prompt = f"{self.materials(trial)}\n\n{instruction}"
        history = [ChatMessage("user", prompt)]
        result = TaskResult(
            Status.INVALID, {}, [TurnLog("system", system), TurnLog("user", prompt)]
        )
        coded = await self.ask(
            provider,
            system,
            history,
            result,
            seed=trial.seed,
            tag="ANSWER",
            allowed={"YES", "NO"},
            instruction=instruction,
        )
        result.status = coded.status
        if coded.status is Status.VALID and coded.value:
            answer = coded.value.lower()
            hit = answer == self.outcome_answer(trial.condition).lower()
            result.outcome = {"answer": answer, self.primary_outcome: int(hit)}
        return result

    def forecast_prompt(self, trial: TrialSpec, persona: Persona | None) -> tuple[str, str]:
        templates = self.stimuli["forecast"]
        labels = ", ".join(self.labels(trial))
        materials = self.materials(trial)
        if trial.method == "population_forecast":
            return templates["population_system"].strip(), templates["population_prompt"].format(
                target_population=self.target_population(), materials=materials, labels=labels
            ).strip()
        if persona is None:
            raise ValueError("persona_forecast needs a persona")
        return templates["persona_system"].strip(), templates["persona_prompt"].format(
            persona=persona.describe(), materials=materials, labels=labels
        ).strip()

    async def _forecast(
        self, trial: TrialSpec, persona: Persona | None, provider: Provider
    ) -> TaskResult:
        system, prompt = self.forecast_prompt(trial, persona)
        result = TaskResult(Status.INVALID, {}, [])
        v = await self.ask_forecast(
            provider, system, prompt, result, seed=trial.seed, labels=self.labels(trial)
        )
        if v is None or v.probabilities is None:
            return result
        result.status = Status.VALID
        result.outcome = {
            "probabilities": v.probabilities,
            self.primary_outcome: v.probabilities[self.outcome_answer(trial.condition)],
        }
        return result
