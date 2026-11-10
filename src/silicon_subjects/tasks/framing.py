from __future__ import annotations

from ..coding import Status
from ..personas import Persona
from ..providers import ChatMessage, Provider
from .base import Task, TaskResult, TrialSpec, TurnLog

LABELS = ["A", "B"]


class FramingTask(Task):
    kind = "framing"
    outcome_type = "binary"

    def factor_levels(self) -> dict[str, list[str]]:
        if self.spec.counterbalance_option_order:
            return {"option_order": ["certain_first", "risky_first"]}
        return {"option_order": ["certain_first"]}

    def materials(self, trial: TrialSpec) -> tuple[str, dict[str, str]]:
        v = self.stimuli["variants"][trial.prompt_variant]
        frame = v["conditions"][trial.condition]
        order = trial.factors.get("option_order", "certain_first")
        kinds = ["certain", "risky"] if order == "certain_first" else ["risky", "certain"]
        letter_to_kind = dict(zip(LABELS, kinds, strict=True))
        options = "\n".join(
            frame[kind].format(label=letter) for letter, kind in letter_to_kind.items()
        )
        text = f"{v['preamble']}\n\n{options}\n\n{v['question']}"
        if trial.design_disclosure:
            text = f"{self.stimuli['design_disclosure'].strip()}\n\n{text}"
        return text, letter_to_kind

    def render(
        self, trial: TrialSpec, persona: Persona | None = None
    ) -> tuple[str, dict[str, str]]:
        materials, letter_to_kind = self.materials(trial)
        return f"{materials}\n\n{self.response_instruction(persona)}", letter_to_kind

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
        prompt, letter_to_kind = self.render(trial, persona)
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
            tag="CHOICE",
            allowed=set(LABELS),
            instruction=self.response_instruction(persona),
        )
        result.status = coded.status
        if coded.status is Status.VALID and coded.value:
            kind = letter_to_kind[coded.value]
            result.outcome = {"choice": kind, self.primary_outcome: int(kind == "certain")}
        return result

    def forecast_prompt(self, trial: TrialSpec, persona: Persona | None) -> tuple[str, str]:
        templates = self.stimuli["forecast"]
        materials, _ = self.materials(trial)
        labels = ", ".join(LABELS)
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
        _, letter_to_kind = self.materials(trial)
        result = TaskResult(Status.INVALID, {}, [])
        v = await self.ask_forecast(
            provider, system, prompt, result, seed=trial.seed, labels=LABELS
        )
        if v is None or v.probabilities is None:
            return result
        by_kind = {letter_to_kind[k]: p for k, p in v.probabilities.items()}
        result.status = Status.VALID
        result.outcome = {
            "probabilities": v.probabilities,
            "probabilities_by_option": by_kind,
            self.primary_outcome: by_kind["certain"],
        }
        return result
