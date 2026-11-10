from __future__ import annotations

from ..coding import Status
from ..personas import Persona
from ..providers import ChatMessage, Provider
from .base import Task, TaskResult, TrialSpec, TurnLog


class SurveyTask(Task):
    kind = "survey"
    outcome_type = "continuous"

    async def run(
        self, trial: TrialSpec, persona: Persona | None, provider: Provider
    ) -> TaskResult:
        v = self.stimuli["variants"][trial.prompt_variant]
        lo, hi = int(v["scale"]["min"]), int(v["scale"]["max"])
        system = self.system_prompt(persona)
        prompt = (
            f"{v['conditions'][trial.condition]}\n\n{v['question']}\n\n"
            f"{v['scale']['description']}\n\n{self.stimuli['response_instruction']}"
        )
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
            allowed={str(i) for i in range(lo, hi + 1)},
            instruction=self.stimuli["response_instruction"],
        )
        result.status = coded.status
        if coded.status is Status.VALID and coded.value:
            result.outcome = {self.primary_outcome: int(coded.value)}
        return result
