from __future__ import annotations

from typing import Any

from ..coding import Status
from ..personas import Persona
from ..providers import ChatMessage, Provider
from .base import Task, TaskResult, TrialSpec, TurnLog

STEP_V = 15
FORECAST_LABELS = ["continue", "stop"]


class ObedienceTask(Task):
    kind = "obedience"
    outcome_type = "binary"

    def _variant(self, trial: TrialSpec) -> dict[str, Any]:
        return dict(self.stimuli["variants"][trial.prompt_variant])

    def _label(self, volts: int) -> str:
        for band in self.stimuli["switch_labels"]:
            if band["from"] <= volts <= band["to"]:
                return str(band["label"])
        raise ValueError(f"no switch label for {volts} V")

    def _reaction(self, volts: int) -> str | None:
        r = self.stimuli["learner_reactions"].get(volts)
        return str(r) if r else None

    async def run(
        self, trial: TrialSpec, persona: Persona | None, provider: Provider
    ) -> TaskResult:
        if trial.method != "choice":
            return await self._forecast(trial, provider)
        if trial.design_disclosure:
            raise ValueError("obedience role-play never discloses the design")
        v = self._variant(trial)
        system = self.system_prompt(persona)
        cond = self.stimuli["conditions"][trial.condition]
        start_v: int = cond["participant_starts_at"]
        end_v: int = self.stimuli["final_decision_volts"]
        instruction = self.response_instruction(persona)
        prods: list[str] = self.stimuli["prods"]

        result = TaskResult(Status.VALID, {}, [TurnLog("system", system)])
        history: list[ChatMessage] = []
        max_administered = 0
        total_prods = 0
        seed = trial.seed

        def say(text: str) -> None:
            history.append(ChatMessage("user", text))
            result.transcript.append(TurnLog("user", text))

        opening = f"{v['intro']}\n\n{cond['narrative']}".strip()
        prev_reaction: str | None = None
        for volts in range(start_v, end_v + 1, STEP_V):
            parts = [opening] if volts == start_v else []
            if prev_reaction:
                parts.append(v["reaction_frame"].format(reaction=prev_reaction))
            parts.append(v["next_level"].format(volts=volts, label=self._label(volts)))
            parts.append(instruction)
            say("\n\n".join(parts))

            prod_idx = 0
            while True:
                seed += 7
                coded = await self.ask(
                    provider,
                    system,
                    history,
                    result,
                    seed=seed,
                    tag="DECISION",
                    allowed={"CONTINUE", "STOP"},
                    instruction=instruction,
                )
                if coded.status is not Status.VALID:
                    result.status = coded.status
                    result.outcome = {"failed_at_volts": volts, "prods_used": total_prods}
                    return result
                if coded.value == "CONTINUE":
                    break
                if prod_idx == len(prods):
                    result.outcome = self._outcome(max_administered, total_prods, stopped_at=volts)
                    return result
                say(f"{v['experimenter_says'].format(prod=prods[prod_idx])}\n\n{instruction}")
                prod_idx += 1
                total_prods += 1

            if volts == end_v:
                result.outcome = self._outcome(max_administered, total_prods, stopped_at=None)
                return result
            max_administered = volts
            prev_reaction = self._reaction(volts)

        raise AssertionError("unreachable")

    def forecast_prompt(self, trial: TrialSpec) -> tuple[str, str]:
        if trial.method != "population_forecast" or not self.spec.forecast_extension:
            raise ValueError(
                "obedience forecasting requires forecast_extension and population_forecast"
            )
        v = self._variant(trial)
        cond = self.stimuli["conditions"][trial.condition]
        f = self.stimuli["forecast"]
        reactions = "; ".join(
            f'{volts} V: "{text}"'
            for volts, text in sorted(self.stimuli["learner_reactions"].items())
        )
        procedure = (
            f["procedure"]
            .format(
                intro=v["intro"].strip(),
                condition=(cond["narrative"] or "").strip(),
                start=cond["participant_starts_at"],
                reactions=reactions,
                prods=" / ".join(f'"{p}"' for p in self.stimuli["prods"]),
            )
            .strip()
        )
        if trial.design_disclosure:
            procedure = f"{self.stimuli['design_disclosure'].strip()}\n\n{procedure}"
        return f["population_system"].strip(), f["population_prompt"].format(
            target_population=self.target_population(),
            procedure=procedure,
            labels=", ".join(FORECAST_LABELS),
        ).strip()

    async def _forecast(self, trial: TrialSpec, provider: Provider) -> TaskResult:
        system, prompt = self.forecast_prompt(trial)
        result = TaskResult(Status.INVALID, {}, [])
        v = await self.ask_forecast(
            provider, system, prompt, result, seed=trial.seed, labels=FORECAST_LABELS
        )
        if v is None or v.probabilities is None:
            return result
        result.status = Status.VALID
        result.outcome = {
            "probabilities": v.probabilities,
            self.primary_outcome: v.probabilities["continue"],
        }
        return result

    def _outcome(self, max_v: int, prods: int, stopped_at: int | None) -> dict[str, Any]:
        return {
            self.primary_outcome: int(stopped_at is None),
            "max_volts_administered": max_v,
            "prods_used": prods,
            "stopped_at_volts": stopped_at,
        }
