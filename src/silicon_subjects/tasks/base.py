from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

from .. import forecast
from ..coding import Coded, Status, code_response, looks_like_refusal
from ..config import TaskSpec
from ..personas import Persona
from ..providers import ChatMessage, Provider

BASELINE_SYSTEM = (
    "In this study you will take part as a research participant. Respond in the first person, "
    "as a participant would."
)

FORMAT_REPAIR = "Please restate your decision using exactly the required final line: {instruction}"


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    model_id: str
    task_id: str
    arm_id: str
    condition: str
    prompt_variant: str
    persona_id: str | None
    replicate: int
    factors: dict[str, str]
    seed: int

    method: str = "choice"
    design_disclosure: bool = False
    persona_fields: str = "all"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnLog:
    role: str
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    status: Status
    outcome: dict[str, Any]
    transcript: list[TurnLog]
    format_repairs: int = 0
    ooc_disclaimer: bool = False
    served_models: set[str] = field(default_factory=set)
    input_tokens: int = 0
    output_tokens: int = 0


class Task(ABC):
    kind: ClassVar[str]
    outcome_type: ClassVar[Literal["binary", "continuous"]]

    def __init__(
        self,
        spec: TaskSpec,
        stimuli: dict[str, Any],
        benchmark: dict[str, Any],
        forecast_tolerance: float = forecast.DEFAULT_TOLERANCE,
    ) -> None:
        self.spec = spec
        self.stimuli = stimuli
        self.benchmark = benchmark
        self.forecast_tolerance = forecast_tolerance
        if "original" not in stimuli["variants"]:
            raise ValueError(f"{spec.id}: stimuli must define an 'original' variant")

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def primary_outcome(self) -> str:
        return str(self.benchmark["outcome"])

    @property
    def contrast(self) -> tuple[str, str]:
        a, b = self.benchmark["contrast"]
        return str(a), str(b)

    def conditions(self) -> list[str]:
        return list(self.benchmark["conditions"])

    def variants(self) -> list[str]:
        return list(self.stimuli["variants"])

    def factor_levels(self) -> dict[str, list[str]]:
        return {}

    def system_prompt(self, persona: Persona | None) -> str:
        return persona.render() if persona else BASELINE_SYSTEM

    @abstractmethod
    async def run(
        self, trial: TrialSpec, persona: Persona | None, provider: Provider
    ) -> TaskResult: ...

    async def ask(
        self,
        provider: Provider,
        system: str,
        history: list[ChatMessage],
        result: TaskResult,
        *,
        seed: int,
        tag: str,
        allowed: set[str] | None,
        instruction: str,
    ) -> Coded:
        for attempt in range(2):
            comp = await provider.complete(system, history, seed=seed + attempt)
            result.served_models.add(comp.served_model or "unknown")
            result.input_tokens += comp.input_tokens or 0
            result.output_tokens += comp.output_tokens or 0
            history.append(ChatMessage("assistant", comp.text, comp.raw_content))
            coded = code_response(
                comp.text, tag=tag, allowed=allowed, platform_refusal=comp.platform_refusal
            )
            result.transcript.append(
                TurnLog(
                    "assistant",
                    comp.text,
                    {
                        "status": coded.status,
                        "value": coded.value,
                        "stop_reason": comp.stop_reason,
                        "served_model": comp.served_model,
                        "request_id": comp.request_id,
                        **comp.extra,
                    },
                )
            )
            result.ooc_disclaimer |= coded.ooc_disclaimer
            if coded.status is not Status.INVALID or attempt == 1:
                return coded
            repair = FORMAT_REPAIR.format(instruction=instruction)
            history.append(ChatMessage("user", repair))
            result.transcript.append(TurnLog("user", repair, {"format_repair": True}))
            result.format_repairs += 1
        raise AssertionError("unreachable")

    def target_population(self) -> str:
        text = self.benchmark.get("target_population")
        if not text:
            raise ValueError(f"{self.id}: benchmark has no target_population for forecasting")
        return str(text).strip()

    async def ask_forecast(
        self,
        provider: Provider,
        system: str,
        prompt: str,
        result: TaskResult,
        *,
        seed: int,
        labels: list[str],
    ) -> forecast.Validated | None:
        history = [ChatMessage("user", prompt)]
        result.transcript += [TurnLog("system", system), TurnLog("user", prompt)]
        schema = forecast.schema(labels)
        errors: list[str] = []
        for attempt in range(2):
            comp = await provider.complete(
                system, history, seed=seed + attempt, response_schema=schema
            )
            result.served_models.add(comp.served_model or "unknown")
            result.input_tokens += comp.input_tokens or 0
            result.output_tokens += comp.output_tokens or 0
            history.append(ChatMessage("assistant", comp.text, comp.raw_content))
            v = (
                forecast.validate(comp.text, labels, self.forecast_tolerance)
                if not comp.platform_refusal
                else forecast.Validated(None, ["platform refusal"])
            )
            errors += [f"attempt {attempt + 1}: {e}" for e in v.errors]
            result.transcript.append(
                TurnLog(
                    "assistant",
                    comp.text,
                    {
                        "stop_reason": comp.stop_reason,
                        "served_model": comp.served_model,
                        "request_id": comp.request_id,
                        "response_schema": schema,
                        "validation_errors": v.errors,
                        **comp.extra,
                    },
                )
            )
            if comp.platform_refusal:
                result.status, result.outcome = Status.REFUSAL_PLATFORM, {"forecast_errors": errors}
                return None
            if v.ok:
                return v
            if attempt == 1 or (
                looks_like_refusal(comp.text) and not comp.text.strip().startswith("{")
            ):
                refused = looks_like_refusal(comp.text)
                result.status = Status.REFUSAL_OOC if refused else Status.INVALID
                result.outcome = {"forecast_errors": errors}
                return None
            repair = (
                "Your reply did not satisfy the required format ("
                + "; ".join(v.errors)
                + "). Reply with only a JSON object whose keys are exactly "
                + ", ".join(labels)
                + " and whose values are probabilities between 0 and 1 that sum to 1."
            )
            history.append(ChatMessage("user", repair))
            result.transcript.append(TurnLog("user", repair, {"format_repair": True}))
            result.format_repairs += 1
        raise AssertionError("unreachable")
