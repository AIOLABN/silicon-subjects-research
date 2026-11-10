from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(_Strict):
    id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    provider: Literal["anthropic", "openai_compatible", "mock"]
    model: str
    base_url: str | None = None
    max_tokens: int = 2048
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None

    sampling: dict[str, float] = Field(default_factory=dict)

    training_cutoff: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _sampling_supported(self) -> ModelSpec:
        if self.provider == "anthropic" and self.sampling:
            raise ValueError(
                f"model {self.id!r}: the Claude API does not accept sampling parameters "
                f"({', '.join(sorted(self.sampling))}); remove `sampling` (variability is set "
                "by the platform defaults and the pinned effort)"
            )
        return self


class TaskSpec(_Strict):
    id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    kind: Literal["framing", "obedience", "survey", "vignette"]
    stimuli: Path
    benchmark: Path

    split: Literal["development", "evaluation"]

    exposure: str
    enabled: bool = True
    counterbalance_option_order: bool = False

    forecast_extension: bool = False

    excluded_persona_fields: list[str] = Field(default_factory=list)

    materials_first_public: str | None = None

    confirmatory_test: bool = True


class AttributeSpec(_Strict):
    name: str

    levels: dict[str, float]

    @model_validator(mode="after")
    def _sums_to_one(self) -> AttributeSpec:
        total = sum(self.levels.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"attribute {self.name!r} proportions sum to {total}, not 1")
        return self


class PersonaSpec(_Strict):
    n_profiles: int = 50
    file: Path
    attributes: list[AttributeSpec]


class SurveyVariable(_Strict):
    name: str
    columns: list[str]

    kind: Literal["category", "age", "number", "race_ethnicity"] = "category"

    labels: dict[str, str] = Field(default_factory=dict)

    group: Literal["demographic", "survey_response"]


class SurveyPersonaSpec(_Strict):
    source: str = "General Social Survey"
    raw_file: Path
    file: Path
    year: int
    year_column: str = "year"
    id_column: str = "id"
    weight_column: str = "wtssps"
    n_profiles: int = 50

    sampling: Literal["pps", "srs"] = "pps"
    variables: list[SurveyVariable]


class ArmSpec(_Strict):
    id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    persona: bool

    persona_source: Literal["demographic", "survey"] = "demographic"
    n_per_cell: int | None = None

    n_per_cell_by_task: dict[str, int] = Field(default_factory=dict)
    replicates_per_profile: int | None = None

    prompt_variants: Literal["original", "paraphrases"] = "original"
    role: Literal["primary", "secondary", "sensitivity"] = "primary"

    method: Literal["choice", "population_forecast", "persona_forecast"] = "choice"

    design_disclosure: bool = False

    tasks: list[str] | None = None

    persona_fields: Literal["all", "demographic_only"] = "all"

    profiles_limit: int | None = None

    @model_validator(mode="after")
    def _sizing(self) -> ArmSpec:
        if self.persona and not self.replicates_per_profile:
            raise ValueError(f"arm {self.id!r}: persona arms need replicates_per_profile")
        if not self.persona and not self.n_per_cell:
            raise ValueError(f"arm {self.id!r}: non-persona arms need n_per_cell")
        if self.method == "population_forecast" and self.persona:
            raise ValueError(f"arm {self.id!r}: population_forecast takes no persona")
        if self.method == "persona_forecast" and not self.persona:
            raise ValueError(f"arm {self.id!r}: persona_forecast needs persona: true")
        if self.persona_fields != "all" and not self.persona:
            raise ValueError(f"arm {self.id!r}: persona_fields applies to persona arms only")
        if self.n_per_cell_by_task and self.persona:
            raise ValueError(f"arm {self.id!r}: n_per_cell_by_task applies to non-persona arms")
        if any(n < 1 for n in self.n_per_cell_by_task.values()):
            raise ValueError(f"arm {self.id!r}: n_per_cell_by_task values must be >= 1")
        return self

    def n_for(self, task_id: str) -> int:
        return int(self.n_per_cell_by_task.get(task_id, self.n_per_cell or 0))


class CalibrationSpec(_Strict):
    enabled: bool = False
    min_development_studies: int = 5

    file: Path | None = None


class AnalysisSpec(_Strict):
    alpha: float = 0.05

    margin_proportion: float = 0.10
    margin_cohens_h: float = 0.20
    margin_smd: float = 0.20
    multiple_comparisons: Literal["holm", "none"] = "holm"
    bootstrap_iterations: int = 5000

    forecast_sum_tolerance: float = 0.001

    calibration: CalibrationSpec = Field(default_factory=CalibrationSpec)


class RunnerSpec(_Strict):
    concurrency: int = 8
    max_attempts: int = Field(default=6, ge=1)
    request_timeout_s: float = 120.0


class StudyConfig(_Strict):
    study_id: str
    seed: int
    models: list[ModelSpec]
    tasks: list[TaskSpec]
    personas: PersonaSpec
    survey_personas: SurveyPersonaSpec | None = None
    arms: list[ArmSpec]
    analysis: AnalysisSpec = AnalysisSpec()
    runner: RunnerSpec = RunnerSpec()

    root: Path = Path()

    @model_validator(mode="after")
    def _unique_ids(self) -> StudyConfig:
        for label, ids in (
            ("model", [m.id for m in self.models]),
            ("task", [t.id for t in self.tasks]),
            ("arm", [a.id for a in self.arms]),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label} ids: {ids}")
        if self.survey_personas is None and any(a.persona_source == "survey" for a in self.arms):
            raise ValueError("an arm uses persona_source: survey but survey_personas is not set")
        task_ids = {t.id for t in self.tasks}
        kinds = {t.id: t for t in self.tasks}
        for arm in self.arms:
            for tid in [*(arm.tasks or []), *arm.n_per_cell_by_task]:
                if tid not in task_ids:
                    raise ValueError(f"arm {arm.id!r}: unknown task {tid!r}")
            for t in self.tasks:
                if not t.enabled or (arm.tasks is not None and t.id not in arm.tasks):
                    continue
                if problem := method_problem(arm, kinds[t.id]):
                    raise ValueError(f"arm {arm.id!r} on task {t.id!r}: {problem}")
        return self

    def path(self, p: Path) -> Path:
        return p if p.is_absolute() else self.root / p

    @property
    def enabled_tasks(self) -> list[TaskSpec]:
        return [t for t in self.tasks if t.enabled]

    def model(self, model_id: str) -> ModelSpec:
        return next(m for m in self.models if m.id == model_id)


def method_problem(arm: ArmSpec, task: TaskSpec) -> str | None:
    if task.kind == "survey" and (arm.method != "choice" or arm.design_disclosure):
        return "survey tasks support only the choice method without design disclosure"
    if task.kind == "obedience":
        if arm.method == "choice" and arm.design_disclosure:
            return (
                "design disclosure would change what the simulated participant knows; it is "
                "only available as a forecasting extension"
            )
        if arm.method != "choice" and not task.forecast_extension:
            return "obedience forecasting is an optional extension (forecast_extension: true)"
        if arm.method == "persona_forecast":
            return "obedience supports population_forecast only"
    return None


def load_config(path: Path | str, root: Path | None = None) -> StudyConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    raw["root"] = str(root if root is not None else path.resolve().parent.parent)
    return StudyConfig.model_validate(raw)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data
