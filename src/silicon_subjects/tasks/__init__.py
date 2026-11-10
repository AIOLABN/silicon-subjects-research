from __future__ import annotations

from ..config import StudyConfig, TaskSpec, load_yaml
from .base import Task, TaskResult, TrialSpec
from .framing import FramingTask
from .obedience import ObedienceTask
from .survey import SurveyTask
from .vignette import VignetteTask

REGISTRY: dict[str, type[Task]] = {
    t.kind: t for t in (FramingTask, ObedienceTask, SurveyTask, VignetteTask)
}


def build_task(cfg: StudyConfig, spec: TaskSpec) -> Task:
    return REGISTRY[spec.kind](
        spec,
        load_yaml(cfg.path(spec.stimuli)),
        load_yaml(cfg.path(spec.benchmark)),
        cfg.analysis.forecast_sum_tolerance,
    )


def build_tasks(cfg: StudyConfig) -> dict[str, Task]:
    return {t.id: build_task(cfg, t) for t in cfg.enabled_tasks}


__all__ = ["REGISTRY", "Task", "TaskResult", "TrialSpec", "build_task", "build_tasks"]
