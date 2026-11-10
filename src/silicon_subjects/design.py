from __future__ import annotations

import itertools
import random
from collections import Counter
from collections.abc import Iterable

from .config import ArmSpec, StudyConfig
from .hashing import derive_seed, stable_id
from .personas import Persona
from .tasks import Task, TrialSpec


def _arm_variants(arm: ArmSpec, task: Task) -> list[str]:
    if arm.prompt_variants == "original":
        return ["original"]
    return [v for v in task.variants() if v != "original"]


def _balanced_factors(task: Task, n: int, rng: random.Random) -> list[dict[str, str]]:
    levels = task.factor_levels()
    combos = [dict(zip(levels, c, strict=True)) for c in itertools.product(*levels.values())] or [
        {}
    ]
    assignment = [combos[i % len(combos)] for i in range(n)]
    rng.shuffle(assignment)
    return assignment


def build_manifest(
    cfg: StudyConfig, tasks: dict[str, Task], persona_sets: dict[str, list[Persona]]
) -> list[TrialSpec]:
    trials: list[TrialSpec] = []
    for model, task, arm in itertools.product(cfg.models, tasks.values(), cfg.arms):
        if arm.tasks is not None and task.id not in arm.tasks:
            continue
        for condition, variant in itertools.product(task.conditions(), _arm_variants(arm, task)):
            cell = (model.id, task.id, arm.id, condition, variant)
            rng = random.Random(derive_seed(cfg.seed, *cell))
            units: list[tuple[str | None, int]]
            if arm.persona:
                reps = arm.replicates_per_profile or 0
                profiles = persona_sets[arm.persona_source][: arm.profiles_limit]
                units = [(p.id, r) for p in profiles for r in range(reps)]
            else:
                units = [(None, r) for r in range(arm.n_for(task.id))]
            for (persona_id, rep), factors in zip(
                units, _balanced_factors(task, len(units), rng), strict=True
            ):
                tid = stable_id(cfg.study_id, *cell, persona_id, rep)
                trials.append(
                    TrialSpec(
                        trial_id=tid,
                        model_id=model.id,
                        task_id=task.id,
                        arm_id=arm.id,
                        condition=condition,
                        prompt_variant=variant,
                        persona_id=persona_id,
                        replicate=rep,
                        factors=factors,
                        seed=derive_seed(cfg.seed, tid),
                        method=arm.method,
                        design_disclosure=arm.design_disclosure,
                        persona_fields=arm.persona_fields,
                    )
                )
    random.Random(cfg.seed).shuffle(trials)
    ids = [t.trial_id for t in trials]
    if len(ids) != len(set(ids)):
        raise RuntimeError("trial id collision")
    return trials


def summarize(trials: Iterable[TrialSpec]) -> Counter[tuple[str, str, str, str, str]]:
    return Counter((t.model_id, t.task_id, t.arm_id, t.condition, t.prompt_variant) for t in trials)
