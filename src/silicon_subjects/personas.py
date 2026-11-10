from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import AttributeSpec, PersonaSpec, StudyConfig


@dataclass(frozen=True)
class Persona:
    id: str
    attributes: dict[str, str]

    meta: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return (
            "In this study you will take part as the participant described below. Respond as "
            "this person would, in the first person. Nothing beyond these facts is known about "
            "them.\n" + self.facts()
        )

    def describe(self) -> str:
        return "Profile of the person:\n" + self.facts() + "\nNothing else is known about them."

    def facts(self) -> str:
        return "\n".join(f"- {_label(k)}: {v}" for k, v in self.attributes.items())

    @property
    def analysis_weight(self) -> float:
        return float(self.meta.get("analysis_weight", 1.0))


def _label(key: str) -> str:
    text = key.replace("_", " ")
    return text[:1].upper() + text[1:]


def _quotas(levels: dict[str, float], n: int) -> list[str]:
    exact = {k: p * n for k, p in levels.items()}
    counts = {k: int(v) for k, v in exact.items()}
    remainder = n - sum(counts.values())
    for k in sorted(exact, key=lambda k: (exact[k] - counts[k], k), reverse=True)[:remainder]:
        counts[k] += 1
    return [k for k, c in counts.items() for _ in range(c)]


def generate(spec: PersonaSpec, seed: int) -> list[Persona]:
    rng = random.Random(seed)
    columns: dict[str, list[str]] = {}
    for attr in spec.attributes:
        column = _quotas(attr.levels, spec.n_profiles)
        rng.shuffle(column)
        columns[attr.name] = column
    return [
        Persona(id=f"p{i:03d}", attributes={a.name: columns[a.name][i] for a in spec.attributes})
        for i in range(spec.n_profiles)
    ]


def save(personas: list[Persona], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in personas:
            row: dict[str, Any] = {"id": p.id, "attributes": p.attributes}
            if p.meta:
                row["meta"] = p.meta
            f.write(json.dumps(row) + "\n")


def load(path: Path) -> list[Persona]:
    with path.open() as f:
        return [Persona(**json.loads(line)) for line in f if line.strip()]


def marginals(personas: list[Persona], attrs: list[AttributeSpec]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for a in attrs:
        counts = dict.fromkeys(a.levels, 0)
        for p in personas:
            counts[p.attributes[a.name]] += 1
        out[a.name] = counts
    return out


def load_sets(cfg: StudyConfig) -> dict[str, list[Persona]]:
    used = {a.persona_source for a in cfg.arms if a.persona}
    sets: dict[str, list[Persona]] = {}
    if "demographic" in used:
        sets["demographic"] = load(cfg.path(cfg.personas.file))
    if "survey" in used and cfg.survey_personas is not None:
        path = cfg.path(cfg.survey_personas.file)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; build it with `sim survey-personas` (see data/survey/README.md)"
            )
        sets["survey"] = load(path)
    return sets


def all_personas(sets: dict[str, list[Persona]]) -> list[Persona]:
    return [p for ps in sets.values() for p in ps]


def field_groups(cfg: StudyConfig, source: str) -> dict[str, str]:
    if source == "survey" and cfg.survey_personas is not None:
        return {v.name: v.group for v in cfg.survey_personas.variables}
    return {a.name: "demographic" for a in cfg.personas.attributes}


def select_fields(
    persona: Persona, *, groups: dict[str, str], demographic_only: bool, excluded: set[str]
) -> Persona:
    keep = {
        k: v
        for k, v in persona.attributes.items()
        if k not in excluded
        and (not demographic_only or groups.get(k, "survey_response") == "demographic")
    }
    return Persona(persona.id, keep, persona.meta)
