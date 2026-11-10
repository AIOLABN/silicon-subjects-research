"""Cost estimates for a plan, from measured token use where a run has measured it.

Assumed per-trial tokens are guesses. A completed run (for example the exploratory pilot,
configs/smoke.yaml) records the real input and output tokens of every trial, including format
repairs and the tokens of retried attempts, so its averages replace the guesses for the same
model, task and method.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import pricing, runstore
from .config import StudyConfig
from .tasks import Task, TrialSpec

# (input, output) tokens per trial before anything is measured. Obedience role-play is a long
# multi-turn conversation whose context grows each turn, so it dwarfs the one-turn tasks.
ASSUMED_TOKENS: dict[str, tuple[int, int]] = {
    "choice": (600, 1000),
    "population_forecast": (700, 800),
    "persona_forecast": (800, 800),
    "obedience_choice": (25_000, 11_000),
}


@dataclass(frozen=True)
class Profile:
    n: int
    mean_in: float
    mean_out: float
    p90_in: float
    p90_out: float


@dataclass(frozen=True)
class Row:
    model_id: str
    task_id: str
    method: str
    persona: bool
    trials: int
    expected_usd: float
    cautious_usd: float
    source: str


def _key(model: str, task_id: str, method: str, persona: bool) -> tuple[str, str, str, bool]:
    return (model, task_id, method, persona)


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]


def measure(run_dirs: Iterable[Path]) -> dict[tuple[str, str, str, bool], Profile]:
    """Per-trial token use by (pinned model, task, method, persona) from completed trials."""
    tokens: dict[tuple[str, str, str, bool], list[tuple[float, float]]] = {}
    for run in run_dirs:
        path = run / "trials.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found: is {run} a run directory?")
        for r in runstore.read_records(path):
            gen = r.get("generation") or {}
            key = _key(
                str(gen.get("model")),
                r["task_id"],
                r.get("method", "choice"),
                r.get("persona_id") is not None,
            )
            tokens.setdefault(key, []).append(
                (
                    (r.get("input_tokens") or 0) + (r.get("discarded_input_tokens") or 0),
                    (r.get("output_tokens") or 0) + (r.get("discarded_output_tokens") or 0),
                )
            )
    return {
        k: Profile(
            n=len(v),
            mean_in=sum(i for i, _ in v) / len(v),
            mean_out=sum(o for _, o in v) / len(v),
            p90_in=_p90([i for i, _ in v]),
            p90_out=_p90([o for _, o in v]),
        )
        for k, v in tokens.items()
    }


def _profile_for(
    model: str,
    task: Task,
    method: str,
    persona: bool,
    measured: dict[tuple[str, str, str, bool], Profile],
    thinking_multiplier: float,
) -> tuple[Profile, str]:
    exact = measured.get(_key(model, task.id, method, persona))
    if exact is not None:
        return exact, f"measured (n={exact.n})"
    if task.kind != "obedience":
        # Short one-turn tasks of the same method cost about the same; never stand in for the
        # multi-turn obedience role-play, or the reverse.
        near = [
            p
            for (m, t, meth, pers), p in measured.items()
            if m == model and meth == method and pers == persona and t != "obedience"
        ]
        if near:
            n = sum(p.n for p in near)
            pooled = Profile(
                n=n,
                mean_in=sum(p.mean_in * p.n for p in near) / n,
                mean_out=sum(p.mean_out * p.n for p in near) / n,
                p90_in=max(p.p90_in for p in near),
                p90_out=max(p.p90_out for p in near),
            )
            return pooled, f"proxy: same method on another one-turn task (n={n})"
    key = "obedience_choice" if task.kind == "obedience" and method == "choice" else method
    tin, tout = ASSUMED_TOKENS[key]
    out = tout * thinking_multiplier
    # Unmeasured: the cautious figure doubles the output tokens (thinking can run long).
    return Profile(0, tin, out, tin, out * 2), "assumed"


def estimate(
    cfg: StudyConfig,
    tasks: dict[str, Task],
    trials: list[TrialSpec],
    measured: dict[tuple[str, str, str, bool], Profile] | None = None,
    thinking_multiplier: float = 1.0,
) -> list[Row]:
    measured = measured or {}
    counts: dict[tuple[str, str, str, bool], int] = {}
    for t in trials:
        k = (t.model_id, t.task_id, t.method, t.persona_id is not None)
        counts[k] = counts.get(k, 0) + 1
    rows = []
    for (model_id, task_id, method, persona), n in sorted(counts.items()):
        spec = cfg.model(model_id)
        prof, source = _profile_for(
            spec.model, tasks[task_id], method, persona, measured, thinking_multiplier
        )
        expected = pricing.cost_for(spec, round(prof.mean_in), round(prof.mean_out))
        cautious = pricing.cost_for(spec, round(prof.p90_in), round(prof.p90_out))
        rows.append(
            Row(
                model_id=model_id,
                task_id=task_id,
                method=method,
                persona=persona,
                trials=n,
                expected_usd=math.nan if expected is None else expected * n,
                cautious_usd=math.nan if cautious is None else cautious * n,
                source=source,
            )
        )
    return rows


def total(rows: list[Row], attr: str = "expected_usd") -> float:
    return sum(v for r in rows if math.isfinite(v := float(getattr(r, attr))))


def as_dicts(rows: list[Row]) -> list[dict[str, Any]]:
    return [r.__dict__.copy() for r in rows]
