from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TOLERANCE = 0.001


def schema(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {label: {"type": "number"} for label in labels},
        "required": list(labels),
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class Validated:
    probabilities: dict[str, float] | None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.probabilities is not None


def validate(text: str, labels: list[str], tolerance: float = DEFAULT_TOLERANCE) -> Validated:
    errors: list[str] = []
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        return Validated(None, [f"not valid JSON: {e}"])
    if not isinstance(data, dict):
        return Validated(None, [f"expected a JSON object, got {type(data).__name__}"])
    missing = [lb for lb in labels if lb not in data]
    extra = [k for k in data if k not in labels]
    if missing:
        errors.append(f"missing labels: {missing}")
    if extra:
        errors.append(f"unexpected labels: {extra}")
    probs: dict[str, float] = {}
    for lb in labels:
        v = data.get(lb)
        if lb not in data:
            continue
        if isinstance(v, bool) or not isinstance(v, int | float):
            errors.append(f"{lb}: not a number ({v!r})")
            continue
        fv = float(v)
        if not math.isfinite(fv):
            errors.append(f"{lb}: not finite ({v!r})")
        elif not 0.0 <= fv <= 1.0:
            errors.append(f"{lb}: outside [0, 1] ({fv})")
        else:
            probs[lb] = fv
    if not errors:
        total = sum(probs.values())
        if abs(total - 1.0) > tolerance:
            errors.append(f"probabilities sum to {total:.6g}, not 1 (tolerance {tolerance})")
    return Validated(None, errors) if errors else Validated(probs, [])


@dataclass(frozen=True)
class Aggregate:
    p: float
    n_units: int
    unit_sd: float


def population(repeats: list[float]) -> Aggregate:
    if not repeats:
        return Aggregate(math.nan, 0, math.nan)
    mean = sum(repeats) / len(repeats)
    sd = (
        math.sqrt(sum((x - mean) ** 2 for x in repeats) / (len(repeats) - 1))
        if len(repeats) > 1
        else math.nan
    )
    return Aggregate(mean, len(repeats), sd)


def personas(per_persona: dict[str, list[float]], weights: dict[str, float]) -> Aggregate:
    means = {pid: sum(v) / len(v) for pid, v in per_persona.items() if v}
    if not means:
        return Aggregate(math.nan, 0, math.nan)
    for pid in means:
        w = weights.get(pid)
        if w is None or not math.isfinite(w) or w <= 0:
            raise ValueError(f"persona {pid}: invalid analysis weight {w!r}")
    total_w = sum(weights[pid] for pid in means)
    mean = sum(weights[pid] * m for pid, m in means.items()) / total_w
    var = sum(weights[pid] * (m - mean) ** 2 for pid, m in means.items()) / total_w
    return Aggregate(mean, len(means), math.sqrt(var))
