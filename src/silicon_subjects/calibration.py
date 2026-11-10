from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import StudyConfig


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Fit:
    a: float
    b: float
    n_studies: int
    studies: list[str]


def fit(points: list[tuple[str, float, float]], min_studies: int) -> Fit:
    studies = [s for s, _, _ in points]
    if len(set(studies)) != len(studies):
        raise CalibrationError("one point per independent study; duplicates found")
    if len(points) < min_studies:
        raise CalibrationError(
            f"only {len(points)} development studies; at least {min_studies} independent "
            "studies with human effects are needed to fit and check a calibration"
        )
    xs = [p for _, p, _ in points]
    ys = [h for _, _, h in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise CalibrationError("predicted effects do not vary across studies")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    return Fit(my - b * mx, b, len(points), studies)


def development_points(
    cfg: StudyConfig, effects: list[dict[str, Any]], method_key: str
) -> list[tuple[str, float, float]]:
    split = {t.id: t.split for t in cfg.tasks}
    points = []
    for e in effects:
        if e["arm_id"] != method_key:
            continue
        if split.get(e["task_id"]) != "development":
            raise CalibrationError(
                f"{e['task_id']} is not a development study; evaluation outcomes may never "
                "be used for calibration"
            )
        if math.isfinite(e.get("pred_effect", math.nan)) and math.isfinite(
            e.get("human_effect", math.nan)
        ):
            points.append((e["task_id"], float(e["pred_effect"]), float(e["human_effect"])))
    return points


def status(cfg: StudyConfig) -> tuple[dict[str, Fit] | None, str]:
    spec = cfg.analysis.calibration
    if not spec.enabled:
        n_dev = sum(1 for t in cfg.enabled_tasks if t.split == "development")
        return None, (
            f"disabled: {n_dev} development studies available, {spec.min_development_studies} "
            "independent studies needed (see docs/methods.md)"
        )
    if spec.file is None or not cfg.path(spec.file).exists():
        return None, "enabled but no frozen calibration file; run `sim calibrate` and re-freeze"
    raw = json.loads(cfg.path(spec.file).read_text())
    return {k: Fit(**v) for k, v in raw["fits"].items()}, f"applied from {spec.file}"


def write(path: Path, fits: dict[str, Fit], source_runs: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "fits": {k: f.__dict__ for k, f in fits.items()},
                "source_runs": source_runs,
                "note": "fitted on development studies only; freeze before evaluation",
            },
            indent=2,
        )
        + "\n"
    )
