from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from typer.testing import CliRunner

from silicon_subjects import analysis, evaluation, personas
from silicon_subjects.cli import app
from silicon_subjects.config import StudyConfig
from silicon_subjects.tasks import build_tasks


def rows(
    arm: str, cond: str, ys: list[float], persona_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    return [
        {
            "model_id": "m1",
            "task_id": "framing",
            "arm_id": arm,
            "prompt_variant": "original",
            "condition": cond,
            "status": "valid",
            "y": y,
            "persona_id": persona_ids[i] if persona_ids else None,
        }
        for i, y in enumerate(ys)
    ]


def evaluate(
    cfg: StudyConfig, df: pd.DataFrame, seal: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = build_tasks(cfg)
    if seal:
        tasks["framing"].benchmark = analysis.seal(tasks["framing"].benchmark)
    ps = personas.all_personas(personas.load_sets(cfg))
    return evaluation.method_evaluation(df, cfg, tasks, ps)


def test_two_arm_tests_and_distortion_flags(demo_cfg: StudyConfig) -> None:

    df = pd.DataFrame(
        rows("baseline", "gain", [1.0] * 98 + [0.0] * 2)
        + rows("baseline", "loss", [1.0] * 42 + [0.0] * 58)
    )
    conds, effects = evaluate(demo_cfg, df)
    gain = conds[conds["condition"] == "gain"].iloc[0]
    loss = conds[conds["condition"] == "loss"].iloc[0]
    assert gain["collapse_flag"].startswith("near-unanimous")
    assert bool(gain["more_extreme_than_humans"]) and gain["extremity_gap"] > 0.25
    assert gain["fisher_p"] < 1e-6 and gain["diff_p"] < 1e-6
    assert loss["collapse_flag"] == "" and not bool(loss["more_extreme_than_humans"])
    assert loss["fisher_p"] > 0.5 and loss["diff_p"] > 0.5
    eff = effects.iloc[0]
    assert math.isclose(eff["pred_effect"], 0.56) and math.isclose(eff["human_effect"], 0.30)
    assert eff["pred_effect_p"] < 1e-6 and eff["human_effect_fisher_p"] < 1e-6
    assert eff["effect_diff_p"] < 1e-4


def test_persona_collapse_and_no_fisher_for_dependent_units(demo_cfg: StudyConfig) -> None:
    ids = [p.id for p in personas.all_personas(personas.load_sets(demo_cfg))][:20]
    same = rows("persona", "gain", [1.0] * 40, persona_ids=ids * 2)
    varied = rows("persona", "loss", [float(i % 2) for i in range(40)], persona_ids=ids * 2)
    conds, _ = evaluate(demo_cfg, pd.DataFrame(same + varied))
    gain = conds[conds["condition"] == "gain"].iloc[0]
    assert gain["collapse_flag"].startswith("personas collapsed")
    assert gain["n_units"] == 20
    assert "fisher_p" not in conds or conds["fisher_p"].isna().all()


def test_sealed_benchmark_gives_ai_side_flags_only(demo_cfg: StudyConfig) -> None:
    df = pd.DataFrame(rows("baseline", "gain", [1.0] * 100) + rows("baseline", "loss", [0.0] * 100))
    conds, effects = evaluate(demo_cfg, df, seal=True)
    assert conds["collapse_flag"].str.startswith("near-unanimous").all()
    for col in ("human_p", "diff_p", "fisher_p", "more_extreme_than_humans"):
        assert col not in conds
    assert "effect_diff_p" not in effects and "pred_effect_p" in effects


def test_between_run_identical_and_divergent(demo_cfg: StudyConfig) -> None:
    tasks = build_tasks(demo_cfg)
    ps = personas.all_personas(personas.load_sets(demo_cfg))
    a = pd.DataFrame(rows("baseline", "gain", [1.0] * 60 + [0.0] * 40))
    same = evaluation.between_run({"r1": a, "r2": a.copy()}, demo_cfg, tasks, ps).iloc[0]
    assert same["run_range"] == 0 and same["cochran_q"] == 0 and same["homogeneity_p"] == 1
    b = pd.DataFrame(rows("baseline", "gain", [1.0] * 20 + [0.0] * 80))
    diff = evaluation.between_run({"r1": a, "r2": b}, demo_cfg, tasks, ps).iloc[0]
    assert math.isclose(diff["run_range"], 0.4) and diff["homogeneity_p"] < 1e-6
    only_one = evaluation.between_run({"r1": a}, demo_cfg, tasks, ps)
    assert only_one.empty


def test_dry_run_calls_nothing_and_writes_nothing(demo_cfg: StudyConfig) -> None:
    cfg = str(demo_cfg.root / "configs" / "demo.yaml")
    res = CliRunner().invoke(app, ["run", "-c", cfg, "--exploratory", "--dry-run", "--limit", "7"])
    assert res.exit_code == 0, res.output
    assert "DRY RUN: no API calls" in res.output and "would start now: 7" in res.output
    assert not (demo_cfg.root / "results").exists() or not any(
        (demo_cfg.root / "results").iterdir()
    )


def test_compare_runs_of_a_deterministic_provider_show_no_variability(
    demo_cfg: StudyConfig,
) -> None:
    cfg = str(demo_cfg.root / "configs" / "demo.yaml")
    run = CliRunner().invoke
    for rid in ("a", "b"):
        res = run(app, ["run", "-c", cfg, "--exploratory", "--run-id", rid, "--limit", "120"])
        assert res.exit_code == 0, res.output
    res = run(app, ["compare-runs", "a", "b", "-c", cfg])
    assert res.exit_code == 0, res.output
    table = pd.read_csv(Path(demo_cfg.root) / "results" / "compare" / "a__b" / "between_run.csv")
    assert not table.empty and (table["run_range"] == 0).all()
    assert run(app, ["compare-runs", "a", "-c", cfg]).exit_code != 0
    assert run(app, ["compare-runs", "a", "missing", "-c", cfg]).exit_code == 2
