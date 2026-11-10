from __future__ import annotations

import math
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from silicon_subjects import analysis, design, personas, stats
from silicon_subjects.config import StudyConfig, load_config
from silicon_subjects.tasks import build_tasks


def test_cluster_se_equals_plain_se_with_one_outcome_per_group() -> None:
    y = [1, 0, 1, 1, 0, 1, 0, 1]
    p = sum(y) / len(y)
    sd = math.sqrt(sum((v - p) ** 2 for v in y) / (len(y) - 1))
    assert stats.cluster_prop_se([[v] for v in y]) == pytest.approx(sd / math.sqrt(len(y)))


def test_cluster_se_counts_groups_not_trials() -> None:

    groups = [[1, 1]] * 30 + [[0, 0]] * 20
    flat = [[v] for g in groups for v in g]
    one_each = [[g[0]] for g in groups]
    assert stats.cluster_prop_se(groups) == pytest.approx(stats.cluster_prop_se(one_each))
    assert stats.cluster_prop_se(groups) > 1.3 * stats.cluster_prop_se(flat)


def test_cohens_h_cell_se_matches_binomial_formula() -> None:
    p, n = 0.3, 120
    assert stats.cohens_h_cell_se(p, math.sqrt(p * (1 - p) / n), n) == pytest.approx(
        1 / math.sqrt(n)
    )
    assert stats.cohens_h_cell_se(1.0, 0.0, 25) == pytest.approx(0.2)


def test_tost_n_for_power_hits_the_target() -> None:
    human_se = stats.cohens_h_se(3174, 3097)
    n = stats.tost_n_for_power(human_se, 0.20, 0.05)
    assert n is not None

    def se(k: int) -> float:
        return math.hypot(stats.cohens_h_se(k, k), human_se)

    assert stats.tost_power(se(n), 0.20) >= 0.8 > stats.tost_power(se(n - 5), 0.20)

    assert stats.tost_n_for_power(stats.cohens_h_se(40, 30), 0.20, 0.05) is None


def test_n_per_cell_by_task_sets_the_trials_per_condition(demo_cfg: StudyConfig) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(
        path.read_text().replace(
            "    n_per_cell: 100\n    role: primary",
            "    n_per_cell: 100\n    n_per_cell_by_task: {framing: 130}\n    role: primary",
            1,
        )
    )
    cfg = load_config(path)
    cells = design.summarize(design.build_manifest(cfg, build_tasks(cfg), personas.load_sets(cfg)))
    assert cells[("opus55", "framing", "baseline", "gain", "original")] == 130
    assert cells[("opus55", "obedience", "baseline", "base", "original")] == 100


def test_n_per_cell_by_task_rejects_unknown_tasks(demo_cfg: StudyConfig) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(
        path.read_text().replace(
            "    n_per_cell: 100\n    role: primary",
            "    n_per_cell: 100\n    n_per_cell_by_task: {nope: 5}\n    role: primary",
            1,
        )
    )
    with pytest.raises(ValidationError, match="unknown task 'nope'"):
        load_config(path)


def _study_cfg(tmp_path: Path) -> StudyConfig:
    # The real study config, with sealed benchmarks stubbed (sample sizes only) when absent.
    repo = Path(__file__).resolve().parents[1]
    for sub in ("configs", "data"):
        shutil.copytree(repo / sub, tmp_path / sub, ignore=shutil.ignore_patterns("raw"))
    sealed = {
        "holdout_hauser4.yaml": {"greater_good": 3968, "side_effect": 3955},
        "holdout_allowed_forbidden_ml1.yaml": {"allow": 3122, "forbid": 3170},
    }
    for name, sizes in sealed.items():
        path = tmp_path / "data" / "benchmarks" / name
        if not path.exists():
            stub = yaml.safe_load((repo / "data/benchmarks/templates" / name).read_text())
            stub["conditions"] = {c: {"k": None, "n": n} for c, n in sizes.items()}
            path.write_text(yaml.safe_dump(stub))
    return load_config(tmp_path / "configs" / "study.yaml")


def test_obedience_gets_a_small_share_of_the_budget(tmp_path: Path) -> None:
    from silicon_subjects import costs

    cfg = _study_cfg(tmp_path)
    tasks = build_tasks(cfg)
    trials = design.build_manifest(cfg, tasks, personas.load_sets(cfg))
    obedience = [t for t in trials if t.task_id == "obedience"]
    assert not any(t.persona_id for t in obedience)
    assert {t.method for t in obedience} == {"choice", "population_forecast"}
    rows = costs.estimate(cfg, tasks, trials)
    share = costs.total([r for r in rows if r.task_id == "obedience"]) / costs.total(rows)
    assert share < 0.35


def test_study_primary_family_and_its_power(tmp_path: Path) -> None:
    cfg = _study_cfg(tmp_path)
    family = analysis.confirmatory_family(cfg)
    held_out = sorted(t.id for t in cfg.enabled_tasks if t.split == "evaluation")
    # Only evaluation studies make confirmatory claims; development studies informed the design.
    assert sorted({(t, a) for _, t, a in family}) == [(t, "baseline") for t in held_out]
    assert cfg.analysis.primary_test == "joint"
    tasks = build_tasks(cfg)
    for tid in held_out:
        tasks[tid].benchmark = analysis.seal(tasks[tid].benchmark)
    power = analysis.effect_power_table(cfg, tasks).set_index(["task_id", "arm_id"])
    for tid in held_out:
        assert power.loc[(tid, "baseline"), "in_confirmatory_family"]
        assert power.loc[(tid, "baseline"), "joint_power_at_holm_strictest_lower_bound"] >= 0.8
    for tid in ("framing", "obedience"):
        assert not power.loc[(tid, "baseline"), "in_confirmatory_family"]


def test_seal_keeps_sample_sizes_only() -> None:
    bench = {"conditions": {"a": {"k": 3, "n": 10}}, "sites": [1], "subgroups": []}
    sealed = analysis.seal(bench)
    assert sealed["conditions"] == {"a": {"k": None, "n": 10}} and "sites" not in sealed


def _records(cfg: StudyConfig) -> list[dict[str, object]]:
    recs: list[dict[str, object]] = []

    def add(arm: str, cond: str, pid: str | None, rep: int, y: int, variant: str) -> None:
        recs.append(
            {
                "trial_id": f"{arm}-{cond}-{pid}-{rep}-{variant}",
                "model_id": "opus55",
                "task_id": "framing",
                "arm_id": arm,
                "condition": cond,
                "prompt_variant": variant,
                "persona_id": pid,
                "replicate": rep,
                "method": "choice",
                "status": "valid",
                "outcome": {"chose_certain": y},
                "format_repairs": 0,
                "served_model_mismatch": False,
                "confirmatory": False,
                "transcript": [],
            }
        )

    for cond, share in (("gain", 0.7), ("loss", 0.4)):
        for i in range(50):
            y = int(i < share * 50)
            add("persona", cond, f"p{i:03d}", 0, y, "original")
            add("persona", cond, f"p{i:03d}", 1, y, "original")
        for arm, variant, n in (
            ("baseline", "original", 100),
            ("design_aware", "original", 100),
            ("paraphrase", "para1", 50),
            ("paraphrase", "para2", 50),
        ):
            for r in range(n):
                add(arm, cond, None, r, int(r < share * n), variant)
    return recs


def _cells(cfg: StudyConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    tasks = build_tasks(cfg)
    df = analysis.load_trials(_records(cfg), tasks)
    return analysis.cell_summary(df, cfg, tasks), tasks


def test_persona_cells_use_personas_as_units(demo_cfg: StudyConfig) -> None:
    cells, _ = _cells(demo_cfg)
    c = cells.set_index(["arm_id", "prompt_variant", "condition"])
    persona, base = c.loc[("persona", "original", "gain")], c.loc[("baseline", "original", "gain")]
    assert persona["n_valid"] == base["n_valid"] == 100
    assert persona["n_units"] == 50 and base["n_units"] == 100
    assert persona["se_method"].startswith("cluster")

    assert persona["h_se"] == pytest.approx(1 / math.sqrt(50), rel=0.02)
    assert base["h_se"] == pytest.approx(1 / math.sqrt(100))


def test_primary_test_uses_persona_level_uncertainty(demo_cfg: StudyConfig) -> None:
    cells, tasks = _cells(demo_cfg)
    eff = analysis.effect_agreement(cells, demo_cfg, tasks).set_index("arm_id")
    width = eff["delta_hi90"] - eff["delta_lo90"]

    assert width["persona"] / width["baseline"] == pytest.approx(math.sqrt(2), rel=0.05)


def test_wording_sensitivity_excludes_the_design_aware_arm(demo_cfg: StudyConfig) -> None:
    cells, tasks = _cells(demo_cfg)
    para = analysis.paraphrase_sensitivity(cells, demo_cfg, tasks)
    assert set(para["variants"]) == {"original,para1,para2"}


def test_holm_family_size_is_the_planned_one(demo_cfg: StudyConfig) -> None:
    cells, tasks = _cells(demo_cfg)
    eff = analysis.effect_agreement(cells, demo_cfg, tasks)
    lev = analysis.level_agreement(cells, demo_cfg, tasks)
    joint = analysis.primary_agreement(eff, lev, demo_cfg, tasks).set_index("arm_id")
    planned = len(analysis.confirmatory_family(demo_cfg))
    assert planned > 1

    row = joint.loc["baseline"]
    assert row["family"].startswith("confirmatory")
    assert row["joint_p_holm"] == pytest.approx(min(1.0, planned * row["joint_p"]))
    assert eff.set_index("arm_id").loc["baseline", "family"] == analysis.JOINT_COMPONENT


def test_effect_only_primary_test_ignores_the_levels(demo_cfg: StudyConfig) -> None:
    cfg = demo_cfg.model_copy(
        update={"analysis": demo_cfg.analysis.model_copy(update={"primary_test": "effect"})}
    )
    cells, tasks = _cells(cfg)
    eff = analysis.effect_agreement(cells, cfg, tasks)
    joint = analysis.primary_agreement(eff, analysis.level_agreement(cells, cfg, tasks), cfg, tasks)
    planned = len(analysis.confirmatory_family(cfg))
    e, j = eff.set_index("arm_id").loc["baseline"], joint.set_index("arm_id").loc["baseline"]
    assert e["family"] == "confirmatory (Holm)"
    assert e["tost_p_holm"] == pytest.approx(min(1.0, planned * e["tost_p"]))
    assert j["components"] == "effect" and j["joint_p"] == pytest.approx(e["tost_p"])


def test_joint_test_needs_the_levels_as_well_as_the_effect(demo_cfg: StudyConfig) -> None:
    # The thesis example: a model can match the gap (90% vs 60% against 60% vs 30%) while missing
    # both levels. The effect component passes, the levels fail, so the study is not reproduced.
    tasks = build_tasks(demo_cfg)
    family = analysis.confirmatory_family(demo_cfg)
    keys = [(m, t, a) for m, t, a in family if t == "framing"]
    eff = pd.DataFrame(
        [{"model_id": m, "task_id": t, "arm_id": a, "tost_p": 0.001} for m, t, a in keys]
    )

    def levels(p_gain: float, p_loss: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"model_id": m, "task_id": t, "arm_id": a, "condition": c, "tost_p": p}
                for m, t, a in keys
                for c, p in (("gain", p_gain), ("loss", p_loss))
            ]
        )

    wrong_levels = analysis.primary_agreement(eff, levels(0.9, 0.8), demo_cfg, tasks)
    assert (wrong_levels["joint_p"] == 0.9).all() and not wrong_levels["reproduces"].any()
    assert set(wrong_levels["components"]) == {"effect, level_gain, level_loss"}

    all_match = analysis.primary_agreement(eff, levels(0.002, 0.003), demo_cfg, tasks)
    assert (all_match["joint_p"] == 0.003).all() and all_match["reproduces"].all()
    m = len(family)
    assert (all_match["joint_p_holm"] <= min(1.0, m * 0.003) + 1e-12).all()

    sealed = analysis.primary_agreement(eff, levels(0.002, float("nan")), demo_cfg, tasks)
    assert sealed["joint_p"].isna().all() and (sealed["missing"] == "level_loss").all()
    assert not sealed["reproduces"].any()


def test_secondary_p_values_are_labelled(demo_cfg: StudyConfig) -> None:
    from silicon_subjects import evaluation

    assert evaluation.p_value_status("population_forecast") == evaluation.NOT_INFERENTIAL
    assert evaluation.p_value_status("choice") == evaluation.EXPLORATORY
    assert evaluation.p_value_status("persona_forecast") == evaluation.EXPLORATORY
    cells, tasks = _cells(demo_cfg)
    level = analysis.level_agreement(cells, demo_cfg, tasks)
    para = analysis.paraphrase_sensitivity(cells, demo_cfg, tasks)
    for table in (level, para):
        assert set(table["p_values"]) == {evaluation.EXPLORATORY}
    df = analysis.load_trials(_records(demo_cfg), tasks)
    people = personas.all_personas(personas.load_sets(demo_cfg))
    conds, effects = evaluation.method_evaluation(df, demo_cfg, tasks, people)
    assert set(conds["p_values"]) == set(effects["p_values"]) == {evaluation.EXPLORATORY}


def test_doctor_blocks_development_studies_in_the_confirmatory_family(
    demo_cfg: StudyConfig,
) -> None:
    from silicon_subjects import doctor

    fails = {c.item: c.detail for c in doctor.run(demo_cfg) if c.level == "FAIL"}
    assert "development study in the confirmatory family" in fails["task framing"]
    fixed = demo_cfg.model_copy(
        update={
            "tasks": [
                t.model_copy(update={"confirmatory_test": False}) if t.id == "framing" else t
                for t in demo_cfg.tasks
            ]
        }
    )
    still = {c.item: c.detail for c in doctor.run(fixed) if c.level == "FAIL"}
    assert "development study" not in still.get("task framing", "")


def test_directional_p_is_one_sided_in_the_required_direction() -> None:
    assert stats.directional_p(0.3, 0.1, 0.5) == pytest.approx(0.0013499, rel=1e-4)
    assert stats.directional_p(0.3, 0.1, -0.5) == pytest.approx(1 - 0.0013499, rel=1e-6)
    assert stats.directional_p(0.0, 0.1, 0.5) == pytest.approx(0.5)
    assert stats.directional_p(0.3, 0.1, 0.0) == 1.0  # no human direction to match
    assert math.isnan(stats.directional_p(0.3, 0.0, 0.5))
    se, alpha = math.sqrt(2 / 900), 0.05 / 6
    need = stats.directional_min_effect(se, alpha)
    assert stats.directional_power(se, need, alpha) == pytest.approx(0.8)
    assert stats.directional_power(se, 0.0, alpha) == pytest.approx(alpha)


def test_requiring_an_effect_adds_a_component_that_a_flat_model_fails(
    demo_cfg: StudyConfig,
) -> None:
    cfg = demo_cfg.model_copy(
        update={"analysis": demo_cfg.analysis.model_copy(update={"primary_requires_effect": True})}
    )
    tasks = build_tasks(cfg)
    keys = [(m, t, a) for m, t, a in analysis.confirmatory_family(cfg) if t == "framing"]
    levels = pd.DataFrame(
        [
            {"model_id": m, "task_id": t, "arm_id": a, "condition": c, "tost_p": 0.001}
            for m, t, a in keys
            for c in ("gain", "loss")
        ]
    )

    def effects(present_p: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model_id": m,
                    "task_id": t,
                    "arm_id": a,
                    "tost_p": 0.001,
                    "effect_present_p": present_p,
                }
                for m, t, a in keys
            ]
        )

    # Equivalent on the effect and on both levels, but its own effect is not distinguishable
    # from zero: not reproduced.
    flat = analysis.primary_agreement(effects(0.4), levels, cfg, tasks)
    assert set(flat["components"]) == {"effect, effect_present, level_gain, level_loss"}
    assert (flat["joint_p"] == 0.4).all() and not flat["reproduces"].any()
    assert analysis.primary_agreement(effects(0.0005), levels, cfg, tasks)["reproduces"].all()
    # Without the requirement the same flat model passes: this is the gap being closed.
    off = analysis.primary_agreement(effects(0.4), levels, demo_cfg, tasks)
    assert set(off["components"]) == {"effect, level_gain, level_loss"}
    assert off["reproduces"].all()

    # The component is computed from the model's own cells and the human direction.
    cells, tasks = _cells(cfg)
    eff = analysis.effect_agreement(cells, cfg, tasks).set_index(["model_id", "task_id", "arm_id"])
    present = [key for key in keys if key in eff.index]
    assert present
    for key in present:
        e, c = eff.loc[key], cells[(cells["prompt_variant"] == "original")]
        c = c[(c["model_id"] == key[0]) & (c["task_id"] == key[1]) & (c["arm_id"] == key[2])]
        c = c.set_index("condition")
        se = math.hypot(c.loc[e["focal"], "h_se"], c.loc[e["reference"], "h_se"])
        assert e["effect_present_p"] == pytest.approx(
            stats.directional_p(e["model_h"], se, e["human_h"])
        )


def test_power_under_bias_starts_at_the_exact_model_power_and_falls(tmp_path: Path) -> None:
    cfg = _study_cfg(tmp_path)
    assert cfg.analysis.primary_requires_effect
    tasks = build_tasks(cfg)
    for t in cfg.enabled_tasks:
        if t.split == "evaluation":
            tasks[t.id].benchmark = analysis.seal(tasks[t.id].benchmark)
    exact = analysis.effect_power_table(cfg, tasks).set_index(["task_id", "arm_id"])
    bias = analysis.primary_power_under_bias(cfg, tasks)
    family = {(t, a) for _, t, a in analysis.confirmatory_family(cfg)}
    assert {(r.task_id, r.arm_id) for r in bias.itertuples()} == family
    alpha = cfg.analysis.alpha / len(analysis.confirmatory_family(cfg))
    for key, g in bias.groupby(["task_id", "arm_id"]):
        g = g.sort_values("gap_as_fraction_of_margin")
        assert list(g["gap_as_fraction_of_margin"]) == list(analysis.BIAS_GRID)
        first, last = g.iloc[0], g.iloc[-1]
        assert first["effect_power_at_holm_strictest"] == pytest.approx(
            exact.loc[key, "power_at_holm_strictest"]
        )
        for col in ("effect_power_at_holm_strictest", "all_levels_power_at_holm_strictest"):
            assert g[col].is_monotonic_decreasing
            # A model exactly on the margin is not equivalent: passing is a false positive.
            assert last[col] <= alpha + 1e-9
        half = g[g["gap_as_fraction_of_margin"] == 0.5].iloc[0]
        assert half["effect_gap_cohens_h"] == pytest.approx(0.10)
        assert half["level_gap_proportion"] == pytest.approx(0.05)
        assert half["effect_power_at_holm_strictest"] < 0.5
        # 900 per condition and three models: about h = 0.15 for the effect-present component.
        assert exact.loc[key, "effect_present_min_h_for_80pct_at_holm_strictest"] == pytest.approx(
            stats.directional_min_effect(math.sqrt(2 / 900), alpha)
        )


def test_primary_result_is_recomputed_under_both_treatments_of_refusals(
    demo_cfg: StudyConfig,
) -> None:
    tasks = build_tasks(demo_cfg)
    df = analysis.load_trials(_records(demo_cfg), tasks)

    def primary(frame: pd.DataFrame) -> pd.DataFrame:
        cells = analysis.cell_summary(frame, demo_cfg, tasks)
        table = analysis.primary_agreement(
            analysis.effect_agreement(cells, demo_cfg, tasks),
            analysis.level_agreement(cells, demo_cfg, tasks),
            demo_cfg,
            tasks,
        )
        return analysis.refusal_robustness(frame, table, demo_cfg, tasks)

    clean = primary(df)
    assert (clean["refusals"].str.split("/").str[0] == "0").all()
    assert clean["robust_to_refusals"].all()
    assert (clean["reproduces_refusals_as_0"] == clean["reproduces"]).all()
    assert (clean["reproduces_refusals_as_1"] == clean["reproduces"]).all()

    # Turn 30 valid responses of one primary cell into refusals.
    model_id, task_id, arm_id = analysis.confirmatory_family(demo_cfg)[0]
    cell = (
        (df["model_id"] == model_id)
        & (df["task_id"] == task_id)
        & (df["arm_id"] == arm_id)
        & (df["prompt_variant"] == "original")
        & (df["condition"] == tasks[task_id].contrast[0])
        & (df["status"] == "valid")
    )
    refused = df.copy()
    hit = refused[cell].index[:30]
    refused.loc[hit, "status"] = "refusal_ooc"
    refused.loc[hit, "y"] = None
    out = primary(refused).set_index(["model_id", "task_id", "arm_id"])
    row = out.loc[(model_id, task_id, arm_id)]
    n_cells = len(tasks[task_id].conditions())
    used = int((cell).sum()) * n_cells
    assert row["refusals"] == f"30/{used}"

    # Each bound equals the primary test run on data where the refusals are that outcome.
    for bound in (0, 1):
        imputed = refused.copy()
        imputed.loc[hit, "status"] = "valid"
        imputed.loc[hit, "y"] = bound
        expected = primary(imputed).set_index(["model_id", "task_id", "arm_id"])
        assert (
            row[f"reproduces_refusals_as_{bound}"]
            == expected.loc[(model_id, task_id, arm_id), "reproduces"]
        )
    assert row["robust_to_refusals"] == (
        row["reproduces_refusals_as_0"] == row["reproduces"] == row["reproduces_refusals_as_1"]
    )
    # Other claims have no refusals and are untouched.
    others = out.drop(index=(model_id, task_id, arm_id))
    assert others["robust_to_refusals"].all()
