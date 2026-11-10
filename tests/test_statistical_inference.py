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


def test_study_primary_family_and_its_power(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    for sub in ("configs", "data"):
        shutil.copytree(repo / sub, tmp_path / sub, ignore=shutil.ignore_patterns("raw"))
    for name, n in (("holdout_hauser4.yaml", (3968, 3955)),):
        path = tmp_path / "data" / "benchmarks" / name
        if not path.exists():
            stub = yaml.safe_load(
                (repo / "data/benchmarks/templates/holdout_hauser4.yaml").read_text()
            )
            stub["conditions"] = {
                "greater_good": {"k": None, "n": n[0]},
                "side_effect": {"k": None, "n": n[1]},
            }
            path.write_text(yaml.safe_dump(stub))
    cfg = load_config(tmp_path / "configs" / "study.yaml")
    family = analysis.confirmatory_family(cfg)
    assert sorted({(t, a) for _, t, a in family}) == [
        ("framing", "baseline"),
        ("holdout", "baseline"),
    ]
    tasks = build_tasks(cfg)
    for tid in [t.id for t in cfg.enabled_tasks if t.split == "evaluation"]:
        tasks[tid].benchmark = analysis.seal(tasks[tid].benchmark)
    power = analysis.effect_power_table(cfg, tasks).set_index(["task_id", "arm_id"])
    for key in (("framing", "baseline"), ("holdout", "baseline")):
        assert power.loc[key, "in_confirmatory_family"]
        assert power.loc[key, "power_at_holm_strictest"] >= 0.8
    assert not power.loc[("obedience", "baseline"), "in_confirmatory_family"]


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
    eff = analysis.effect_agreement(cells, demo_cfg, tasks).set_index("arm_id")
    planned = len(analysis.confirmatory_family(demo_cfg))
    assert planned > 1

    row = eff.loc["baseline"]
    assert row["family"].startswith("confirmatory")
    assert row["tost_p_holm"] == pytest.approx(min(1.0, planned * row["tost_p"]))


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
