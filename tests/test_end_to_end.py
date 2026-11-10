import json

from silicon_subjects import analysis, design, personas, prereg
from silicon_subjects.config import StudyConfig
from silicon_subjects.runner import Runner
from silicon_subjects.tasks import build_tasks


async def test_mock_run_resume_and_analysis(demo_cfg: StudyConfig) -> None:
    prereg.freeze(demo_cfg)
    status = prereg.verify(demo_cfg)
    tasks = build_tasks(demo_cfg)
    ps = personas.load_sets(demo_cfg)
    trials = design.build_manifest(demo_cfg, tasks, ps)
    out = demo_cfg.root / "results" / "t"

    first = await Runner(
        demo_cfg,
        tasks,
        personas.all_personas(ps),
        out,
        prereg_hash=status.expected,
        confirmatory=True,
    ).run(trials, limit=300)
    rest = await Runner(
        demo_cfg,
        tasks,
        personas.all_personas(ps),
        out,
        prereg_hash=status.expected,
        confirmatory=True,
    ).run(trials)
    assert first.completed == 300 and first.pending == len(trials) - 300 and first.ok
    assert rest.previously_completed == 300 and rest.completed == len(trials) - 300
    assert rest.study_complete and rest.ok
    ids = [json.loads(line)["trial_id"] for line in (out / "trials.jsonl").open()]
    assert len(ids) == len(set(ids)) == len(trials)

    out_run = analysis.analyze_run(out)
    primary = out_run.result.tables["primary_agreement"]
    assert set(primary["task_id"]) == {"framing", "obedience", "holdout"}
    family = primary[primary["family"].str.startswith("confirmatory")]
    assert set(family["task_id"]) == {"framing", "holdout"} and set(family["arm_id"]) == {
        "baseline"
    }
    assert family["joint_p_holm"].notna().all()
    assert set(family["components"]) == {
        "effect, level_gain, level_loss",
        "effect, level_greater_good, level_side_effect",
    }
    effects = out_run.result.tables["primary_effect_agreement"]
    components = effects[effects["family"] == analysis.JOINT_COMPONENT]
    assert set(components["task_id"]) == {"framing", "holdout"}
    descriptive = effects[effects["task_id"] == "obedience"]
    assert descriptive["tost_p_holm"].isna().all() and not descriptive["equivalent"].any()
    assert descriptive["model_rd_inside_human_ci"].notna().all()
    assert not primary[primary["task_id"] == "obedience"]["reproduces"].any()
    assert out_run.report.exists()
    assert out_run.provenance["analysis_status"] == "confirmatory"
