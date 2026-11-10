from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from silicon_subjects import analysis, design, personas, runstore
from silicon_subjects.cli import app
from silicon_subjects.config import StudyConfig
from silicon_subjects.providers.mock import MockProvider
from silicon_subjects.runner import Runner
from silicon_subjects.tasks import build_tasks


def cli(cfg_path: Path, *args: str) -> Any:
    return CliRunner().invoke(app, [*args, "-c", str(cfg_path)])


def fast(cfg: StudyConfig) -> Path:
    path = cfg.root / "configs" / "demo.yaml"
    path.write_text(
        path.read_text().replace("bootstrap_iterations: 5000", "bootstrap_iterations: 100")
    )
    return path


@pytest.fixture()
def collected(demo_cfg: StudyConfig) -> Path:
    cfg_path = fast(demo_cfg)
    for args in (["freeze"], ["run", "--run-id", "r"]):
        res = cli(cfg_path, *args)
        assert res.exit_code == 0, res.output
    return cfg_path


def run_dir(cfg_path: Path) -> Path:
    return cfg_path.parent.parent / "results" / "r"


def drop_one_result(run: Path) -> None:
    lines = (run / "trials.jsonl").read_text().splitlines()
    (run / "trials.jsonl").write_text("\n".join(lines[:-1]) + "\n")


def tree(d: Path) -> dict[str, str]:
    return {
        str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(d.rglob("*"))
        if p.is_file()
    }


def test_missing_plan_file_is_refused(collected: Path) -> None:
    run = run_dir(collected)
    drop_one_result(run)
    (run / "plan.jsonl").unlink()
    before = tree(run)
    res = cli(collected, "analyze", "r")
    assert res.exit_code == 2 and "plan" in res.output
    assert tree(run) == before and not (run / "report.md").exists()


def test_missing_plan_metadata_is_refused(collected: Path) -> None:
    run = run_dir(collected)
    m = json.loads((run / "manifest.json").read_text())
    m.pop("plan")
    (run / "manifest.json").write_text(json.dumps(m))
    res = cli(collected, "analyze", "r")
    assert res.exit_code == 2 and "plan" in res.output
    with pytest.raises(analysis.AnalysisProvenanceError, match="plan"):
        analysis.analyze_run(run, partial=True)


def test_explicit_unverified_analysis_is_labelled_and_separate(collected: Path) -> None:
    run = run_dir(collected)
    drop_one_result(run)
    (run / "plan.jsonl").unlink()
    res = cli(collected, "analyze", "r", "--reanalyze")
    assert res.exit_code == 0, res.output
    (out,) = (run / "unverified").iterdir()
    prov = json.loads((out / "analysis_provenance.json").read_text())
    assert prov["analysis_status"].startswith("unverified")
    assert prov["completeness"]["checked"] is False
    assert "Analysis status: **unverified" in (out / "report.md").read_text()
    assert not (run / "report.md").exists() and not (run / "reanalysis").exists()


def make_runner(cfg: StudyConfig) -> tuple[Runner, list[Any], dict[str, int]]:
    tasks = build_tasks(cfg)
    sets = personas.load_sets(cfg)
    calls = {"n": 0}

    class Counting(MockProvider):
        async def complete(self, *a: Any, **k: Any) -> Any:
            calls["n"] += 1
            return await super().complete(*a, **k)

    r = Runner(
        cfg,
        tasks,
        personas.all_personas(sets),
        cfg.root / "results" / "run",
        prereg_hash=None,
        confirmatory=False,
        providers={m.id: Counting(m) for m in cfg.models},
    )
    return r, design.build_manifest(cfg, tasks, sets), calls


@pytest.mark.parametrize("damage", ["condition", "unexpected"])
def test_resume_rejects_records_that_contradict_the_plan(
    demo_cfg: StudyConfig, damage: str
) -> None:
    r, trials, _ = make_runner(demo_cfg)
    asyncio.run(r.run(trials))
    p = demo_cfg.root / "results" / "run" / "trials.jsonl"
    lines = p.read_text().splitlines()
    rec = json.loads(lines[0])
    if damage == "condition":
        rec["condition"] = "WRONG"
        lines[0] = json.dumps(rec)
    else:
        rec["trial_id"] = "not-in-plan"
        lines.append(json.dumps(rec))
    p.write_text("\n".join(lines) + "\n")
    before = p.read_bytes()

    r, trials, calls = make_runner(demo_cfg)
    with pytest.raises(runstore.ResultsCorruptError, match="contradict the run's plan"):
        asyncio.run(r.run(trials))
    assert calls["n"] == 0 and p.read_bytes() == before


def test_cli_resume_with_contradicting_record_exits_nonzero(demo_cfg: StudyConfig) -> None:
    cfg_path = fast(demo_cfg)
    assert cli(cfg_path, "run", "--exploratory", "--run-id", "x", "--limit", "3").exit_code == 0
    p = demo_cfg.root / "results" / "x" / "trials.jsonl"
    rec = json.loads(p.read_text().splitlines()[0])
    rec["condition"] = "WRONG"
    p.write_text(json.dumps(rec) + "\n" + "\n".join(p.read_text().splitlines()[1:]) + "\n")
    res = cli(cfg_path, "run", "--exploratory", "--run-id", "x")
    assert res.exit_code == 2 and "mismatched" in res.output


def test_snapshot_is_self_contained_with_absolute_paths(
    demo_cfg: StudyConfig, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    cfg_path = fast(demo_cfg)
    outside = tmp_path_factory.mktemp("outside_study_root") / "obedience.yaml"
    shutil.copy(demo_cfg.root / "data" / "benchmarks" / "obedience_burger2009.yaml", outside)
    framing = (demo_cfg.root / "data" / "stimuli" / "framing.yaml").resolve()
    text = cfg_path.read_text()
    text = text.replace(
        "benchmark: data/benchmarks/obedience_burger2009.yaml", f"benchmark: {outside}"
    )
    text = text.replace("stimuli: data/stimuli/framing.yaml", f"stimuli: {framing}")
    cfg_path.write_text(text)
    for args in (["freeze"], ["run", "--run-id", "r"]):
        res = cli(cfg_path, *args)
        assert res.exit_code == 0, res.output
    original = analysis.analyze_run(run_dir(cfg_path))
    assert original.provenance["analysis_status"] == "confirmatory"

    moved = tmp_path / "archive" / "r"
    shutil.move(str(run_dir(cfg_path)), moved)
    outside.unlink()
    framing.unlink()
    shutil.rmtree(demo_cfg.root / "data")
    again = analysis.analyze_run(moved)
    assert again.provenance["analysis_status"] == "confirmatory"
    pd.testing.assert_frame_equal(
        again.result.tables["level_agreement"], original.result.tables["level_agreement"]
    )

    mapping = json.loads((moved / "inputs" / "paths.json").read_text())
    assert not any(Path(v).is_absolute() or ".." in Path(v).parts for v in mapping.values())
    assert mapping["obedience.benchmark"].startswith("external/")
    (moved / "inputs" / mapping["obedience.benchmark"]).unlink()
    with pytest.raises(analysis.AnalysisProvenanceError, match=r"obedience\.benchmark"):
        analysis.analyze_run(moved)


def test_relative_snapshot_still_works_after_move(collected: Path, tmp_path: Path) -> None:
    moved = tmp_path / "moved" / "r"
    shutil.move(str(run_dir(collected)), moved)
    shutil.rmtree(collected.parent.parent / "data")
    assert analysis.analyze_run(moved).provenance["analysis_status"] == "confirmatory"


def test_cell_without_valid_responses_is_reported_unavailable(collected: Path) -> None:
    run = run_dir(collected)
    target = ("opus55", "framing", "paraphrase", "gain", "para1")
    lines = []
    for line in (run / "trials.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if (
            rec["model_id"],
            rec["task_id"],
            rec["arm_id"],
            rec["condition"],
            rec["prompt_variant"],
        ) == target:
            rec["status"], rec["outcome"] = "refusal_platform", {}
        lines.append(json.dumps(rec))
    (run / "trials.jsonl").write_text("\n".join(lines) + "\n")

    out = analysis.analyze_run(run)
    cells = out.result.tables["cells"].set_index(
        ["model_id", "task_id", "arm_id", "condition", "prompt_variant"]
    )
    assert cells.loc[target, "n_valid"] == 0 and cells.loc[target, "n_refusal_platform"] == 50
    para = out.result.tables["paraphrase_sensitivity"].set_index(
        ["model_id", "task_id", "condition"]
    )
    row = para.loc[("opus55", "framing", "gain")]
    assert pd.isna(row["p_value"]) and not bool(row["available"])
    assert "para1" in row["reason"] and "no valid responses" in row["reason"]
    assert "50 platform refusals" in row["reason"]
    other = para.loc[("opus55", "framing", "loss")]
    assert bool(other["available"]) and pd.notna(other["p_value"])
    assert out.report.exists()


def test_chi2_ignores_nothing_silently() -> None:
    from silicon_subjects import stats

    chi2, p = stats.chi2_homogeneity([(5, 10), (0, 0), (7, 10)])
    assert pd.isna(chi2) and pd.isna(p)
