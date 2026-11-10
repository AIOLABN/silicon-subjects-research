from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

import silicon_subjects
from silicon_subjects import prereg
from silicon_subjects.cli import app
from silicon_subjects.config import StudyConfig

FAST = "bootstrap_iterations: 200"


@pytest.fixture()
def collected(demo_cfg: StudyConfig) -> Path:
    cfg_path = demo_cfg.root / "configs" / "demo.yaml"
    cfg_path.write_text(cfg_path.read_text().replace("bootstrap_iterations: 5000", FAST))
    for args in (["freeze"], ["run", "--run-id", "r"], ["analyze", "r"]):
        res = CliRunner().invoke(app, [*args, "-c", str(cfg_path)])
        assert res.exit_code == 0, res.output
    return cfg_path


def run_dir(cfg_path: Path) -> Path:
    return cfg_path.parent.parent / "results" / "r"


def analyze(cfg_path: Path, *extra: str) -> object:
    return CliRunner().invoke(app, ["analyze", "r", "-c", str(cfg_path), *extra])


def equivalents(table: Path) -> int:
    import pandas as pd

    return int(pd.read_csv(table)["equivalent"].sum())


def files(d: Path) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def test_unchanged_inputs_reproduce_the_result(collected: Path) -> None:
    out = run_dir(collected)
    before = files(out / "analysis")
    res = analyze(collected)
    assert res.exit_code == 0, res.output
    assert files(out / "analysis") == before


def test_edited_margin_does_not_change_default_analysis(collected: Path) -> None:
    out = run_dir(collected)
    primary = out / "analysis" / "primary_effect_agreement.csv"
    original = primary.read_bytes()
    collected.write_text(
        collected.read_text().replace("margin_cohens_h: 0.20", "margin_cohens_h: 5.0")
    )
    res = analyze(collected)
    assert res.exit_code == 0, res.output
    assert primary.read_bytes() == original


def test_edited_benchmark_does_not_change_default_analysis(collected: Path) -> None:
    out = run_dir(collected)
    level = out / "analysis" / "level_agreement.csv"
    original = level.read_bytes()
    bench = collected.parent.parent / "data" / "benchmarks" / "obedience_burger2009.yaml"
    bench.write_text(bench.read_text().replace("{k: 28, n: 40}", "{k: 39, n: 40}"))
    assert analyze(collected).exit_code == 0
    assert level.read_bytes() == original


def test_explicit_reanalysis_uses_changed_inputs_and_is_exploratory(collected: Path) -> None:
    out = run_dir(collected)
    original_report = (out / "report.md").read_bytes()
    n_orig = equivalents(out / "analysis" / "primary_effect_agreement.csv")
    collected.write_text(
        collected.read_text().replace("margin_cohens_h: 0.20", "margin_cohens_h: 5.0")
    )
    res = analyze(collected, "--reanalyze")
    assert res.exit_code == 0, res.output
    (re_dir,) = (out / "reanalysis").iterdir()
    assert equivalents(re_dir / "primary_effect_agreement.csv") > n_orig
    report = (re_dir / "report.md").read_text()
    assert "Analysis status: **exploratory reanalysis**" in report
    assert "config.analysis" in report
    prov = json.loads((re_dir / "analysis_provenance.json").read_text())
    assert prov["analysis_status"] == "exploratory reanalysis"
    assert "config.analysis" in prov["deviations"]
    assert (out / "report.md").read_bytes() == original_report
    assert (out / "analysis_provenance.json").exists()


def test_changed_analysis_code_cannot_stay_confirmatory(
    collected: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = run_dir(collected)
    before = files(out)
    copy = tmp_path / "pkg" / "silicon_subjects"
    shutil.copytree(
        Path(silicon_subjects.__file__).parent, copy, ignore=shutil.ignore_patterns("__pycache__")
    )
    monkeypatch.setattr(prereg, "SOURCE_ROOT", copy)
    f = copy / "stats.py"
    f.write_text(f.read_text().replace("z = stats.norm.ppf(1 - alpha / 2)", "z = 1.0", 1))
    res = analyze(collected)
    assert res.exit_code != 0
    assert "code:silicon_subjects/stats.py" in res.output and "--reanalyze" in res.output
    assert files(out) == before


def test_report_separates_collection_and_analysis_status(collected: Path) -> None:
    report = (run_dir(collected) / "report.md").read_text()
    assert "Collection status: **confirmatory**" in report
    assert "Analysis status: **confirmatory**" in report


def test_exploratory_collection_is_not_confirmatory_analysis(demo_cfg: StudyConfig) -> None:
    cfg_path = demo_cfg.root / "configs" / "demo.yaml"
    cfg_path.write_text(cfg_path.read_text().replace("bootstrap_iterations: 5000", FAST))
    for args in (["run", "--run-id", "r", "--exploratory"], ["analyze", "r"]):
        assert CliRunner().invoke(app, [*args, "-c", str(cfg_path)]).exit_code == 0
    report = (run_dir(cfg_path) / "report.md").read_text()
    assert "Collection status: **exploratory**" in report
    assert "Analysis status: **exploratory**" in report


def test_corrupted_snapshot_fails_clearly(collected: Path) -> None:
    out = run_dir(collected)
    snap = out / "inputs" / "data" / "stimuli" / "framing.yaml"
    snap.write_text(snap.read_text().replace("600", "700", 1))
    before = files(out)
    res = analyze(collected)
    assert res.exit_code != 0 and "framing.stimuli" in res.output
    assert files(out) == before


def test_missing_snapshot_fails_clearly(collected: Path) -> None:
    out = run_dir(collected)
    shutil.rmtree(out / "inputs")
    res = analyze(collected)
    assert res.exit_code != 0 and "snapshot" in res.output


def test_legacy_run_is_not_confirmatory(collected: Path) -> None:
    out = run_dir(collected)
    m = json.loads((out / "manifest.json").read_text())
    m.pop("experiment_identity")
    (out / "manifest.json").write_text(json.dumps(m))
    shutil.rmtree(out / "inputs")
    assert analyze(collected).exit_code != 0
    res = analyze(collected, "--reanalyze")
    assert res.exit_code == 0, res.output

    (re_dir,) = (out / "unverified").iterdir()
    assert (
        "Analysis status: **unverified exploratory reanalysis" in (re_dir / "report.md").read_text()
    )


def test_python_entry_point_is_protected(
    collected: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from silicon_subjects import analysis

    copy = tmp_path / "pkg" / "silicon_subjects"
    shutil.copytree(
        Path(silicon_subjects.__file__).parent, copy, ignore=shutil.ignore_patterns("__pycache__")
    )
    monkeypatch.setattr(prereg, "SOURCE_ROOT", copy)
    (copy / "analysis.py").write_text((copy / "analysis.py").read_text() + "\n# edit\n")
    with pytest.raises(analysis.AnalysisProvenanceError, match=r"analysis\.py"):
        analysis.analyze_run(run_dir(collected))
    assert not hasattr(analysis, "analyze")
