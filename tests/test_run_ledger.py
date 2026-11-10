from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from silicon_subjects import analysis, prereg
from silicon_subjects.cli import app
from silicon_subjects.config import StudyConfig


def cli(cfg_path: Path, *args: str) -> Any:
    return CliRunner().invoke(app, [*args, "-c", str(cfg_path)])


def _frozen(cfg: StudyConfig) -> Path:
    path = cfg.root / "configs" / "demo.yaml"
    assert cli(path, "freeze").exit_code == 0
    return path


def test_second_confirmatory_run_needs_replication_flag(demo_cfg: StudyConfig) -> None:
    path = _frozen(demo_cfg)
    assert cli(path, "run", "--run-id", "first", "--limit", "5").exit_code == 0
    res = cli(path, "run", "--run-id", "second", "--limit", "5")
    assert res.exit_code == 2 and "already has its primary confirmatory run ('first'" in res.output
    assert not (demo_cfg.root / "results" / "second").exists()

    assert cli(path, "run", "--run-id", "first", "--limit", "5").exit_code == 0
    res = cli(path, "run", "--run-id", "second", "--limit", "5", "--replication")
    assert res.exit_code == 0, res.output
    ledger = prereg.read_ledger(demo_cfg)
    assert [(e["run_id"], e["role"], e["primary_run"]) for e in ledger] == [
        ("first", "primary", "first"),
        ("second", "replication", "first"),
    ]
    manifest = json.loads((demo_cfg.root / "results" / "second" / "manifest.json").read_text())
    assert manifest["confirmatory_role"] == {"role": "replication", "primary_run": "first"}


def test_registered_run_cannot_be_recollected_after_deletion(demo_cfg: StudyConfig) -> None:
    path = _frozen(demo_cfg)
    assert cli(path, "run", "--run-id", "first", "--limit", "3").exit_code == 0
    shutil.rmtree(demo_cfg.root / "results" / "first")
    res = cli(path, "run", "--run-id", "first", "--limit", "3")
    assert res.exit_code == 2 and "already registered" in res.output


def test_exploratory_runs_are_not_registered(demo_cfg: StudyConfig) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    assert cli(path, "run", "--exploratory", "--run-id", "x", "--limit", "3").exit_code == 0
    assert prereg.read_ledger(demo_cfg) == []
    res = cli(path, "run", "--exploratory", "--run-id", "y", "--limit", "3", "--replication")
    assert res.exit_code == 2 and "confirmatory runs only" in res.output


def test_replication_is_labelled_in_the_analysis(demo_cfg: StudyConfig) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(
        path.read_text().replace("bootstrap_iterations: 200", "bootstrap_iterations: 50")
    )
    _frozen(demo_cfg)
    assert cli(path, "run", "--run-id", "first").exit_code == 0
    assert cli(path, "run", "--run-id", "second", "--replication").exit_code == 0
    primary = analysis.analyze_run(demo_cfg.root / "results" / "first")
    assert primary.provenance["analysis_status"] == "confirmatory"
    assert primary.provenance["confirmatory_role"]["role"] == "primary"
    rep = analysis.analyze_run(demo_cfg.root / "results" / "second")
    assert rep.provenance["analysis_status"] == "confirmatory replication (primary run: first)"
    assert "Confirmatory run role: **replication**" in rep.report.read_text()
