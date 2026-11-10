from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from silicon_subjects import runstore
from silicon_subjects.cli import app
from silicon_subjects.config import StudyConfig
from silicon_subjects.providers.mock import MockProvider


def cli(cfg_path: Path, *args: str) -> Any:
    return CliRunner().invoke(app, [*args, "-c", str(cfg_path)])


@pytest.fixture()
def crashed(demo_cfg: StudyConfig) -> Path:
    path = demo_cfg.root / "configs" / "demo.yaml"
    assert cli(path, "run", "--exploratory", "--run-id", "r", "--limit", "6").exit_code == 0
    p = demo_cfg.root / "results" / "r" / "trials.jsonl"
    p.write_bytes(p.read_bytes()[:-40])
    return path


def test_recover_repairs_without_calling_any_model(
    crashed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def forbidden(*a: object, **k: object) -> object:
        raise AssertionError("recover must not call a model")

    monkeypatch.setattr(MockProvider, "complete", forbidden)
    res = cli(crashed, "recover", "r")
    assert res.exit_code == 0, res.output
    run = crashed.parent.parent / "results" / "r"
    assert len(runstore.read_records(run / "trials.jsonl")) == 5
    assert "kept 5 records" in res.output and (run / "recovery.jsonl").exists()
    assert cli(crashed, "recover", "r").exit_code == 0


def test_recover_works_after_the_experiment_changed(crashed: Path) -> None:
    stim = crashed.parent.parent / "data" / "stimuli" / "framing.yaml"
    stim.write_text(stim.read_text().replace("600 people", "601 people", 1))
    assert cli(crashed, "recover", "r").exit_code == 0
    run = crashed.parent.parent / "results" / "r"
    assert len(runstore.read_records(run / "trials.jsonl")) == 5


def test_truncation_message_points_to_recover_only_command(crashed: Path) -> None:
    res = cli(crashed, "run", "--exploratory", "--run-id", "r")
    assert res.exit_code == 2 and "sim recover r" in res.output


def test_recover_refuses_interior_corruption(demo_cfg: StudyConfig) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    assert cli(path, "run", "--exploratory", "--run-id", "r", "--limit", "3").exit_code == 0
    p = demo_cfg.root / "results" / "r" / "trials.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0], "{broken", *lines[1:]]) + "\n")
    before = p.read_bytes()
    res = cli(path, "recover", "r")
    assert res.exit_code == 2 and "line 2" in res.output and p.read_bytes() == before


def test_audit_refuses_damaged_results(crashed: Path) -> None:
    res = cli(crashed, "audit", "r")
    assert res.exit_code == 2 and "incomplete record" in res.output


def test_audit_export_is_blinded(demo_cfg: StudyConfig) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    assert cli(path, "run", "--exploratory", "--run-id", "a", "--limit", "40").exit_code == 0
    res = cli(path, "audit", "a", "--per-status", "5")
    assert res.exit_code == 0, res.output
    out = demo_cfg.root / "results" / "a" / "audit"
    items = runstore.read_records(out / "to_code.jsonl", key="audit_id")
    assert items
    for item in items:
        for turn in item["transcript"]:
            assert set(turn) == {"role", "content"}, turn
    keys = runstore.read_records(out / "key.jsonl", key="audit_id")
    sizes: dict[str, int] = {}
    for k in keys:
        sizes[k["status"]] = sizes.get(k["status"], 0) + 1
        assert k["sampling_weight"] == k["stratum_size"] / min(5, k["stratum_size"])
    assert all(n <= 5 for n in sizes.values())
