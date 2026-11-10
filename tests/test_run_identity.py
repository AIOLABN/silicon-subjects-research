from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import silicon_subjects
from silicon_subjects import design, personas, prereg, runner
from silicon_subjects.config import StudyConfig, load_config
from silicon_subjects.providers.base import ChatMessage, Completion
from silicon_subjects.providers.mock import MockProvider
from silicon_subjects.runner import RunIdentityError, Runner
from silicon_subjects.tasks import TrialSpec, build_tasks


class Counting:
    name = "counting"

    def __init__(self, inner: MockProvider) -> None:
        self.inner = inner
        self.calls = 0

    async def complete(
        self, system: str, messages: list[ChatMessage], *, seed: int, **kw: Any
    ) -> Completion:
        self.calls += 1
        return await self.inner.complete(system, messages, seed=seed, **kw)

    async def aclose(self) -> None:
        return None


def providers(cfg: StudyConfig) -> dict[str, Counting]:
    return {m.id: Counting(MockProvider(m)) for m in cfg.models}


def plan(cfg: StudyConfig) -> tuple[Runner, list[TrialSpec], dict[str, Counting]]:
    tasks = build_tasks(cfg)
    sets = personas.load_sets(cfg)
    ps = providers(cfg)
    r = Runner(
        cfg,
        tasks,
        personas.all_personas(sets),
        cfg.root / "results" / "run",
        prereg_hash=None,
        confirmatory=False,
        providers=dict(ps),
    )
    return r, design.build_manifest(cfg, tasks, sets), ps


def go(cfg: StudyConfig, out: str = "run", limit: int | None = None) -> dict[str, Counting]:
    r, trials, ps = plan(cfg)
    r.paths = runner.RunPaths(cfg.root / "results" / out)
    asyncio.run(r.run(trials, limit=limit))
    return ps


def snapshot(d: Path) -> dict[str, str]:
    return {
        str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(d.rglob("*"))
        if p.is_file()
    }


def calls(ps: dict[str, Counting]) -> int:
    return sum(p.calls for p in ps.values())


def reload(cfg: StudyConfig) -> StudyConfig:
    return load_config(cfg.root / "configs" / "demo.yaml")


def ids(d: Path) -> list[str]:
    return [json.loads(line)["trial_id"] for line in (d / "trials.jsonl").open()]


def test_unchanged_interrupted_run_resumes_without_rerunning(demo_cfg: StudyConfig) -> None:
    go(demo_cfg, limit=40)
    out = demo_cfg.root / "results" / "run"
    first = ids(out)
    assert len(first) == 40
    go(demo_cfg)
    after = ids(out)
    assert after[:40] == first
    assert len(after) == len(set(after)) == len(plan(demo_cfg)[1])


def assert_rejected(cfg: StudyConfig, match: str) -> None:
    out = cfg.root / "results" / "run"
    before = snapshot(out)
    r, trials, ps = plan(cfg)
    with pytest.raises(RunIdentityError, match=match) as exc:
        asyncio.run(r.run(trials))
    assert "--run-id" in str(exc.value)
    assert calls(ps) == 0
    assert snapshot(out) == before


def test_changed_model_is_rejected(demo_cfg: StudyConfig) -> None:
    go(demo_cfg, limit=10)
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(path.read_text().replace("model: claude-sonnet-5-5", "model: claude-x-1", 1))
    assert_rejected(reload(demo_cfg), r"sonnet55: claude-sonnet-5-5 -> claude-x-1")


def test_changed_stimulus_is_rejected(demo_cfg: StudyConfig) -> None:
    go(demo_cfg, limit=10)
    stim = demo_cfg.path(demo_cfg.tasks[0].stimuli)
    stim.write_text(stim.read_text().replace("600 people", "601 people", 1))
    assert_rejected(demo_cfg, "framing.stimuli")


def test_changed_persona_is_rejected(demo_cfg: StudyConfig) -> None:
    go(demo_cfg, limit=10)
    f = demo_cfg.path(demo_cfg.personas.file)
    f.write_text(f.read_text().replace("woman", "man", 1))
    assert_rejected(demo_cfg, "personas")


def test_changed_seed_is_rejected(demo_cfg: StudyConfig) -> None:
    go(demo_cfg, limit=10)
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(path.read_text().replace("seed: 20261012", "seed: 1", 1))
    assert_rejected(reload(demo_cfg), "config.seed")


def test_changed_code_is_rejected(
    demo_cfg: StudyConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = tmp_path / "src" / "silicon_subjects"
    shutil.copytree(
        Path(silicon_subjects.__file__).parent, copy, ignore=shutil.ignore_patterns("__pycache__")
    )
    monkeypatch.setattr(prereg, "SOURCE_ROOT", copy)
    go(demo_cfg, limit=10)
    f = copy / "coding.py"
    f.write_text(f.read_text().replace(r'r"\bas an ai\b",', r'r"\bas an ai model\b",', 1))
    assert_rejected(demo_cfg, "code:silicon_subjects/coding.py")


def test_changed_experiment_succeeds_under_new_run_id(demo_cfg: StudyConfig) -> None:
    go(demo_cfg, limit=10)
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(path.read_text().replace("seed: 20261012", "seed: 1", 1))
    cfg = reload(demo_cfg)
    go(cfg, out="run2", limit=10)
    m1 = json.loads((cfg.root / "results" / "run" / "manifest.json").read_text())
    m2 = json.loads((cfg.root / "results" / "run2" / "manifest.json").read_text())
    assert m1["experiment_identity"]["hash"] != m2["experiment_identity"]["hash"]
    assert m1["prereg_hash"] is None and m1["confirmatory"] is False


def test_every_run_has_identity_and_plan(demo_cfg: StudyConfig) -> None:
    go(demo_cfg, limit=1)
    out = demo_cfg.root / "results" / "run"
    m = json.loads((out / "manifest.json").read_text())
    assert m["experiment_identity"]["hash"]
    assert m["plan"]["n_trials"] == len(plan(demo_cfg)[1])
    planned = [json.loads(line) for line in (out / "plan.jsonl").open()]
    assert [p["trial_id"] for p in planned] == [t.trial_id for t in plan(demo_cfg)[1]]


def test_tampered_plan_is_rejected(demo_cfg: StudyConfig) -> None:
    go(demo_cfg, limit=5)
    p = demo_cfg.root / "results" / "run" / "plan.jsonl"
    lines = p.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["condition"] = "loss" if rec["condition"] != "loss" else "gain"
    p.write_text("\n".join([json.dumps(rec), *lines[1:]]) + "\n")
    assert_rejected(demo_cfg, "plan")


def test_legacy_run_directory_is_rejected(demo_cfg: StudyConfig) -> None:
    go(demo_cfg, limit=5)
    m = demo_cfg.root / "results" / "run" / "manifest.json"
    legacy = json.loads(m.read_text())
    for k in ("experiment_identity", "plan"):
        legacy.pop(k)
    m.write_text(json.dumps(legacy))
    assert_rejected(demo_cfg, "legacy")


def test_concurrency_may_change_on_resume(demo_cfg: StudyConfig) -> None:
    go(demo_cfg, limit=5)
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(path.read_text().replace("concurrency: 8", "concurrency: 2", 1))
    ps = go(reload(demo_cfg), limit=5)
    assert calls(ps) > 0 and len(ids(demo_cfg.root / "results" / "run")) == 10


def test_max_attempts_may_not_change_on_resume(demo_cfg: StudyConfig) -> None:
    go(demo_cfg, limit=5)
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(path.read_text().replace("max_attempts: 6", "max_attempts: 9", 1))
    assert_rejected(reload(demo_cfg), "runner")


def test_rejection_happens_before_providers_are_built(
    demo_cfg: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    go(demo_cfg, limit=5)
    stim = demo_cfg.path(demo_cfg.tasks[0].stimuli)
    stim.write_text(stim.read_text().replace("600 people", "601 people", 1))

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("provider built before identity check")

    monkeypatch.setattr(runner, "build_provider", boom)
    tasks = build_tasks(demo_cfg)
    sets = personas.load_sets(demo_cfg)
    r = Runner(
        demo_cfg,
        tasks,
        personas.all_personas(sets),
        demo_cfg.root / "results" / "run",
        prereg_hash=None,
        confirmatory=False,
    )
    with pytest.raises(RunIdentityError):
        asyncio.run(r.run(design.build_manifest(demo_cfg, tasks, sets)))


def test_switching_collection_status_is_rejected(demo_cfg: StudyConfig) -> None:
    go(demo_cfg, limit=5)
    prereg.freeze(demo_cfg)
    r, trials, ps = plan(demo_cfg)
    r.prereg_hash, r.confirmatory = prereg.verify(demo_cfg).expected, True
    with pytest.raises(RunIdentityError, match="confirmatory"):
        asyncio.run(r.run(trials))
    assert calls(ps) == 0


def test_cli_rejects_changed_resume_with_instructions(demo_cfg: StudyConfig) -> None:
    from typer.testing import CliRunner

    from silicon_subjects.cli import app

    cfg_path = str(demo_cfg.root / "configs" / "demo.yaml")
    ok = CliRunner().invoke(
        app, ["run", "-c", cfg_path, "--exploratory", "--run-id", "x", "--limit", "3"]
    )
    assert ok.exit_code == 0, ok.output
    stim = demo_cfg.path(demo_cfg.tasks[0].stimuli)
    stim.write_text(stim.read_text().replace("600 people", "601 people", 1))
    bad = CliRunner().invoke(app, ["run", "-c", cfg_path, "--exploratory", "--run-id", "x"])
    assert bad.exit_code == 2
    assert "framing.stimuli" in bad.output and "--run-id" in bad.output
