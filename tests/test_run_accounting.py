from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from silicon_subjects import design, personas
from silicon_subjects.cli import app
from silicon_subjects.coding import Status
from silicon_subjects.config import StudyConfig
from silicon_subjects.providers import TransientProviderError
from silicon_subjects.providers.mock import MockProvider
from silicon_subjects.runner import Runner
from silicon_subjects.tasks import TaskResult, TrialSpec, build_tasks


class Scripted:
    def __init__(self, script: dict[str, list[str]]) -> None:
        self.script = {k: list(v) for k, v in script.items()}
        self.attempts: dict[str, int] = {}

    async def run(self, trial: TrialSpec, persona: object, provider: object) -> TaskResult:
        self.attempts[trial.trial_id] = self.attempts.get(trial.trial_id, 0) + 1
        steps = self.script.get(trial.trial_id, [])
        action = steps.pop(0) if steps else "ok"
        if action == "transient":
            raise TransientProviderError("429 rate limited")
        if action == "permanent":
            raise ValueError("400 bad request")
        if action == "refusal":
            return TaskResult(Status.REFUSAL_PLATFORM, {}, [])
        return TaskResult(Status.VALID, {"y": 1}, [])

    async def aclose(self) -> None:
        return None


class Closing(MockProvider):
    closed = 0

    async def aclose(self) -> None:
        type(self).closed += 1


def setup(
    cfg: StudyConfig, script: dict[str, list[str]], out: str = "run"
) -> tuple[Runner, list[TrialSpec], Scripted]:
    real = build_tasks(cfg)
    sets = personas.load_sets(cfg)
    trials = design.build_manifest(cfg, real, sets)
    fake = Scripted(script)
    r = Runner(
        cfg,
        {k: fake for k in real},
        personas.all_personas(sets),
        cfg.root / "results" / out,
        prereg_hash=None,
        confirmatory=False,
        providers={m.id: Closing(m) for m in cfg.models},
        retry_delay=lambda attempt: 0.0,
    )
    return r, trials, fake


def first(trials: list[TrialSpec], n: int) -> list[str]:
    return [t.trial_id for t in trials[:n]]


def errors(cfg: StudyConfig) -> list[dict[str, Any]]:
    p = cfg.root / "results" / "run" / "errors.jsonl"
    return [json.loads(line) for line in p.open()] if p.exists() else []


def test_all_selected_succeed(demo_cfg: StudyConfig) -> None:
    r, trials, _ = setup(demo_cfg, {})
    rep = asyncio.run(r.run(trials, limit=5))
    assert (rep.planned, rep.previously_completed, rep.selected, rep.completed) == (
        len(trials),
        0,
        5,
        5,
    )
    assert rep.failed == [] and rep.pending == len(trials) - 5
    assert rep.ok and not rep.study_complete


def test_all_selected_fail_permanently(demo_cfg: StudyConfig) -> None:
    _, trials, _ = setup(demo_cfg, {})
    r, _, fake = setup(demo_cfg, {t: ["permanent"] for t in first(trials, 3)})
    rep = asyncio.run(r.run(trials, limit=3))
    assert (rep.selected, rep.completed, len(rep.failed)) == (3, 0, 3)
    assert not rep.ok
    assert all(fake.attempts[t] == 1 for t in first(trials, 3))
    errs = errors(demo_cfg)
    assert {e["trial_id"] for e in errs} == set(first(trials, 3))
    assert errs[0]["kind"] == "permanent" and "400 bad request" in errs[0]["error"]
    assert not (demo_cfg.root / "results" / "run" / "trials.jsonl").exists()


def test_transient_then_success(demo_cfg: StudyConfig) -> None:
    _, trials, _ = setup(demo_cfg, {})
    tid = first(trials, 1)[0]
    r, _, fake = setup(demo_cfg, {tid: ["transient", "transient", "ok"]})
    rep = asyncio.run(r.run(trials, limit=1))
    assert rep.ok and rep.completed == 1 and fake.attempts[tid] == 3
    rec = json.loads((demo_cfg.root / "results" / "run" / "trials.jsonl").read_text())
    assert rec["attempts"] == 3 and len(rec["attempt_errors"]) == 2


def test_retry_exhaustion(demo_cfg: StudyConfig) -> None:
    _, trials, _ = setup(demo_cfg, {})
    tid = first(trials, 1)[0]
    r, _, fake = setup(demo_cfg, {tid: ["transient"] * 50})
    rep = asyncio.run(r.run(trials, limit=1))
    assert not rep.ok and rep.completed == 0
    assert fake.attempts[tid] == demo_cfg.runner.max_attempts
    (err,) = errors(demo_cfg)
    assert (
        err["kind"] == "retries_exhausted"
        and len(err["attempt_errors"]) == demo_cfg.runner.max_attempts
    )


def test_mixture_and_refusals_are_observations(demo_cfg: StudyConfig) -> None:
    _, trials, _ = setup(demo_cfg, {})
    a, b, c, _ = first(trials, 4)
    r, _, _ = setup(demo_cfg, {a: ["permanent"], b: ["refusal"], c: ["transient", "ok"]})
    rep = asyncio.run(r.run(trials, limit=4))
    assert (rep.completed, [f["trial_id"] for f in rep.failed]) == (3, [a])
    assert rep.pending == len(trials) - 3


def test_resume_completes_previously_failed(demo_cfg: StudyConfig) -> None:
    _, trials, _ = setup(demo_cfg, {})
    a, _ = first(trials, 2)
    r, _, _ = setup(demo_cfg, {a: ["permanent"]})
    assert not asyncio.run(r.run(trials, limit=2)).ok
    r2, _, _ = setup(demo_cfg, {})
    rep = asyncio.run(r2.run(trials, limit=1))
    assert rep.previously_completed == 1 and rep.completed == 1 and rep.ok
    assert rep.unresolved_failures == []
    assert errors(demo_cfg)[0]["trial_id"] == a


def test_providers_closed_on_failure_and_cancellation(demo_cfg: StudyConfig) -> None:
    Closing.closed = 0
    _, trials, _ = setup(demo_cfg, {})
    r, _, _ = setup(demo_cfg, {t: ["permanent"] for t in first(trials, 2)})
    asyncio.run(r.run(trials, limit=2))
    assert Closing.closed == len(demo_cfg.models)

    Closing.closed = 0
    r2, _, fake = setup(demo_cfg, {}, out="run2")

    async def hang(*a: object, **k: object) -> TaskResult:
        await asyncio.sleep(3600)
        raise AssertionError

    fake.run = hang

    async def main() -> None:
        job = asyncio.create_task(r2.run(trials, limit=2))
        await asyncio.sleep(0.05)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job

    asyncio.run(main())
    assert Closing.closed == len(demo_cfg.models)


def cli(cfg_path: Path, *args: str) -> Any:
    return CliRunner().invoke(app, [*args, "-c", str(cfg_path)])


def test_cli_exit_codes(demo_cfg: StudyConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = demo_cfg.root / "configs" / "demo.yaml"
    ok = cli(cfg_path, "run", "--exploratory", "--run-id", "a", "--limit", "5")
    assert ok.exit_code == 0, ok.output
    assert "completed 5" in ok.output and "pending" in ok.output

    async def broken(self: object, *a: object, **k: object) -> object:
        raise ValueError("invalid api key")

    monkeypatch.setattr(MockProvider, "complete", broken)
    bad = cli(cfg_path, "run", "--exploratory", "--run-id", "b", "--limit", "3")
    assert bad.exit_code == 1
    assert "failed 3" in bad.output and "ran 3 trials" not in bad.output


@pytest.fixture()
def collected(demo_cfg: StudyConfig) -> Path:
    cfg_path = demo_cfg.root / "configs" / "demo.yaml"
    cfg_path.write_text(
        cfg_path.read_text().replace("bootstrap_iterations: 5000", "bootstrap_iterations: 200")
    )
    for args in (["freeze"], ["run", "--run-id", "r"]):
        assert cli(cfg_path, *args).exit_code == 0
    return cfg_path


def trials_file(cfg_path: Path) -> Path:
    return cfg_path.parent.parent / "results" / "r" / "trials.jsonl"


def edit_lines(path: Path, fn: Any) -> None:
    lines = path.read_text().splitlines()
    path.write_text("\n".join(fn(lines)) + "\n")


def test_partial_collection_rejected_unless_partial(collected: Path) -> None:
    edit_lines(trials_file(collected), lambda ls: ls[:-10])
    res = cli(collected, "analyze", "r")
    assert res.exit_code == 2 and "10 missing" in res.output
    assert not (collected.parent.parent / "results" / "r" / "report.md").exists()
    res = cli(collected, "analyze", "r", "--partial")
    assert res.exit_code == 0, res.output
    (out,) = (collected.parent.parent / "results" / "r" / "provisional").iterdir()
    report = (out / "report.md").read_text()
    assert "Analysis status: **provisional" in report and "10 missing" in report
    assert not (collected.parent.parent / "results" / "r" / "report.md").exists()


def test_duplicate_records_rejected_even_when_partial(collected: Path) -> None:
    edit_lines(trials_file(collected), lambda ls: [*ls, ls[0]])
    for extra in ([], ["--partial"]):
        res = cli(collected, "analyze", "r", *extra)
        assert res.exit_code == 2 and "duplicate" in res.output


def test_unexpected_records_rejected(collected: Path) -> None:
    def add(ls: list[str]) -> list[str]:
        rec = json.loads(ls[0])
        rec["trial_id"] = "not-in-plan"
        return [*ls, json.dumps(rec)]

    edit_lines(trials_file(collected), add)
    res = cli(collected, "analyze", "r", "--partial")
    assert res.exit_code == 2 and "unexpected" in res.output


def test_record_that_contradicts_its_plan_is_rejected(collected: Path) -> None:
    def flip(ls: list[str]) -> list[str]:
        rec = json.loads(ls[0])
        rec["condition"] = "loss" if rec["condition"] != "loss" else "gain"
        return [json.dumps(rec), *ls[1:]]

    edit_lines(trials_file(collected), flip)
    res = cli(collected, "analyze", "r")
    assert res.exit_code == 2 and "does not match its planned specification" in res.output
