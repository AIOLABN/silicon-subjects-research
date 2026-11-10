from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from silicon_subjects import costs, design, personas
from silicon_subjects.cli import app
from silicon_subjects.config import StudyConfig
from silicon_subjects.tasks import build_tasks


def _write_run(run: Path, records: list[dict[str, object]]) -> Path:
    run.mkdir(parents=True)
    (run / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return run


_IDS = iter(range(10**6))


def _rec(task: str, method: str, tin: int, tout: int, **kw: object) -> dict[str, object]:
    return {
        "trial_id": f"t{next(_IDS)}",
        "task_id": task,
        "method": method,
        "persona_id": None,
        "input_tokens": tin,
        "output_tokens": tout,
        "discarded_input_tokens": 0,
        "discarded_output_tokens": 0,
        "generation": {"model": "claude-opus-5-5"},
        **kw,
    }


def _plan(cfg: StudyConfig) -> tuple[dict[str, object], list[object]]:
    tasks = build_tasks(cfg)
    return tasks, design.build_manifest(cfg, tasks, personas.load_sets(cfg))


def test_unmeasured_rows_use_the_assumptions(demo_cfg: StudyConfig) -> None:
    tasks, trials = _plan(demo_cfg)
    rows = costs.estimate(demo_cfg, tasks, trials)
    assert {r.source for r in rows} == {"assumed"}
    row = next(
        r
        for r in rows
        if (r.model_id, r.task_id, r.method, r.persona) == ("opus55", "obedience", "choice", False)
    )
    # 25k input at $4/M + 11k output at $20/M per trial
    assert row.expected_usd == pytest.approx(row.trials * 0.32)
    assert row.cautious_usd == pytest.approx(row.trials * (0.10 + 0.44))


def test_measured_tokens_replace_assumptions_including_retries(
    demo_cfg: StudyConfig, tmp_path: Path
) -> None:
    tasks, trials = _plan(demo_cfg)
    recs = [_rec("obedience", "choice", 10_000, 1_000) for _ in range(9)]
    recs.append(
        _rec(
            "obedience",
            "choice",
            10_000,
            1_000,
            discarded_input_tokens=20_000,
            discarded_output_tokens=2_000,
        )
    )
    measured = costs.measure([_write_run(tmp_path / "pilot", recs)])
    prof = measured[("claude-opus-5-5", "obedience", "choice", False)]
    assert prof.n == 10 and prof.mean_in == 12_000 and prof.mean_out == 1_200
    assert prof.p90_in == 10_000  # 9 of 10 trials are at 10k
    rows = costs.estimate(demo_cfg, tasks, trials, measured)
    row = next(
        r
        for r in rows
        if (r.model_id, r.task_id, r.method, r.persona) == ("opus55", "obedience", "choice", False)
    )
    assert row.source == "measured (n=10)"
    assert row.expected_usd == pytest.approx(row.trials * (12_000 * 4 + 1_200 * 20) / 1e6)


def test_one_turn_tasks_borrow_measurements_but_obedience_never_does(
    demo_cfg: StudyConfig, tmp_path: Path
) -> None:
    tasks, trials = _plan(demo_cfg)
    measured = costs.measure([_write_run(tmp_path / "pilot", [_rec("framing", "choice", 500, 50)])])
    rows = {
        (r.model_id, r.task_id, r.method, r.persona): r
        for r in costs.estimate(demo_cfg, tasks, trials, measured)
    }
    assert rows[("opus55", "framing", "choice", False)].source == "measured (n=1)"
    assert rows[("opus55", "holdout", "choice", False)].source.startswith("proxy")
    assert rows[("opus55", "obedience", "choice", False)].source == "assumed"
    assert rows[("sonnet55", "framing", "choice", False)].source == "assumed"  # other model
    assert rows[("opus55", "framing", "choice", True)].source == "assumed"  # persona differs


def test_cli_estimate_from_a_run(demo_cfg: StudyConfig, tmp_path: Path) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    _write_run(demo_cfg.root / "results" / "pilot", [_rec("framing", "choice", 500, 50)])
    res = CliRunner().invoke(app, ["estimate", "-c", str(path), "--from-run", "pilot"])
    assert res.exit_code == 0, res.output
    assert "measured (n=1)" in res.output and "cautious $" in res.output
    missing = CliRunner().invoke(app, ["estimate", "-c", str(path), "--from-run", "nope"])
    assert missing.exit_code == 2 and "not found" in missing.output
