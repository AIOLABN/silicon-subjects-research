from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from silicon_subjects import design, personas, runstore
from silicon_subjects.config import StudyConfig
from silicon_subjects.providers.mock import MockProvider
from silicon_subjects.runner import Runner
from silicon_subjects.tasks import build_tasks


def rec(i: int, **extra: object) -> bytes:
    return (json.dumps({"trial_id": f"t{i}", "v": i, **extra}, ensure_ascii=False) + "\n").encode()


def write(tmp_path: Path, data: bytes) -> Path:
    p = tmp_path / "trials.jsonl"
    p.write_bytes(data)
    return p


def test_valid_file_reads_unchanged(tmp_path: Path) -> None:
    p = write(tmp_path, rec(1) + rec(2))
    assert [r["trial_id"] for r in runstore.read_records(p)] == ["t1", "t2"]


def test_truncated_tail_is_recoverable_error(tmp_path: Path) -> None:
    p = write(tmp_path, rec(1) + rec(2)[:15])
    with pytest.raises(runstore.TruncatedTailError, match="sim recover"):
        runstore.read_records(p)


def test_truncated_utf8_tail_is_recoverable(tmp_path: Path) -> None:
    full = rec(2, text="café été")
    cut = full.index("é".encode()) + 1
    p = write(tmp_path, rec(1) + full[:cut])
    with pytest.raises(runstore.TruncatedTailError):
        runstore.read_records(p)
    runstore.recover_tail(p)
    assert [r["trial_id"] for r in runstore.read_records(p)] == ["t1"]


def test_malformed_interior_record_is_rejected_with_location(tmp_path: Path) -> None:
    p = write(tmp_path, rec(1) + b'{"trial_id": "t2", oops\n' + rec(3))
    with pytest.raises(runstore.ResultsCorruptError, match=r"line 2 \(byte 27\)") as e:
        runstore.read_records(p)
    assert not isinstance(e.value, runstore.TruncatedTailError)
    with pytest.raises(runstore.ResultsCorruptError):
        runstore.recover_tail(p)


def test_invalid_utf8_interior_is_rejected(tmp_path: Path) -> None:
    p = write(tmp_path, rec(1) + b'{"trial_id": "t2", "x": "\xff"}\n' + rec(3))
    with pytest.raises(runstore.ResultsCorruptError, match="line 2"):
        runstore.read_records(p)


def test_blank_interior_line_is_not_silently_skipped(tmp_path: Path) -> None:
    p = write(tmp_path, rec(1) + b"\n" + rec(2))
    with pytest.raises(runstore.ResultsCorruptError, match="line 2"):
        runstore.read_records(p)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    p = write(tmp_path, rec(1) + rec(2) + rec(1))
    with pytest.raises(runstore.ResultsCorruptError, match="duplicate trial_id t1"):
        runstore.read_records(p)


def test_valid_final_record_without_newline_survives_append(tmp_path: Path) -> None:
    p = write(tmp_path, rec(1) + rec(2).rstrip(b"\n"))
    assert [r["trial_id"] for r in runstore.read_records(p)] == ["t1", "t2"]
    runstore.ensure_trailing_newline(p)
    runstore.append_record(p, {"trial_id": "t3"})
    assert [r["trial_id"] for r in runstore.read_records(p)] == ["t1", "t2", "t3"]


def test_recovery_preserves_bytes_logs_and_is_idempotent(tmp_path: Path) -> None:
    original = rec(1) + rec(2) + rec(3)[:9]
    p = write(tmp_path, original)
    action = runstore.recover_tail(p)
    assert action is not None
    assert Path(action["backup"]).read_bytes() == original
    assert action["removed_bytes"] == 9 and action["kept_records"] == 2
    assert p.read_bytes() == rec(1) + rec(2)
    log = [json.loads(line) for line in (tmp_path / "recovery.jsonl").open()]
    assert log == [action]
    assert runstore.recover_tail(p) is None
    assert p.read_bytes() == rec(1) + rec(2)
    assert len(list(tmp_path.glob("trials.jsonl.*.bak"))) == 1


def test_interrupted_atomic_write_leaves_old_or_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "manifest.json"

    def crash(*a: object) -> None:
        raise OSError("power cut")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError):
        runstore.atomic_write_text(target, '{"complete": true}')
    assert not target.exists()
    target.write_text('{"v": 1}')
    with pytest.raises(OSError):
        runstore.atomic_write_text(target, '{"v": 2}')
    assert target.read_text() == '{"v": 1}'


def runner_for(cfg: StudyConfig) -> tuple[Runner, list, dict]:
    tasks = build_tasks(cfg)
    sets = personas.load_sets(cfg)
    calls: dict[str, int] = {"n": 0}

    class Counting(MockProvider):
        async def complete(self, *a: object, **k: object):
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


def trials_path(cfg: StudyConfig) -> Path:
    return cfg.root / "results" / "run" / "trials.jsonl"


def test_resume_after_valid_record_without_newline(demo_cfg: StudyConfig) -> None:
    r, trials, _ = runner_for(demo_cfg)
    asyncio.run(r.run(trials, limit=5))
    p = trials_path(demo_cfg)
    p.write_bytes(p.read_bytes().rstrip(b"\n"))
    r, trials, _ = runner_for(demo_cfg)
    asyncio.run(r.run(trials, limit=5))
    ids = [x["trial_id"] for x in runstore.read_records(p)]
    assert len(ids) == len(set(ids)) == 10


def test_truncated_tail_blocks_resume_until_explicit_recovery(demo_cfg: StudyConfig) -> None:
    r, trials, _ = runner_for(demo_cfg)
    asyncio.run(r.run(trials, limit=5))
    p = trials_path(demo_cfg)
    good = p.read_bytes()
    kept = [x["trial_id"] for x in runstore.read_records(p)][:4]
    lines = good.splitlines(keepends=True)
    p.write_bytes(b"".join(lines[:4]) + lines[4][:40])
    damaged = p.read_bytes()

    r, trials, calls = runner_for(demo_cfg)
    with pytest.raises(runstore.TruncatedTailError):
        asyncio.run(r.run(trials, limit=5))
    assert calls["n"] == 0 and p.read_bytes() == damaged

    r, trials, calls = runner_for(demo_cfg)
    asyncio.run(r.run(trials, limit=1, recover_tail=True))
    ids = [x["trial_id"] for x in runstore.read_records(p)]
    assert ids[:4] == kept and len(ids) == 5 and len(set(ids)) == 5
    planned_order = [t.trial_id for t in trials]
    assert ids[4] == planned_order[4]
    assert (p.parent / "recovery.jsonl").exists()


def test_duplicate_records_block_resume(demo_cfg: StudyConfig) -> None:
    r, trials, _ = runner_for(demo_cfg)
    asyncio.run(r.run(trials, limit=3))
    p = trials_path(demo_cfg)
    p.write_bytes(p.read_bytes() + p.read_bytes().splitlines(keepends=True)[0])
    r, trials, calls = runner_for(demo_cfg)
    with pytest.raises(runstore.ResultsCorruptError, match="duplicate"):
        asyncio.run(r.run(trials))
    assert calls["n"] == 0


def test_cli_recover_tail_flow(demo_cfg: StudyConfig) -> None:
    from typer.testing import CliRunner

    from silicon_subjects.cli import app

    cfg = str(demo_cfg.root / "configs" / "demo.yaml")
    base = ["run", "-c", cfg, "--exploratory", "--run-id", "run"]
    assert CliRunner().invoke(app, [*base, "--limit", "4"]).exit_code == 0
    p = trials_path(demo_cfg)
    p.write_bytes(p.read_bytes()[:-30])
    blocked = CliRunner().invoke(app, [*base, "--limit", "1"])
    assert blocked.exit_code == 2 and "sim recover run" in blocked.output
    fixed = CliRunner().invoke(app, [*base, "--limit", "1", "--recover-tail"])
    assert fixed.exit_code == 0, fixed.output
    assert len(runstore.read_records(p)) == 4


def test_analysis_refuses_truncated_results(demo_cfg: StudyConfig) -> None:
    from silicon_subjects import analysis

    r, trials, _ = runner_for(demo_cfg)
    asyncio.run(r.run(trials, limit=3))
    p = trials_path(demo_cfg)
    p.write_bytes(p.read_bytes()[:-5])
    with pytest.raises(analysis.AnalysisProvenanceError, match="incomplete record"):
        analysis.analyze_run(p.parent, partial=True)
