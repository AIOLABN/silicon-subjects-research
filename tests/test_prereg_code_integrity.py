from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import silicon_subjects
from silicon_subjects import prereg
from silicon_subjects.config import StudyConfig

PACKAGE = Path(silicon_subjects.__file__).resolve().parent


@pytest.fixture()
def src(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    copy = tmp_path / "checkout" / "src" / "silicon_subjects"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(prereg, "SOURCE_ROOT", copy, raising=False)
    return copy


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, f"{old!r} not in {path}"
    path.write_text(text.replace(old, new, 1))


def test_unchanged_inputs_verify(demo_cfg: StudyConfig, src: Path) -> None:
    prereg.freeze(demo_cfg)
    status = prereg.verify(demo_cfg)
    assert status.matches and status.changed == []


def test_python_defined_prompt_change_fails(demo_cfg: StudyConfig, src: Path) -> None:
    prereg.freeze(demo_cfg)
    _edit(src / "tasks" / "base.py", "as a participant would.", "as a typical participant would.")
    status = prereg.verify(demo_cfg)
    assert not status.matches
    assert status.changed == ["code:silicon_subjects/tasks/base.py"]


def test_coding_rule_change_fails(demo_cfg: StudyConfig, src: Path) -> None:
    prereg.freeze(demo_cfg)
    _edit(src / "coding.py", r'r"\bas an ai\b",', r'r"\bas an ai model\b",')
    status = prereg.verify(demo_cfg)
    assert not status.matches
    assert status.changed == ["code:silicon_subjects/coding.py"]


def test_analysis_change_fails(demo_cfg: StudyConfig, src: Path) -> None:
    prereg.freeze(demo_cfg)
    _edit(src / "stats.py", "z = stats.norm.ppf(1 - alpha / 2)", "z = 1.96")
    status = prereg.verify(demo_cfg)
    assert not status.matches
    assert status.changed == ["code:silicon_subjects/stats.py"]


def test_added_or_removed_source_file_fails(demo_cfg: StudyConfig, src: Path) -> None:
    prereg.freeze(demo_cfg)
    (src / "tasks" / "extra.py").write_text("X = 1\n")
    (src / "probes.py").unlink()
    status = prereg.verify(demo_cfg)
    assert status.changed == [
        "code:silicon_subjects/probes.py",
        "code:silicon_subjects/tasks/extra.py",
    ]


def test_documented_out_of_scope_files_do_not_matter(demo_cfg: StudyConfig, src: Path) -> None:
    prereg.freeze(demo_cfg)
    (demo_cfg.root / "README.md").write_text("unrelated edit\n")
    results = demo_cfg.root / "results" / "run1"
    results.mkdir(parents=True)
    (results / "trials.jsonl").write_text("{}\n")
    (src / "__pycache__").mkdir(exist_ok=True)
    (src / "__pycache__" / "x.cpython-313.pyc").write_bytes(b"\x00compiled")
    assert prereg.verify(demo_cfg).matches


def test_same_source_at_different_paths_has_same_fingerprint(tmp_path: Path) -> None:
    a = tmp_path / "a" / "silicon_subjects"
    b = tmp_path / "somewhere" / "else" / "silicon_subjects"
    shutil.copytree(PACKAGE, a, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PACKAGE, b, ignore=shutil.ignore_patterns("__pycache__"))
    assert prereg.source_fingerprint(a) == prereg.source_fingerprint(b)


def test_line_endings_do_not_change_fingerprint(tmp_path: Path) -> None:
    a = tmp_path / "a" / "silicon_subjects"
    b = tmp_path / "b" / "silicon_subjects"
    shutil.copytree(PACKAGE, a, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PACKAGE, b, ignore=shutil.ignore_patterns("__pycache__"))
    f = b / "coding.py"
    f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))
    assert prereg.source_fingerprint(a) == prereg.source_fingerprint(b)


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_committing_only_the_lock_keeps_it_valid_and_dirty_edits_fail(
    demo_cfg: StudyConfig, src: Path
) -> None:
    root = demo_cfg.root
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "experiment")
    (root / "notes.txt").write_text("uncommitted\n")
    lock = prereg.freeze(demo_cfg)
    saved = json.loads(lock.read_text())
    assert saved["provenance"]["git_commit"] == _git(root, "rev-parse", "HEAD")
    assert saved["provenance"]["git_dirty"] is True
    assert saved["provenance"]["git_dirty_paths"] == ["notes.txt"]
    assert saved["provenance"]["dependency_locking"].endswith("not enforced")
    assert "file" in saved["provenance"]["dependency_lock"]
    assert saved["provenance"]["distributions"]["pydantic"]

    _git(root, "add", lock.name)
    _git(root, "commit", "-q", "-m", "lock")
    assert prereg.verify(demo_cfg).matches

    _edit(src / "tasks" / "obedience.py", "STEP_V = 15", "STEP_V = 30")
    status = prereg.verify(demo_cfg)
    assert not status.matches
    assert status.changed == ["code:silicon_subjects/tasks/obedience.py"]


def _legacy_lock(cfg: StudyConfig) -> Path:
    parts = prereg.data_fingerprint(cfg)
    lock = prereg.lock_path(cfg)
    lock.write_text(
        json.dumps(
            {
                "study_id": cfg.study_id,
                "frozen_at": "2026-10-21T00:00:00+00:00",
                "git_commit": None,
                "hash": prereg.combined_hash(parts),
                "parts": parts,
            },
            indent=2,
        )
        + "\n"
    )
    return lock


def test_legacy_lock_is_unverified_not_upgraded(demo_cfg: StudyConfig, src: Path) -> None:
    lock = _legacy_lock(demo_cfg)
    before = lock.read_bytes()
    status = prereg.verify(demo_cfg)
    assert status.locked and status.legacy
    assert not status.matches
    assert status.changed == []
    assert lock.read_bytes() == before


def test_legacy_lock_reports_data_changes(demo_cfg: StudyConfig, src: Path) -> None:
    _legacy_lock(demo_cfg)
    stim = demo_cfg.path(demo_cfg.tasks[0].stimuli)
    stim.write_text(stim.read_text().replace("600 people", "601 people", 1))
    assert prereg.verify(demo_cfg).changed == ["framing.stimuli"]


def test_refreeze_preserves_the_superseded_lock(demo_cfg: StudyConfig, src: Path) -> None:
    lock = _legacy_lock(demo_cfg)
    old = lock.read_text()
    with pytest.raises(FileExistsError):
        prereg.freeze(demo_cfg)
    prereg.freeze(demo_cfg, force=True)
    archived = list(demo_cfg.root.glob("prereg.demo-mock.lock.superseded-*.json"))
    assert len(archived) == 1 and archived[0].read_text() == old
    new = json.loads(lock.read_text())
    assert new["supersedes"] == {
        "hash": json.loads(old)["hash"],
        "file": archived[0].name,
        "plan_version": 1,
    }
    assert new["plan_version"] == 2
    assert prereg.verify(demo_cfg).matches


def test_run_manifest_records_code_identity_and_runtime(demo_cfg: StudyConfig, src: Path) -> None:
    import asyncio

    from silicon_subjects import design, personas
    from silicon_subjects.runner import Runner
    from silicon_subjects.tasks import build_tasks

    prereg.freeze(demo_cfg)
    status = prereg.verify(demo_cfg)
    tasks = build_tasks(demo_cfg)
    sets = personas.load_sets(demo_cfg)
    out = demo_cfg.root / "results" / "r"
    runner = Runner(
        demo_cfg,
        tasks,
        personas.all_personas(sets),
        out,
        prereg_hash=status.expected,
        confirmatory=True,
    )
    asyncio.run(runner.run(design.build_manifest(demo_cfg, tasks, sets), limit=3))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["code_fingerprint"] == prereg.source_fingerprint()
    assert manifest["runtime"]["distributions"]["numpy"]
    assert manifest["runtime"]["dependency_locking"].endswith("not enforced")
    assert "file" in manifest["dependency_lock"]
    assert "git_commit" in manifest["provenance"]


def test_cli_refuses_confirmatory_run_under_legacy_lock(demo_cfg: StudyConfig, src: Path) -> None:
    from typer.testing import CliRunner

    from silicon_subjects.cli import app

    _legacy_lock(demo_cfg)
    cfg_path = str(demo_cfg.root / "configs" / "demo.yaml")
    result = CliRunner().invoke(app, ["verify", "-c", cfg_path])
    assert result.exit_code == 1 and "LEGACY" in result.output
    result = CliRunner().invoke(app, ["run", "-c", cfg_path, "--limit", "1"])
    assert result.exit_code == 2 and "LEGACY" in result.output
    assert not (demo_cfg.root / "results").exists()
