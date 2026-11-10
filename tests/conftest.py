from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from silicon_subjects import personas
from silicon_subjects.config import StudyConfig, load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def demo_cfg(tmp_path: Path) -> StudyConfig:
    for sub in ("configs", "data"):
        for source in (REPO / sub).rglob("*"):
            relative = source.relative_to(REPO)
            if "raw" in relative.parts or source.suffix == ".jsonl":
                continue
            target = tmp_path / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)

    path = tmp_path / "configs" / "demo.yaml"
    path.write_text(
        path.read_text().replace("bootstrap_iterations: 5000", "bootstrap_iterations: 200")
    )
    cfg = load_config(path)
    personas.save(personas.generate(cfg.personas, cfg.seed), cfg.path(cfg.personas.file))
    return cfg
