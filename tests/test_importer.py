from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from silicon_subjects import importer
from silicon_subjects.cli import app
from silicon_subjects.config import StudyConfig

SPEC = {
    "source": "synthetic test file",
    "condition_column": "grp",
    "condition_values": {"1": "gain", "2": "loss"},
    "outcome_column": "dv",
    "success_values": ["1"],
    "failure_values": ["2"],
    "site_column": "site",
    "min_site_n": 3,
}


def raw_frame() -> pd.DataFrame:
    rows = []

    for site, grp, dvs in [
        ("A", 1, [1, 1, 1, 2]),
        ("A", 2, [1, 2, 2, 2]),
        ("B", 1, [1, 1, 2]),
        ("B", 2, [2, 2, 2]),
        ("C", 1, [1]),
        ("C", 2, [2]),
    ]:
        rows += [{"site": site, "grp": grp, "dv": dv} for dv in dvs]
    rows += [
        {"site": "A", "grp": 9, "dv": 1},
        {"site": "A", "grp": 1, "dv": None},
        {"site": "B", "grp": 2, "dv": 7},
    ]
    return pd.DataFrame(rows)


def write(tmp: Path, spec: dict[str, Any], fmt: str = "csv") -> tuple[Path, Path]:
    raw = tmp / f"raw.{fmt}"
    df = raw_frame()
    if fmt == "csv":
        df.to_csv(raw, index=False)
    else:
        df.astype({"grp": "float64", "dv": "float64"}).to_stata(raw, write_index=False)
    sp = tmp / "spec.yaml"
    sp.write_text(yaml.safe_dump(spec))
    return raw, sp


@pytest.mark.parametrize("fmt", ["csv", "dta"])
def test_counts_sites_and_exclusions(demo_cfg: StudyConfig, tmp_path: Path, fmt: str) -> None:
    raw, sp = write(tmp_path, SPEC, fmt)
    base = demo_cfg.path(demo_cfg.tasks[0].benchmark)
    out = tmp_path / "out.yaml"
    bench = importer.import_benchmark(base, raw, sp, out)
    assert bench["conditions"] == {"gain": {"k": 6, "n": 8}, "loss": {"k": 1, "n": 8}}
    assert [s["site"] for s in bench["sites"]] == ["A", "B"]
    assert bench["sites"][1]["conditions"]["gain"] == {"k": 2, "n": 3}
    ex = bench["import_provenance"]["excluded"]
    assert ex["unmapped condition"] == 1 and ex["missing outcome"] == 1
    assert ex["unrecognized outcome value"] == 1
    assert bench["import_provenance"]["raw_sha256"] and bench["verified"] is False
    assert yaml.safe_load(out.read_text())["conditions"] == bench["conditions"]
    assert bench["target_population"]


def test_overlapping_codes_and_missing_columns_are_refused(
    demo_cfg: StudyConfig, tmp_path: Path
) -> None:
    base = demo_cfg.path(demo_cfg.tasks[0].benchmark)
    raw, sp = write(tmp_path, {**SPEC, "failure_values": ["1", "2"]})
    with pytest.raises(importer.ImportError_, match="both success and failure"):
        importer.import_benchmark(base, raw, sp, tmp_path / "o.yaml")
    raw, sp = write(tmp_path, {**SPEC, "outcome_column": "nope"})
    with pytest.raises(importer.ImportError_, match="nope"):
        importer.import_benchmark(base, raw, sp, tmp_path / "o.yaml")


def test_evaluation_import_never_prints_outcomes(demo_cfg: StudyConfig, tmp_path: Path) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(
        path.read_text().replace(
            "  - id: framing\n    split: development", "  - id: framing\n    split: evaluation", 1
        )
    )
    raw, sp = write(tmp_path, SPEC)
    out = tmp_path / "sealed.yaml"
    res = CliRunner().invoke(
        app,
        [
            "import-benchmark",
            "-c",
            str(path),
            "--task",
            "framing",
            "--raw",
            str(raw),
            "--spec",
            str(sp),
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "'k'" not in res.output and "sealed" in res.output and "'n': 8" in res.output
    assert yaml.safe_load(out.read_text())["conditions"]["gain"]["k"] == 6


def test_imported_sites_feed_the_replication_range(demo_cfg: StudyConfig, tmp_path: Path) -> None:
    from silicon_subjects import evaluation
    from silicon_subjects.tasks import build_tasks

    raw, sp = write(tmp_path, SPEC)
    base = demo_cfg.path(demo_cfg.tasks[0].benchmark)
    importer.import_benchmark(base, raw, sp, base)
    tasks = {"framing": build_tasks(demo_cfg)["framing"]}
    effects = pd.DataFrame(
        [
            {
                "task_id": "framing",
                "model_id": "m",
                "arm_id": "a",
                "method": "choice",
                "prompt_variant": "original",
                "pred_effect": 0.5,
            }
        ]
    )
    rr = evaluation.replication_range(pd.DataFrame(), effects, tasks)
    row = rr.iloc[0]
    assert row["n_sites"] == 2
    assert row["site_effect_min"] == pytest.approx(0.5)
    assert row["site_effect_max"] == pytest.approx(2 / 3)
    assert bool(row["pred_effect_within_site_range"]) is True
