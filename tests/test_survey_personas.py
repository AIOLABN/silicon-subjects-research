from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silicon_subjects import design, personas, survey_personas
from silicon_subjects.config import StudyConfig, SurveyPersonaSpec, load_config
from silicon_subjects.tasks import build_tasks

LABELS = {
    "sex": {1: "male", 2: "female"},
    "race": {1: "white", 2: "black", 3: "other"},
    "hispanic": {1: "not hispanic", 2: "mexican, mexican american, chicano/a"},
    "partyid": {0: "strong democrat", 6: "strong republican", 3: "independent"},
    "relig": {1: "protestant", 4: "none"},
    "childs": {8: "eight or more"},
}


def make_gss(path: Path, n: int = 3000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    relig = rng.choice([1, 4], n, p=[0.7, 0.3])
    p_dem = np.where(relig == 4, 0.85, 0.25)
    party = np.where(rng.random(n) < p_dem, 0, 6).astype(float)
    party[rng.random(n) < 0.05] = np.nan
    df = pd.DataFrame(
        {
            "year": np.where(np.arange(n) < n - 100, 2022, 2018),
            "id": np.arange(1, n + 1),
            "wtssps": rng.uniform(0.2, 3.0, n),
            "age": rng.integers(18, 90, n),
            "sex": rng.choice([1, 2], n),
            "race": rng.choice([1, 2, 3], n, p=[0.7, 0.2, 0.1]),
            "hispanic": rng.choice([1, 2], n, p=[0.8, 0.2]),
            "partyid": party,
            "relig": relig,
            "childs": rng.choice([0, 1, 2, 3, 8], n),
        }
    )
    df.to_stata(path, write_index=False, value_labels=LABELS, version=118)


def spec(tmp_path: Path, n_profiles: int = 50) -> SurveyPersonaSpec:
    return SurveyPersonaSpec(
        raw_file=tmp_path / "gss.dta",
        file=tmp_path / "out.jsonl",
        year=2022,
        n_profiles=n_profiles,
        variables=[
            {"name": "age", "columns": ["age"], "kind": "age", "group": "demographic"},
            {"name": "sex", "columns": ["sex"], "group": "demographic"},
            {
                "name": "race_ethnicity",
                "columns": ["race", "hispanic"],
                "kind": "race_ethnicity",
                "group": "demographic",
                "labels": {"white (not hispanic)": "White (not Hispanic)"},
            },
            {"name": "political_party", "columns": ["partyid"], "group": "survey_response"},
            {"name": "religion", "columns": ["relig"], "group": "demographic"},
            {
                "name": "number_of_children",
                "columns": ["childs"],
                "kind": "number",
                "group": "demographic",
            },
        ],
    )


@pytest.fixture()
def gss(tmp_path: Path) -> Path:
    make_gss(tmp_path / "gss.dta")
    return tmp_path / "gss.dta"


def test_profiles_are_real_reported_answers_only(tmp_path: Path, gss: Path) -> None:
    ps, _, prov = survey_personas.build(spec(tmp_path), gss, seed=1)
    assert len(ps) == 50 and len({p.meta["respondent_id"] for p in ps}) == 50
    assert all(p.id.startswith("s") for p in ps)
    raw = pd.read_stata(gss).set_index("id")
    for p in ps:
        row = raw.loc[p.meta["respondent_id"]]
        assert row["year"] == 2022
        assert p.attributes["age"].startswith(str(int(row["age"])))
        if pd.isna(row["partyid"]):
            assert "political_party" not in p.attributes
        assert "weight" not in p.render() and "respondent" not in p.render()
    assert prov["n_eligible"] == 2900


def test_race_ethnicity_derivation_and_relabel(tmp_path: Path, gss: Path) -> None:
    _, _, prov = survey_personas.build(spec(tmp_path, 200), gss, seed=1)
    seen = set(prov["labels_seen"]["race_ethnicity"])
    assert "White (not Hispanic)" in seen and "Hispanic or Latino" in seen
    assert "black (not Hispanic)" in seen


def test_deterministic(tmp_path: Path, gss: Path) -> None:
    a, _, _ = survey_personas.build(spec(tmp_path), gss, seed=7)
    b, _, _ = survey_personas.build(spec(tmp_path), gss, seed=7)
    assert [p.meta["respondent_id"] for p in a] == [p.meta["respondent_id"] for p in b]


def test_joint_distribution_is_preserved(tmp_path: Path, gss: Path) -> None:
    ps, _, _ = survey_personas.build(spec(tmp_path, 400), gss, seed=3)

    def dem_rate(relig: str) -> float:
        group = [
            p for p in ps if p.attributes["religion"] == relig and "political_party" in p.attributes
        ]
        return sum(p.attributes["political_party"] == "strong democrat" for p in group) / len(group)

    assert dem_rate("none") - dem_rate("protestant") > 0.4


def test_representativeness_table(tmp_path: Path, gss: Path) -> None:
    _, rep, _ = survey_personas.build(spec(tmp_path, 400), gss, seed=3)
    for attr, g in rep.groupby("attribute"):
        assert g["population_share"].sum() == pytest.approx(1.0)
        assert g["sample_share"].sum() == pytest.approx(1.0)
        assert g["attribute_tvd"].iloc[0] < 0.15, attr


def test_survey_arm_in_design(demo_cfg: StudyConfig, tmp_path: Path, gss: Path) -> None:
    raw = demo_cfg.root / "configs" / "demo.yaml"
    text = raw.read_text().replace(
        "  - id: paraphrase",
        "  - id: survey_persona\n    persona: true\n    persona_source: survey\n"
        "    replicates_per_profile: 2\n    role: secondary\n  - id: paraphrase",
    )
    text += (
        f"\nsurvey_personas:\n  raw_file: {gss}\n  file: data/personas/gss_test.jsonl\n"
        "  year: 2022\n  n_profiles: 50\n  variables:\n"
        "    - {name: religion, columns: [relig], group: demographic}\n"
    )
    raw.write_text(text)
    cfg = load_config(raw)
    assert cfg.survey_personas is not None
    ps, _, _ = survey_personas.build(cfg.survey_personas, gss, cfg.seed)
    personas.save(ps, cfg.path(cfg.survey_personas.file))
    sets = personas.load_sets(cfg)
    trials = design.build_manifest(cfg, build_tasks(cfg), sets)
    survey_trials = [t for t in trials if t.arm_id == "survey_persona"]
    assert {t.persona_id for t in survey_trials} == {p.id for p in sets["survey"]}
    assert (
        design.summarize(trials)[("opus55", "framing", "survey_persona", "gain", "original")] == 100
    )


def test_number_kind_uses_top_code_label(tmp_path: Path, gss: Path) -> None:
    _, _, prov = survey_personas.build(spec(tmp_path, 300), gss, seed=2)
    assert set(prov["labels_seen"]["number_of_children"]) == {"0", "1", "2", "3", "eight or more"}


def test_missing_column_is_a_clear_error(tmp_path: Path, gss: Path) -> None:
    raw = spec(tmp_path).model_dump()
    raw["variables"] = [
        raw["variables"][0],
        {"name": "x", "columns": ["nosuchvar"], "group": "demographic"},
    ]
    bad = SurveyPersonaSpec.model_validate(raw)
    with pytest.raises(KeyError, match="nosuchvar"):
        survey_personas.build(bad, gss, seed=1)
