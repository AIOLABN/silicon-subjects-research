from __future__ import annotations

import asyncio
import json
import math
import shutil
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest
from typer.testing import CliRunner

from silicon_subjects import analysis, calibration, evaluation, forecast, personas, prereg
from silicon_subjects.cli import app
from silicon_subjects.config import StudyConfig, load_config
from silicon_subjects.personas import Persona, select_fields
from silicon_subjects.tasks import TrialSpec, build_tasks


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("not json", "not valid JSON"),
        ("[0.5, 0.5]", "expected a JSON object"),
        ('{"A": 0.5}', "missing labels"),
        ('{"A": 0.5, "B": 0.4, "C": 0.1}', "unexpected labels"),
        ('{"A": "0.5", "B": 0.5}', "not a number"),
        ('{"A": true, "B": 0.0}', "not a number"),
        ('{"A": NaN, "B": 0.5}', "not finite"),
        ('{"A": Infinity, "B": 0}', "not finite"),
        ('{"A": -0.1, "B": 1.1}', "outside [0, 1]"),
        ('{"A": 0.6, "B": 0.6}', "sum to 1.2"),
        ('{"A": 0.5, "B": 0.4985}', "sum to 0.9985"),
    ],
)
def test_invalid_forecasts_are_rejected_not_repaired(text: str, fragment: str) -> None:
    v = forecast.validate(text, ["A", "B"])
    assert not v.ok and v.probabilities is None
    assert any(fragment in e for e in v.errors), v.errors


def test_valid_forecast_within_tolerance_is_kept_unnormalized() -> None:
    v = forecast.validate('{"A": 0.3005, "B": 0.7}', ["A", "B"])
    assert v.ok and v.probabilities == {"A": 0.3005, "B": 0.7}


def test_schema_has_explicit_labels_and_no_extras() -> None:
    s = forecast.schema(["A", "B"])
    assert s["required"] == ["A", "B"] and s["additionalProperties"] is False


def test_persona_aggregation_averages_repeats_then_weights_personas() -> None:
    agg = forecast.personas({"p1": [0.2, 0.4], "p2": [0.9]}, {"p1": 1.0, "p2": 3.0})

    assert agg.p == pytest.approx(0.75) and agg.n_units == 2


def test_invalid_weights_are_refused() -> None:
    with pytest.raises(ValueError, match="invalid analysis weight"):
        forecast.personas({"p1": [0.5]}, {"p1": 0.0})


def test_pps_sample_is_not_reweighted_but_srs_is(tmp_path: Path) -> None:
    from test_survey_personas import make_gss, spec

    from silicon_subjects import survey_personas

    make_gss(tmp_path / "gss.dta")
    pps, _, _ = survey_personas.build(spec(tmp_path), tmp_path / "gss.dta", seed=1)
    assert {p.analysis_weight for p in pps} == {1.0}
    srs_spec = spec(tmp_path).model_copy(update={"sampling": "srs"})
    srs, _, _ = survey_personas.build(srs_spec, tmp_path / "gss.dta", seed=1)
    assert all(p.analysis_weight == p.meta["survey_weight"] for p in srs)


def test_evaluation_uses_analysis_weights_for_persona_arms(demo_cfg: StudyConfig) -> None:
    rows = [
        {
            "model_id": "m",
            "task_id": "framing",
            "arm_id": "persona_forecast",
            "persona_id": pid,
            "condition": cond,
            "prompt_variant": "original",
            "status": "valid",
            "y": y,
            "method": "persona_forecast",
        }
        for pid, y in (("p000", 0.2), ("p001", 0.8))
        for cond in ("gain", "loss")
    ]
    ps = [
        Persona("p000", {}, {"analysis_weight": 3.0}),
        Persona("p001", {}, {"analysis_weight": 1.0}),
    ]
    conds, _ = evaluation.method_evaluation(pd.DataFrame(rows), demo_cfg, build_tasks(demo_cfg), ps)
    assert conds.iloc[0]["pred_p"] == pytest.approx((0.2 * 3 + 0.8 * 1) / 4)


def test_effect_errors_signs_and_inflation_flags() -> None:
    assert evaluation.inflation(0.4, 0.2, 0.05) == {
        "inflation_ratio": 2.0,
        "inflation_flag": "ok",
        "sign_disagreement": False,
    }
    assert evaluation.inflation(0.1, 0.0, 0.05)["inflation_flag"].startswith("undefined")
    assert evaluation.inflation(0.1, 0.05, 0.05)["inflation_flag"].startswith("unstable")
    neg = evaluation.inflation(-0.1, 0.3, 0.05)
    assert neg["sign_disagreement"] and neg["inflation_flag"].startswith("sign")


def test_effect_table_against_known_values(demo_cfg: StudyConfig) -> None:

    rows = []
    for cond, ones in (("gain", 9), ("loss", 5)):
        for i in range(10):
            rows.append(
                {
                    "model_id": "m",
                    "task_id": "framing",
                    "arm_id": "baseline",
                    "persona_id": None,
                    "condition": cond,
                    "prompt_variant": "original",
                    "status": "valid",
                    "y": 1 if i < ones else 0,
                    "method": "choice",
                }
            )
    conds, effects = evaluation.method_evaluation(
        pd.DataFrame(rows), demo_cfg, build_tasks(demo_cfg), []
    )
    e = effects[effects["task_id"] == "framing"].iloc[0]
    assert e["pred_effect"] == pytest.approx(0.4)
    assert e["human_effect"] == pytest.approx(0.3)
    assert e["signed_effect_error"] == pytest.approx(0.1)
    gain = conds[(conds["task_id"] == "framing") & (conds["condition"] == "gain")].iloc[0]
    assert gain["signed_error"] == pytest.approx(0.9 - 0.72)
    assert gain["tvd"] == pytest.approx(abs(0.9 - 0.72))
    assert gain["brier"].startswith("not computed")


def trial(method: str, cond: str = "gain", disclose: bool = False, **f: str) -> TrialSpec:
    return TrialSpec(
        "t",
        "m",
        "framing",
        "a",
        cond,
        "original",
        None,
        0,
        {"option_order": "certain_first", **f},
        1,
        method,
        disclose,
    )


def test_forecast_prompts_never_contain_benchmark_results(demo_cfg: StudyConfig) -> None:
    bench = demo_cfg.path(demo_cfg.tasks[0].benchmark)
    bench.write_text(bench.read_text().replace("{k: 720, n: 1000}", "{k: 98761, n: 99999}"))
    task = build_tasks(demo_cfg)["framing"]
    person = Persona("p", {"age": "30-44 years old"})
    for method in ("population_forecast", "persona_forecast"):
        for cond in ("gain", "loss"):
            for disclose in (False, True):
                system, prompt = task.forecast_prompt(trial(method, cond, disclose), person)
                text = system + prompt
                for leak in ("98761", "99999", "Many Labs", "Klein", "Tversky", "Kahneman"):
                    assert leak not in text
    _, pop = task.forecast_prompt(trial("population_forecast"), None)
    assert task.target_population() in pop


def test_design_disclosure_reveals_design_not_results(demo_cfg: StudyConfig) -> None:
    task = build_tasks(demo_cfg)["framing"]
    plain, _ = task.render(trial("choice"))
    disclosed, _ = task.render(trial("choice", disclose=True))
    assert "randomly assigned" in disclosed and "randomly assigned" not in plain
    lower = task.stimuli["design_disclosure"].lower()
    for word in ("more likely", "most people", "percent", "%", "prefer", "effect"):
        assert word not in lower


def test_excluded_and_survey_fields_never_reach_prompts() -> None:
    p = Persona(
        "s1",
        {
            "age": "50 years old",
            "political_party": "strong democrat",
            "risk_tolerance": "high",
            "undocumented": "x",
        },
    )
    groups = {
        "age": "demographic",
        "political_party": "survey_response",
        "risk_tolerance": "survey_response",
    }
    demog = select_fields(p, groups=groups, demographic_only=True, excluded={"risk_tolerance"})
    assert demog.attributes == {"age": "50 years old"}
    full = select_fields(p, groups=groups, demographic_only=False, excluded={"risk_tolerance"})
    assert "risk_tolerance" not in full.describe() and "strong democrat" in full.describe()


def _with_arm(cfg_path: Path, arm: str, task_patch: tuple[str, str] | None = None) -> None:
    text = cfg_path.read_text().replace("  - id: paraphrase", arm + "  - id: paraphrase", 1)
    if task_patch:
        text = text.replace(*task_patch, 1)
    cfg_path.write_text(text)


@pytest.mark.parametrize(
    ("arm", "message"),
    [
        (
            "  - id: x\n    method: population_forecast\n    persona: false\n    n_per_cell: 2\n"
            "    tasks: [obedience]\n",
            "optional extension",
        ),
        (
            "  - id: x\n    persona: false\n    n_per_cell: 2\n    design_disclosure: true\n"
            "    tasks: [obedience]\n",
            "changes what the simulated participant knows".replace("changes", "change"),
        ),
        (
            "  - id: x\n    method: persona_forecast\n    persona: false\n    n_per_cell: 2\n",
            "persona_forecast needs persona",
        ),
    ],
)
def test_config_rejects_unsupported_method_combinations(
    demo_cfg: StudyConfig, arm: str, message: str
) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    _with_arm(path, arm)
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_obedience_forecast_extension_when_enabled(demo_cfg: StudyConfig) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    _with_arm(
        path,
        "  - id: obf\n    method: population_forecast\n    persona: false\n"
        "    n_per_cell: 2\n    tasks: [obedience]\n",
        ("    kind: obedience\n", "    kind: obedience\n    forecast_extension: true\n"),
    )
    cfg = load_config(path)
    task = build_tasks(cfg)["obedience"]
    t = TrialSpec(
        "t", "m", "obedience", "obf", "base", "original", None, 0, {}, 1, "population_forecast"
    )
    system, prompt = task.forecast_prompt(t)
    assert "continue past 150 volts" in prompt
    for leak in ("28", "70.0", "63.3", "Milgram", "Burger", "obedience"):
        assert leak not in system + prompt, leak


def _cli(cfg_path: Path, *args: str) -> Any:
    return CliRunner().invoke(app, [*args, "-c", str(cfg_path)])


@pytest.fixture()
def collected(demo_cfg: StudyConfig) -> Path:
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(
        path.read_text().replace("bootstrap_iterations: 5000", "bootstrap_iterations: 100")
    )
    for args in (["freeze"], ["run", "--run-id", "r"]):
        res = _cli(path, *args)
        assert res.exit_code == 0, res.output
    return path


def test_methods_are_reported_separately_and_never_pooled(collected: Path) -> None:
    run = collected.parent.parent / "results" / "r"
    out = analysis.analyze_run(run)
    records = [json.loads(line) for line in (run / "trials.jsonl").open()]
    choice_ids = {r["trial_id"] for r in records if r["method"] == "choice"}
    cells = out.result.tables["cells"]
    assert cells["n_trials"].sum() == len(choice_ids)
    conds = out.result.tables["method_conditions"]
    assert set(conds["arm_id"]) >= {
        "baseline",
        "design_aware",
        "population_forecast",
        "persona_forecast",
    }
    pop = conds[(conds["arm_id"] == "population_forecast") & (conds["task_id"] == "framing")]
    assert set(pop["method"]) == {"population_forecast"}

    for _, row in pop.iterrows():
        ys = [
            r["outcome"]["chose_certain"]
            for r in records
            if r["arm_id"] == "population_forecast"
            and r["condition"] == row["condition"]
            and r["status"] == "valid"
            and r["model_id"] == row["model_id"]
        ]
        assert row["pred_p"] == pytest.approx(sum(ys) / len(ys))
        assert row["n_units"] == len(ys)
    diag = out.result.tables["method_diagnostics"]
    bad = diag[diag["method"] == "population_forecast"]["parse_failure"]
    assert all("/" in x for x in bad)


def test_invalid_forecasts_are_recorded_with_raw_output(collected: Path) -> None:
    run = collected.parent.parent / "results" / "r"
    records = [json.loads(line) for line in (run / "trials.jsonl").open()]
    repaired = [r for r in records if r["method"] != "choice" and r["format_repairs"]]
    assert repaired, "the mock emits ~3% malformed forecasts"
    r = repaired[0]
    first = next(t for t in r["transcript"] if t["role"] == "assistant")
    assert first["meta"]["validation_errors"] and first["content"].startswith("{")
    assert r["generation"]["structured_output"] is True


def test_verified_confirmatory_run(collected: Path) -> None:
    run = collected.parent.parent / "results" / "r"
    out = analysis.analyze_run(run)
    assert out.provenance["analysis_status"] == "confirmatory"
    assert out.provenance["plan_version"] == 1


@pytest.mark.parametrize(
    "tamper", ["delete_lock_copy", "edit_lock_copy", "edit_prereg_hash", "plan_version"]
)
def test_unverifiable_confirmatory_claim_is_refused(collected: Path, tamper: str) -> None:
    run = collected.parent.parent / "results" / "r"
    lock = run / "inputs" / "prereg.lock.json"
    m = json.loads((run / "manifest.json").read_text())
    if tamper == "delete_lock_copy":
        lock.unlink()
    elif tamper == "edit_lock_copy":
        data = json.loads(lock.read_text())
        data["parts"]["framing.stimuli"] = "0" * 64
        lock.write_text(json.dumps(data))
    elif tamper == "edit_prereg_hash":
        m["prereg_hash"] = "f" * 64
    else:
        m["plan_version"] = 7
    (run / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(analysis.AnalysisProvenanceError, match="claims to be confirmatory"):
        analysis.analyze_run(run)
    assert not (run / "report.md").exists()


def test_exploratory_run_edited_to_confirmatory_is_refused(demo_cfg: StudyConfig) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(
        path.read_text().replace("bootstrap_iterations: 5000", "bootstrap_iterations: 100")
    )
    assert _cli(path, "run", "--run-id", "e", "--exploratory").exit_code == 0
    run = demo_cfg.root / "results" / "e"
    m = json.loads((run / "manifest.json").read_text())
    m["confirmatory"] = True
    m["prereg_hash"] = prereg.combined_hash(
        {k: v for k, v in m["experiment_identity"]["parts"].items() if k != "runner"}
    )
    (run / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(analysis.AnalysisProvenanceError, match="no copy of the frozen plan"):
        analysis.analyze_run(run)


def test_changed_method_requires_new_plan_version(collected: Path) -> None:
    stim = collected.parent.parent / "data" / "stimuli" / "framing.yaml"
    stim.write_text(
        stim.read_text().replace(
            "You are reading one of the two versions.", "You read one version."
        )
    )
    res = _cli(collected, "run", "--run-id", "r2")
    assert res.exit_code == 2
    assert _cli(collected, "freeze", "--force").exit_code == 0
    lock = json.loads((collected.parent.parent / "prereg.demo-mock.lock.json").read_text())
    assert lock["plan_version"] == 2 and lock["supersedes"]["plan_version"] == 1


def test_evaluation_outcomes_sealed_unless_confirmatory(demo_cfg: StudyConfig) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    text = path.read_text().replace("bootstrap_iterations: 5000", "bootstrap_iterations: 100")
    path.write_text(
        text.replace(
            "  - id: framing\n    split: development", "  - id: framing\n    split: evaluation", 1
        )
    )
    assert _cli(path, "run", "--run-id", "x", "--exploratory").exit_code == 0
    out = analysis.analyze_run(demo_cfg.root / "results" / "x")
    conds = out.result.tables["method_conditions"]
    framing = conds[conds["task_id"] == "framing"]
    assert "human_p" not in framing or framing["human_p"].isna().all()
    assert "720" not in (out.report.read_text())
    assert out.provenance["sealed_evaluation_studies"] == ["framing", "holdout"]
    shutil.rmtree(demo_cfg.root / "results")
    for args in (["freeze"], ["run", "--run-id", "c"]):
        assert _cli(path, *args).exit_code == 0
    out = analysis.analyze_run(demo_cfg.root / "results" / "c")
    conds = out.result.tables["method_conditions"]
    assert conds[conds["task_id"] == "framing"]["human_p"].notna().any()
    assert out.provenance["sealed_evaluation_studies"] == []


def test_task_split_is_required(demo_cfg: StudyConfig) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(path.read_text().replace("    split: development\n", "", 1))
    with pytest.raises(ValueError, match="split"):
        load_config(path)


def test_calibration_refuses_evaluation_studies(demo_cfg: StudyConfig) -> None:
    cfg = demo_cfg.model_copy(
        update={
            "tasks": [
                t.model_copy(update={"split": "evaluation"}) if t.id == "framing" else t
                for t in demo_cfg.tasks
            ]
        }
    )
    effects = [
        {"arm_id": "baseline", "task_id": "framing", "pred_effect": 0.4, "human_effect": 0.3}
    ]
    with pytest.raises(calibration.CalibrationError, match="not a development study"):
        calibration.development_points(cfg, effects, "baseline")


def test_calibration_needs_enough_independent_studies() -> None:
    with pytest.raises(calibration.CalibrationError, match="at least 5"):
        calibration.fit([("a", 0.1, 0.05), ("b", 0.4, 0.2)], min_studies=5)
    f = calibration.fit(
        [(f"s{i}", x, 0.5 * x + 0.01) for i, x in enumerate([0.1, 0.2, 0.3, 0.5, 0.6])], 5
    )
    assert f.a == pytest.approx(0.01) and f.b == pytest.approx(0.5)


def test_calibration_disabled_by_default_with_reason(demo_cfg: StudyConfig) -> None:
    fits, note = calibration.status(demo_cfg)
    assert fits is None and note.startswith("disabled") and "needed" in note


def test_anthropic_request_uses_structured_output_and_replays_blocks() -> None:
    pytest.importorskip("anthropic")
    from silicon_subjects.config import ModelSpec
    from silicon_subjects.providers.anthropic_provider import AnthropicProvider
    from silicon_subjects.providers.base import ChatMessage

    sent: dict[str, Any] = {}

    class Resp:
        content: ClassVar[list[Any]] = []
        stop_reason = "end_turn"
        model = "claude-opus-5-5"

        class usage:
            input_tokens = 1
            output_tokens = 1

    class Messages:
        async def create(self, **kw: Any) -> Resp:
            sent.update(kw)
            return Resp()

    class Client:
        messages = Messages()

        async def close(self) -> None:
            return None

    p = AnthropicProvider(
        ModelSpec(id="m", provider="anthropic", model="claude-opus-5-5", effort="medium"), 10
    )
    p._client = Client()
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "DECISION: STOP"},
    ]
    asyncio.run(
        p.complete(
            "sys",
            [
                ChatMessage("user", "hi"),
                ChatMessage("assistant", "DECISION: STOP", blocks),
                ChatMessage("user", "go on"),
            ],
            seed=1,
            response_schema=forecast.schema(["A", "B"]),
        )
    )
    assert sent["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": forecast.schema(["A", "B"])},
    }
    assert sent["messages"][1]["content"] == blocks
    assert "temperature" not in sent


def test_persona_sets_have_analysis_weights(demo_cfg: StudyConfig) -> None:
    ps = personas.all_personas(personas.load_sets(demo_cfg))
    assert all(math.isclose(p.analysis_weight, 1.0) for p in ps)


def test_power_table_counts_independent_units(demo_cfg: StudyConfig) -> None:
    from silicon_subjects.analysis import power_table

    path = demo_cfg.root / "configs" / "demo.yaml"
    _with_arm(
        path,
        "  - id: small\n    persona: true\n    replicates_per_profile: 2\n"
        "    profiles_limit: 10\n    tasks: [framing]\n    role: secondary\n",
    )
    cfg = load_config(path)
    table = power_table(cfg, build_tasks(cfg)).set_index(["task_id", "arm_id", "condition"])

    assert table.loc[("framing", "persona", "gain"), "n_model"] == 50
    assert table.loc[("framing", "small", "gain"), "n_model"] == 10
    assert table.loc[("framing", "baseline", "gain"), "n_model"] == 100

    assert ("obedience", "small", "base") not in table.index


def test_forecast_only_run_still_reports_benchmark_warnings(demo_cfg: StudyConfig) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    text = path.read_text().replace("bootstrap_iterations: 5000", "bootstrap_iterations: 100")
    head, _, tail = text.partition("arms:\n")
    arms = (
        "arms:\n  - id: population_forecast\n    method: population_forecast\n"
        "    persona: false\n    n_per_cell: 3\n    tasks: [framing]\n    role: primary\n\n"
    )
    path.write_text(head + arms + tail[tail.index("analysis:") :])
    assert _cli(path, "run", "--run-id", "f", "--exploratory").exit_code == 0
    out = analysis.analyze_run(demo_cfg.root / "results" / "f")
    assert any("not marked verified" in w for w in out.result.warnings)
    assert not out.result.tables["method_effects"].empty
    for name in (
        "method_conditions",
        "method_diagnostics",
        "replication_range",
        "subgroup_evaluation",
    ):
        assert name in out.result.tables


def _tree(d: Path) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("mode", ["default", "reanalysis"])
def test_failed_publish_leaves_previous_outputs_untouched(
    collected: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from silicon_subjects import report

    run = collected.parent.parent / "results" / "r"
    analysis.analyze_run(run)
    before = _tree(run)

    def crash(*a: object, **k: object) -> Path:
        raise OSError("disk full while writing the report")

    monkeypatch.setattr(report, "render", crash)
    kwargs = {"reanalyze_with": load_config(collected)} if mode == "reanalysis" else {}
    with pytest.raises(OSError, match="disk full"):
        analysis.analyze_run(run, **kwargs)
    assert _tree(run) == before
    assert not any(p.name.startswith(".") for p in run.iterdir() if p.is_dir())
