from __future__ import annotations

import json
from pathlib import Path

import yaml

from silicon_subjects import design, personas, probes
from silicon_subjects.coding import Status
from silicon_subjects.config import StudyConfig
from silicon_subjects.providers.base import ChatMessage, Completion
from silicon_subjects.tasks import TrialSpec, build_tasks

REPO = Path(__file__).resolve().parents[1]


class Scripted:
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.seen: list[list[ChatMessage]] = []
        self.schemas: list[object] = []

    async def complete(
        self, system: str, messages: list[ChatMessage], *, seed: int, **kw: object
    ) -> Completion:
        self.seen.append(list(messages))
        self.schemas.append(kw.get("response_schema"))
        return Completion(self.replies.pop(0), "end_turn", "scripted-model")

    async def aclose(self) -> None:
        return None


def trial(
    condition: str, order: str = "yes_first", method: str = "choice", **kw: bool
) -> TrialSpec:
    return TrialSpec(
        "t1",
        "m",
        "holdout",
        "baseline",
        condition,
        "original",
        None,
        0,
        {"answer_order": order},
        1,
        method=method,
        **kw,
    )


async def test_yes_is_coded_one_whatever_the_answer_order(demo_cfg: StudyConfig) -> None:
    task = build_tasks(demo_cfg)["holdout"]
    for order, reply, expected in (
        ("yes_first", "Hard call.\nANSWER: Yes", 1),
        ("no_first", "**ANSWER:** no", 0),
        ("no_first", "ANSWER: YES.", 1),
    ):
        p = Scripted([reply])
        res = await task.run(trial("side_effect", order), None, p)
        assert res.status is Status.VALID and res.outcome["judged_permissible"] == expected
        prompt = p.seen[0][0].content
        first, second = ("Yes", "No") if order == "yes_first" else ("No", "Yes")
        assert prompt.rstrip().endswith(f"ANSWER: {first} or ANSWER: {second}")


async def test_unparseable_answer_gets_one_repair_then_invalid(demo_cfg: StudyConfig) -> None:
    task = build_tasks(demo_cfg)["holdout"]
    res = await task.run(trial("greater_good"), None, Scripted(["Maybe.", "ANSWER: No"]))
    assert res.status is Status.VALID and res.outcome["judged_permissible"] == 0
    assert res.format_repairs == 1
    res = await task.run(trial("greater_good"), None, Scripted(["Maybe.", "Perhaps."]))
    assert res.status is Status.INVALID and res.outcome == {}


async def test_each_condition_shows_only_its_own_scenario(demo_cfg: StudyConfig) -> None:
    task = build_tasks(demo_cfg)["holdout"]
    stim = task.stimuli["variants"]["original"]["conditions"]
    for cond, other in (("greater_good", "side_effect"), ("side_effect", "greater_good")):
        text = task.materials(trial(cond))
        assert stim[cond]["scenario"] in text and stim[cond]["question"] in text
        assert stim[other]["scenario"] not in text
        assert task.stimuli["design_disclosure"].strip() not in text
        assert text.startswith(task.stimuli["design_disclosure"].strip()[:40]) is False
        disclosed = task.materials(trial(cond, design_disclosure=True))
        assert disclosed.startswith(task.stimuli["design_disclosure"].strip())


async def test_forecast_returns_probability_of_yes(demo_cfg: StudyConfig) -> None:
    task = build_tasks(demo_cfg)["holdout"]
    p = Scripted([json.dumps({"No": 0.3, "Yes": 0.7})])
    res = await task.run(trial("side_effect", "no_first", method="population_forecast"), None, p)
    assert res.status is Status.VALID and res.outcome["judged_permissible"] == 0.7
    assert p.schemas[0] == {
        "type": "object",
        "properties": {"No": {"type": "number"}, "Yes": {"type": "number"}},
        "required": ["No", "Yes"],
        "additionalProperties": False,
    }
    prompt = p.seen[0][0].content
    assert task.target_population() in prompt and "keys No, Yes" in prompt
    bad = Scripted([json.dumps({"Yes": 0.7, "No": 0.7}), json.dumps({"Yes": 0.9})])
    res = await task.run(trial("side_effect", method="population_forecast"), None, bad)
    assert res.status is Status.INVALID and res.format_repairs == 1


async def test_persona_forecast_uses_the_persona(demo_cfg: StudyConfig) -> None:
    task = build_tasks(demo_cfg)["holdout"]
    persona = personas.load(demo_cfg.path(demo_cfg.personas.file))[0]
    p = Scripted([json.dumps({"Yes": 0.4, "No": 0.6})])
    res = await task.run(trial("greater_good", method="persona_forecast"), persona, p)
    assert res.outcome["judged_permissible"] == 0.4
    assert persona.describe() in p.seen[0][0].content


def test_design_counterbalances_answer_order(demo_cfg: StudyConfig) -> None:
    tasks = build_tasks(demo_cfg)
    trials = design.build_manifest(demo_cfg, tasks, personas.load_sets(demo_cfg))
    hold = [t for t in trials if t.task_id == "holdout" and t.arm_id == "baseline"]
    assert hold
    for cond in ("greater_good", "side_effect"):
        orders = [t.factors["answer_order"] for t in hold if t.condition == cond]
        assert orders.count("yes_first") == orders.count("no_first")


def test_stimuli_match_the_benchmark_and_carry_attribution() -> None:
    stim = yaml.safe_load((REPO / "data/stimuli/trolley_hauser2007.yaml").read_text())
    template = yaml.safe_load((REPO / "data/benchmarks/templates/holdout_hauser4.yaml").read_text())
    original = stim["variants"]["original"]
    assert set(original["conditions"]) == set(template["conditions"]) == set(template["contrast"])
    assert "Hauser" in original["source"] and "Appendix" in original["source"]
    assert list(stim["variants"]) == ["original"]
    for c in original["conditions"].values():
        assert c["question"].startswith("Is it morally permissible for ")
        assert c["question"].endswith("?") and len(c["scenario"]) > 500
    assert "{first}" in stim["response_instruction"] and "{second}" in stim["response_instruction"]


def test_forecast_prompts_name_no_study_and_no_result() -> None:
    template = yaml.safe_load((REPO / "data/benchmarks/templates/holdout_hauser4.yaml").read_text())
    stim = yaml.safe_load((REPO / "data/stimuli/trolley_hauser2007.yaml").read_text())
    text = (template["target_population"] + json.dumps(stim["forecast"])).lower()
    text += stim["design_disclosure"].lower()
    for word in ("hauser", "many labs", "trolley", "double effect", "%", "permissible rate"):
        assert word not in text


def test_probe_uses_the_focal_condition(demo_cfg: StudyConfig) -> None:
    task = build_tasks(demo_cfg)["holdout"]
    text = probes.stimulus_text(task)
    focal = task.stimuli["variants"]["original"]["conditions"][task.contrast[0]]
    assert focal["scenario"] in text and focal["question"] in text
