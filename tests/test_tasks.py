from silicon_subjects.config import StudyConfig
from silicon_subjects.providers.base import ChatMessage, Completion
from silicon_subjects.tasks import TrialSpec, build_tasks


class Scripted:
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.seen: list[list[ChatMessage]] = []

    async def complete(
        self, system: str, messages: list[ChatMessage], *, seed: int, **kw: object
    ) -> Completion:
        self.seen.append(list(messages))
        return Completion(self.replies.pop(0), "end_turn", "scripted-model")

    async def aclose(self) -> None:
        return None


def trial(task_id: str, condition: str, **factors: str) -> TrialSpec:
    return TrialSpec("t1", "m", task_id, "baseline", condition, "original", None, 0, factors, 1)


async def test_framing_maps_letters_through_counterbalancing(demo_cfg: StudyConfig) -> None:
    task = build_tasks(demo_cfg)["framing"]
    res = await task.run(
        trial("framing", "gain", option_order="risky_first"), None, Scripted(["CHOICE: B"])
    )
    assert res.outcome == {"choice": "certain", "chose_certain": 1}


async def test_obedience_full_compliance_reaches_endpoint(demo_cfg: StudyConfig) -> None:
    task = build_tasks(demo_cfg)["obedience"]
    p = Scripted(["DECISION: CONTINUE"] * 11)
    res = await task.run(trial("obedience", "base"), None, p)
    assert res.outcome["continued_past_150"] == 1
    assert res.outcome["max_volts_administered"] == 150
    assert "Get me out of here" in p.seen[-1][-1].content


async def test_obedience_stops_after_four_prods(demo_cfg: StudyConfig) -> None:
    task = build_tasks(demo_cfg)["obedience"]
    replies = ["DECISION: CONTINUE"] * 7 + ["DECISION: STOP"] * 5
    res = await task.run(trial("obedience", "base"), None, Scripted(replies))
    assert res.outcome["continued_past_150"] == 0
    assert res.outcome["stopped_at_volts"] == 120
    assert res.outcome["prods_used"] == 4


async def test_obedience_prod_sequence_resets_and_counts(demo_cfg: StudyConfig) -> None:
    task = build_tasks(demo_cfg)["obedience"]
    replies = ["DECISION: STOP", "DECISION: CONTINUE"] * 2 + ["DECISION: CONTINUE"] * 3
    res = await task.run(trial("obedience", "modeled_refusal"), None, Scripted(replies))
    assert res.outcome == {
        "continued_past_150": 1,
        "max_volts_administered": 150,
        "prods_used": 2,
        "stopped_at_volts": None,
    }


async def test_format_repair_then_refusal_is_recorded(demo_cfg: StudyConfig) -> None:
    task = build_tasks(demo_cfg)["framing"]
    res = await task.run(
        trial("framing", "loss", option_order="certain_first"),
        None,
        Scripted(["I think A.", "As an AI, I can't take part in this."]),
    )
    assert res.format_repairs == 1
    assert res.status == "refusal_ooc"


def persona_trial(task_id: str, condition: str, **factors: str) -> TrialSpec:
    return TrialSpec("t1", "m", task_id, "persona", condition, "original", "p1", 0, factors, 1)


async def test_persona_trials_never_ask_the_model_to_answer_as_itself(
    demo_cfg: StudyConfig,
) -> None:
    from silicon_subjects.personas import Persona

    persona = Persona("p1", {"age": "30-44 years old"})
    tasks = build_tasks(demo_cfg)
    cases = [
        ("framing", persona_trial("framing", "gain", option_order="certain_first"), "CHOICE: A"),
        (
            "holdout",
            persona_trial("holdout", "side_effect", answer_order="yes_first"),
            "ANSWER: No",
        ),
        ("obedience", persona_trial("obedience", "base"), "DECISION: STOP"),
    ]
    for task_id, t, reply in cases:
        p = Scripted([reply] * 5)
        res = await tasks[task_id].run(t, persona, p)
        system = res.transcript[0].content
        assert "described below" in system
        for turn in p.seen[0]:
            assert "as yourself" not in turn.content, task_id
            assert "participant described to you" in turn.content, task_id


async def test_baseline_prompt_has_no_persona_instruction(demo_cfg: StudyConfig) -> None:
    task = build_tasks(demo_cfg)["framing"]
    p = Scripted(["CHOICE: A"])
    await task.run(trial("framing", "gain", option_order="certain_first"), None, p)
    assert p.seen[0][0].content.endswith(task.stimuli["response_instruction"])
    assert "Answer as yourself" in p.seen[0][0].content


async def test_persona_format_repair_restates_the_persona_instruction(
    demo_cfg: StudyConfig,
) -> None:
    from silicon_subjects.personas import Persona

    task = build_tasks(demo_cfg)["framing"]
    p = Scripted(["I would pick the first one.", "CHOICE: A"])
    res = await task.run(
        persona_trial("framing", "gain", option_order="certain_first"),
        Persona("p1", {"age": "65 or older"}),
        p,
    )
    assert res.format_repairs == 1 and res.status == "valid"
    repair = p.seen[1][-1].content
    assert "participant described to you" in repair and "as yourself" not in repair


def test_plan_fails_early_without_a_persona_instruction(demo_cfg: StudyConfig) -> None:
    import pytest

    from silicon_subjects import design, personas

    tasks = build_tasks(demo_cfg)
    del tasks["framing"].stimuli["persona_response_instruction"]
    with pytest.raises(ValueError, match="persona_response_instruction"):
        design.build_manifest(demo_cfg, tasks, personas.load_sets(demo_cfg))
