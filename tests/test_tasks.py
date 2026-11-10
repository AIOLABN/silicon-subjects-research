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
