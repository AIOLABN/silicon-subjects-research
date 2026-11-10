from collections import Counter

from silicon_subjects import design, personas
from silicon_subjects.config import StudyConfig
from silicon_subjects.tasks import build_tasks


def test_manifest_is_deterministic_and_sized(demo_cfg: StudyConfig) -> None:
    tasks = build_tasks(demo_cfg)
    ps = personas.load_sets(demo_cfg)
    a = design.build_manifest(demo_cfg, tasks, ps)
    b = design.build_manifest(demo_cfg, tasks, ps)
    assert [t.trial_id for t in a] == [t.trial_id for t in b]
    cells = design.summarize(a)
    assert cells[("opus55", "framing", "baseline", "gain", "original")] == 100
    assert cells[("opus55", "framing", "persona", "gain", "original")] == 100
    assert cells[("opus55", "obedience", "paraphrase", "base", "para1")] == 50


def test_persona_arm_uses_every_profile_twice(demo_cfg: StudyConfig) -> None:
    tasks = build_tasks(demo_cfg)
    ps = personas.load_sets(demo_cfg)
    trials = [
        t
        for t in design.build_manifest(demo_cfg, tasks, ps)
        if (t.model_id, t.task_id, t.arm_id, t.condition)
        == ("opus55", "framing", "persona", "loss")
    ]
    counts = Counter(t.persona_id for t in trials)
    assert len(counts) == 50 and set(counts.values()) == {2}


def test_option_order_exactly_balanced(demo_cfg: StudyConfig) -> None:
    tasks = build_tasks(demo_cfg)
    ps = personas.load_sets(demo_cfg)
    trials = [
        t
        for t in design.build_manifest(demo_cfg, tasks, ps)
        if (t.model_id, t.task_id, t.arm_id, t.condition)
        == ("opus55", "framing", "baseline", "gain")
    ]
    assert Counter(t.factors["option_order"] for t in trials) == {
        "certain_first": 50,
        "risky_first": 50,
    }


def test_persona_marginals_exact(demo_cfg: StudyConfig) -> None:
    ps = personas.load(demo_cfg.path(demo_cfg.personas.file))
    m = personas.marginals(ps, demo_cfg.personas.attributes)
    assert m["gender"] == {"woman": 26, "man": 24}
    assert all(sum(v.values()) == 50 for v in m.values())


def test_render_keeps_acronyms() -> None:
    text = personas.Persona("x", {"region_of_the_US": "South"}).render()
    assert "- Region of the US: South" in text
