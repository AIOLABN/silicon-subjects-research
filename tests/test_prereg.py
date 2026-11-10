from silicon_subjects import prereg
from silicon_subjects.config import StudyConfig


def test_freeze_then_tamper_is_detected(demo_cfg: StudyConfig) -> None:
    prereg.freeze(demo_cfg)
    assert prereg.verify(demo_cfg).matches
    stim = demo_cfg.path(demo_cfg.tasks[0].stimuli)
    stim.write_text(stim.read_text().replace("600 people", "601 people", 1))
    status = prereg.verify(demo_cfg)
    assert not status.matches
    assert status.changed == ["framing.stimuli"]
