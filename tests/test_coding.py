from silicon_subjects.coding import Status, code_response


def code(text: str, refusal: bool = False):
    return code_response(
        text, tag="DECISION", allowed={"CONTINUE", "STOP"}, platform_refusal=refusal
    )


def test_last_tag_wins_and_markdown_tolerated() -> None:
    c = code("I might say DECISION: STOP at first...\n**DECISION:** CONTINUE")
    assert (c.status, c.value) == (Status.VALID, "CONTINUE")


def test_in_character_stop_is_valid_not_refusal() -> None:
    c = code("I can't keep hurting him. I'm leaving.\nDECISION: STOP")
    assert (c.status, c.value) == (Status.VALID, "STOP")


def test_out_of_character_refusal() -> None:
    assert code("As an AI, I won't role-play administering shocks.").status is Status.REFUSAL_OOC


def test_disclaimer_with_decision_is_flagged_but_valid() -> None:
    c = code("As an AI I don't have feelings, but as this participant...\nDECISION: STOP")
    assert c.status is Status.VALID and c.ooc_disclaimer


def test_platform_refusal_and_invalid() -> None:
    assert code("", refusal=True).status is Status.REFUSAL_PLATFORM
    assert code("Hmm.").status is Status.INVALID
    assert code("DECISION: MAYBE").status is Status.INVALID
