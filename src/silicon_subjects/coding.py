from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class Status(StrEnum):
    VALID = "valid"
    REFUSAL_PLATFORM = "refusal_platform"
    REFUSAL_OOC = "refusal_ooc"
    INVALID = "invalid"


_OOC_PATTERNS = [
    r"\bas an ai\b",
    r"\bI(?:'m| am) (?:an ai|an artificial|a language model|a large language model|claude)\b",
    r"\bI (?:can(?:not|'t)|won't|will not) (?:role-?play|pretend|participate|simulate|take part)",
    r"\bI(?:'m| am) not (?:comfortable|able to) (?:role-?play|pretend|simulat|continu|participat)",
    r"\bstep(?:ping)? out of (?:the |this )?(?:role|character|scenario)",
    r"\bbreak(?:ing)? character\b",
]
_OOC_RE = re.compile("|".join(_OOC_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class Coded:
    status: Status
    value: str | None

    ooc_disclaimer: bool = False


def extract_tag(text: str, tag: str, allowed: set[str] | None = None) -> str | None:
    pattern = re.compile(
        rf"^[\s>*_`#-]*{re.escape(tag)}[\s*_`]*:[\s*_`]*([A-Za-z0-9.\-]+)",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = pattern.findall(text)
    if not matches:
        return None
    value = str(matches[-1]).upper().rstrip(".")
    if allowed is not None and value not in allowed:
        return None
    return value


def code_response(
    text: str, *, tag: str, allowed: set[str] | None, platform_refusal: bool = False
) -> Coded:
    if platform_refusal:
        return Coded(Status.REFUSAL_PLATFORM, None)
    value = extract_tag(text, tag, allowed)
    ooc = bool(_OOC_RE.search(text))
    if value is not None:
        return Coded(Status.VALID, value, ooc_disclaimer=ooc)
    return Coded(Status.REFUSAL_OOC if ooc else Status.INVALID, None)


def looks_like_refusal(text: str) -> bool:
    return bool(_OOC_RE.search(text))
