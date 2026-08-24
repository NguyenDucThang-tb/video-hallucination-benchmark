from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ParseResult:
    value: str | None
    status: str
    error: str | None = None


def _unique_matches(pattern: str, text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(pattern, text, flags=re.IGNORECASE)))


def parse_yes_no(text: str) -> ParseResult:
    matches = [x.lower() for x in _unique_matches(r"\b(yes|no)\b", text)]
    if not matches:
        return ParseResult(None, "unparseable", "no yes/no token")
    if len(matches) > 1:
        return ParseResult(None, "ambiguous", "both yes and no present")
    return ParseResult(matches[0], "valid")


def parse_leading_yes_no(text: str) -> ParseResult:
    """EventHallusion's official parser accepts yes/no only at the start."""
    match = re.match(r"\s*(yes|no)\b", text, flags=re.IGNORECASE)
    if match is None:
        return ParseResult(None, "unparseable", "response does not start with yes/no")
    return ParseResult(match.group(1).lower(), "valid")


def parse_ab_ba(text: str) -> ParseResult:
    compact = re.sub(r"[^A-Za-z]", "", text).upper()
    exact = re.search(r"\b(AB|BA)\b", text.upper())
    if exact:
        return ParseResult(exact.group(1), "valid")
    if compact in {"AB", "BA"}:
        return ParseResult(compact, "valid")
    lowered = text.lower()
    if "not clear" in lowered or "no clear" in lowered:
        return ParseResult(None, "unparseable", "model reported unclear action order")
    mentions = list(re.finditer(r"action\s*([ab])", lowered))
    if len(mentions) >= 2:
        first, second = mentions[0].group(1).upper(), mentions[1].group(1).upper()
        between = lowered[mentions[0].end():mentions[1].start()]
        if "after" in between:
            first, second = second, first
        value = first + second
        if value in {"AB", "BA"}:
            return ParseResult(value, "valid")
    return ParseResult(None, "unparseable", "no standalone AB/BA answer")


def parse_vidhalluc_tsh_official(text: str) -> ParseResult:
    """Port of VidHalluc ``eval/evaluation/eval_tsh.py``.

    The upstream parser intentionally does not strip punctuation or surrounding
    whitespace before its exact AB/BA checks. Keep that behavior here so the
    reproduction metric remains distinguishable from the more permissive local
    diagnostic parser above.
    """
    answer = text.lower()
    if answer in {"ab", "ba", "a", "b"}:
        return ParseResult(answer.upper(), "valid")
    if "not clear" in answer or "no clear" in answer:
        return ParseResult(None, "unparseable", "official parser returned None")

    has_action_a = "action a" in answer
    has_action_b = "action b" in answer
    if has_action_a and has_action_b:
        if "before" in answer:
            match = re.search(
                r"(action [ab]|[ab][\.\)])[^before]+before[^action]*?(action [ab]|[ab][\.\)])",
                answer,
            )
            if match:
                return ParseResult(match.group(1)[-1].upper() + match.group(2)[-1].upper(), "valid")
        elif "then" in answer:
            match = re.search(
                r"(action [ab]|[ab][\.\)])[^then]+then[^action]*?(action [ab]|[ab][\.\)])",
                answer,
            )
            if match:
                return ParseResult(match.group(1)[-1].upper() + match.group(2)[-1].upper(), "valid")
        elif "after" in answer:
            match = re.search(
                r"(action [ab]|[ab][\.\)])[^after]+after[^action]*?(action [ab]|[ab][\.\)])",
                answer,
            )
            if match:
                return ParseResult(match.group(2)[-1].upper() + match.group(1)[-1].upper(), "valid")

        positions = [
            (match.start(), match.group(1)[-1].upper())
            for match in re.finditer(r"(action [ab]|[ab][\.\)])", answer)
        ]
        positions.sort()
        return ParseResult("".join(action for _, action in positions), "valid")
    if has_action_a:
        return ParseResult("A", "valid")
    if has_action_b:
        return ParseResult("B", "valid")
    return ParseResult(None, "unparseable", "official parser returned None")


def parse_vidhalluc_sth(text: str) -> tuple[ParseResult, str | None]:
    """Parse the two fields consumed by VidHalluc's official STH evaluator."""
    if ", Locations: " in text:
        scene_part, locations = text.split(", Locations: ", 1)
    else:
        scene_part, locations = text, None
    # Preserve upstream's split behavior, then expose invalid values as None
    # instead of silently mutating them to the negative class at record level.
    scene_change = scene_part.split(": ", 1)[-1].split(",", 1)[0].strip().lower()
    if scene_change not in {"yes", "no"}:
        return ParseResult(None, "unparseable", "official STH field is not yes/no"), locations
    return ParseResult(scene_change, "valid"), locations.strip() if locations is not None else None


def parse_mcq(text: str, choices: Mapping[str, str] | None = None) -> ParseResult:
    letters = [x.upper() for x in _unique_matches(r"\b([A-D])\b", text)]
    if len(letters) == 1:
        return ParseResult(letters[0], "valid")
    if len(letters) > 1:
        return ParseResult(None, "ambiguous", f"multiple choices: {letters}")
    if choices:
        lowered = text.casefold()
        hits = [key.upper() for key, value in choices.items() if value.casefold() in lowered]
        if len(hits) == 1:
            return ParseResult(hits[0], "valid")
        if len(hits) > 1:
            return ParseResult(None, "ambiguous", f"multiple option texts: {hits}")
    return ParseResult(None, "unparseable", "no A-D choice")
