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


def parse_ab_ba(text: str) -> ParseResult:
    compact = re.sub(r"[^A-Za-z]", "", text).upper()
    exact = re.search(r"\b(AB|BA)\b", text.upper())
    if exact:
        return ParseResult(exact.group(1), "valid")
    if compact in {"AB", "BA"}:
        return ParseResult(compact, "valid")
    return ParseResult(None, "unparseable", "no standalone AB/BA answer")


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
