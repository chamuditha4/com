"""Output guardrails used by the Validator node.

Pure functions only, so each rule is unit-testable and every rejection is explainable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_CITATION = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")

_PAN_CANDIDATE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("api_key", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{20,}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("password_assignment", re.compile(r"(?i)\bpassword\s*[:=]\s*\S{6,}")),
)

# Brand safety for a regulated bank: no promises of returns, no personalised investment or
# legal advice, no abusive language.
_BRAND_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("guaranteed_returns", re.compile(r"(?i)\bguaranteed? (returns?|profits?|gains?)\b")),
    ("investment_advice", re.compile(r"(?i)\byou should (buy|sell|invest in)\b")),
    ("legal_advice", re.compile(r"(?i)\b(this constitutes|consider this) legal advice\b")),
    ("profanity", re.compile(r"(?i)\b(damn|shit|fuck\w*|crap|idiot|stupid customers?)\b")),
)


def extract_citations(text: str) -> list[int]:
    found: list[int] = []
    for group in _CITATION.findall(text):
        for part in group.split(","):
            number = int(part.strip())
            if number not in found:
                found.append(number)
    return found


def _luhn_valid(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass(frozen=True)
class RedactionResult:
    text: str
    kinds: list[str]


def redact_sensitive(text: str) -> RedactionResult:
    kinds: list[str] = []

    def pan(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            kinds.append("card_number")
            return f"[REDACTED CARD ****{digits[-4:]}]"
        return match.group(0)

    redacted = _PAN_CANDIDATE.sub(pan, text)
    for kind, pattern in _SECRET_PATTERNS:
        if pattern.search(redacted):
            kinds.append(kind)
            redacted = pattern.sub("[REDACTED SECRET]", redacted)
    return RedactionResult(redacted, kinds)


def brand_safety_violations(text: str) -> list[str]:
    return [name for name, pattern in _BRAND_RULES if pattern.search(text)]


def contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(n and n in text for n in needles)
