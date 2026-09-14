"""Prompt-injection detection and untrusted-content sanitization.

Treat heuristics as one layer, not the defense. The structural defenses are:
1. untrusted text is always wrapped in labeled delimiters and described as *data* in the system
   prompt (`agents/prompts.py`);
2. tools and data access are enforced server-side from the verified principal, so an injected
   instruction cannot widen what the agent is allowed to do;
3. the validator inspects every answer before it leaves the graph.

This module adds detection (to flag, log and, for egregious user input, block) and
sanitization (so untrusted text cannot forge our delimiters).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# (pattern, weight, signal name). Weights are additive; see thresholds below.
_SIGNALS: tuple[tuple[re.Pattern[str], float, str], ...] = tuple(
    (re.compile(p, re.IGNORECASE), w, name)
    for p, w, name in (
        (
            r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|system)\b.{0,20}\b(instructions?|rules|prompts?|guidelines)",
            0.6,
            "instruction_override",
        ),
        (
            r"\b(reveal|print|show|repeat|leak|output)\b.{0,40}\b(system prompt|hidden prompt|instructions|developer message|secret|api[ _-]?keys?|passwords?|credentials)",
            0.6,
            "exfiltration_request",
        ),
        (r"\byou are (now|no longer)\b", 0.3, "persona_hijack"),
        (r"\b(developer|god|jailbreak|dan) mode\b", 0.5, "jailbreak_mode"),
        (
            r"\bact as\b.{0,30}\b(unrestricted|unfiltered|jailbroken|without (any )?restrictions)",
            0.5,
            "jailbreak_persona",
        ),
        (r"(<\|im_start\|>|<\|system\|>|\[/?INST\]|###\s*system|BEGIN SYSTEM PROMPT)", 0.5, "control_tokens"),
        (r"</?\s*(evidence|untrusted|system|memory|tool_result)\b", 0.4, "delimiter_forgery"),
        (r"\b(new|updated) (instructions|rules)\s*:", 0.3, "instruction_injection"),
        (
            r"\b(send|post|upload|exfiltrate|forward)\b.{0,50}\b(https?://|webhook|email address)",
            0.5,
            "data_exfil_channel",
        ),
        (
            r"\b(other|another|all) (users?|customers?)('s)?\b.{0,30}\b(data|conversations?|history|accounts?|memories)",
            0.4,
            "cross_user_access",
        ),
    )
)

FLAG_THRESHOLD = 0.3
BLOCK_THRESHOLD = 0.9

_CONTROL_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069]")
_DELIMITER_TAG = re.compile(r"<(/?)\s*(evidence|untrusted|system|memory|tool_result)\b", re.IGNORECASE)


@dataclass(frozen=True)
class InjectionAssessment:
    score: float
    signals: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return self.score >= FLAG_THRESHOLD

    @property
    def should_block(self) -> bool:
        return self.score >= BLOCK_THRESHOLD


def normalize_text(text: str) -> str:
    """NFKC-normalize and strip invisible/control characters used to hide instructions."""
    return _CONTROL_CHARS.sub("", unicodedata.normalize("NFKC", text))


def assess_injection(text: str) -> InjectionAssessment:
    normalized = normalize_text(text)
    hits = [(w, name) for pattern, w, name in _SIGNALS if pattern.search(normalized)]
    return InjectionAssessment(score=round(min(1.0, sum(w for w, _ in hits)), 2), signals=[n for _, n in hits])


def sanitize_untrusted(text: str, *, max_chars: int = 6000) -> str:
    """Make untrusted text safe to embed inside our prompt delimiters.

    Content is preserved for the model to reason over; only our own delimiter syntax is
    defanged so a document cannot close an `<evidence>` block and start speaking as the system.
    """
    cleaned = normalize_text(text)
    cleaned = _DELIMITER_TAG.sub(lambda m: f"‹{m.group(1)}{m.group(2)}", cleaned)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + " …[truncated]"
    return cleaned
