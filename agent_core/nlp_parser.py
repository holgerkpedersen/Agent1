"""NLP parsing and intent classification for the chat_nlp entry point.

This module defines the contract between free-text user input and the
natural-language tool loop (``agent.chat_nlp``):

System-prompt contract
----------------------
``chat_nlp`` builds the loop's system prompt itself (workspace facts, shell
hints, tool rules); ``NLPParser`` is NOT the loop's parser — the loop uses
native OpenAI tool-calling.  This module powers the lighter REPL pre-routing
intent checks (e.g. ``clear``/``help``/``exit`` keyword routing) and any
caller that needs a structured ``(intent, entities)`` view of user input.

Intent classes
--------------
:class:`IntentType` — ``IMPLEMENT``, ``CLEAR``, ``CHAT``, ``HELP``,
``EXIT``, ``UNKNOWN``.  ``classify_intent`` scores keyword overlap per
intent (``_score_intent``); below the confidence threshold the input is
treated as ``CHAT`` (free conversation for the tool loop).

Extraction guarantees
---------------------
- ``ParsedInput`` is immutable (frozen dataclass) with ``intent``,
  ``raw_text``, ``entities``, ``keywords``, ``confidence``.
- ``extract_files`` returns file-like tokens (word characters with dots,
  length > 2, not all-uppercase) — a heuristic, not a path validator.
- ``_score_intent`` is deterministic and pure: same text, same score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class IntentType(Enum):
    """Supported user intents for command routing."""

    IMPLEMENT = "implement"
    CLEAR = "clear"
    CHAT = "chat"
    HELP = "help"
    EXIT = "exit"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ParsedInput:
    """Structured result of NLP parsing."""

    intent: IntentType
    raw_text: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    confidence: float = 0.0


_INTENT_KEYWORDS: dict[IntentType, set[str]] = {
    IntentType.IMPLEMENT: {"implement", "create", "generate", "write", "build", "add"},
    IntentType.CLEAR: {"clear", "reset", "wipe", "flush"},
    IntentType.HELP: {"help", "usage", "how to", "what can"},
    IntentType.EXIT: {"exit", "quit", "bye", "stop"},
}

_FILE_PATTERN = re.compile(r"\b([a-zA-Z_]\w*(?:\.\w+)?)\b")


def _score_intent(text_lower: str, keywords: set[str]) -> float:
    """Return a simple keyword-match score in [0, 1]."""
    if not text_lower or not keywords:
        return 0.0
    words = set(re.findall(r"\w+", text_lower))
    hits = len(words & keywords)
    return min(hits / max(len(keywords), 1), 1.0)


def _extract_entities(text: str) -> dict[str, list[str]]:
    """Extract file-like entities from *text*."""
    candidates: set[str] = set()
    for match in _FILE_PATTERN.finditer(text):
        token = match.group(1)
        if len(token) > 2 and not token.isupper():
            candidates.add(token)
    return {"files": sorted(candidates)}


def _detect_keywords(text: str) -> list[str]:
    """Return a deduplicated list of significant tokens."""
    seen: set[str] = set()
    result: list[str] = []
    for token in re.findall(r"\w+", text.lower()):
        if len(token) > 2 and token not in seen:
            seen.add(token)
            result.append(token)
    return result


class NLPParser:
    """Lightweight intent classifier and entity extractor."""

    def __init__(self, confidence_threshold: float = 0.3) -> None:
        self._threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, text: str) -> ParsedInput:
        """Parse *text* and return a structured ``ParsedInput``."""
        intent, confidence = self.classify_intent(text)
        entities = _extract_entities(text)
        keywords = _detect_keywords(text)
        return ParsedInput(
            intent=intent,
            raw_text=text,
            entities=entities,
            keywords=keywords,
            confidence=confidence,
        )

    def classify_intent(self, text: str) -> tuple[IntentType, float]:
        """Return ``(intent, confidence)`` for *text*."""
        lower = text.strip().lower()
        best_intent: IntentType = IntentType.UNKNOWN
        best_score: float = 0.0

        for intent_type, keywords in _INTENT_KEYWORDS.items():
            score = _score_intent(lower, keywords)
            if score > best_score:
                best_score = score
                best_intent = intent_type

        if best_score < self._threshold and best_intent != IntentType.UNKNOWN:
            return IntentType.CHAT, 0.0

        return best_intent, round(best_score, 2)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_command(self, text: str) -> bool:
        """Return ``True`` when *text* maps to a non-chat intent."""
        intent, _ = self.classify_intent(text)
        return intent not in (IntentType.CHAT, IntentType.UNKNOWN)

    def extract_files(self, text: str) -> list[str]:
        """Shortcut for file entity extraction."""
        return _extract_entities(text).get("files", [])


# Module-level convenience instance
_default_parser = NLPParser()


def parse_input(text: str) -> ParsedInput:
    """Module-level shortcut wrapping ``NLPParser.parse``."""
    return _default_parser.parse(text)


def classify_intent(text: str) -> tuple[IntentType, float]:
    """Module-level shortcut wrapping ``NLPParser.classify_intent``."""
    return _default_parser.classify_intent(text)