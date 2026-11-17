"""Distil a finished chat turn into durable, ranked semantic memories.

This module is the WRITER the ledger never had: storage
(:func:`agent_core.memory.load_semantic_memory` /
:func:`agent_core.memory.save_semantic_memory`), embedding ranking and the
``SEMANTIC MEMORY`` prompt block all existed, but nothing ever appended an
entry, so the block was permanently empty.

Design constraints, in order of importance:

1. **No LLM calls.**  Extraction is regex + heuristics only, so remembering a
   turn costs no VRAM and no LM Studio round trip.  A summariser would defeat
   the point of a memory store that exists to *save* context.
2. **Never fail a turn.**  :func:`record_turn` swallows every error; a broken
   ledger must not kill a chat turn (the same contract as
   :func:`agent_core.memory.load_semantic_memory`).
3. **Write only what matters.**  A turn that changed no file and reported no
   error or decision stores nothing — otherwise the ledger fills with chat
   noise and the prompt block degrades into a transcript.
4. **Bounded.**  Entries are capped at
   :data:`agent_core.memory.types.MAX_ENTRY_CHARS`, the ledger at
   :data:`agent_core.memory.types.MAX_ENTRIES`, and consolidation merges
   near-duplicates so a long session accumulates facts, not repetitions.

The extraction/update shape (extract facts, then merge into the store rather
than append blindly) follows Mem0 (arXiv 2504.19413); the "what is worth
remembering" split into errors / decisions / preferences / files follows the
importance signal of Generative Agents (arXiv 2304.03442).
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from .scoring import ENTRY_KIND_IMPORTANCE, rank_memories
from .types import (
    MAX_ENTRIES,
    MAX_ENTRY_CHARS,
    load_semantic_memory,
    save_semantic_memory,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DUPLICATE_JACCARD",
    "record_turn",
    "extract_signals",
    "consolidate",
]

#: Token-set overlap at or above which two entries of the same ``kind`` are the
#: same fact re-stated.  Chosen instead of a cosine threshold on purpose: the
#: writer must not need an embedder.
DUPLICATE_JACCARD = 0.92

#: Usage count after which ``last_used_at`` write-back is debounced (>1 hour).
_REUSE_BUMP_MAX = 1000

# Lifecycle markers printed by the run harness.  ``[verify] ruff style ...`` is
# ADVISORY and must not count as an error; only a non-zero exit code, a failed
# test id or a traceback does.
_FAILED_LINE_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)
_TRACEBACK_RE = re.compile(r"^Traceback \(most recent call last\)", re.MULTILINE)
_EXIT_CODE_RE = re.compile(r"\[EXIT CODE: ([1-9]\d*)\]")
_DECISION_RE = re.compile(r"^\s*Decision\s*[:#]\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_ERROR_LINE_RE = re.compile(
    r"^(?:[A-Za-z_][\w.]*(?:Error|Exception)|AssertionError)\s*:\s*(.+)$",
    re.MULTILINE,
)

# Preference markers.  Deliberately narrow: only the user's own messages are
# scanned, a question is skipped, and one preference per turn is kept.
_PREFERENCE_RE = re.compile(
    r"\b(always|never|from now on|prefer|don't|do not|avoid|make sure|must)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

#: Prefix per entry kind, so the rendered line states what it is.
_KIND_PREFIX = {
    "file": "Touched",
    "error": "Error",
    "decision": "Decision",
    "preference": "Preference",
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _entry_id(kind: str, text: str) -> str:
    """Stable 12-hex id for an entry — the primitive behind deduplication.

    The same fact re-derived in a later turn hashes to the same id, so the
    writer bumps ``use_count`` instead of appending a near-identical line.
    """
    digest = hashlib.sha256(f"{kind}\x00{text}".encode("utf-8", "replace"))
    return digest.hexdigest()[:12]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{3,}", str(text or "").lower()))


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 1.0 if a == b else 0.0
    return len(a & b) / len(a | b)


def _clip(text: str) -> str:
    """Whitespace-normalise and cap *text* at ``MAX_ENTRY_CHARS``."""
    clean = " ".join(str(text or "").split())
    return clean[:MAX_ENTRY_CHARS].strip()


def _message_texts(messages: Iterable[Any], roles: Sequence[str]) -> list[str]:
    """``content`` of the messages whose ``role`` is in *roles*, in order."""
    out: list[str] = []
    for message in messages or []:
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role") or "").strip().lower() not in roles:
            continue
        content = str(message.get("content") or "").strip()
        if content:
            out.append(content)
    return out


def _first_error(texts: Sequence[str]) -> str | None:
    """The most specific failure marker in *texts* (``None`` when clean).

    Priority: a traceback's own exception line, then a failed test id, then a
    bare non-zero exit code.  Checks newest text first so a fixed error from
    earlier in the turn is not remembered as still open.
    """
    for text in reversed(list(texts)):
        match = _ERROR_LINE_RE.search(text)
        if match:
            return _clip(match.group(1))
        match = _FAILED_LINE_RE.search(text)
        if match:
            return _clip(f"failing: {match.group(1)}")
        if _TRACEBACK_RE.search(text):
            return _clip(f"traceback in {text[:80]}")
    for text in reversed(list(texts)):
        match = _EXIT_CODE_RE.search(text)
        if match:
            return _clip(f"exit code {match.group(1)}")
    return None


def _first_decision(texts: Sequence[str]) -> str | None:
    """The last explicit ``Decision:`` line across *texts* (``None`` if none)."""
    found: str | None = None
    for text in texts:
        for match in _DECISION_RE.finditer(text):
            found = _clip(match.group(1))
    return found


def _first_preference(user_texts: Sequence[str]) -> str | None:
    """One imperative preference from the user's messages (``None`` if none).

    Skips questions — "why must this fail?" is a question, not a rule — and
    returns the first qualifying sentence of the NEWEST user message.
    """
    for text in reversed(list(user_texts)):
        for sentence in _SENTENCE_SPLIT_RE.split(" ".join(text.split())):
            candidate = sentence.strip()
            if not candidate or candidate.endswith("?") or len(candidate) < 12:
                continue
            if not _PREFERENCE_RE.search(candidate):
                continue
            return _clip(candidate)
    return None


def _new_entry(kind: str, text: str, *, now: str | None = None) -> dict[str, Any]:
    """Build a ledger entry for one distilled fact.

    Every field lives inside ``metadata`` because
    :func:`agent_core.memory.load_semantic_memory` rebuilds entries from
    ``metadata.text`` alone — a top-level field would be dropped on reload.
    """
    body = _clip(text)
    stamp = now or _now_iso()
    return {
        "metadata": {
            "text": body,
            "kind": kind,
            "id": _entry_id(kind, body),
            "importance": ENTRY_KIND_IMPORTANCE.get(kind, 0.5),
            "created_at": stamp,
            "last_used_at": stamp,
            "use_count": 1,
            "schema": 1,
        }
    }


def extract_signals(
    turn_messages: Sequence[Any] | None,
    mutated_files: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Distil one turn's messages into ledger entries (no I/O, no LLM).

    Extracts, in this order: the turn's error (if it failed), the turn's
    decision (if it recorded one), the user's stated preference (if any), and
    one entry per mutated file.  Returns ``[]`` for a turn that changed
    nothing and reported nothing — the gate that keeps chat noise out of the
    ledger.
    """
    entries: list[dict[str, Any]] = []
    messages = list(turn_messages or [])
    assistant_texts = _message_texts(messages, ("assistant",))
    user_texts = _message_texts(messages, ("user",))
    tool_texts = _message_texts(messages, ("tool",))
    # Failure markers appear in assistant prose ("the suite failed with ...")
    # and in captured tool output; scan both, newest last.
    error = _first_error(assistant_texts + tool_texts)
    if error:
        entries.append(_new_entry("error", error))
    decision = _first_decision(assistant_texts)
    if decision:
        entries.append(_new_entry("decision", decision))
    preference = _first_preference(user_texts)
    if preference:
        entries.append(_new_entry("preference", preference))
    seen: set[str] = set()
    for path in mutated_files or []:
        clean = str(path or "").strip().replace("\\", "/").strip("./")
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        entries.append(_new_entry("file", clean))
    return entries


def _merge(existing: dict[str, Any], new: dict[str, Any]) -> None:
    """Fold *new* into *existing* in place (usage bump, never duplication)."""
    meta = existing.setdefault("metadata", {})
    now = _now_iso()
    try:
        meta["use_count"] = int(meta.get("use_count") or 1) + 1
    except (TypeError, ValueError):
        meta["use_count"] = 2
    if meta["use_count"] > _REUSE_BUMP_MAX:
        meta["use_count"] = _REUSE_BUMP_MAX
    meta["last_used_at"] = now
    # Keep the strongest importance seen for this fact.
    try:
        meta["importance"] = max(
            float(meta.get("importance") or 0.0),
            float(new.get("metadata", {}).get("importance") or 0.0),
        )
    except (TypeError, ValueError):
        pass
    meta.setdefault("created_at", now)


def consolidate(
    existing: Sequence[Any] | None,
    incoming: Iterable[dict[str, Any]],
    *,
    max_entries: int = MAX_ENTRIES,
) -> list[dict[str, Any]]:
    """Merge *incoming* entries into *existing*, deduplicated and bounded.

    Three rules, applied per incoming entry:

    * same ``id`` → a usage bump on the stored entry (exact repeat);
    * same ``kind`` and token-set overlap ``>= DUPLICATE_JACCARD`` → the stored
      entry absorbs it (re-stated fact), and the stored text is kept because it
      is already in the ledger;
    * otherwise → append.

    When the result exceeds *max_entries* it is pruned to the highest fused
    score (:func:`agent_core.memory.scoring.rank_memories`: recency +
    importance, since there is no query at prune time), which drops the
    stalest, least important lines first.
    """
    merged: list[dict[str, Any]] = []
    for entry in existing or []:
        if not isinstance(entry, Mapping):
            continue
        metadata = entry.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {"text": entry.get("text", "")}
        text = str(metadata.get("text") or "").strip()
        if not text:
            continue
        merged.append({"metadata": dict(metadata)})
    for new in incoming or []:
        new_meta = new.get("metadata") or {}
        text = str(new_meta.get("text") or "").strip()
        if not text:
            continue
        kind = str(new_meta.get("kind") or "note")
        new_id = str(new_meta.get("id") or _entry_id(kind, text))
        absorbed = False
        for candidate in merged:
            meta = candidate["metadata"]
            if str(meta.get("id") or "") == new_id:
                _merge(candidate, new)
                absorbed = True
                break
            if str(meta.get("kind") or "") != kind:
                continue
            if _jaccard(str(meta.get("text") or ""), text) >= DUPLICATE_JACCARD:
                _merge(candidate, new)
                absorbed = True
                break
        if not absorbed:
            merged.append({"metadata": dict(new_meta, id=new_id)})
    if max_entries > 0 and len(merged) > max_entries:
        ranked = rank_memories(merged, k=max_entries)
        merged = [{"metadata": item["metadata"]} for item in ranked]
    return merged


def record_turn(
    workspace: Any,
    turn_messages: Sequence[Any] | None,
    mutated_files: Sequence[str] | None = None,
    *,
    goal: str | None = None,
    max_entries: int = MAX_ENTRIES,
) -> int:
    """Distil the finished turn into the workspace ledger; return entries added.

    The one entry point ``Agent._finish_turn`` calls, AFTER the turn's messages
    were captured and BEFORE they are lost to ``_trim_chat_history``.  Reads
    the ledger, merges the turn's facts into it, prunes to *max_entries* and
    writes it back atomically.

    *goal* is accepted for signature stability and future goal-weighted
    pruning — the injected ranking already weights the goal
    (:func:`agent_core.memory.scoring.rank_memories`), so the writer does not
    need it.

    Returns the number of NEW entries stored (0 when the turn held nothing
    worth remembering, which is the common case).  Never raises.
    """
    try:
        incoming = extract_signals(turn_messages, mutated_files)
        if not incoming:
            return 0
        existing = load_semantic_memory(workspace)
        merged = consolidate(existing, incoming, max_entries=max_entries)
        before = len(existing)
        if save_semantic_memory(workspace, merged, max_entries=max_entries) is None:
            return 0
        return max(0, len(merged) - before)
    except Exception:  # noqa: BLE001 - remembering must never fail a turn
        logger.debug("turn distillation failed (no-op)", exc_info=True)
        return 0
