"""Weighted ranking for stored semantic memories (pure, no I/O).

The stored ledger is a flat list of distilled lines.  Handing that list to the
prompt in file order answers "what did I ever learn?", not "what matters for
THIS prompt?" — the relevance question the caller actually has.  This module
fuses four signals into one score in ``[0, 1]``:

``score = 0.45*relevance + 0.25*goal_alignment + 0.15*recency + 0.15*importance``

* **relevance** — cosine similarity to the live user message when the embedding
  ranking produced one, else lexical overlap (so the ranking still works when
  LM Studio is down).
* **goal_alignment** — overlap with the session's pinned original goal, so a
  long run does not drift away from what the user actually asked for.
* **recency** — ``0.5 ** (age_days / 7)``: a one-week-old memory scores half.
* **importance** — the writer's own 0..1 judgement of the entry's kind
  (an error or an explicit decision outranks a passing file mention).

This weighting follows the Generative Agents retrieval function (relevance +
recency + importance, arXiv 2304.03442) with the goal term added for the
long-session case that MemGPT-style context management targets
(arXiv 2310.08560).  Every function here is pure: no ledger read, no partial
config, no clock except the injectable ``now``.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Mapping, Sequence

__all__ = [
    "W_RELEVANCE",
    "W_GOAL_ALIGNMENT",
    "W_RECENCY",
    "W_IMPORTANCE",
    "RECENCY_HALF_LIFE_DAYS",
    "ENTRY_KIND_IMPORTANCE",
    "recency_score",
    "importance_of",
    "lexical_relevance",
    "goal_alignment",
    "fuse",
    "rank_memories",
]

#: Signal weights.  They sum to 1.0, so a fully-relevant, on-goal, fresh,
#: important entry scores exactly ``1.0``.
W_RELEVANCE = 0.45
W_GOAL_ALIGNMENT = 0.25
W_RECENCY = 0.15
W_IMPORTANCE = 0.15

#: Age at which the recency term halves (Generative Agents uses a ~0.995/tick
#: decay; a week-long half-life is the equivalent for a coding session).
RECENCY_HALF_LIFE_DAYS = 7.0

#: Default ``importance`` per entry ``kind``.  The writer stamps an explicit
#: value; this table is the fallback for an older entry that has none, and the
#: single place where "how much does this kind matter" is decided.
ENTRY_KIND_IMPORTANCE: dict[str, float] = {
    "decision": 0.9,
    "error": 0.85,
    "preference": 0.8,
    "goal": 0.75,
    "file": 0.5,
    "note": 0.5,
}

#: Importance assumed when an entry carries neither a value nor a known kind.
DEFAULT_IMPORTANCE = 0.5

#: Recency assumed when an entry has no parseable timestamp: neither fresh nor
#: stale, so an undated entry is neither promoted nor buried.
UNKNOWN_RECENCY = 0.5


def _clamp01(value: Any, default: float) -> float:
    """Coerce *value* to a float in ``[0, 1]`` (``default`` when unusable)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return 0.0 if number < 0.0 else 1.0 if number > 1.0 else number


def _parse_timestamp(value: Any) -> datetime | None:
    """Best-effort parse of an ISO 8601 timestamp (``None`` when unusable)."""
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def recency_score(created_at: Any, *, now: datetime | None = None) -> float:
    """Freshness of an entry in ``[0, 1]`` — ``0.5 ** (age_days / 7)``.

    Future timestamps (clock skew) score ``1.0``; an unparseable or missing
    timestamp scores :data:`UNKNOWN_RECENCY` rather than ``0.0``, so an older
    ledger written before timestamps existed is not buried outright.
    """
    stamp = _parse_timestamp(created_at)
    if stamp is None:
        return UNKNOWN_RECENCY
    reference = now or datetime.now()
    if stamp.tzinfo is not None and reference.tzinfo is None:
        reference = reference.astimezone()
    try:
        age_days = (reference - stamp).total_seconds() / 86400.0
    except TypeError:  # naive vs aware mismatch
        age_days = (
            reference.replace(tzinfo=None) - stamp.replace(tzinfo=None)
        ).total_seconds() / 86400.0
    if age_days <= 0.0:
        return 1.0
    score = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    return _clamp01(score, UNKNOWN_RECENCY)


def importance_of(entry: Any) -> float:
    """The entry's stored importance, falling back to its ``kind`` weight.

    Reads ``metadata`` of a ledger dict (or a bare string).  Explicit
    ``importance`` wins; otherwise the ``kind`` table decides; otherwise
    :data:`DEFAULT_IMPORTANCE`.
    """
    metadata: Mapping[str, Any] = {}
    if isinstance(entry, Mapping):
        raw = entry.get("metadata")
        metadata = raw if isinstance(raw, Mapping) else {}
    stored = metadata.get("importance")
    if stored is not None:
        return _clamp01(stored, DEFAULT_IMPORTANCE)
    kind = str(metadata.get("kind") or "").strip().lower()
    return ENTRY_KIND_IMPORTANCE.get(kind, DEFAULT_IMPORTANCE)


def _content_tokens(text: Any) -> set[str]:
    """Content words of *text* (delegates to the session-wide tokenizer).

    Reuses :func:`agent_core.history_compaction._content_tokens` so this module
    introduces no second stopword list — the same tokens then drive the goal
    relevance check and the memory ranking.  Imported lazily because
    ``history_compaction`` imports this package.
    """
    try:
        from agent_core.history_compaction import _content_tokens as tokenizer

        return set(tokenizer(str(text or "")))
    except Exception:  # noqa: BLE001 - ranking must never raise
        return set(re.findall(r"[a-z0-9_]{3,}", str(text or "").lower()))


def lexical_relevance(text: Any, query: Any) -> float:
    """Share of the *query*'s content words that *text* also contains.

    Bounded ``[0, 1]`` and asymmetric on purpose: the query is short, so
    "how much of what was asked does this memory cover" is the useful
    direction.  Returns ``0.0`` when the query has no content words.
    """
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = _content_tokens(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def goal_alignment(text: Any, goal: Any) -> float:
    """Share of the original goal's content words that *text* contains.

    ``0.0`` when there is no goal, so a session without one ranks on the other
    three signals exactly as before.
    """
    return lexical_relevance(text, goal)


def fuse(
    relevance: float,
    goal: float,
    recency: float,
    importance: float,
) -> float:
    """Combine the four signals into one ``[0, 1]`` score."""
    total = (
        W_RELEVANCE * _clamp01(relevance, 0.0)
        + W_GOAL_ALIGNMENT * _clamp01(goal, 0.0)
        + W_RECENCY * _clamp01(recency, UNKNOWN_RECENCY)
        + W_IMPORTANCE * _clamp01(importance, DEFAULT_IMPORTANCE)
    )
    return _clamp01(total, 0.0)


def _entry_text(entry: Any) -> str:
    """The ``metadata.text`` of a ledger entry ("" when absent)."""
    if isinstance(entry, Mapping):
        metadata = entry.get("metadata")
        if isinstance(metadata, Mapping):
            return str(metadata.get("text") or "").strip()
        return str(entry.get("text") or "").strip()
    return str(entry or "").strip()


def _entry_created_at(entry: Any) -> Any:
    """The entry's creation stamp, trying the known metadata keys in order."""
    if isinstance(entry, Mapping):
        metadata = entry.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("created_at", "timestamp", "ts"):
                if metadata.get(key):
                    return metadata.get(key)
    return None


def rank_memories(
    memories: Sequence[Any] | None,
    query: str | None = None,
    goal: str | None = None,
    *,
    semantic_scores: Mapping[str, float] | None = None,
    now: datetime | None = None,
    k: int | None = None,
) -> list[dict[str, Any]]:
    """Score, sort and (optionally) truncate ledger entries, most relevant first.

    *memories* accepts the shapes :func:`agent_core.memory.load_semantic_memory`
    returns (``{"metadata": {...}, "similarity_score": float}``) or plain
    strings.  The result keeps the entry's ``metadata`` intact — the prompt
    renderer reads ``text``/``timestamp`` from it — and REPLACES
    ``similarity_score`` with the fused score, so the rendered ``(score 0.72)``
    is the weighted judgement rather than the raw cosine.

    *semantic_scores* maps entry text to a cosine score (from
    :func:`agent_core.memory.semantic_search_memories`).  When an entry has no
    usable cosine — embeddings down, or the entry was not in the indexed
    sample — the lexical overlap is used instead, so the ranking degrades
    instead of collapsing to file order.

    Ordering is a total order: score descending, then text, so the block is
    deterministic across runs.
    """
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for entry in memories or []:
        text = _entry_text(entry)
        if not text:
            continue
        metadata: dict[str, Any] = {}
        if isinstance(entry, Mapping):
            raw = entry.get("metadata")
            if isinstance(raw, Mapping):
                metadata = dict(raw)
        metadata.setdefault("text", text)
        cosine: float | None = None
        if semantic_scores is not None and text in semantic_scores:
            cosine = _clamp01(semantic_scores[text], 0.0)
        elif isinstance(entry, Mapping):
            stored = entry.get("similarity_score")
            if stored:
                cosine = _clamp01(stored, 0.0)
        relevance = (
            cosine if cosine and cosine > 0.0 else lexical_relevance(text, query)
        )
        score = fuse(
            relevance,
            goal_alignment(text, goal),
            recency_score(_entry_created_at(entry), now=now),
            importance_of(entry),
        )
        scored.append((score, text, {"metadata": metadata, "similarity_score": score}))
    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [entry for _, _, entry in scored]
    if k is not None and k >= 0:
        ranked = ranked[:k]
    return ranked
