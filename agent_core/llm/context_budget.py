"""Per-provider context budgeting.

Local accelerators differ wildly in context window: a Radeon iGPU model may
accept 128k tokens while an NPU (FastFlowLM) model can be loaded with 4k.  The
agent's history trim is a single global character budget, so a prompt that fits
the big model can overflow the small one and the whole request is rejected
("Max length reached!").  This module adds the missing piece: a per-provider
token budget derived from the provider's REAL context window, plus a message
trimmer that keeps the system prompt and the newest turns.
"""
from __future__ import annotations

from typing import Any, Sequence

#: Rough characters-per-token ratio used for the estimate (matches the
#: prefill sizing heuristic in ``lmstudio._estimate_prompt_tokens``).
CHARS_PER_TOKEN = 3.5

#: Note inserted where older turns were dropped, so the model knows the
#: conversation was compacted rather than silently missing context.
TRIM_NOTE = "[earlier turns trimmed to fit this model's context window]"


def estimate_tokens(messages: Sequence[dict[str, Any]]) -> int:
    """Estimate the prompt size in tokens (char/3.5 heuristic)."""
    total_chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif content is not None:
            total_chars += len(str(content))
        # Tool-call payloads count too.
        for call in message.get("tool_calls") or []:
            total_chars += len(str(call))
    return int(total_chars / CHARS_PER_TOKEN) + 1


def provider_context_limit(provider: Any) -> int | None:
    """The provider's context window in tokens, or ``None`` when unknown.

    Providers opt in by exposing a positive integer ``context_limit`` attribute
    (:class:`~agent_core.llm.lemonade_provider.LemonadeProvider` fills it from
    the server's ``/models`` response).
    """
    limit = getattr(provider, "context_limit", None)
    try:
        value = int(limit) if limit is not None else 0
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def char_budget_for(
    provider: Any,
    *,
    reserve_tokens: int = 1024,
    safety: float = 0.8,
) -> int | None:
    """Character budget for a provider, or ``None`` when the context is unknown."""
    limit = provider_context_limit(provider)
    if limit is None:
        return None
    usable = max(0, limit - max(0, reserve_tokens))
    return int(usable * CHARS_PER_TOKEN * max(0.0, min(1.0, safety)))


def trim_messages_to_context(
    messages: Sequence[dict[str, Any]],
    *,
    provider: Any = None,
    limit_tokens: int | None = None,
    reserve_tokens: int = 1024,
) -> list[dict[str, Any]]:
    """Drop oldest turns so *messages* fit the provider's context window.

    Returns the input unchanged when the limit is unknown or already fits.
    Leading system messages are always kept, the newest body turns are kept
    until the budget is spent, and a :data:`TRIM_NOTE` user turn is inserted
    where older turns were dropped.  A leading orphan ``tool`` message (its
    assistant ``tool_calls`` was dropped) is skipped so strict servers do not
    reject the payload.
    """
    msgs = list(messages)
    if limit_tokens is None:
        limit_tokens = provider_context_limit(provider)
    if not limit_tokens or limit_tokens <= 0:
        return msgs
    budget_chars = int(max(0, limit_tokens - max(0, reserve_tokens)) * CHARS_PER_TOKEN)
    if sum(len(str(m.get("content") or "")) for m in msgs) <= budget_chars:
        return msgs

    system = [m for m in msgs if m.get("role") == "system"]
    body = [m for m in msgs if m.get("role") != "system"]
    kept: list[dict[str, Any]] = []
    used = 0
    for message in reversed(body):
        size = len(str(message.get("content") or ""))
        if kept and used + size > budget_chars:
            break
        kept.append(message)
        used += size
    kept.reverse()
    # Never start the kept tail with an orphan tool result.
    while kept and kept[0].get("role") == "tool":
        kept.pop(0)
    if len(kept) == len(body):
        return msgs
    return [*system, {"role": "user", "content": TRIM_NOTE}, *kept]


__all__: list[str] = [
    "CHARS_PER_TOKEN",
    "TRIM_NOTE",
    "char_budget_for",
    "estimate_tokens",
    "provider_context_limit",
    "trim_messages_to_context",
]
