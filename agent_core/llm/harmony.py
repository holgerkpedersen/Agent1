"""Harmony (gpt-oss) prompt shaping for the llama provider.

gpt-oss models are served through llama-server's harmony chat template,
which renders ``messages[0]`` (role ``system`` or ``developer``) as the
``# Instructions`` developer block — the highest-priority instruction slot
the model actually reads.  Every later ``system`` message is demoted to a
``user`` note by ``sanitize_message_roles``, so safety rules placed anywhere
but the leading message are effectively advisory.

This module guarantees that a gpt-oss conversation carries an explicit
"do not make unnecessary fatal changes" addendum in that leading slot, and
that the shaping is idempotent (a long tool loop re-shapes the same history
every iteration and must not stack duplicates).

Detection is name-based (``is_harmony_model``): the routing label
(``llama/oss-20b``) and the cached server id (``gpt-oss-20b-MXFP4``) are
both consulted, because either may be the only identifier available when
the payload is built.
"""
from __future__ import annotations

from typing import Any

#: Model-name tokens that identify the harmony (gpt-oss) family.  Checked
#: against the lowercased routing label and the cached server model id.
_HARMONY_TOKENS: tuple[str, ...] = ("gpt-oss", "oss-20b", "oss-120b")

#: Safety addendum injected as the leading system message content for
#: gpt-oss models.  Kept as one block so ``apply_harmony_prompt`` can detect
#: prior injection by substring match (idempotency) and so tests can pin
#: the exact dangerous commands the model must refuse to run unasked.
HARMONY_SAFETY_ADDENDUM = (
    "\n\nSAFETY RULES (harmony/gpt-oss — non-negotiable):\n"
    "- NEVER make unnecessary fatal changes: no rm -rf, no git reset --hard, "
    "no git clean -f, no force push, no rewriting files wholesale when a "
    "targeted edit works, no deleting tests or assertions to make a suite pass.\n"
    "- Prefer the smallest targeted edit that fixes the problem. If a fix "
    "looks destructive, stop and ask the user before acting.\n"
    "- Destructive shell commands are blocked by the harness anyway — do not "
    "try to work around the block; report the block instead.\n"
    "- Verify with tests before claiming a fix; never weaken a test to get "
    "green."
)


def is_harmony_model(*names: str) -> bool:
    """True when any of *names* identifies a harmony (gpt-oss) model.

    Matches on the lowercased routing label and/or the cached server model
    id — callers pass whichever identifiers they have.
    """
    for name in names:
        lowered = str(name or "").lower()
        if any(token in lowered for token in _HARMONY_TOKENS):
            return True
    return False


def apply_harmony_prompt(
    messages: list[dict[str, Any]], model_name: str
) -> list[dict[str, Any]]:
    """Return *messages* with the safety addendum in the leading system slot.

    - Non-harmony model -> the list is returned unchanged (same object).
    - Leading message is already ``system`` -> the addendum is appended to
      its content unless already present (idempotent).
    - No leading ``system`` message -> a new one carrying only the addendum
      is prepended.

    Never mutates the caller's messages.
    """
    if not is_harmony_model(model_name):
        return messages
    marker = HARMONY_SAFETY_ADDENDUM.strip()
    out = list(messages)
    if out and out[0].get("role") == "system":
        first = dict(out[0])
        content = str(first.get("content") or "")
        if marker not in content:
            first["content"] = content + HARMONY_SAFETY_ADDENDUM
        out[0] = first
    else:
        out.insert(0, {"role": "system", "content": marker})
    return out
