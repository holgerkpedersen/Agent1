"""Chat-history compaction: repair, project and trim the message list.

Pure functions over the message list — no class state, no logging, no
I/O, and no reads of names tests patch on ``agent`` (verified: the only
free names in this module are stdlib, the constants below, and each
other).  Extracted from ``agent.py`` as the first refactor step for
exactly that reason: it is the region with the fewest patch-couplings.

Re-exported by ``agent.py``, so ``from agent import _trim_chat_history`` and
monkeypatches on ``agent._trim_chat_history`` keep working.
"""
from __future__ import annotations

import json
import re
from typing import Any

from agent_core.memory import original_goal_from_block


#: Content-word stopwords for the cross-session goal relevance check
#: (:func:`_content_tokens` / :meth:`Agent._prompts_related`) — glue words
#: that carry no topic signal, so "refactor the pytest timeout code" vs a
#: benchmark-comparison goal overlap on nothing topic-bearing.
_GOAL_TOPIC_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into", "your",
    "please", "can", "you", "should", "would", "about", "when", "then",
    "them", "they", "been", "were", "what", "which", "make", "need",
    "must", "will", "just", "also", "than", "over", "under", "after",
    "before", "between", "through", "during", "without", "within",
    "agent", "session", "task", "prompt", "message", "goal", "user",
})

_MAX_CHAT_MESSAGES = 60

#: Rough character budget for the chat-history BODY (system prompt excluded;
#: ~4 chars per token, so ~15k chars ≈ 4k tokens — conservative enough for
#: small local models (bonsai-27b@4.4GB) that choke on large prefill, while
#: still leaving room for tool schemas and the answer).  When the body
#: exceeds it, oldest messages are dropped (see _trim_chat_history).
_HISTORY_CHAR_BUDGET = 15000

_HISTORY_TRIM_NOTE = (
    "[context compaction] {dropped} earlier message(s) were dropped to fit "
    "the context window. Earlier file contents are no longer visible — "
    "re-read anything you still need."
)

#: How much of the pinned goal the compaction note repeats.  The full goal
#: lives in the system prompt (ORIGINAL TASK block); this is only a recency
#: reminder placed near the newest messages, so a short excerpt is enough.
_HISTORY_TRIM_GOAL_CAP = 300


def _drop_orphan_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop ``tool`` messages that are orphans or duplicate responses.

    - **Orphan**: a ``tool`` message whose ``tool_call_id`` has no matching
      assistant ``tool_calls`` message.
    - **Duplicate**: a second (or later) ``tool`` message for a
      ``tool_call_id`` already answered — strict gateways reject extras with
      HTTP 400 ("Messages with role 'tool' must be a response to a preceding
      message with 'tool_calls'").

    Trimming can cut between an assistant tool_calls message and its tool
    result, leaving an orphan; a backfill/sanitizer bug can leave duplicates.
    """
    valid_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    valid_ids.add(str(tc["id"]))
    # Dedupe WITHIN a contiguous run of tool results (reset on any non-tool
    # message): a later turn may legitimately reuse an id, so a global set
    # would wrongly drop a valid response.
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "tool":
            tid = str(m.get("tool_call_id") or "")
            if tid not in valid_ids or tid in seen:
                continue
            seen.add(tid)
            out.append(m)
        else:
            seen = set()
            out.append(m)
    return out


def _repair_unanswered_tool_calls(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip assistant ``tool_calls`` that have no matching tool response.

    The inverse of :func:`_drop_orphan_tool_messages`: an assistant message
    can announce a ``tool_calls`` batch whose results never arrive.  The
    projection removes loop-steering results ("NOTE: This ...") as cross-session
    noise, which orphans the assistant call that requested them; trimming a
    restored window can also cut the results.  Strict gateways reject the
    payload with HTTP 400 ("An assistant message with 'tool_calls' must be
    followed by tool messages responding to each 'tool_call_id'").

    The assistant's text content is preserved — only the unanswered calls are
    removed.  A message left with neither content nor calls is dropped.
    """
    answered = {
        str(m.get("tool_call_id") or "")
        for m in messages
        if m.get("role") == "tool"
    }
    out: list[dict[str, Any]] = []
    for m in messages:
        calls = m.get("tool_calls")
        if m.get("role") == "assistant" and calls:
            kept = [
                tc for tc in calls
                if isinstance(tc, dict) and str(tc.get("id") or "") in answered
            ]
            if len(kept) != len(calls):
                if kept:
                    m = {**m, "tool_calls": kept}
                else:
                    m = {k: v for k, v in m.items() if k != "tool_calls"}
                    if not str(m.get("content") or ""):
                        continue
        out.append(m)
    return out


def _strip_tool_args(m: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *m* with large tool-call arguments truncated.

    The LLM only needs to know WHICH tool was called (function name) and
    the tool result — not the full multi-KB argument payload (e.g. a
    ``write`` call with 10KB of markdown).  Stripping these before sending
    history to the LLM dramatically cuts prefill time and avoids HTTP 500
    on small local models.
    """
    tcs = m.get("tool_calls")
    if not tcs or not isinstance(tcs, list):
        return m
    stripped = []
    for tc in tcs:
        if not isinstance(tc, dict):
            stripped.append(tc)
            continue
        func = tc.get("function") or {}
        args = func.get("arguments") or ""
        if isinstance(args, str) and len(args) > 500:
            # Must remain VALID JSON: the gateway validates function.arguments
            # and rejects a "[truncated]" fragment with HTTP 400 ("Assistant
            # tool call function.arguments must be valid JSON").  A compact
            # placeholder keeps the call identifiable without a half-cut string.
            func = {
                **func,
                "arguments": json.dumps(
                    {"_truncated": True, "_original_len": len(args)}
                ),
            }
            stripped.append({**tc, "function": func})
        else:
            stripped.append(tc)
    return {**m, "tool_calls": stripped}


def _message_size(m: dict[str, Any]) -> int:
    """Approximate character size of one message (4 chars ≈ 1 token).

    Tool-call argument JSON counts too — a single big write call can dwarf
    its tiny ``content`` string.
    """
    total = len(str(m.get("content") or ""))
    for tc in m.get("tool_calls") or []:
        if isinstance(tc, dict):
            func = tc.get("function") or {}
            total += len(str(func.get("arguments") or ""))
            total += len(str(func.get("name") or ""))
    return total


def _content_tokens(text: str) -> set[str]:
    """Lowercased content words of *text* for the goal relevance check.

    Regex tokens of 3+ chars, minus :data:`_GOAL_TOPIC_STOPWORDS` glue words —
    the topic-bearing remainder used by :meth:`Agent._prompts_related` to
    decide whether a restored goal belongs to the live prompt's task.
    """
    return {
        t for t in re.findall(r"[a-z0-9_]{3,}", text.lower())
        if t not in _GOAL_TOPIC_STOPWORDS
    }


def _compaction_note_goal(text: str) -> str:
    """The ``Original task`` excerpt a compaction note carries ("" when none).

    Compaction notes are the ONE place the dropped first prompt still speaks:
    ``_trim_chat_history`` appends an excerpt of the pinned goal to the note it
    leaves behind, so a later session can recover the request from it.
    """
    marker = "Original task (do not lose sight of it): "
    if marker not in text:
        return ""
    excerpt = text.split(marker, 1)[1].strip()
    if excerpt.endswith("..."):
        excerpt = excerpt[:-3].strip()
    return excerpt


def _is_compaction_note(text: str) -> bool:
    return text.strip().startswith("[context compaction]")


def _strip_turn_wrappers(text: str) -> str:
    """Remove harness steering wrappers from a stored user message.

    ``_append_user_turn`` prepends a plan-mode note and/or skill hints to the
    message it stores, so the first user message in a restored history may not
    BE the raw request.  Both wrappers are one paragraph followed by a blank
    line — peel them off until the remainder looks like the raw prompt.
    """
    text = text.strip()
    while True:
        if text.lower().startswith("[plan mode]"):
            parts = text.split("\n\n", 1)
            text = parts[1].strip() if len(parts) == 2 else ""
        elif text.startswith("Skill hints (load with read_skill"):
            parts = text.split("\n\n", 1)
            text = parts[1].strip() if len(parts) == 2 else ""
        else:
            return text


def _recover_original_goal(messages: list[dict[str, Any]]) -> str | None:
    """Recover the session's ORIGINAL request from a restored history.

    Sources, most faithful first:

    1. The ``ORIGINAL TASK`` block in the system prompt — the goal itself,
       persisted verbatim by every session that had one.
    2. The ``Original task`` excerpt in a compaction note — the request as the
       last trim remembered it (capped, but never wrong).
    3. The first REAL user message (steering wrappers stripped), but only when
       no compaction note precedes it — a note ahead of it means the true
       first prompt was already dropped and this message is merely a LATER
       request.

    Returns ``""`` when the original is provably gone and ``None`` when the
    history holds nothing to judge (a fresh session — the pending input is the
    original then).  A wrong pin is worse than none: the block claims to be
    "the user's original request" and steers every later answer towards it.
    """
    try:
        compacted = False
        for m in messages:
            role = m.get("role")
            if role == "system":
                goal = original_goal_from_block(str(m.get("content") or ""))
                if goal:
                    return goal
            elif role == "user":
                text = _first_user_message_text([m])
                if not text:
                    continue
                if _is_compaction_note(text):
                    compacted = True
                    return _compaction_note_goal(text) or ""
                return _strip_turn_wrappers(text) or ""
        return "" if compacted else None
    except Exception:  # noqa: BLE001 - recovery must not kill a session
        return None


def _first_user_message_text(
    messages: list[dict[str, Any]],
) -> str:
    """The text of the first user message in *messages* ("" when there is none).

    Multimodal user messages carry a content array whose text block holds the
    actual request; a pure-image message yields nothing and the search
    continues.  Used to recover the session's original request from a restored
    history (see :meth:`Agent._ensure_original_goal`).
    """
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = ""
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = str(block.get("text") or "").strip()
                    if text:
                        break
        else:
            text = ""
        if text:
            return text
    return ""


def _trim_chat_history(
    messages: list[dict[str, Any]], goal: str | None = None,
) -> list[dict[str, Any]]:
    """Keep the conversation within a bounded context; returns a new list.

    *goal* is the session's pinned original request (see
    ``Agent._original_goal``).  When compaction has to drop messages, a short
    excerpt of it is repeated in the compaction note so the newest messages
    still carry what the session was originally about — the first user prompt
    is otherwise exactly what the caps below drop first.

    Two caps apply (plan item B-#5 — the old code had only the count cap, so
    a handful of huge read/write messages could blow the context window while
    staying under 60 messages):

    - ``_MAX_CHAT_MESSAGES``: at most this many messages (system prompt +
      newest tail), as before.
    - ``_HISTORY_CHAR_BUDGET``: the body must also fit a rough char budget;
      when it does not, the OLDEST body messages are dropped until it fits
      (contiguous tail — no holes), and a compaction note tells the next turn
      that earlier context was dropped.

    Orphan tool messages are dropped both before and after the cut (see
    :func:`_drop_orphan_tool_messages`): a count-based slice can cut between
    an assistant tool_calls message and its tool result, and strict gateways
    reject those orphans with HTTP 400.  The reverse case — an assistant
    ``tool_calls`` message whose results were dropped — is repaired by
    :func:`_repair_unanswered_tool_calls`.
    """
    messages = _drop_orphan_tool_messages(messages)
    count_dropped = 0
    if len(messages) <= _MAX_CHAT_MESSAGES:
        head = messages[:1]
        body = list(messages[1:])
    else:
        head = messages[:1]
        body = list(messages[-(_MAX_CHAT_MESSAGES - 1):])
        count_dropped = (len(messages) - 1) - len(body)

    # Strip large tool-call arguments from assistant messages in the body:
    # the LLM only needs to know WHICH tool was called, not the full payload
    # (e.g. a 10KB write() call).  This keeps the char-budget accurate for
    # prefill without losing conversation structure.
    body = [_strip_tool_args(m) for m in body]

    # Char-budget trim: walk from the NEWEST body message backwards until the
    # budget is exhausted.  A candidate is only accepted when it is not an
    # assistant-tool_calls/tool boundary split — i.e. we may drop a prefix
    # ending anywhere EXCEPT between an assistant(tool_calls) message and its
    # following tool result.
    total = sum(_message_size(m) for m in body)
    if total <= _HISTORY_CHAR_BUDGET:
        trimmed = body
        dropped = count_dropped
    else:
        keep_from = 0
        running = total
        for i, m in enumerate(body):
            if running <= _HISTORY_CHAR_BUDGET:
                keep_from = i
                break
            running -= _message_size(m)
            role = m.get("role")
            prev_role = body[i - 1].get("role") if i else None
            boundary_safe = not (
                role == "tool" or prev_role == "assistant" and "tool_calls" in body[i - 1]
            )
            if i == len(body) - 1 or boundary_safe:
                keep_from = i + 1
        trimmed = body[keep_from:]
        dropped = count_dropped + keep_from
    if not dropped:
        return _repair_unanswered_tool_calls(
            _drop_orphan_tool_messages(head + body)
        )
    note_content = _HISTORY_TRIM_NOTE.format(dropped=dropped)
    goal_text = " ".join(str(goal).split()) if goal else ""
    if goal_text:
        excerpt = goal_text[:_HISTORY_TRIM_GOAL_CAP]
        if len(goal_text) > _HISTORY_TRIM_GOAL_CAP:
            excerpt += "..."
        note_content += "\nOriginal task (do not lose sight of it): " + excerpt
    note = {
        "role": "user",
        "content": note_content,
    }
    # The compaction note is a message too — it must not push the result over
    # the count cap (the note replaces what it dropped, it does not add to it).
    if len(head) + 1 + len(trimmed) > _MAX_CHAT_MESSAGES and _MAX_CHAT_MESSAGES > 2:
        trimmed = trimmed[-(_MAX_CHAT_MESSAGES - 2):]
    return _repair_unanswered_tool_calls(
        _drop_orphan_tool_messages(head + [note] + trimmed)
    )


def _strip_image_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return *messages* with any multimodal image content stripped out.

    Vision turns carry base64 data URLs that can be several megabytes each;
    persisting them to ``chat_history.json`` would bloat the file and waste
    context on reload.  A user message that was purely an image (no text block)
    is dropped entirely; a mixed text+image message keeps its text and loses
    the image blocks so the conversation stays coherent without the payload.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            text_blocks = [
                b for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            image_blocks = [
                b for b in content
                if isinstance(b, dict) and b.get("type") == "image_url"
            ]
            if not image_blocks:
                out.append(m)
                continue
            if not text_blocks:
                # Purely an image message — drop it (nothing to continue on).
                continue
            out.append({**m, "content": text_blocks})
            continue
        out.append(m)
    return out


def _project_chat_history(
    messages: list[dict[str, Any]], goal: str | None = None,
) -> list[dict[str, Any]]:
    """Project a conversation down to what the NEXT session should see.

    Keeps the system prompt plus a bounded multi-exchange window (see
    :func:`_trim_chat_history`) so a fresh session continues the dialogue
    instead of forgetting it.  Loop steering notes (``NOTE: This ...`` tool
    messages), empty assistant placeholders, and orphan tool messages are
    dropped as cross-session noise.

    *goal* is forwarded to the trimmer so the compaction note this projection
    may leave behind carries the ``Original task`` reminder — the saved file is
    exactly where the next session must be able to recover it from.
    """
    if not messages:
        return []
    head = messages[:1] if messages and messages[0].get("role") == "system" else []
    body = messages[1:] if head else messages
    cleaned = []
    for m in body:
        content = str(m.get("content") or "")
        if m.get("role") == "tool" and content.startswith("NOTE: This"):
            continue  # loop steering noise — meaningless in a fresh session
        if m.get("role") == "assistant" and not content and not m.get("tool_calls"):
            continue  # empty placeholder with no action
        cleaned.append(m)
    return _trim_chat_history(head + cleaned, goal)
