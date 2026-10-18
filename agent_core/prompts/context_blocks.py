"""Prompt context-block builders extracted from Agent (refactor step r05).

Each former ``Agent._*_block`` method is a plain module-level function that
takes the owning :class:`Agent` instance as its first parameter.  Names that
tests monkeypatch on the ``agent`` module (``decisions_as_system_prompt``,
``CHAT_HISTORY_JSON_PATH``, ``_resolve_display_mode``, ``_effective_ws_dir``)
are resolved lazily from it so patching keeps working; agent-instance
methods are forwarded via ``getattr(agent, name)`` at call time.
"""
from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from agent_core.colors import yellow
from agent_core.config import AgentDisplayMode
from agent_core.file_history import file_notes_block
from agent_core.habits import habits_block, load_habits
from agent_core.history_compaction import _content_tokens, _is_compaction_note, _recover_original_goal, _strip_turn_wrappers
from agent_core.memory import load_semantic_memory, semantic_memory_block, semantic_search_memories
from agent_core.memory.types import MAX_GOAL_CHARS, original_goal_block
from agent_core.skills import discover_skills, load_skill_index, match_skills_for_input, skill_hint_block
from agent_core.types import _GOAL_RELATED_THRESHOLD

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent import Agent

logger = logging.getLogger(__name__)


def _agent_mod() -> Any:
    """The ``agent`` module, resolved lazily (agent.py imports this module)."""
    import agent as _agent_mod
    return _agent_mod


def _decision_constraints_block(agent: "Agent") -> str:
    """Design-decision constraints for the chat system prompt.

    Mirrors what ``implement``/``fix`` already inject (decision ledger
    ``.decisions.json`` via :func:`decisions_as_system_prompt`) so the
    conversational loop cannot contradict a recorded decision mid-chat.
    Matched on the files read this session; empty string when no decision
    touches them — the prompt stays byte-identical to before in that case.
    Never raises: a broken/absent ledger must not kill a chat turn.
    """
    try:
        if not agent._files_read:
            return ''
        return str(_agent_mod().decisions_as_system_prompt(agent.workspace, sorted(agent._files_read)))
    except Exception:
        logger.exception('Decision constraints unavailable:\n')
        return ''


def _skill_index_block(agent: "Agent") -> str:
    """Compact index of workspace skills for the chat system prompt.

    Only ``- name — description`` lines are injected (see
    :func:`agent_core.skills.skill_index_block`); the runbook BODY is read
    on demand via the ``read_skill`` tool.  Rebuilt every turn so a skill
    added to ``skills/`` mid-session is advertised on the next turn.
    Empty string when the workspace has no skills — the prompt then stays
    byte-identical to before.  Never raises: a broken ``SKILL.md`` must not
    kill a chat turn (same rule as :meth:`_decision_constraints_block`).
    """
    try:
        return load_skill_index(getattr(agent, '_effective_ws_dir')())
    except Exception:
        logger.exception('Skill index unavailable:\n')
        return ''


def _skill_hint_block(agent: "Agent", user_input: str) -> str:
    """Per-turn skill hint prepended to the user message when a skill matches.

    Only name + one-line description are injected (see
    :func:`agent_core.skills.skill_hint_block`); the runbook BODY still
    only enters context via an explicit ``read_skill`` call.  Never raises:
    a broken match must not kill a chat turn (same rule as
    :meth:`_skill_index_block`).

    Side-effects on a successful match: increments the shared
    ``skill.hint.matched`` metrics counter (mirrored to the event file so
    a standalone --serve dashboard sees it) and prints a short
    ``[skill-hints]`` status line suppressed in QUIET mode.
    """
    try:
        skills = discover_skills(getattr(agent, '_effective_ws_dir')())
        matched = match_skills_for_input(skills, user_input or '')
        if not matched:
            return ''
        try:
            collector = agent.get_metrics_collector()
            collector.increment_counter('skill.hint.matched')
            from agent_core.monitoring.metrics_file import append_event as _ae
            _ae('counter', 'skill.hint.matched', 1.0)
        except Exception:
            pass
        if _agent_mod()._resolve_display_mode() != AgentDisplayMode.QUIET:
            names = ', '.join((s.name for s in matched))
            print(yellow(f'  [skill-hints] matched: {names}'))
        return skill_hint_block(matched)
    except Exception:
        logger.exception('Skill hint unavailable:\n')
        return ''


def _habits_block(agent: "Agent") -> str:
    """Mined user-habits block for the chat system prompt.

    Only the compact block (marker + at most 8 lines, see
    :func:`agent_core.habits.habits_block`) is injected; it is rebuilt
    every turn from the workspace ledger so a habit pinned or mined
    mid-session is advertised on the next turn.  Empty string when the
    workspace has no habits — the prompt then stays byte-identical to
    before.  Never raises: a broken ``.habits.json`` must not kill a chat
    turn (same rule as :meth:`_decision_constraints_block`).
    """
    try:
        return habits_block(load_habits(getattr(agent, '_effective_ws_dir')()))
    except Exception:
        logger.exception('Habits block unavailable:\n')
        return ''


def _file_history_block(agent: "Agent", query: str | None=None) -> str:
    """Per-file execution-history block for the chat system prompt.

    The workspace already keeps a trace corpus and an execution ledger
    (``reports/traces``, ``reports/history/executions.jsonl``); only
    implement/fix ever read them.  This method makes the same history
    visible to the conversational loop: the files *query* mentions are
    looked up with :func:`harnessfix.history.file_history` (the real
    matcher — a directory arg counts for its direct children) and the
    most important past events render as one compact block.

    Rebuilt every turn from the current workspace, same contract as
    :meth:`_habits_block`: empty string when nothing is relevant, so the
    prompt stays byte-identical to before; never raises — a broken
    ledger must not kill a chat turn.
    """
    try:
        return file_notes_block(query, getattr(agent, '_effective_ws_dir')())
    except Exception:
        logger.exception('File-history block unavailable:\n')
        return ''


def _semantic_memory_block(agent: "Agent", query: str | None=None) -> str:
    """Embedding/KG memory block for the chat system prompt.

    Builds a compact block from indexed memories (see
    :func:`agent_core.memory.semantic_memory_block`).  It is
    rebuilt every turn from the workspace ledger so a memory indexed or
    updated mid-session is advertised on the next turn.  Empty string when
    the workspace has no indexed memories — the prompt then stays
    byte-identical to before.  Never raises: a broken ``.semantic_memory``
    must not kill a chat turn (same rule as
    :meth:`_decision_constraints_block`).

    When *query* (the current user message) is known, the ledger entries
    are ranked with :func:`semantic_search_memories`
    (:meth:`MemoryStore.semantic_search`) and the top-5 most relevant ones
    are injected, prefixed with a ``QUERY:`` line.  When embeddings are
    unavailable the ranking returns ``[]`` and the block falls back to the
    unranked ledger order WITHOUT the query line (byte-identical to the
    old behaviour).
    """
    try:
        indexed_memories = load_semantic_memory(getattr(agent, '_effective_ws_dir')())
        if query and indexed_memories:
            try:
                ranked = semantic_search_memories(indexed_memories, query, k=5, embedding_service=getattr(agent, '_semantic_embedding_service', None))
            except Exception:
                ranked = []
            if ranked:
                return semantic_memory_block(ranked, query=query, k=5)
        return semantic_memory_block(indexed_memories, k=5)
    except Exception:
        logger.exception('Semantic memory block unavailable:\n')
        return ''


def _ensure_original_goal(agent: "Agent", pending_user_input: str | None=None) -> None:
    """Record the session's original request once, and only once.

    The goal is whatever the user asked FIRST: the first user message
    already in a restored history, or the input of the first turn of a
    fresh session (``pending_user_input`` — captured before the plan-mode
    note and skill hints are prepended to it).  Later prompts never replace
    it: the point is to pin what the session was *originally* about, which
    is exactly what chat-history compaction drops first.

    Cross-session exception: when the goal was RESTORED (agent_memory.json
    or the previous session's history, tracked by ``_goal_from_restore``)
    and the first live prompt of this session is topically unrelated to it
    (:meth:`_prompts_related` over the goal plus the recent user texts),
    the restored goal is REPLACED by the live prompt and the old
    conversation is suspended (:meth:`_suspend_history_for_new_task`).  A
    stale goal block otherwise steers every answer of the new session away
    from what the user is actually asking now.

    Capped at ``MAX_GOAL_CHARS`` so a first prompt that is itself a file
    dump cannot bloat the system prompt or ``agent_memory.json``.
    """
    if agent._original_goal:
        if agent._goal_from_restore and pending_user_input:
            agent._goal_from_restore = False
            if not getattr(agent, '_prompts_related')(pending_user_input, [agent._original_goal, *getattr(agent, '_recent_user_texts')()]):
                agent._original_goal = pending_user_input[:MAX_GOAL_CHARS]
                getattr(agent, '_suspend_history_for_new_task')()
        return
    recovered = _recover_original_goal(agent._chat_history)
    if recovered is not None:
        source = recovered
        if recovered:
            agent._goal_from_restore = True
    elif pending_user_input:
        source = pending_user_input
    else:
        source = ''
    if source:
        agent._original_goal = source[:MAX_GOAL_CHARS]


def _recent_user_texts(agent: "Agent", limit: int=10) -> list[str]:
    """The newest *limit* raw user texts from the current chat history.

    Steering wrappers (plan-mode note, skill hints) are stripped and
    compaction notes skipped, so the texts are comparable with a live
    prompt for the relevance check in :meth:`_ensure_original_goal`.
    """
    texts: list[str] = []
    for m in reversed(agent._chat_history):
        if m.get('role') != 'user':
            continue
        content = m.get('content')
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = ' '.join((str(b.get('text') or '') for b in content if isinstance(b, dict) and b.get('type') == 'text')).strip()
        else:
            text = ''
        if text and (not _is_compaction_note(text)):
            texts.append(_strip_turn_wrappers(text))
        if len(texts) >= limit:
            break
    texts.reverse()
    return texts


def _prompts_related(agent: "Agent", new_prompt: str, prior_texts: list[str], threshold: float=_GOAL_RELATED_THRESHOLD) -> bool:
    """Whether *new_prompt* looks topically related to any prior text.

    Token-overlap heuristic over content words (:func:`_content_tokens`):
    a short prompt (<2 content tokens, e.g. "continue") is conservatively
    treated as related so a bare nudge can never evict the goal.
    """
    new_tokens = _content_tokens(new_prompt)
    if len(new_tokens) < 2:
        return True
    prior: set[str] = set()
    for text in prior_texts:
        prior |= _content_tokens(text or '')
    if not prior:
        return False
    overlap = new_tokens & prior
    return len(overlap) / len(new_tokens) >= threshold


def _suspend_history_for_new_task(agent: "Agent") -> None:
    """Archive a restored conversation whose goal just got replaced.

    The previous session's history belonged to a different task; keeping
    it inline would let old context steer (and contaminate) the new task.
    The body is archived in ``_prior_chat_history`` and — best-effort — on
    disk as ``chat_history_prior.json`` (path derived from the CURRENT
    ``CHAT_HISTORY_JSON_PATH`` so test monkeypatches are honoured), so
    nothing is lost.  Only the system message stays inline (rebuilt fresh
    by ``_refresh_system_message`` on the next turn).  Task-agnostic
    memory stores (files_read, semantic index, knowledge graph) are kept.
    """
    body = [dict(m) for m in agent._chat_history if m.get('role') != 'system']
    if not body:
        return
    system = [dict(m) for m in agent._chat_history if m.get('role') == 'system'][:1]
    agent._prior_chat_history = body
    agent._chat_history = system
    agent._turn_start_index = len(agent._chat_history)
    prior_path = os.path.join(os.path.dirname(_agent_mod().CHAT_HISTORY_JSON_PATH) or '.', 'chat_history_prior.json')
    try:
        with open(prior_path, 'w', encoding='utf-8') as f:
            json.dump(body, f, ensure_ascii=False, indent=2)
    except OSError:
        logger.debug('Could not archive prior chat history to %s', prior_path, exc_info=True)


def _original_goal_block(agent: "Agent") -> str:
    """The pinned ORIGINAL TASK block for the chat system prompt.

    Rebuilt every turn from ``_original_goal`` (same contract as
    :meth:`_decision_constraints_block`); empty string when no goal is
    recorded, so the prompt stays byte-identical to before in that case.
    Never raises: a missing goal must not kill a chat turn.
    """
    try:
        return original_goal_block(agent._original_goal)
    except Exception:
        logger.exception('Original goal block unavailable:\n')
        return ''
