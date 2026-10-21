"""Memory persistence extracted from Agent (refactor step r06).

Chat-history and cross-session-memory load/save, plan-answer persistence,
memory stats, history clearing, and the LLM-experience ledger.  Each former
``Agent`` method is a plain module-level function taking the owning
:class:`Agent` instance as its first parameter.  The persisted-state paths
(``CHAT_HISTORY_JSON_PATH`` / ``AGENT_MEMORY_JSON_PATH``) are resolved
lazily from the ``agent`` module so test monkeypatches are honoured.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent_core.bootstrap import LOOP_NOTE_TAG_KEY
from agent_core.colors import yellow
from agent_core.history_compaction import _project_chat_history, _strip_image_blocks
from agent_core.json_utils import _read_json_quarantining
from agent_core.suppress_log import _suppress_and_log

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent import Agent

logger = logging.getLogger(__name__)


def _agent_mod() -> Any:
    """The ``agent`` module, resolved lazily (agent.py imports this module)."""
    import agent as _agent_mod

    return _agent_mod


def _persist_plan_answer(agent: "Agent", plan_text: str) -> None:
    """Save a plan-mode final answer to ``.docs/<ts>/plan_proposed.md``.

    Plan-mode answers used to evaporate as terminal text; persisting them
    gives the user a durable artifact to hand to ``implement``/``fix``
    later (plan item D-#14).  Best-effort: a write failure must never
    lose the already-printed answer or crash the turn.
    """
    with _suppress_and_log("Could not persist plan-mode answer:\n"):
        from agent_core.commands.doc_paths import new_run_dir

        out = new_run_dir(agent.workspace) / "plan_proposed.md"
        out.write_text(
            f"# Proposed plan\n\n_Generated in plan mode on {datetime.now():%Y-%m-%d %H:%M}. Review before applying — switch to build mode to implement._\n\n{plan_text}\n",
            encoding="utf-8",
        )
        print(yellow(f"\n  [plan] Saved to {out}"))


def memory_stats(agent: "Agent") -> dict[str, Any]:
    """Return summary of current memory state."""
    stale = getattr(agent, "check_stale_files")()
    return {
        "chat_history": len(agent._chat_history),
        "files_read": len(agent._files_read),
        "stale_files": len(stale),
        "history": len(agent._history),
        "working_memory": len(agent._working_memory),
        "semantic_index": len(agent._semantic_index),
        "knowledge_graph": len(agent._knowledge_graph),
    }


def clear_history(agent: "Agent") -> None:
    """Clear all agent state."""
    agent._history = []
    agent._chat_history.clear()
    if hasattr(agent, "_conversation"):
        agent._conversation.clear()
    agent._turn_start_index = 0
    agent._turn_run_id = None
    agent._read_streak = 0
    agent._delegating = False
    agent._active_subagents.clear()
    agent._delegate_counter = 0
    agent._pending_effects = None
    agent._original_goal = ""
    agent._goal_from_restore = False
    agent._prior_chat_history = []
    agent._files_read.clear()
    agent._file_mtimes.clear()
    agent._knowledge_graph.clear()
    agent._working_memory.clear()
    agent._semantic_index.clear()
    try:
        os.remove(_agent_mod().CHAT_HISTORY_JSON_PATH)
        os.remove(_agent_mod().AGENT_MEMORY_JSON_PATH)
    except OSError as e:
        logger.warning("Could not remove persisted state files: %s", e)
    try:
        agent._full_run_gate.reset()
    except Exception:
        logger.debug("full-run gate reset failed", exc_info=True)


def _load_chat_history(agent: "Agent") -> list[dict[str, Any]]:
    """Load the persisted NLP conversation from the previous session.

    The persisted history holds a bounded multi-exchange window (see
    ``_project_chat_history``), so a fresh session continues the dialogue
    instead of forgetting it.  A corrupt file is quarantined (see
    :func:`_read_json_quarantining`) instead of being silently dropped.
    """
    data = _read_json_quarantining(_agent_mod().CHAT_HISTORY_JSON_PATH, "chat history")
    if not isinstance(data, list):
        return []
    messages = [
        m
        for m in data
        if isinstance(m, dict)
        and m.get("role") in ("system", "user", "assistant", "tool")
        and (LOOP_NOTE_TAG_KEY not in m)
    ]
    for m in messages:
        for tc in m.get("tool_calls", []):
            func = tc.get("function") or {}
            args_str = func.get("arguments", "{}")
            try:
                parsed = json.loads(args_str)
                if not isinstance(parsed, dict):
                    func["arguments"] = "{}"
            except (json.JSONDecodeError, TypeError):
                func["arguments"] = "{}"
    return _project_chat_history(
        _strip_image_blocks(messages), getattr(agent, "_original_goal", "")
    )


def _save_chat_history(agent: "Agent") -> None:
    """Persist the NLP conversation so the next session can continue it.

    Image blocks are stripped first (see :func:`_strip_image_blocks`) so a
    vision turn does not bloat the on-disk history with base64 blobs.

    The write is atomic: content goes to ``chat_history.json.tmp`` first
    and is then ``os.replace()``d into place, so a crash mid-write can
    never leave a half-written JSON that would be silently dropped on the
    next load.
    """
    payload = json.dumps(
        _project_chat_history(
            _strip_image_blocks(agent._chat_history), agent._original_goal
        ),
        ensure_ascii=False,
        indent=2,
    )
    chat_path = _agent_mod().CHAT_HISTORY_JSON_PATH
    tmp_path = chat_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, chat_path)
    except OSError as e:
        logger.warning("Failed to save chat history: %s", e)


def _load_memory(agent: "Agent") -> None:
    """Load cross-session memory from the previous session.

    Restores files read, the semantic index, the knowledge graph, and
    working memory so a fresh session resumes with accumulated context
    instead of starting from zero.  A corrupt file is quarantined as
    ``agent_memory.json.bad-<timestamp>`` before falling back to the
    empty defaults, so the bytes stay inspectable instead of being
    clobbered by the next save.
    """
    data = _read_json_quarantining(_agent_mod().AGENT_MEMORY_JSON_PATH, "agent memory")
    if not isinstance(data, dict):
        return
    files = data.get("files_read")
    if isinstance(files, list):
        agent._files_read = {str(p) for p in files}
    index = data.get("semantic_index")
    if isinstance(index, dict):
        agent._semantic_index = defaultdict(
            set, {k: set(v) for k, v in index.items() if isinstance(v, list)}
        )
    kg = data.get("knowledge_graph")
    if isinstance(kg, dict):
        agent._knowledge_graph = kg
    wm = data.get("working_memory")
    if isinstance(wm, list):
        agent._working_memory = wm
    hist = data.get("history")
    if isinstance(hist, list):
        agent._history = hist
    goal = data.get("original_goal")
    if isinstance(goal, str):
        agent._original_goal = goal
        if goal:
            agent._goal_from_restore = True


def _save_memory(agent: "Agent") -> None:
    """Persist cross-session memory so the next session resumes with it.

    Atomic like :meth:`_save_chat_history` (tmp file + ``os.replace``).
    """
    data = {
        "files_read": sorted(agent._files_read),
        "semantic_index": {k: sorted(v) for k, v in agent._semantic_index.items()},
        "knowledge_graph": agent._knowledge_graph,
        "working_memory": agent._working_memory,
        "history": agent._history,
        "original_goal": agent._original_goal,
    }
    memory_path = _agent_mod().AGENT_MEMORY_JSON_PATH
    tmp_path = memory_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, memory_path)
    except OSError:
        logger.warning("Failed to save agent memory:\n%s", traceback.format_exc())


def _record_llm_experience(
    agent: "Agent",
    action: str,
    outcome: float,
    context: dict[str, Any] | None = None,
    success: bool | None = None,
    decision_llm: str | None = None,
    decision_prompt: Any = None,
    run_id: str | None = None,
    latency_ms: float | None = None,
    token_usage: int | None = None,
) -> int | None:
    """Record an LLM decision/outcome into agent_memory.db (experiences).

    Mirrors the MCP ``record_experience`` schema so a direct write is
    reconcilable with the memory server's own inserts.  outcome must be in
    [0.0, 1.0]; success defaults to ``outcome >= 0.5``.

    This is harness-layer only (decision #014): it never alters model or
    prompt behavior — it only persists observability of decisions for the
    self-improvement loop.  It must NEVER raise: if agent_memory.db is
    absent, unwritable, or the table missing, we silently no-op so untraced
    / non-memory runs stay byte-identical (decisions #048/#049).

    Decision attribution (plan: memory-collection LLM attribution)
    -------------------------------------------------------------
    Every row is stamped with the LLM that *decided* it, so a later model
    swap can be measured against ground-truth memory outcomes:

    ``decision_llm``
        The deciding model.  Defaults to this agent's active model
        (:func:`agent_core.memory.attribution.resolve_decision_llm`), so
        the stamp cannot be forgotten by a call site.
    ``decision_prompt_hash``
        SHA-256 of the prompt/conversation behind the decision (canonical,
        tag-insensitive — see ``prompt_sha256``), or ``NULL`` when the
        caller passes no prompt.  Never a hash of the empty string, which
        would falsely claim "the empty prompt was the decision input".
    ``decision_timestamp``
        When the decision was taken (same instant as ``timestamp`` for a
        direct write).

    An attributed write also appends one ``llm_decisions`` provenance row
    (run_id, model, prompt hash, the new experience rowid, latency, token
    usage).  The invariant is exact: one ``llm_decisions`` row per
    experience whose ``decision_llm`` is non-NULL.

    Schema compatibility: the first five columns keep the MCP server's
    exact names/order/types; the three attribution columns are APPENDED
    and nullable.  The server reads with ``SELECT *`` and formats rows by
    column name, and both writers name their columns explicitly — so old
    rows read back ``NULL`` and the server is unaffected.  Additive DDL
    only: no row is ever rewritten, so MCP-authored tables are extended in
    place, never clobbered.
    """
    try:
        outcome = float(outcome)
        if not 0.0 <= outcome <= 1.0:
            return None
        if success is None:
            success = 1 if outcome >= 0.5 else 0
        ctx_str = json.dumps(context or {}, ensure_ascii=False, sort_keys=True)
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        import sqlite3
        from agent_core.memory.attribution import (
            ensure_attribution_schema,
            prompt_sha256,
            record_llm_decision,
            resolve_decision_llm,
        )

        if decision_llm is None:
            decision_llm = resolve_decision_llm(agent)
        prompt_hash = (
            prompt_sha256(decision_prompt) if decision_prompt is not None else None
        )
        db_path = _agent_mod().AGENT_MEMORY_JSON_PATH.replace(".json", ".db")
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            ensure_attribution_schema(conn)
            if decision_llm is None:
                cur = conn.execute(
                    "INSERT INTO experiences (timestamp, action, outcome, context, success) VALUES (?, ?, ?, ?, ?)",
                    (ts, action, outcome, ctx_str, 1 if success else 0),
                )
                conn.commit()
                return int(cur.lastrowid)
            cur = conn.execute(
                "INSERT INTO experiences (timestamp, action, outcome, context, success, decision_llm, decision_prompt_hash, decision_timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    action,
                    outcome,
                    ctx_str,
                    1 if success else 0,
                    decision_llm,
                    prompt_hash,
                    ts,
                ),
            )
            rowid = int(cur.lastrowid)
            record_llm_decision(
                conn,
                run_id=run_id,
                model_name=decision_llm,
                prompt_sha256=prompt_hash,
                outcome_experience_id=rowid,
                latency_ms=latency_ms,
                token_usage=token_usage,
            )
            conn.commit()
            return rowid
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("Could not record LLM experience (no-op): %s", exc)
        return None
