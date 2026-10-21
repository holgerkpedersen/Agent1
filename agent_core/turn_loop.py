"""Turn-loop core extracted from Agent (refactor step r04).

Each former ``Agent`` method is a plain module-level function that takes the
owning :class:`Agent` instance as its first parameter.  Agent-instance
methods are forwarded via ``getattr(agent, name)`` at call time so
monkeypatching and subclass overrides keep working; module-level names that
still live in ``agent.py`` are resolved lazily from the ``agent`` module
(circular-import safe).
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent_core.bootstrap import (
    CORRELATION_ID_CTX, CorrelationIdContext, get_last_metrics
)
from agent_core.colors import gray, green, magenta, red, yellow
from agent_core.config import AgentDisplayMode
from agent_core.history_compaction import _trim_chat_history
from agent_core.llm.context_budget import provider_context_limit
from agent_core.modes import filter_tool_schemas
from agent_core.run_utils import (
    append_turn_log as _append_turn_log,
    final_answer_fallback as _final_answer_fallback,
    resolve_stream_fn as _resolve_stream_fn,
    turn_quality_score as _turn_quality_score,
)
from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent import Agent
    from agent_core.llm.tool_loop import ToolLoopRunner

logger = logging.getLogger(__name__)


def _agent_mod() -> Any:
    """The ``agent`` module, resolved lazily (agent.py imports this module)."""
    import agent as _agent_mod
    return _agent_mod


async def _run_chained_tool_loop(agent: "Agent", user_input: str, messages: list[dict[str, Any]], display_mode: AgentDisplayMode, seen_calls: dict[tuple[str, str], int]) -> tuple[str, list[dict[str, Any]], str | None, ToolLoopRunner]:
    """Run :class:`ToolLoopRunner`, auto-continuing unfinished tasks.

    Returns ``(final_text, final_messages, llm_error, loop)``.  *llm_error*
    carries a provider-level failure (reasoning-budget exhaustion, HTTP
    error, unreachable server, ...) — such a failure must never be mistaken
    for a model answer, fed into auto-continue chaining, or printed as the
    green final answer.  *loop* exposes observability stats for the
    final-answer fallback.

    Auto-continue rules: a "cap" verdict (budget ran out while
    progressing), an "answer" that signals unfinished work, or a
    "no_progress" verdict whose forced answer STILL signals unfinished
    work justify another run (up to ``_MAX_CHAINED_RUNS``).  Plain
    "stuck"/"no_progress" verdicts mean the model is not making progress —
    continuing would only re-enter the same loop.  A byte-identical
    repeated answer stops chaining: the model is stuck, not working.
    """
    mod = _agent_mod()
    llm_error: list[str] = []

    async def llm_chat_fn(msgs: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Call the LLM with tools; parse a JSON tool_calls message back into
        the message list so ToolLoopRunner can execute them."""
        raw = await agent.llm.chat(msgs, tools=tools, disable_thinking=True)
        if raw.strip() == '(no output)':
            raw = ''
        if raw.startswith('[Error') or raw.startswith('[LM Studio'):
            if not llm_error:
                llm_error.append(raw)
            return (raw, msgs)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get('tool_calls'):
            parsed.pop('role', None)
            for tc in parsed['tool_calls']:
                func = tc.get('function') or {}
                args_str = func.get('arguments', '{}')
                try:
                    json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    func['arguments'] = json.dumps(mod._repair_tool_arguments(args_str), ensure_ascii=False)
            updated = list(msgs)
            updated.append({'role': 'assistant', 'content': parsed.get('content') or '', **parsed})
            text = str(parsed.get('content') or '')
            return (text, updated)
        updated = list(msgs)
        updated.append({'role': 'assistant', 'content': raw})
        return (raw, updated)
    final_text = ''
    final_messages = messages
    continuations = 0
    last_answer = ''
    with CorrelationIdContext() as turn_cid:
        agent._turn_run_id = turn_cid
        mod._begin_trace_turn()
        while True:
            trace_writer = mod.TraceWriter(meta={'model': agent.model_name, 'profile': getattr(agent.llm, '_profile_name', None) or ''}) if mod.trace_enabled() and mod.TraceWriter is not None else None
            if trace_writer is not None:
                trace_writer.emit_task_begin(user_input)
            agent._active_trace_writer = trace_writer
            agent._pending_effects = [] if trace_writer is not None else None
            # Extract the model's REAL context window (providers opt in via a
            # `context_limit` attribute — LM Studio from the native /models
            # catalog, Lemonade from its /models) so in-loop compaction
            # bounds the prompt to what this model can actually accept.
            _ctx_tokens = provider_context_limit(getattr(agent.llm, '_provider', None))
            loop = mod.ToolLoopRunner(
                max_iterations=150, display_mode=display_mode, trace=trace_writer,
                context_limit_tokens=_ctx_tokens,
            )
            final_text, final_messages = await loop.run(messages=final_messages, llm_chat_fn=llm_chat_fn, execute_tool_fn=getattr(agent, '_execute_tool_call'), tools=filter_tool_schemas(NLP_TOOL_SCHEMAS, agent.mode), seen_calls=seen_calls, effects_fn=getattr(agent, '_take_trace_effects'), stream_fn=_resolve_stream_fn(agent.llm))
            agent._pending_effects = None
            agent._active_trace_writer = None
            if trace_writer is not None:
                trace_writer.close()
            reason = loop.termination_reason
            _run_outcome = 1.0 if not llm_error else 0.0
            _run_success = bool(not llm_error and reason in ('answer', 'cap'))
            _metrics = get_last_metrics(getattr(agent.llm, '_provider', None))
            getattr(agent, '_record_llm_experience')(action='llm_decision', outcome=_run_outcome, context={'model': agent.model_name, 'profile': getattr(agent.llm, '_profile_name', None) or '', 'verdict': reason, 'continuations': continuations, 'final_text_len': len(final_text)}, success=_run_success, decision_prompt=final_messages, run_id=agent._turn_run_id or CORRELATION_ID_CTX.get() or None, latency_ms=getattr(_metrics, 'latency_ms', None), token_usage=getattr(_metrics, 'total_tokens', None))
            if llm_error:
                break
            unverified = reason in ('answer', 'no_progress') and mod._unverified_action_claim(user_input, getattr(loop, 'tool_calls_made', 0) or 0)
            needs_more = reason == 'cap' or unverified or (reason in ('answer', 'no_progress') and mod._looks_incomplete(final_text))
            if needs_more and continuations < mod._MAX_CHAINED_RUNS:
                if final_text and final_text == last_answer:
                    if display_mode != AgentDisplayMode.QUIET:
                        print(magenta('\n  [stopped] The model repeated the same answer ') + yellow('twice — ending the turn. ') + gray('Rephrase or ask something more specific to continue.\n'))
                    break
                last_answer = final_text
                continuations += 1
                note = mod._VERIFY_NOTE if unverified else mod._CONTINUE_NOTE
                if display_mode != AgentDisplayMode.QUIET:
                    why = {'cap': 'iteration budget exhausted', 'answer': 'answer signals unfinished work', 'no_progress': 'the final answer signals unfinished work'}.get(reason, reason)
                    if unverified:
                        why = 'answer claimed repo state without calling git'
                    print(magenta(f'\n  [auto-continue] Run {continuations}: ') + yellow(f'{why} — continuing automatically.\n'))
                final_messages = list(final_messages) + [{'role': 'user', 'content': note, mod._CONTINUE_NOTE_TAG_KEY: mod._CONTINUE_NOTE_TAG}]
                continue
            if reason in ('stuck', 'no_progress') and display_mode != AgentDisplayMode.QUIET:
                print(magenta('\n  [stopped] The model stopped making progress ') + yellow('stuck on repeated calls' if reason == 'stuck' else 'too many calls without making new progress') + gray('The answer above is the best it produced — rephrase or ask something more specific to continue.\n'))
            break
    final_messages = [m for m in final_messages if m.get(mod._CONTINUE_NOTE_TAG_KEY) != mod._CONTINUE_NOTE_TAG]
    return (final_text, final_messages, llm_error[0] if llm_error else None, loop)


def _reset_failover_notice(agent: "Agent") -> None:
    """Drop any pending failover notice on the wrapped provider.

    Called once at turn start so a failover that happened in an EARLIER
    turn is never reported again (the record would otherwise be stale
    forever).  No-op when the provider is not a failover chain.
    """
    reset = getattr(getattr(agent.llm, '_provider', None), 'reset_failover_notice', None)
    if callable(reset):
        try:
            reset()
        except Exception:
            logger.debug('failover notice reset failed (no-op)', exc_info=True)


def _report_failover(agent: "Agent") -> None:
    """Print which LLM answered when a failover happened this turn.

    Failover is currently invisible to the user: the switch is only a
    ``logger.warning`` in :class:`FailoverProvider`, so the REPL silently
    answers from a different model than the one the user selected — which
    looks like a hallucination when the answer style changes.  This prints
    one line naming the model that answered and what was skipped.

    Honours the display-mode contract: suppressed in QUIET, which promises
    only the final answer.  Never raises — reporting must not kill a turn.
    """
    try:
        notice = getattr(getattr(agent.llm, '_provider', None), 'last_failover', None)
        if not notice:
            return
        if _agent_mod()._resolve_display_mode() == AgentDisplayMode.QUIET:
            return
        answered = notice.get('answered_model') or notice.get('answered_provider') or 'next provider'
        attempted = notice.get('attempted') or []
        skipped = ', '.join((str(a.get('model') or a.get('provider') or '?') for a in attempted))
        print(magenta(f'\n  [failover] answered by {answered}' + (f' after {skipped} unreachable' if skipped else '')))
        for entry in attempted:
            err = str(entry.get('error') or '').strip()
            if err:
                print(gray(f'    skipped {err[:160]}'))
    except Exception:
        logger.debug('failover report failed (no-op)', exc_info=True)


def _finish_turn(agent: "Agent", final_text: str, llm_error: str | None, loop: ToolLoopRunner, display_mode: AgentDisplayMode) -> None:
    """Bound + persist the conversation and print the turn outcome.

    The final answer is ALWAYS printed — in every display mode, including
    QUIET (which only hides intermediate tool output, per the display-mode
    contract: "only the final answer is printed").
    """
    agent._chat_history = _trim_chat_history(agent._chat_history, agent._original_goal)
    mutated_files = getattr(agent, '_mutating_files_this_turn')()
    try:
        started = getattr(agent, '_turn_started_at', None)
        duration = round((datetime.now() - started).total_seconds(), 3) if started else None
        tail = str(getattr(agent, '_last_user_input', '') or '')[-200:]
        record = {'ts': datetime.now().isoformat(timespec='seconds'), 'user_input_tail': tail, 'mutated_files': list(mutated_files or []), 'llm_error': bool(llm_error), 'duration': duration}
        _append_turn_log(record)
        getattr(agent, '_record_llm_experience')(action='chat_turn', outcome=0.0 if llm_error else 1.0 if mutated_files else 0.7, context={'plan_mode': bool(agent.is_plan_mode()), 'files': list(mutated_files or [])}, decision_prompt=str(getattr(agent, '_last_user_input', '') or '') or None, run_id=agent._turn_run_id or CORRELATION_ID_CTX.get() or None)
        _quality = _turn_quality_score(llm_error, mutated_files)
        _agent_mod().record_turn_outcome(str(getattr(agent, '_last_user_input', '') or ''), profile_name=getattr(agent.llm, '_profile_name', None), success=not llm_error, latency_seconds=duration, provider=getattr(agent.llm, '_provider', None), mutated_files=list(mutated_files or []), quality=_quality)
        if _quality is not None:
            from agent_core.monitoring.metrics_file import append_event
            append_event('turn', 'quality', float(_quality))
    except Exception:
        logger.debug('Turn-outcome hooks failed (no-op)', exc_info=True)
    getattr(agent, '_save_chat_history')()
    getattr(agent, '_save_memory')()
    clean = re.sub('</?tool_call>', '', final_text)
    clean = re.sub('</?function_call>', '', clean)
    if llm_error:
        err = llm_error.strip()
        print(yellow('\n  [llm-error] The model did not produce a usable response:'))
        print(red(f'  {err[:400]}'))
        if 'reasoning' in err.lower():
            print(yellow('  The reasoning model exhausted its thinking budget. ') + gray('Switch to a non-reasoning model (model <name>) or retry the request.\n'))
        agent._nlp_workspace = None
        return
    if display_mode == AgentDisplayMode.CLEAN:
        _NARRATION_PREFIX = re.compile("^\\s*(i\\s+(will|'ll|going\\sto|am\\ngoing)|let\\s+me)\\b", re.I)
        if _NARRATION_PREFIX.match(clean):
            clean = 'Plan: ' + clean.strip()
    if clean.strip():
        if not getattr(loop, 'final_answer_streamed', False):
            print(green(clean))
    else:
        print(yellow(_final_answer_fallback()))
    if agent.is_plan_mode() and clean.strip():
        getattr(agent, '_persist_plan_answer')(clean)
    if mutated_files and clean.strip():
        getattr(agent, '_print_self_review_note')(mutated_files)
    agent._nlp_workspace = None


def _mutating_files_this_turn(agent: "Agent") -> list[str]:
    """Files written/edited by this turn (plan item B-#6).

    The NLP write/edit handlers append a ``[verify] py_compile ✓`` line to
    their result — that line is the per-file mutation marker.  Scanning it
    keeps this zero-cost while the loop runs; no extra bookkeeping.

    Only messages from the CURRENT turn (at or after ``_turn_start_index``,
    set at the top of :meth:`chat_nlp`) are scanned.  A session restored
    from chat_history.json contains tool results from previous sessions;
    without this boundary the self-review note would list stale files
    after every restart.
    """
    marker = '[verify] py_compile'
    files: list[str] = []
    history = agent._chat_history
    start = max(0, min(agent._turn_start_index, len(history)))
    for idx in range(start, len(history)):
        m = history[idx]
        if m.get('role') != 'tool':
            continue
        content = str(m.get('content') or '')
        if marker not in content:
            continue
        first_line = content.splitlines()[0] if content else ''
        for prefix in ('Written ', 'Edited ', 'Skipped '):
            if first_line.startswith(prefix):
                path = first_line[len(prefix):]
                if path.endswith(')'):
                    path = path.rsplit('(', 1)[0].strip()
                if path and path not in files:
                    files.append(path)
                break
    return files


def _print_self_review_note(agent: "Agent", mutated_files: list[str]) -> None:
    """Post-mutation self-review reminder (plan item B-#6).

    py_compile proves a file PARSES — not that it does what was asked.
    This prints one short nudge listing the changed files so the user
    knows exactly what to double-check (or ask the agent to re-verify).
    """

    def _short(f: str) -> str:
        try:
            return os.path.relpath(f, agent.workspace).replace('\\', '/')
        except ValueError:
            return f
    shown = ', '.join((_short(f) for f in mutated_files[:6]))
    more = f' (+{len(mutated_files) - 6} more)' if len(mutated_files) > 6 else ''
    print(magenta(f"\n  [self-review] This turn changed {len(mutated_files)} file(s): {shown}{more}. py_compile verified syntax only — run 'tests' or 'git diff' to confirm behaviour."))


def check_stale_files(agent: "Agent") -> list[str]:
    """Return files whose mtime has changed since last read."""
    stale = []
    for path in list(agent._file_mtimes):
        try:
            if os.path.getmtime(path) != agent._file_mtimes[path]:
                stale.append(path)
        except FileNotFoundError:
            stale.append(path)
        except OSError as e:
            logger.warning('Could not stat %s: %s', path, e)
    return stale


def invalidate_stale(agent: "Agent") -> int:
    """Remove stale entries from memory. Returns count of invalidated files."""
    stale = check_stale_files(agent)
    for path in stale:
        agent._file_mtimes.pop(path, None)
        agent._files_read.discard(path)
    if stale:
        agent._semantic_index.clear()
    return len(stale)
