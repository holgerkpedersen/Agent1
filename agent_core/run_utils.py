"""Run utilities: process management, kanban bridge, trace, and stream helpers.

Extracted from agent.py as the t00 run-concern block (step t01).
Contains helpers that cross-cut the run loop and tool execution.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from agent_core.config import AgentDisplayMode

from agent_core.llm.provider import provider_supports_streaming

logger = logging.getLogger(__name__)

# ── Kanban inbound ───────────────────────────────────────────────────────────


def _start_kanban_inbound() -> None:
    """Start the inbound kanban bridge processor.

    Delegates to kanban_bridge.start_inbound_processor() so the agent
    picks up tasks handed to it via the kanban file.
    """
    try:
        from agent_core.orchestration.kanban_bridge import start_inbound_processor
        start_inbound_processor()
    except ImportError:
        logger.debug("kanban_bridge not available — running without inbound processor")


# ── Process tree ─────────────────────────────────────────────────────────────


def kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Kill a process and all its children, cross-platform."""
    try:
        if sys.platform == "win32":
            # On Windows, terminate the process tree using taskkill
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=10,
            )
        else:
            # On POSIX, kill the whole process group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.debug("killpg failed: %s", exc)
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
    except Exception as exc:
        logger.debug("kill_process_tree failed: %s", exc)


# ── Stream / display helpers ─────────────────────────────────────────────────


async def _stream_via_llm(llm: Any, messages: list[dict[str, Any]]) -> str:
    """Await the LLM's ``chat_stream`` and return the complete text."""
    return await llm.chat_stream(messages)


def resolve_stream_fn(
    stream: bool | Any,
    display_mode: "AgentDisplayMode" | None = None,
    *,
    llm: Any | None = None,
) -> Callable[..., Any] | None:
    """Return the streaming function for this LLM/configuration, or None.

    Supports two calling conventions:
    * ``resolve_stream_fn(stream: bool, display_mode, llm=llm)`` — explicit
      boolean gate; the LLM must be supplied as a keyword argument.
    * ``resolve_stream_fn(llm)`` — legacy API preserved while call sites and
      tests migrate: checks ``settings.stream_final_answer`` and the LLM
      provider's real streaming capability.

    Returns ``None`` when streaming is disabled, the provider does not really
    print tokens as they arrive, or no LLM is available.
    """
    # Legacy single-argument LLM call: not a bool and no display_mode supplied.
    if display_mode is None and not isinstance(stream, bool):
        llm = stream
        try:
            # Access via module attribute so tests/monkeypatches can replace it.
            from agent_core import config as _config_module
            settings = _config_module.load_agent_settings()
            if not getattr(settings, "stream_final_answer", False):
                return None
        except Exception:
            return None
        provider = getattr(llm, "_provider", None)
        if not provider_supports_streaming(provider):
            return None
        from functools import partial
        return partial(_stream_via_llm, llm)
    if stream and llm is not None:
        provider = getattr(llm, "_provider", None)
        if not provider_supports_streaming(provider):
            return None
        from functools import partial
        return partial(_stream_via_llm, llm)
    return None


def resolve_display_mode() -> "AgentDisplayMode":
    """Return the agent's display mode from settings/env (default: verbose).

    Reads ``AGENT_DISPLAY_MODE`` via :func:`load_agent_settings` (process env
    wins, then the ``.env`` found from the cwd upward — so ``display intent``
    in the REPL takes effect on the next turn), falling back to a direct env
    parse when settings are unavailable, and finally ``VERBOSE``.

    Always returns a real ``AgentDisplayMode`` member.  The t00 extraction
    shipped a ``getattr(settings, "display_mode", AgentDisplayMode.ANSI)``
    whose default referenced a nonexistent member: arguments evaluate before
    ``getattr`` runs, so every call raised ``AttributeError``, was swallowed,
    and returned ``0`` — which ``ToolLoopRunner`` mapped to VERBOSE, silently
    killing intent/quiet/clean while every test kept passing (they patch the
    resolver instead of calling it).
    """
    from agent_core.config import AgentDisplayMode, _parse_display_mode, load_agent_settings

    try:
        return load_agent_settings().display_mode
    except Exception:
        logger.debug("load_agent_settings failed in resolve_display_mode", exc_info=True)
    return _parse_display_mode(os.environ.get("AGENT_DISPLAY_MODE"), AgentDisplayMode.VERBOSE)


def final_answer_fallback() -> str:
    """Return the fallback text when no final answer is available."""
    return "(no answer produced)"


# ── Turn quality ─────────────────────────────────────────────────────────────


def turn_quality_score(
    llm_error: str | None,
    mutated_files: list[str],
) -> float | None:
    """B3: per-turn quality score in [0,1] for the shared metrics log.

    With harnessfix tracing enabled and a turn armed via ``begin_turn()``
    (the tool-loop caller arms it once per turn), the turn's own trace
    events are drained and scored with ``score_run`` -- one turn is never
    scored from another turn's events.  A provider error always scores
    0.0, whatever the trace recorded.  Otherwise the cheap proxy:
    ``llm_error`` -> 0.0, mutated files -> 0.8, clean turn -> 0.7.
    Never raises; returns ``None`` only if the check itself fails
    (nothing is recorded then).
    """
    try:
        from harnessfix.tracing import trace_enabled

        if trace_enabled():
            import harnessfix.tracing as _tr

            events: list[dict[str, Any]] | None = None
            for _name in ("take_turn_events", "drain_turn_events",
                          "current_turn_events", "turn_events"):
                _getter = getattr(_tr, _name, None)
                if callable(_getter):
                    events = _getter()
                    if events:
                        break
            if llm_error:
                return 0.0
            if events:
                from harnessfix.evolution_metrics import score_run

                return float(score_run(events))
        if llm_error:
            return 0.0
        if mutated_files:
            return 0.8
        return 0.7
    except Exception:  # noqa: BLE001 - observability must not break turns
        return None


# ── Trace helpers ───────────────────────────────────────────────────────────


def begin_trace_turn(turn_id: str | None = None) -> None:
    """Arm turn-scoped trace-event collection for the coming turn.

    Plan item #4 residual: the turn's own trace events are buffered in
    memory (``harnessfix.tracing``) so :func:`turn_quality_score` can feed
    them to ``score_run`` at turn end.  Degrades to a no-op when harnessfix
    (or the API) is unavailable; never raises.
    """
    try:
        import harnessfix.tracing as _tr

        starter = getattr(_tr, "begin_turn", None) or getattr(_tr, "start_turn", None)
        if callable(starter):
            starter()
    except Exception:  # noqa: BLE001 - observability must not break turns
        logger.debug("Could not arm trace turn (no-op)", exc_info=True)


def append_turn_log(record: dict[str, Any]) -> None:
    """Append one turn-outcome line to ``turn_log.jsonl`` (bounded, atomic-ish).

    Harness-layer observability only (decision #014): the log feeds habit
    mining and is rewritten to its newest 500 lines whenever it grows past
    that bound, so it can never bloat the state dir.  Never raises - a broken
    or missing state dir must not break a finished turn.
    """
    try:
        import json
        import os
        from agent_core import constants as _constants_module
        path = _constants_module.TURN_LOG_PATH
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        lines: list[str] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        lines.append(json.dumps(record, ensure_ascii=False, default=str))
        if len(lines) > 500:
            lines = lines[-500:]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - observability must not break turns
        logger.debug("Could not append turn log (no-op)", exc_info=True)


# ── Dynamic system blocks ─────────────────────────────────────────────────────


def strip_dynamic_system_blocks(text: str) -> str:
    """Remove previously injected dynamic blocks from a system prompt.

    ``chat_nlp`` rebuilds the constraints block (decision ledger), original
    goal, skill index, habits, semantic memory, file history, and plan-mode
    suffix on every turn; without stripping, a long-lived session would
    accumulate one stale copy of each block per turn.  The BASE prompt -
    everything before the first dynamic marker - is what survives.
    """
    markers = (
        "\n\nCRITICAL DESIGN CONSTRAINTS",
        "\n\nSESSION MODE: PLAN",
        "\n\nORIGINAL TASK",
        "\n\nSKILLS (on-demand procedural knowledge)",
        "\n\nUSER HABITS",
        "\n\nSEMANTIC MEMORY",
        "\n\nRECENT FILE NOTES",
        "\n\n[habits]",
        "\n\n[file history]",
        "\n\n[semantic memory]",
    )
    cut = len(text)
    for marker in markers:
        pos = text.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    return text[:cut]


# ── Action claim helper ──────────────────────────────────────────────────────


def unverified_action_claim(user_input: str, tool_calls_made: int) -> bool:
    """Return True if the user's input looks like an unverified action claim.

    Guard: warn when a user says "done / fixed / done, all tests pass" without
    the agent having run any tools (decision #090).
    """
    claim_words = {"done", "fixed", "all", "tests", "pass", "complete", "finished"}
    input_words = set(user_input.lower().split())
    matched = claim_words & input_words
    return bool(matched) and tool_calls_made == 0
