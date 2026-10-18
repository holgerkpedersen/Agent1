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


def resolve_stream_fn(
    stream: bool | Any, display_mode: "AgentDisplayMode" | None = None
) -> Callable[..., Any] | None:
    """Return the appropriate streaming function, or None.

    Supports two calling conventions:
    * ``resolve_stream_fn(stream: bool, display_mode: AgentDisplayMode)`` — new
      generic API: return the stream function when *stream* is True.
    * ``resolve_stream_fn(llm)`` — legacy API preserved while call sites and
      tests migrate: checks ``settings.stream_final_answer`` and the LLM
      provider's real streaming capability.
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
        try:
            from agent_core.llm.streaming import stream_chat_sync
            return stream_chat_sync
        except ImportError:
            return None
    if stream:
        try:
            from agent_core.llm.streaming import stream_chat_sync
            return stream_chat_sync
        except ImportError:
            pass
    return None


def resolve_display_mode() -> "AgentDisplayMode":
    """Return the display mode from agent settings or default."""
    try:
        from agent_core.config import AgentDisplayMode, load_agent_settings
        settings = load_agent_settings()
        return getattr(settings, "display_mode", AgentDisplayMode.ANSI)
    except Exception:
        pass
    try:
        from agent_core.config import AgentDisplayMode
        return AgentDisplayMode.ANSI
    except Exception:
        return 0  # type: ignore[return-value]


def final_answer_fallback() -> str:
    """Return the fallback text when no final answer is available."""
    return "(no answer produced)"


# ── Turn quality ─────────────────────────────────────────────────────────────


def turn_quality_score(
    llm_error: str | None,
    mutated_files: list[str],
) -> float:
    """Score the quality of a turn (0.0–1.0) based on error and mutations.

    Lower scores indicate problems; 1.0 is perfect.
    """
    score = 1.0
    if llm_error:
        score -= 0.4
    if mutated_files:
        score -= min(0.2, len(mutated_files) * 0.05)
    return max(0.0, score)


# ── Trace helpers ───────────────────────────────────────────────────────────


def begin_trace_turn(turn_id: str | None = None) -> None:
    """Begin a new trace turn, if tracing is enabled."""
    try:
        from agent_core.monitoring.trace_writer import TraceWriter
        if TraceWriter.is_enabled():
            TraceWriter.get_instance().begin_turn(turn_id)
    except Exception:
        pass


def append_turn_log(record: dict[str, Any]) -> None:
    """Append a turn log record (used by _finish_turn)."""
    try:
        from agent_core.monitoring.trace_writer import TraceWriter
        if TraceWriter.is_enabled():
            TraceWriter.get_instance().append_turn_log(record)
    except Exception:
        pass


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
