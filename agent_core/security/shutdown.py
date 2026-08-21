"""Cleanup-aware signal handlers for SIGBREAK (Windows Ctrl+Break / taskkill).

Replaces the bare ``os._exit(1)`` handler in ``agent.py`` that previously
skipped cleanup hooks (``_save_memory``, trace writer close), risking data
loss mid-session.  This module performs best-effort cleanup — memory save +
trace flush/close — before exiting, and never raises so a failing hook can't
mask the shutdown itself.

The store directory can be redirected with ``AGENT1_SECRETS_DIR`` (tests/portable).
"""
from __future__ import annotations

import logging
import os
import signal
import traceback
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def safe_signal_break_handler(
    memory_path: str | os.PathLike[str],
    trace_writer_close: Callable[[], Any] | None,
    save_memory_fn: Callable[[], Any] | None,
) -> None:
    """Perform cleanup then exit on SIGBREAK.

    Order matters: persist agent memory first (so cross-session state like
    files read / knowledge graph survives), then flush + close the trace writer
    so no in-flight tool-call effects are lost from ``reports/traces/*.jsonl``.
    Each step is best-effort — a failing hook logs and continues rather than
    aborting shutdown.  Finally calls ``os._exit(1)`` (the signal handler can't
    raise out of C-level input() on Windows).

    Parameters
    ----------
    memory_path : str | PathLike
        Agent memory file path (for logging context only — the actual save is
        delegated to *save_memory_fn*).
    trace_writer_close : callable or None
        Callable that flushes and closes an active TraceWriter, if any.
    save_memory_fn : callable or None
        Callable that persists agent cross-session memory (``_save_memory``).
    """
    try:
        if callable(save_memory_fn):
            save_memory_fn()
            logger.info("Memory saved during signal break handler (%s)", memory_path)
    except Exception as exc:  # noqa: BLE001 — never abort shutdown for a hook failure
        logger.error(
            "Error saving memory during signal break:\n%s", traceback.format_exc(),
        )

    try:
        if callable(trace_writer_close):
            trace_writer_close()
            logger.info("Trace writer closed during signal break handler")
    except Exception as exc:  # noqa: BLE001 — never abort shutdown for a hook failure
        logger.error(
            "Error closing trace writer during signal break:\n%s", traceback.format_exc(),
        )

    os._exit(1)


def register_signal_break_handler(
    memory_path: str,
    save_memory_fn: Callable[[], Any] | None = None,
    trace_writer_close: Callable[[], Any] | None = None,
) -> bool:
    """Register a cleanup-aware SIGBREAK handler.

    Returns True when the handler was installed (SIGBREAK is Windows-only),
    False on platforms without it so callers can fall back to SIGTERM.
    """
    sig = getattr(signal, "SIGBREAK", None) or signal.SIGTERM

    def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001 — signature required by signal API
        safe_signal_break_handler(memory_path, trace_writer_close, save_memory_fn)

    try:
        signal.signal(sig, _handler)
    except (ValueError, OSError):
        logger.warning("Could not install SIGBREAK handler on this platform")
        return False
    return True


def shutdown_for_exit(
    memory_path: str | os.PathLike[str],
    save_memory_fn: Callable[[], Any] | None = None,
    trace_writer_close: Callable[[], Any] | None = None,
) -> None:
    """Non-signal cleanup path (e.g. normal REPL exit): same hooks, no hard exit."""
    try:
        if callable(save_memory_fn):
            save_memory_fn()
            logger.info("Memory saved during shutdown (%s)", memory_path)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Error saving memory during shutdown:\n%s", traceback.format_exc(),
        )

    try:
        if callable(trace_writer_close):
            trace_writer_close()
            logger.info("Trace writer closed during shutdown")
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Error closing trace writer during shutdown:\n%s", traceback.format_exc(),
        )
