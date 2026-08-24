"""Shared JSONL event log for cross-process dashboard metrics.

Every agent session appends one line per metric write to a repo-root-anchored
file; the dashboard server tails that file on each API request and replays
new events into its own in-memory collector (skipping lines written by its
own process).  This lets a standalone ``--serve`` dashboard show activity
from any other session in this workspace instead of an always-empty view.

Design notes:
- Append-only, one JSON object per line: crash-safe and multi-process safe.
- Every failure is swallowed: metrics must never break the agent or serving.
- Lines carry ``pid`` so combined REPL+dashboard processes don't double-count
  their own writes (they already went into the in-process collector).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict

#: Shared event file lives at the repository root so every session — however
#: it is launched — reads and writes the same log regardless of cwd.
EVENTS_FILENAME = ".metrics_events.jsonl"


def default_events_path() -> Path:
    """Repo-root-anchored path of the shared metrics event file."""
    return Path(__file__).resolve().parent.parent.parent / EVENTS_FILENAME


def append_event(kind: str, name: str, value: float) -> None:
    """Best-effort append of one metric event line. Never raises.

    Args:
        kind: "counter" (``value`` is the delta), "gauge" or "histogram"
            (``value`` is the sample).
        name: Metric name as written to the collector.
        value: Delta/sample associated with the write.
    """
    try:
        line = json.dumps(
            {"kind": kind, "name": name, "value": value, "pid": os.getpid()}
        ) + "\n"
        with open(default_events_path(), "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def replay_new_events(collector: Any, path: Path, offset: int) -> int:
    """Replay complete event lines from byte *offset* into *collector*.

    Returns the new byte offset.  An incomplete trailing line (another
    process mid-write) is left for the next call.  Events written by this
    process are skipped so combined REPL+dashboard mode doesn't double-count.
    """
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return offset
    if not data:
        return offset
    end = data.rfind(b"\n")
    if end == -1:
        return offset  # no complete line yet
    complete = data[: end + 1]
    own_pid = os.getpid()
    for raw in complete.splitlines():
        try:
            ev = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(ev, Dict):
            continue
        if ev.get("pid") == own_pid:
            continue
        name = ev.get("name")
        kind = ev.get("kind")
        value = ev.get("value")
        if not isinstance(name, str) or not isinstance(value, (int, float)):
            continue
        try:
            if kind == "counter":
                collector.increment_counter(name, float(value))
            elif kind == "gauge":
                collector.set_gauge(name, float(value))
            elif kind == "histogram":
                collector.record_histogram(name, float(value))
        except Exception:
            continue
    return offset + len(complete)


def make_event_tailer(path: Path | None = None) -> Callable[[Any], None]:
    """Return a thread-safe refresh callback suitable for DashboardAPIServer.

    The returned callable receives the server's collector, tails new events
    from the shared file (byte 0 on first call → full history on first poll),
    and swallows all failures.
    """
    target = Path(path) if path is not None else default_events_path()
    state = {"offset": 0}
    lock = threading.Lock()

    def tail(collector: Any) -> None:
        try:
            with lock:
                state["offset"] = replay_new_events(collector, target, state["offset"])
        except Exception:
            pass  # metrics must never break request handling

    return tail


__all__ = [
    "EVENTS_FILENAME",
    "default_events_path",
    "append_event",
    "replay_new_events",
    "make_event_tailer",
]
