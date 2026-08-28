"""Live progress beacon for the autonomous self-improvement loop.

The loop only writes ``summary.json`` at the END of each iteration (a single
iteration runs the full pytest suite and can take several minutes).  To let the
dashboard show *what the autonomous agent is doing right now*, both the driver
(``scripts/autonomous_self_improve.py``) and ``harnessfix.loop.run_loop`` emit
lightweight phase beacons here.

Two artifacts are maintained under ``reports/harnessfix/``:

- ``run_status.json``  -- the current live state (phase, iteration, current
  repair target, gate in progress, started timestamps).  Overwritten on every
  phase change; read by the dashboard as the real-time signal.
- ``run_history.jsonl`` -- one line per finished iteration, so the dashboard can
  render a results table across all iterations of a run even after the process
  has exited.

All writes are atomic (temp file + ``os.replace``) so a half-written beacon can
never be read by the dashboard, and reads tolerate a missing/corrupt file.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "reports" / "harnessfix"
STATUS_PATH = OUTPUT_DIR / "run_status.json"
HISTORY_PATH = OUTPUT_DIR / "run_history.jsonl"


def output_dir() -> Path:
    """Directory holding the live beacons.

    Defined as a function (not a bare constant) so tests can redirect beacon
    writes to a temp dir by monkeypatching ``OUTPUT_DIR`` -- a bare constant
    would be copied into callers' namespaces and ignore the override.
    """
    return OUTPUT_DIR


def status_path() -> Path:
    """Path of the live run_status.json beacon (see :func:`output_dir`)."""
    return STATUS_PATH


def history_path() -> Path:
    """Path of the live run_history.jsonl beacon (see :func:`output_dir`)."""
    return HISTORY_PATH

#: Phases the loop moves through, in rough order.  Used only for stable labels
#: and human-readable ordering in the UI; the beacon stores a free-form string.
PHASES = (
    "idle",
    "collecting_traces",
    "diagnosing",
    "evaluating_candidate",
    "applying_repair",
    "running_test_gate",
    "running_security_gate",
    "running_harness_gate",
    "finalizing",
    "finished",
)


def _now() -> float:
    return time.time()


def write_progress(update: Dict[str, Any]) -> None:
    """Merge ``update`` into the live run_status.json beacon atomically.

    Only the keys present in ``update`` are overwritten; other live fields
    (e.g. the driver's ``iteration`` set before the loop runs) are preserved.
    A ``ts`` heartbeat is stamped on every write so the dashboard can tell
    whether the process is still alive.
    """
    OUTPUT_DIR = output_dir()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if status_path().is_file():
        try:
            existing = json.loads(status_path().read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing.update(update)
    existing["ts"] = _now()
    tmp = status_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, status_path())


def set_phase(phase: str, **extra: Any) -> None:
    """Convenience: write a phase change plus optional extra fields."""
    write_progress({"phase": phase, **extra})


def append_history(record: Dict[str, Any]) -> None:
    """Append one finished-iteration record to run_history.jsonl atomically.

    Implemented as a single-line append; the temp+replace dance is unnecessary
    here because appends are independent and the dashboard never reads a
    half-line (jsonlines is line-delimited).  We still guard the parent dir.
    """
    output_dir().mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with history_path().open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read_progress() -> Dict[str, Any]:
    """Return the live run_status.json, or an empty dict if absent/corrupt."""
    if not status_path().is_file():
        return {}
    try:
        data = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def read_history(limit: int | None = None) -> list[Dict[str, Any]]:
    """Return finished-iteration records, oldest first (newest appended last)."""
    if not history_path().is_file():
        return []
    records: list[Dict[str, Any]] = []
    try:
        with history_path().open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
    except OSError:
        return []
    if limit is not None:
        records = records[-limit:]
    return records


def clear_progress() -> None:
    """Mark the run finished/idle so the dashboard stops showing 'live'."""
    set_phase("idle", ended=_now(), running=False)


__all__ = [
    "STATUS_PATH",
    "HISTORY_PATH",
    "PHASES",
    "write_progress",
    "set_phase",
    "append_history",
    "read_progress",
    "read_history",
    "clear_progress",
]
