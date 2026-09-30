"""Gate that keeps FULL pytest runs rare, earned and late.

A full run of this suite costs ~2 minutes and ~2500 tests.  Observed failure
mode: the agent launches ``python -m pytest -q --no-cov`` in the middle of a
*read-only* task (a "what's next?" question), because bare ``pytest`` is by
convention a full run.  Worse, the only file it had touched was a scratch
probe (``_tmp_probe.py``), so the run verified nothing.

Two deterministic rules, both free:

1. **A full run must be EARNED** — since the session baseline (or the last
   full run) at least one *real* source file must differ.  Scratch paths
   (``_tmp_*``, ``tmp/*``, ``*.bak``, ``__pycache__`` ...) and non-testable
   artifacts never count (:func:`is_scratch_path`, :func:`is_testable_source`).
2. **A full run must be RARE** — at most ``AGENT_MAX_FULL_PYTEST_RUNS`` (default
   1) per session, so it is naturally spent near the END of the work, once.

When the gate refuses, the caller returns the refusal text to the model
instead of running pytest, so the model learns the cheap lanes (``--lf``,
``--testmon``, explicit paths) from a tool result rather than from a two-minute
waste.

Escape hatch: ``AGENT_PYTEST_FULL_RUN_GATE=off`` disables the gate entirely
(human-in-the-loop sessions, or a deliberate second full pass).
"""

from __future__ import annotations

import os
import subprocess
from typing import Iterable

__all__ = [
    "DEFAULT_MAX_FULL_RUNS",
    "GATE_ENV",
    "MAX_FULL_RUNS_ENV",
    "FullRunGate",
    "changed_real_files",
    "gate_enabled",
    "is_scratch_path",
    "is_testable_source",
    "max_full_runs_from_env",
]

#: Environment override for the per-session full-run budget.
MAX_FULL_RUNS_ENV = "AGENT_MAX_FULL_PYTEST_RUNS"
#: Set to ``off``/``0``/``false`` to disable the gate completely.
GATE_ENV = "AGENT_PYTEST_FULL_RUN_GATE"
#: One full run per session: the point is that it happens ONCE, at the end.
DEFAULT_MAX_FULL_RUNS = 1

#: Directory prefixes that never count as a real change (generated / runtime).
_IGNORED_DIR_PREFIXES = (
    "__pycache__",
    "reports/",
    ".docs/",
    "backups/",
    "generated/",
    "htmlcov/",
    ".pytest_cache/",
    "tmp/",
    "temp/",
    "scratch/",
    ".git/",
)

#: Filename prefixes/patterns that are scratch by convention.
_SCRATCH_PATTERNS = (
    "_tmp_",
    "tmp_",
    "_scratch_",
    "scratch_",
    "test_tmp_",
)

_SCRATCH_SUFFIXES = (".bak", ".orig", ".rej", ".tmp", ".log")

#: Extensions a full-suite run can actually regress on.
_TESTABLE_SUFFIXES = (
    ".py",
    ".pyi",
    ".toml",
    ".ini",
    ".cfg",
    ".json",
    ".yml",
    ".yaml",
    ".txt",
)

#: Plain git porcelain status codes that mean "this path differs from HEAD".
_MEANINGFUL_STATUS = ("M", "A", "D", "R", "C", "T")

_GIT_TIMEOUT_S = 15


def is_scratch_path(path: str) -> bool:
    """True for throwaway probe/scratch files (``_tmp_probe.py`` and friends).

    A scratch file is deliberately excluded from the "real change" set: writing
    one must never entitle the agent to burn a two-minute full-suite run.
    """
    norm = path.replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    if not norm:
        return False
    lowered = norm.lower()
    if any(
        lowered.startswith(p) or ("/" + p) in lowered for p in _IGNORED_DIR_PREFIXES
    ):
        return True
    name = lowered.rsplit("/", 1)[-1]
    if name.endswith(_SCRATCH_SUFFIXES):
        return True
    stem = name.rsplit(".", 1)[0]
    return any(stem.startswith(p) for p in _SCRATCH_PATTERNS)


def is_testable_source(path: str) -> bool:
    """True when *path* is a tracked, non-scratch file the suite can regress on."""
    if is_scratch_path(path):
        return False
    lowered = path.replace("\\", "/").lower()
    return lowered.endswith(_TESTABLE_SUFFIXES)


def _porcelain_paths(output: str) -> Iterable[str]:
    """Yield the path from each ``git status --porcelain -uall`` line."""
    for line in output.splitlines():
        if len(line) < 4:
            continue
        entry = line[3:].strip()
        if not entry:
            continue
        code = line[:2]
        if all(ch in (" ", "?", "!") for ch in code):
            continue  # untracked or ignored
        if not any(ch in _MEANINGFUL_STATUS for ch in code):
            continue
        # Renames/copies: "old -> new"; the new name is the one on disk.
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        yield entry.strip().strip('"')


def changed_real_files(root: str) -> set[str]:
    """Real (tracked, testable) paths that differ from HEAD under *root*.

    Uses ``git status --porcelain -uall``; untracked files are excluded because
    a scratch/one-off probe is precisely what must not trigger a full run.
    Best effort: any git failure yields an empty set (never a raised error),
    and the gate then simply refuses a full run until something real appears.
    """
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "-uall"],
            cwd=root or None,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # git missing / timeout
        if os.environ.get("AGENT_PYTEST_GATE_DEBUG"):
            import logging

            logging.getLogger(__name__).warning("git status unavailable: %s", exc)
        return set()
    if proc.returncode != 0:
        return set()
    return {
        p.replace("\\", "/")
        for p in _porcelain_paths(proc.stdout)
        if is_testable_source(p)
    }


_GATE_OFF = {"off", "0", "false", "no", "none"}


def gate_enabled() -> bool:
    """False when ``AGENT_PYTEST_FULL_RUN_GATE`` explicitly disables the gate."""
    return (os.environ.get(GATE_ENV) or "").strip().lower() not in _GATE_OFF


def max_full_runs_from_env() -> int:
    """Per-session full-run budget: ``AGENT_MAX_FULL_PYTEST_RUNS`` or the default."""
    raw = (os.environ.get(MAX_FULL_RUNS_ENV) or "").strip()
    if not raw:
        return DEFAULT_MAX_FULL_RUNS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_FULL_RUNS
    return max(0, value)


class FullRunGate:
    """Allow a FULL pytest run only when it is earned (:meth:`check` returns None).

    Usage: construct once per :class:`agent.Agent` (baseline = the tracked
    changes already present at session start), call :meth:`check` before every
    full run and :meth:`record_full_run` after one is spent.
    """

    def __init__(
        self,
        root: str,
        max_full_runs: int | None = None,
        baseline: set[str] | None = None,
    ) -> None:
        self.root = root
        self._max = (
            max_full_runs if max_full_runs is not None else max_full_runs_from_env()
        )
        self._baseline = changed_real_files(root) if baseline is None else set(baseline)
        self._runs = 0

    @property
    def runs(self) -> int:
        """Full runs spent so far this session."""
        return self._runs

    @property
    def max_runs(self) -> int:
        return self._max

    def pending_real_changes(self) -> set[str]:
        """Real files that changed since the session baseline."""
        return changed_real_files(self.root) - self._baseline

    def check(self) -> str | None:
        """Return a refusal message, or None when a full run is allowed.

        The budget is a HARD per-session cap: a new real change does not buy
        extra full runs (that is what the cheap lanes are for).
        """
        if not gate_enabled():
            return None
        if self._runs >= self._max:
            return (
                "Error: full-suite pytest REFUSED - the per-session budget of "
                f"{self._max} full run(s) is already spent. Spend the rest of "
                "the session on cheap lanes: "
                "`python -m pytest --lf -q --no-cov` (only what failed), "
                "`python -m pytest --testmon -q --no-cov` (only what changed), or "
                "`python -m pytest <path>::<TestName> -q --no-cov` (targeted). "
                f"Raise {MAX_FULL_RUNS_ENV} deliberately only when the change is "
                "wide enough to justify another full pass."
            )
        pending = self.pending_real_changes()
        if not pending:
            shown = ", ".join(sorted(self._baseline)[:5]) or "(none)"
            return (
                "Error: full-suite pytest REFUSED - no real source file has changed "
                "since the session baseline (scratch/probe files such as _tmp_*.py "
                f"never count; real files already dirty at baseline: {shown}). A "
                "full run would verify nothing. Use a cheap lane instead: "
                "`python -m pytest --lf -q --no-cov` (only what failed), "
                "`python -m pytest --testmon -q --no-cov` (only what changed), or "
                "`python -m pytest <path>::<TestName> -q --no-cov` (targeted). "
                "Reserve the full suite for the END of the work, after real edits."
            ).format(shown=shown)
        return None

    def record_full_run(self) -> None:
        """Record that a full run was spent (the budget is a hard cap)."""
        self._runs += 1
