"""Refuse an IDENTICAL pytest run when nothing in the tree has changed.

The failure mode (2026-11-27): verification is repeated for no reason.  A turn
ends with a green targeted run, the next turn re-runs the SAME command "to be
sure", or a green run is repeated after a commit that changed no source line.
A second identical run over the same tree cannot produce a different result, so
it is pure cost - and it is what makes a session look busy while nothing is
being learned.

This module owns ONE deterministic rule: *verification stays valid until the
repository changes* (the state-change rule).  It records, per session and per
command, the result summary plus a fingerprint of the tree pytest would read,
and REFUSES a repeat whose fingerprint is unchanged - returning the PREVIOUS
result to the model instead of spending another run.

Design rules (each one is deliberate):

* **Deterministic and free.**  The fingerprint comes from
  ``git status --porcelain -uall`` plus content hashes of the paths that
  differ - no LLM, no clock, no randomness.  Only files git already reports
  are hashed, so the cost is proportional to the EDITS, not to the repo.
* **Fail open.**  No git, an unreadable tree, an oversized workspace or any
  other uncertainty yields ``None`` and the run is ALLOWED.  The guard may
  never block work it cannot prove redundant.
* **Per session, like** :class:`agent_core.pytest_gate.FullRunGate`.  The memo
  is memory only; ``clear`` resets it and a new session starts empty, so a
  stale record from yesterday can never silence today's run.
* **Escape hatch.**  ``AGENT_PYTEST_REPEAT_GUARD=off`` (process env) disables
  it for a deliberate flakiness check - the one legitimate reason to re-run an
  identical command without a change.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence

__all__ = [
    "GUARD_ENV",
    "MAX_RECORDS",
    "PytestRepeatGuard",
    "RunRecord",
    "disabled",
    "guard_enabled",
    "normalize_command",
    "summarize_pytest_output",
    "tree_fingerprint",
]

#: Set to ``off``/``0``/``false`` to disable the guard (deliberate re-run).
GUARD_ENV = "AGENT_PYTEST_REPEAT_GUARD"

#: How many (command -> record) entries one session keeps.  A long session can
#: run many distinct lanes; the oldest entry is dropped first (FIFO), so the
#: memo is bounded without ever growing with the number of runs.
MAX_RECORDS = 64

_GIT_TIMEOUT_S = 15

#: Directories that never hold source pytest reads (or that a run CREATES).
_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".sessions",
        ".worktrees",
        ".scratch-quarantine",
    }
)

#: Files bigger than this are identified by size+mtime instead of a content
#: hash: hashing a multi-MB artifact (a model blob, a trace bundle) on every
#: check would cost more than the run it is trying to save.
_MAX_HASH_BYTES = 1024 * 1024
#: Upper bound on files hashed per fingerprint (the rest use size+mtime).
_MAX_HASHED_FILES = 256
#: Upper bound on entries in the non-git fallback scan; beyond it we give up
#: (``None``) rather than walk an enormous tree on every check.
_MAX_SCAN_ENTRIES = 5000

_OFF = {"off", "0", "false", "no", "none"}


def guard_enabled() -> bool:
    """False when ``AGENT_PYTEST_REPEAT_GUARD`` explicitly disables the guard."""
    return (os.environ.get(GUARD_ENV) or "").strip().lower() not in _OFF


def disabled() -> bool:
    """Readable inverse of :func:`guard_enabled` (call sites read better)."""
    return not guard_enabled()


def normalize_command(command: str | Sequence[str]) -> str:
    """Canonical key for a command: whitespace collapsed, case folded.

    ``python -m pytest -q --no-cov`` and ``Python  -M  Pytest -q --no-cov``
    are the same run on Windows, so they must hit the same record.  Token
    ORDER is deliberately preserved: ``-k "not slow"`` pairs a flag with its
    value, so sorting tokens would merge different selections.
    """
    if not isinstance(command, str):
        command = " ".join(str(token) for token in command)
    return re.sub(r"\s+", " ", command).strip().casefold()


#: Final pytest line, e.g. ``==== 1 failed, 12 passed in 0.83s ====`` or
#: ``==== no tests ran in 0.02s ====``.  Matched on the STRIPPED line, so the
#: ``=`` padding only appears where pytest actually indents a result word.
_RESULT_WORD_RE = re.compile(
    r"\b(?:passed|failed|error|errors|skipped|xfailed|xpassed"
    r"|deselected|warning|warnings)\b"
)
_IN_LINE_RE = re.compile(
    r"^(?P<line>[^=\n]*?\bin \d+(?:\.\d+)?s[^=\n]*?)\s*=*$"
)


def summarize_pytest_output(output: str, exit_code: int | None = None) -> str:
    """One-line description of a pytest result, for the refusal message.

    Prefers pytest's own final summary line (the last line that names a
    passed/failed/error count), so the model sees the number it would
    otherwise re-discover by re-running.  Falls back to the exit code and
    finally to ``"no summary captured"`` - never raises, never guesses.
    """
    text = output or ""
    for raw_line in reversed(text.splitlines()):
        line = raw_line.strip().strip("=").strip()
        if not line:
            continue
        if _RESULT_WORD_RE.search(line) and re.search(r"\d", line):
            return re.sub(r"\s+", " ", line)[:160]
        match = _IN_LINE_RE.match(line)
        if match:
            return re.sub(r"\s+", " ", match.group("line")).strip()[:160]
    if exit_code is not None:
        return f"exit code {exit_code} (no summary line in the captured output)"
    return "no summary captured"


def _status_entries(root: str) -> list[tuple[str, str]] | None:
    """``git status --porcelain -uall`` as sorted ``(code, path)`` pairs.

    ``None`` means "no git view" (git missing, not a repository, timeout) -
    the caller then falls back to the mtime scan, and if that is unusable the
    fingerprint is ``None`` (fail open).
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
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    entries: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        code = line[:2]
        path = line[3:].strip().strip('"')
        if " -> " in path:  # rename/copy: the new name is the one on disk
            path = path.split(" -> ", 1)[1].strip().strip('"')
        if path:
            entries.append((code, path))
    return sorted(entries)


def _path_descriptor(full: str, hashed: int) -> tuple[str, int]:
    """Identity of one path: a content hash when cheap, else size+mtime.

    Returns ``(descriptor, new_hashed_count)``.  A missing path is ``absent``,
    which makes a deletion a real state change.
    """
    try:
        stat = os.stat(full)
    except OSError:
        return "absent", hashed
    if not os.path.isfile(full):
        return f"dir:{stat.st_mtime_ns}", hashed
    if stat.st_size <= _MAX_HASH_BYTES and hashed < _MAX_HASHED_FILES:
        try:
            digest = hashlib.sha256()
            with open(full, "rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
            return f"sha256:{digest.hexdigest()}", hashed + 1
        except OSError:
            pass  # fall through to the stat identity
    return f"stat:{stat.st_size}:{stat.st_mtime_ns}", hashed


def _scan_descriptors(root: str) -> list[str] | None:
    """Bounded ``relpath:identity`` scan for workspaces that are not a repo."""
    descriptors: list[str] = []
    hashed = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORED_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace("\\", "/")
            identity, hashed = _path_descriptor(full, hashed)
            descriptors.append(f"{rel}={identity}")
            if len(descriptors) > _MAX_SCAN_ENTRIES:
                return None
    return descriptors


def _is_housekeeping_path(path: str) -> bool:
    """True for paths a pytest run itself creates or that no source lives in.

    ``.pytest_cache/``, ``.git/``, ``__pycache__/`` and the virtualenv trees are
    excluded from the fingerprint on purpose: they are (a) invisible to git in a
    repo that lists them in ``.gitignore`` and (b) WRITTEN BY the run being
    guarded.  Counting them would make every run look like it changed the tree.
    """
    parts = path.replace("\\", "/").split("/")
    return any(part in _IGNORED_DIRS for part in parts[:-1])


def tree_fingerprint(root: str) -> str | None:
    """Digest of everything under *root* that a pytest run could read.

    ``None`` means "cannot prove the tree is unchanged" - callers MUST treat
    that as a reason to allow the run.  The fingerprint covers the worktree as
    pytest sees it: every path git reports as differing from HEAD or as
    untracked, identified by content (or size+mtime for oversized files).
    Housekeeping trees a run writes itself (``.pytest_cache/``,
    ``__pycache__/``) are excluded, so a run can never make its own repetition
    look legitimate.
    """
    try:
        root = os.path.abspath(root or ".")
    except (TypeError, ValueError):
        return None
    if not os.path.isdir(root):
        return None

    entries = _status_entries(root)
    if entries is None:
        scanned = _scan_descriptors(root)
        if scanned is None:
            return None
        return hashlib.sha256("\n".join(scanned).encode("utf-8")).hexdigest()

    parts: list[str] = []
    hashed = 0
    for code, rel in entries:
        if _is_housekeeping_path(rel):
            continue
        full = os.path.join(root, rel.replace("/", os.sep))
        identity, hashed = _path_descriptor(full, hashed)
        parts.append(f"{rel}={identity}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunRecord:
    """What was learned from one completed pytest run."""

    command: str
    fingerprint: str
    summary: str
    finished_at: str


class PytestRepeatGuard:
    """Allow the first run of a command; refuse repeats over an unchanged tree.

    Usage (mirrors :class:`agent_core.pytest_gate.FullRunGate`)::

        refusal, fingerprint = guard.check(command, repo_root)
        if refusal:
            return refusal          # nothing changed: hand back the old result
        ...run pytest...
        guard.record(command, fingerprint, output=..., exit_code=...)

    ``fingerprint_fn`` is injectable so tests can drive the state change
    deterministically; production always uses :func:`tree_fingerprint`.
    """

    def __init__(
        self,
        root: str,
        *,
        fingerprint_fn: Callable[[str], str | None] | None = None,
        max_records: int = MAX_RECORDS,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.root = root
        self._fingerprint = fingerprint_fn or tree_fingerprint
        self._max_records = max(1, int(max_records))
        self._clock = clock or _default_clock
        self._records: dict[tuple[str, str], RunRecord] = {}

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _root_key(root: str) -> str:
        return os.path.normcase(os.path.abspath(root or "."))

    def _key(self, command: str | Sequence[str], root: str) -> tuple[str, str]:
        return (self._root_key(root), normalize_command(command))

    def _refusal(self, record: RunRecord, root: str) -> str:
        return (
            "Error: pytest re-run REFUSED - this exact command already ran at "
            f"{record.finished_at} and nothing in {os.path.abspath(root)} has "
            f"changed since (last result: {record.summary}).\n"
            "A second identical run cannot produce a different result, so it "
            "was skipped: verification stays valid until the repository "
            "changes (state-change rule). Do one of these instead:\n"
            "  - change the code, then re-run;\n"
            "  - run a DIFFERENT lane (`python -m pytest <path>::<Test> -q "
            "--no-cov`, or `--lf` for only the tests that failed);\n"
            "  - for a deliberate flakiness check (state that intent "
            f"explicitly), set {GUARD_ENV}=off for this process."
        )

    def _trim(self) -> None:
        while len(self._records) > self._max_records:
            oldest = next(iter(self._records))
            del self._records[oldest]

    # -- public API --------------------------------------------------------

    @property
    def records(self) -> int:
        """Number of commands currently memoised (each is one distinct lane)."""
        return len(self._records)

    def check(
        self,
        command: str | Sequence[str],
        root: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Return ``(refusal, fingerprint_of_the_tree_now)``.

        ``refusal`` is ``None`` when the run may proceed - on the first run of
        a command, after any real change, when the tree cannot be fingerprinted
        (fail open), or when the guard is disabled.  The fingerprint is handed
        back so :meth:`record` stores the state the result actually belongs to
        (the run itself may leave cache/report files behind; those must not
        make the next identical run look "new").
        """
        if not guard_enabled():
            return None, None
        target = self.root if root is None else root
        try:
            fingerprint = self._fingerprint(target)
        except Exception:  # noqa: BLE001 - a broken probe must never block work
            return None, None
        if not fingerprint:
            return None, None
        record = self._records.get(self._key(command, target))
        if record is None or record.fingerprint != fingerprint:
            return None, fingerprint
        return self._refusal(record, target), None

    def record(
        self,
        command: str | Sequence[str],
        fingerprint: str | None,
        *,
        root: str | None = None,
        output: str = "",
        exit_code: int | None = None,
    ) -> None:
        """Memoise a COMPLETED run so an identical repeat can be refused.

        Call this only after a run actually produced a result: a refusal, a
        timeout or a launch error is not evidence, and recording it would let
        the guard suppress the retry that is needed.  A ``None`` fingerprint
        means "the tree could not be fingerprinted" - nothing is stored, so the
        next run is allowed.
        """
        if not guard_enabled() or not fingerprint:
            return
        target = self.root if root is None else root
        key = self._key(command, target)
        record = RunRecord(
            command=normalize_command(command),
            fingerprint=str(fingerprint),
            summary=summarize_pytest_output(output, exit_code),
            finished_at=self._clock(),
        )
        # Re-insert so the dict order tracks recency (oldest dropped first).
        self._records.pop(key, None)
        self._records[key] = record
        self._trim()

    def reset(self) -> None:
        """Forget every record (``clear`` / a fresh session starts clean)."""
        self._records.clear()


def _default_clock() -> str:
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")
