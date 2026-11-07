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
import re
import subprocess
from typing import Iterable

__all__ = [
    "DEFAULT_MAX_FULL_RUNS",
    "DEFAULT_FULL_SUITE_TIMEOUT",
    "FULL_SUITE_MARGIN",
    "FULL_SUITE_TIMEOUT_ENV",
    "GATE_ENV",
    "LAST_FULL_RUN_ENV",
    "MAX_FULL_RUNS_ENV",
    "FullRunGate",
    "changed_real_files",
    "effective_repo_root",
    "find_repo_root",
    "full_suite_timeout",
    "gate_enabled",
    "is_scratch_path",
    "is_testable_source",
    "max_full_runs_from_env",
    "record_full_run_seconds",
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


def find_repo_root(start: str) -> str:
    """Walk up from *start* to the git repository root (a ``.git`` dir/file).

    Returns *start* unchanged when no ``.git`` marker is found (e.g. the
    workspace is not a checkout) — :func:`changed_real_files` then degrades to
    an empty set, which is the gate's fail-closed default.
    """
    cur = os.path.abspath(start or ".")
    while True:
        if os.path.exists(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start or ".")
        cur = parent


#: One shell segment that changes the directory: ``cd``, ``Set-Location``,
#: ``sl``, ``chdir`` — with the cmd ``/d`` and PowerShell ``-Path`` switches,
#: and optional quotes.
_CD_SEGMENT_RE = re.compile(
    r"^\s*(?:cd|Set-Location|sl|chdir)"
    r"(?:\s+(?:/[dm]|-Path|-LiteralPath|-PSPath))?"
    r"\s+[\"']?(?P<path>[^\s\"';|]+)[\"']?\s*$",
    re.IGNORECASE,
)

#: Shell segment separators (POSIX + PowerShell).
_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||;|\||\r?\n")

#: Git-Bash mount style ``/c/Dev/Kanban`` -> ``C:\\Dev\\Kanban``.
_BASH_MOUNT_RE = re.compile(r"^/([a-z])/(.*)$")


def _normalize_cd_target(path: str, cwd: str) -> str:
    """Resolve one ``cd`` target *path* against *cwd* to a Windows-style path."""
    target = path.strip().rstrip(";,")
    mount = _BASH_MOUNT_RE.match(target)
    if mount:
        target = f"{mount.group(1).upper()}:\\{mount.group(2)}"
    if target in (".", "..", "", "~"):
        # ``cd ..`` chains are common; ``.``/``~`` are no-ops for repo lookup.
        if target == "..":
            return os.path.dirname(cwd)
        return cwd
    if not os.path.isabs(target):
        target = os.path.join(cwd, target)
    return os.path.normpath(target)


def effective_repo_root(command: str, start_cwd: str) -> str:
    """The git repo a shell command runs pytest in, following its ``cd`` segments.

    The NLP ``run`` tool launches every command at ``_effective_ws_dir`` and the
    model then chains ``cd <other-repo>; python -m pytest ...`` (the Kanban
    session).  A dirtiness check pinned to the SESSION workspace then sees a
    clean tree and refuses a run whose edits live in the OTHER repo — the full
    suite becomes unreachable no matter how much real work was done.  This
    helper tracks every ``cd``/``Set-Location`` segment in command order and
    resolves the final directory's repository root, so the gate asks the right
    repo "did real files change here?".

    Best-effort by design: a target that does not exist keeps the previous cwd
    (a typo must not poison the lookup), and any unparseable command yields the
    repo of *start_cwd* — the pre-cross-repo behavior.
    """
    cwd = os.path.abspath(start_cwd or ".")
    for segment in _SEGMENT_SPLIT_RE.split(command or ""):
        match = _CD_SEGMENT_RE.match(segment)
        if not match:
            continue
        candidate = _normalize_cd_target(match.group("path"), cwd)
        if os.path.isdir(candidate):
            cwd = candidate
    return find_repo_root(cwd)


_GATE_OFF = {"off", "0", "false", "no", "none"}


def gate_enabled() -> bool:
    """False when ``AGENT_PYTEST_FULL_RUN_GATE`` explicitly disables the gate."""
    return (os.environ.get(GATE_ENV) or "").strip().lower() not in _GATE_OFF


#: Repo-root ``.env`` consulted for the full-run COUNT, mirroring how the
#: timeout budget already works (``agent._read_env_value``): process env wins,
#: then ``.env``.
_ENV_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
)


def _read_env_file_value(key: str) -> str:
    """Value of *key* from the repo-root ``.env``, or ``""`` when absent."""
    try:
        with open(_ENV_FILE_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                env_key, _, val = stripped.partition("=")
                if env_key.strip() == key:
                    return val.strip()
    except OSError:
        pass
    return ""


#: Environment key holding the floor for a pytest run's timeout budget.
FULL_SUITE_TIMEOUT_ENV = "PYTEST_FULL_SUITE_TIMEOUT"
#: Environment key holding the last measured full-suite duration in seconds,
#: written back after every full run (see :func:`record_full_run_seconds`).
LAST_FULL_RUN_ENV = "PYTEST_LAST_FULL_RUN_SECONDS"

#: Timeout floor when nothing has been measured yet.  Chosen from measurement,
#: not taste: a real full run of this suite took **861.5s** on the development
#: box and **608s** in CI, so the old 600s floor was already below the real
#: cost and killed runs mid-suite.  1500s survives an unmapped machine; as soon
#: as one run is recorded, the 50% margin over that measurement takes over.
DEFAULT_FULL_SUITE_TIMEOUT = 1500.0

#: Headroom granted over the last measured full-run duration.  MUST stay in
#: sync with ``conftest._FULL_SUITE_MARGIN`` (a regression test asserts it) so
#: the tool timeout and the in-suite watchdog agree.
FULL_SUITE_MARGIN = 0.50


def full_suite_timeout() -> float:
    """Seconds a pytest run may take: ``max(floor, last_full_run * 1.5)``.

    The FLOOR (``PYTEST_FULL_SUITE_TIMEOUT``) is a setting, so process env wins
    over the repo ``.env`` — that is how CI raises it per job.  The MEASURED
    duration (``PYTEST_LAST_FULL_RUN_SECONDS``) is read from the ``.env`` file
    only: the ubuntu and windows CI jobs write it on the same checkout, and a
    process-env override would let the faster job shrink the slower job's
    budget.  Mirrors ``conftest._get_env_value``.
    """
    raw_floor = os.environ.get(FULL_SUITE_TIMEOUT_ENV)
    if raw_floor is None:
        raw_floor = _read_env_file_value(FULL_SUITE_TIMEOUT_ENV)
    try:
        floor = float((raw_floor or "").strip() or DEFAULT_FULL_SUITE_TIMEOUT)
    except ValueError:
        floor = DEFAULT_FULL_SUITE_TIMEOUT

    raw_last = _read_env_file_value(LAST_FULL_RUN_ENV)
    try:
        last = float((raw_last or "").strip() or 0.0)
    except ValueError:
        last = 0.0
    return max(floor, last * (1.0 + FULL_SUITE_MARGIN))


def _write_env_file_value(key: str, value: str) -> None:
    """Set *key* in the repo-root ``.env``, preserving comments and other keys.

    Never raises: losing a timing measurement must not break the run that
    produced it.
    """
    try:
        with open(_ENV_FILE_PATH, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        lines = []
    out: list[str] = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if (
            not stripped.startswith("#")
            and "=" in stripped
            and stripped.partition("=")[0].strip() == key
        ):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        if out and out[-1].strip():
            out.append("")
        out.append(f"{key}={value}")
    try:
        with open(_ENV_FILE_PATH, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")
    except OSError:
        pass


def record_full_run_seconds(seconds: float, *, completed: bool = True) -> None:
    """Persist how long a full pytest run really took.

    A COMPLETED run is a measurement and overwrites the stored value.  A run
    killed by a timeout only proves it lasted *at least* that long, so it is
    stored as a lower bound (``max(stored, seconds)``) — the budget then grows
    past the point that killed it instead of being pinned to a too-short
    guess, which is the failure mode that makes a harness kill the same run
    over and over.  Never raises.
    """
    try:
        seconds = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return
    if not completed:
        try:
            stored = float(_read_env_file_value(LAST_FULL_RUN_ENV) or 0.0)
        except ValueError:
            stored = 0.0
        seconds = max(stored, seconds)
    _write_env_file_value(LAST_FULL_RUN_ENV, f"{seconds:.1f}")


def max_full_runs_from_env() -> int:
    """Per-session full-run budget: ``AGENT_MAX_FULL_PYTEST_RUNS`` or the default.

    Process env wins; when the variable is *absent* the repo-root ``.env`` is
    consulted — that is where ``PYTEST_FULL_SUITE_TIMEOUT`` already lives, and
    the refusal text tells the model to "raise AGENT_MAX_FULL_PYTEST_RUNS
    deliberately", which until now had nowhere to be raised except the launch
    environment.  A present-but-unparsable value still falls back to the
    default rather than to ``.env``.
    """
    raw = os.environ.get(MAX_FULL_RUNS_ENV)
    if raw is None:
        raw = _read_env_file_value(MAX_FULL_RUNS_ENV)
    raw = (raw or "").strip()
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

    The gate follows the command's repository: :meth:`check` accepts the repo
    root the run will execute in (see :func:`effective_repo_root`), so work in
    a SECOND repo (e.g. ``cd C:\\Dev\\Kanban; python -m pytest``) is judged
    against THAT repo's dirt, not the session workspace's.  Each foreign repo
    is baselined at first sight, so its pre-existing tracked changes never
    entitle a run.  The run BUDGET stays session-wide: one full pass per
    session, whichever repo earns it.
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
        #: Baselines per repo root (normalized).  The primary root keeps the
        #: constructor baseline; secondary roots are captured at first check.
        self._baselines: dict[str, set[str]] = {
            self._norm(root): set(self._baseline),
        }
        self._runs = 0

    @staticmethod
    def _norm(root: str) -> str:
        """Case/shape-insensitive key for a repo root path."""
        return os.path.normcase(os.path.abspath(root or "."))

    @property
    def runs(self) -> int:
        """Full runs spent so far this session."""
        return self._runs

    @property
    def max_runs(self) -> int:
        return self._max

    def baseline_for(self, root: str) -> set[str]:
        """The baseline of *root*: the primary repo keeps the session baseline;
        a FOREIGN repo has none.

        The primary baseline exists to stop a read-only probe in the session's
        own workspace from burning the suite on pre-existing dirt.  A repo the
        command explicitly ``cd``-s into and runs pytest against IS the work
        target: any real tracked change there is by definition this session's
        work (the Kanban session was already dirty at the gate's first check,
        so a lazily-captured baseline refused it forever).
        """
        key = self._norm(root)
        if key in self._baselines:
            return self._baselines[key]
        return set()

    def pending_real_changes(self, root: str | None = None) -> set[str]:
        """Real files that changed since the baseline of *root* (default: primary)."""
        target = self.root if root is None else root
        return changed_real_files(target) - self.baseline_for(target)

    def check(self, root: str | None = None) -> str | None:
        """Return a refusal message, or None when a full run is allowed.

        The budget is a HARD per-session cap: a new real change does not buy
        extra full runs (that is what the cheap lanes are for).  *root* is the
        repository the run will execute in; the dirtiness rule is evaluated
        there, so cross-repo work is judged on its own tree.
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
        target = self.root if root is None else root
        pending = self.pending_real_changes(target)
        if not pending:
            base = self.baseline_for(target)
            shown = ", ".join(sorted(base)[:5]) or "(none)"
            repo_note = (
                ""
                if self._norm(target) == self._norm(self.root)
                else f" (checked repo: {os.path.abspath(target)})"
            )
            return (
                "Error: full-suite pytest REFUSED - no real source file has changed "
                "since the session baseline (scratch/probe files such as _tmp_*.py "
                f"never count; real files already dirty at baseline: {shown}). A "
                "full run would verify nothing. Use a cheap lane instead: "
                "`python -m pytest --lf -q --no-cov` (only what failed), "
                "`python -m pytest --testmon -q --no-cov` (only what changed), "
                "or `python -m pytest <path>::<TestName> -q --no-cov` (targeted). "
                "Reserve the full suite for the END of the work, after real edits."
                f"{repo_note}"
            )
        return None

    def record_full_run(self) -> None:
        """Record that a full run was spent (the budget is a hard cap)."""
        self._runs += 1

    def reset(self) -> None:
        """Re-baseline and hand the full-run budget back for a fresh session.

        :meth:`agent.Agent.clear_history` calls this so a cleared session
        starts with a full budget and a baseline of *current* tracked changes,
        not the dirty files an earlier session left behind. Without it the
        gate is ghost state: ``clear`` wipes chat history and the memory
        JSONs, yet a spent full-run budget silently carries into every later
        session (every subsequent full run refused). Baseline re-read is best
        effort — :func:`changed_real_files` already degrades to an empty set
        when git is unavailable.
        """
        self._baseline = changed_real_files(self.root)
        self._baselines = {self._norm(self.root): set(self._baseline)}
        self._runs = 0
