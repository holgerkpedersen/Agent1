"""Pytest command policy for the run/tests tools (taskplan step t06).

Owns the FULL-vs-targeted-vs-subset classification of a shell command and the
suite timeout budget, so ``agent._is_full_pytest_command`` and friends have one
authoritative implementation.

History: this block lived inline in ``agent.py`` (lines ~4530-4691).  Commit
``ac9a3df`` (the REFACTOR_TASKPLAN rewrite) deleted it while rewiring agent.py's
import block to ``agent_core.shell_utils``, whose naive
``is_full_pytest_command(args: list[str])`` became the effective classifier --
but the run tool passes it a STRING, so its marker check
``"tests/" in args`` degenerated into a substring search over the whole
command.  Two symmetric failures (31 + 11 test failures, identical at the
baseline commit):

* ``python -m pytest "tests/test_x.py::test_y" ...`` CONTAINS ``tests/`` ->
  misclassified as a FULL run -> the full-run gate refused it once the
  per-session budget was spent (reported in the REPL), and a targeted run
  wrongly received the suite timeout;
* bare ``python -m pytest -q`` has no ``tests/`` -> misclassified as
  targeted -> the full-run gate never fired (the exact leak the gate exists
  to prevent) and the suite timeout was never injected.

The budget helpers read ``agent._ENV_FILE_PATH`` at CALL time (through the
``agent`` shim, same pattern as ``_agent_mod()``), because tests monkeypatch
that name on the ``agent`` module.
"""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Sequence

from agent_core.pytest_gate import DEFAULT_FULL_SUITE_TIMEOUT, FULL_SUITE_MARGIN

#: Repo-root .env consulted for the full-pytest budget (conftest reads it too).
#: Two-level __file__ form: this module lives in ``agent_core/``.
_ENV_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
)
_PYTEST_FULL_SUITE_TIMEOUT_KEY = "PYTEST_FULL_SUITE_TIMEOUT"
_PYTEST_LAST_FULL_RUN_KEY = "PYTEST_LAST_FULL_RUN_SECONDS"
#: Must match conftest._FULL_SUITE_MARGIN so tool and watchdog agree; the
#: authoritative copy lives in agent_core.pytest_gate.
_FULL_SUITE_MARGIN = FULL_SUITE_MARGIN

#: Tokens that are part of the interpreter/pytest invocation itself, not a path.
_PYTEST_BARE_TOKENS = {
    "python", "python2", "python3", "py", "pytest", "py.test", "m", "set",
    "call", "&", "&&", "|", "||", ";",
}

#: pytest options that consume the FOLLOWING token as their value.  Without
#: this the value (``no:cacheprovider`` in ``-p no:cacheprovider``, ``not slow``
#: in ``-m "not slow"``, ``4`` in ``-n 4``) is mistaken for a test path, the
#: command is misclassified as TARGETED, and a full run is killed at the 600s
#: default instead of the suite budget.  Values attached with ``=`` are handled
#: by the generic flag check.
_PYTEST_VALUE_FLAGS = {
    "-p", "-k", "-m", "-o", "-c", "-n", "-W", "-r",
    "--deselect", "--ignore", "--ignore-glob", "--maxfail", "--tb",
    "--durations", "--rootdir", "--confcutdir", "--basetemp", "--junitxml",
    "--junit-prefix", "--log-level", "--log-file", "--log-format", "--dist",
    "--import-mode", "--cache-dir", "--override-ini", "--numprocesses",
    "--lfnf", "--last-failed-no-failures",
}

#: Flags that SELECT A SUBSET, so the invocation is not a full run: ``--lf``
#: / ``--nf`` (only failed / only new) and ``--testmon`` (only what changed).
#: Without this, ``pytest --lf`` looked like a full run and got both the suite
#: budget and the full-run gate - i.e. the cheap lanes were penalised exactly
#: like the expensive one.  Mirrors conftest._SUBSET_FLAGS.
_PYTEST_SUBSET_FLAGS = {
    "--lf", "--last-failed", "--lfnf", "--last-failed-no-failures",
    "--nf", "--new-first", "--failed-first", "--testmon",
}


def _env_file_path() -> str:
    """The .env path: ``agent._ENV_FILE_PATH`` when the shim binds it.

    Tests redirect the budget file by patching ``agent._ENV_FILE_PATH`` (the
    pre-refactor implementation read the same module global), so the lookup
    must happen at call time on the shim, not on this module.
    """
    try:
        import agent as _agent_mod
        value = getattr(_agent_mod, "_ENV_FILE_PATH", None)
        if value:
            return str(value)
    except Exception:  # noqa: BLE001 - agent not importable: fall back below
        pass
    return _ENV_FILE_PATH


def _read_env_value(key: str, default: str = "") -> str:
    """Budget value for *key*: process env wins, else the repo .env."""
    if key in os.environ:
        return os.environ[key].strip()
    try:
        with open(_env_file_path(), "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                env_key, _, val = stripped.partition("=")
                if env_key.strip() == key:
                    return val.strip()
    except OSError:
        pass
    return default


def _pytest_full_suite_timeout() -> float:
    """Timeout (seconds) to grant a pytest run.

    The model regularly guesses a timeout (often 300-600s) for
    ``python -m pytest ...``, which is below the suite's real cost -- the run
    tool then killed it mid-suite and the whole run was wasted.  Reuse the SAME
    budget conftest's watchdog enforces:
    ``max(PYTEST_FULL_SUITE_TIMEOUT, PYTEST_LAST_FULL_RUN_SECONDS * 1.5)``,
    read from the process env then the repo ``.env``.  The floor comes from
    measurement (a real full run took 861.5s here, 608s in CI), not from taste
    -- see :data:`agent_core.pytest_gate.DEFAULT_FULL_SUITE_TIMEOUT`.
    """
    try:
        floor = float(
            _read_env_value(_PYTEST_FULL_SUITE_TIMEOUT_KEY)
            or DEFAULT_FULL_SUITE_TIMEOUT
        )
    except ValueError:
        floor = DEFAULT_FULL_SUITE_TIMEOUT
    try:
        last = float(_read_env_value(_PYTEST_LAST_FULL_RUN_KEY) or 0.0)
    except ValueError:
        last = 0.0
    return max(floor, last * (1.0 + _FULL_SUITE_MARGIN))


def _as_command(command: str | Sequence[str]) -> str:
    """Normalize *command* to a string (callers pass either form)."""
    if isinstance(command, str):
        return command
    return " ".join(str(token) for token in command)


def _is_full_pytest_command(command: str | Sequence[str]) -> bool:
    """True when *command* invokes pytest with no explicit test path.

    Mirrors conftest's definition of a full run: no positional path at all, or
    the whole test tree spelled out (``tests`` / ``tests/`` / ``.`` -- the
    configured ``testpaths``).  Spelling the suite explicitly (``pytest
    tests/``) used to be misread as a targeted run, so it got the 120s default
    instead of the suite budget and was killed mid-run.  A subset selector
    (``--lf`` / ``--nf`` / ``--testmon``) is NOT a full run.  Segments without
    pytest, ``K=V`` assignments and redirections (``2>&1``, ``>nul``) are
    ignored; values consumed by :data:`_PYTEST_VALUE_FLAGS` are not paths;
    any other positional token marks a targeted run.
    """
    command = _as_command(command)
    for tokens in _pytest_segments_tokens(command):
        skip_next = False
        for token in tokens:
            if skip_next:
                skip_next = False
                continue
            if token in _PYTEST_SUBSET_FLAGS:
                # Checked BEFORE the value-flag list: ``--lfnf`` is in both and
                # is boolean in pytest, so it must not swallow the next token.
                return False  # runs a subset, not the suite
            if token in _PYTEST_VALUE_FLAGS:
                skip_next = True
                continue
            if token.startswith("-"):
                continue
            if token in _PYTEST_BARE_TOKENS or "=" in token:
                continue
            if re.match(r"^\d*>&?\S*$", token):
                continue
            if token.replace("\\", "/").rstrip("/") in ("tests", "."):
                continue  # the whole test tree spelled out is still a full run
            return False
        return True
    return False


def _is_pytest_command(command: str | Sequence[str]) -> bool:
    """True when *command* invokes pytest at all (full, subset or targeted).

    The whole-suite budget bounds the cost of ANY pytest invocation -- a subset
    can select most of the suite -- so every pytest command gets it as its
    timeout floor.  Targeted path runs are the exception (see
    :func:`_is_subset_pytest_command`); non-pytest commands stay capped.
    """
    return bool(_pytest_segments_tokens(command))


def _is_subset_pytest_command(command: str | Sequence[str]) -> bool:
    """True for a pytest run limited by a SUBSET selector, not by a path.

    ``pytest --lf`` / ``--nf`` / ``--testmon`` is not a full run (the
    full-run gate stays off), but it still selects from the whole suite and
    can cost as much as one -- so it deserves the same timeout floor.  This is
    the reported bug: ``python -m pytest --lf -q --no-cov`` was killed at the
    model's guessed 300s because it matched neither the full-run path nor a
    targeted path.
    """
    for tokens in _pytest_segments_tokens(command):
        if any(t in _PYTEST_SUBSET_FLAGS for t in tokens):
            return True
    return False


def _pytest_segments_tokens(command: str | Sequence[str]) -> list[list[str]]:
    """Tokenize each ``&&``/``||``/``;``/``|`` segment that invokes pytest.

    Shared by every classifier in this module (full/subset/pytest detection and
    the conflict guard below) so they all see the SAME tokenization (``shlex``
    with a plain-split fallback for unbalanced quotes).
    """
    command = _as_command(command)
    segments: list[list[str]] = []
    for segment in re.split(r"&&|\|\||[;|]", command):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            tokens = segment.split()
        if any(t in ("pytest", "py.test") for t in tokens):
            segments.append(tokens)
    return segments


def _testmon_cacheprovider_conflict(tokens: list[str]) -> str | None:
    """Detect ``--testmon`` + ``-p no:cacheprovider`` (pytest-testmon crash).

    pytest-testmon 2.2.0's ``pytest_configure`` -> ``_get_noselect_reasons``
    does a hard ``options["lf"]`` lookup (``testmon/configure.py``).  ``lf`` is
    registered by pytest's *cacheprovider* plugin; ``-p no:cacheprovider``
    unregisters it, so the key is missing and the KeyError is re-raised by
    pytest as ``INTERNALERROR`` -- before any test runs (unrelated to
    ``--tb=``).  cacheprovider does not otherwise conflict with testmon, so
    the fix is simply to drop ``-p no:cacheprovider``.
    """
    has_testmon = any(t == "--testmon" for t in tokens)
    if not has_testmon:
        return None
    disables_cache = False
    for i, tok in enumerate(tokens):
        if tok == "-p" and i + 1 < len(tokens) and tokens[i + 1] == "no:cacheprovider":
            disables_cache = True
            break
        if tok in ("-p=no:cacheprovider", "-pno:cacheprovider"):
            disables_cache = True
            break
    if not disables_cache:
        return None
    return (
        "--testmon cannot be combined with -p no:cacheprovider: pytest-testmon's "
        "pytest_configure reads the 'lf' option registered by cacheprovider and "
        "crashes with INTERNALERROR (KeyError: 'lf'). Drop -p no:cacheprovider."
    )


def pytest_flag_conflict(command: str | Sequence[str]) -> str:
    """Return a refusal message for a known-bad pytest flag combo, else ``""``.

    Guards the run/tests tools against invocations that CRASH pytest
    (INTERNALERROR / hang) rather than producing a normal usage error -- the
    model would otherwise burn a turn on a stack trace it cannot fix.  Each
    combo is a small predicate over one pytest segment's tokens; the first
    match wins and its message is returned verbatim (callers prefix ``Error:``).
    """
    for tokens in _pytest_segments_tokens(command):
        conflict = _testmon_cacheprovider_conflict(tokens)
        if conflict:
            return conflict
    return ""
