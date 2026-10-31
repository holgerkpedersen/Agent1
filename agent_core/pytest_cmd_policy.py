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
    "python",
    "python2",
    "python3",
    "py",
    "pytest",
    "py.test",
    "m",
    "set",
    "call",
    "&",
    "&&",
    "|",
    "||",
    ";",
}

#: pytest options that consume the FOLLOWING token as their value.  Without
#: this the value (``no:cacheprovider`` in ``-p no:cacheprovider``, ``not slow``
#: in ``-m "not slow"``, ``4`` in ``-n 4``) is mistaken for a test path, the
#: command is misclassified as TARGETED, and a full run is killed at the 600s
#: default instead of the suite budget.  Values attached with ``=`` are handled
#: by the generic flag check.
_PYTEST_VALUE_FLAGS = {
    "-p",
    "-k",
    "-m",
    "-o",
    "-c",
    "-n",
    "-W",
    "-r",
    "--deselect",
    "--ignore",
    "--ignore-glob",
    "--maxfail",
    "--tb",
    "--durations",
    "--rootdir",
    "--confcutdir",
    "--basetemp",
    "--junitxml",
    "--junit-prefix",
    "--log-level",
    "--log-file",
    "--log-format",
    "--dist",
    "--import-mode",
    "--cache-dir",
    "--override-ini",
    "--numprocesses",
    "--lfnf",
    "--last-failed-no-failures",
}

#: Flags that SELECT A SUBSET, so the invocation is not a full run: ``--lf``
#: / ``--nf`` (only failed / only new) and ``--testmon`` (only what changed).
#: Without this, ``pytest --lf`` looked like a full run and got both the suite
#: budget and the full-run gate - i.e. the cheap lanes were penalised exactly
#: like the expensive one.  Mirrors conftest._SUBSET_FLAGS.
_PYTEST_SUBSET_FLAGS = {
    "--lf",
    "--last-failed",
    "--lfnf",
    "--last-failed-no-failures",
    "--nf",
    "--new-first",
    "--failed-first",
    "--testmon",
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


# ── Non-Python test targets (decision #124) ─────────────────────────────────
#
# History (2026-10-31): a documentation turn ended with
# ``run(command="python -m pytest docs/PROSPECTOR_PLAN.md -q")`` - pytest aimed
# at a Markdown PLAN document.  pytest collects nothing from a ``.md`` file, so
# the run printed "no tests ran" (exit code 5) and the transcript then looked
# as if the plan had been "tested".  A plan is verified by its content and by
# structural checks, never by a test runner, so that command is refused before
# any process starts.

#: File suffixes pytest/unittest cannot collect a test from.
_NON_PYTHON_TARGET_SUFFIXES = frozenset(
    {
        ".md",
        ".rst",
        ".txt",
        ".json",
        ".jsonl",
        ".toml",
        ".cfg",
        ".ini",
        ".yml",
        ".yaml",
        ".xml",
        ".csv",
        ".tsv",
        ".log",
        ".html",
        ".css",
        ".js",
        ".ts",
        ".ps1",
        ".bat",
        ".cmd",
        ".sh",
        ".sql",
        ".ipynb",
    }
)

#: Extensionless (or dot-prefixed) files that are documentation/config too.
_NON_PYTHON_TARGET_NAMES = frozenset(
    {
        ".gitignore",
        ".gitattributes",
        ".env",
        ".decisions.json",
        "agents",
        "readme",
        "license",
        "licence",
        "changelog",
        "changes",
        "contributing",
        "usage",
        "makefile",
        "dockerfile",
    }
)

#: Refusal returned when a test runner is pointed at a non-Python file.
#: The placeholders are ``{target}`` only; everything else is literal so the
#: message cannot drift from the rule it states.
NON_PYTHON_TARGET_REFUSAL = (
    "Error: {target} is not a Python test target - pytest and unittest cannot "
    "collect tests from it. Do not run a test runner against a plan, document, "
    "data or config file: a green 'no tests ran' proves nothing about it "
    "(decision #124).\n"
    "Use a real test file instead, for example "
    "`python -m pytest tests/test_x.py -q --no-cov`, or verify a non-Python "
    "file with a structural check such as "
    "`python scripts/audit_invariants.py`."
)


def _is_non_python_test_target(token: str) -> bool:
    """True when *token* names a file a test runner cannot collect from.

    Only FILES are judged (a name carrying a known non-Python suffix, a
    well-known doc/config name, or a dotfile), so a legitimate directory or
    dotted word (``tests/v1.0``, ``tests.test_x``) is never mistaken for one.
    """
    name = token.strip().strip("\"'`").replace("\\", "/").rstrip("/")
    if not name or name in (".", "..") or name.endswith("/"):
        return False
    if "::" in name:  # node id: the file part is what pytest collects
        name = name.split("::", 1)[0]
    base = name.rsplit("/", 1)[-1].strip().lower()
    if not base:
        return False
    if base in _NON_PYTHON_TARGET_NAMES:
        return True
    if base.startswith(".") and base.count(".") == 1:  # dotfile: .gitignore
        return True
    suffix = base[base.rfind(".") :] if "." in base else ""
    return suffix in _NON_PYTHON_TARGET_SUFFIXES


#: Segment heads that can actually start a test-runner invocation, so a
#: wrapper such as ``echo pytest plan.md`` is never read as a pytest run.
_INVOCATION_HEADS = frozenset(
    {"python", "python2", "python3", "py", "pytest", "py.test"}
)

#: Interpreter modules that run tests; the positionals after them are targets.
_TEST_MODULES = frozenset({"pytest", "py.test", "unittest"})


def _pytest_target_tokens(command: str | Sequence[str]) -> list[list[str]]:
    """Positional TARGET tokens of every segment that really invokes a runner.

    Stricter than :func:`_pytest_segments_tokens` on purpose: this feeds a
    REFUSAL, so a segment must start with an interpreter/test-runner head
    before its positionals count as targets.  ``echo pytest plan.md`` and
    ``git status && python -m pytest -q`` therefore yield no targets, while
    ``python -m pytest docs/PLAN.md`` yields ``docs/PLAN.md``.
    """
    targets: list[list[str]] = []
    for segment in re.split(r"&&|\|\||[;|]", _as_command(command)):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            tokens = segment.split()
        # Trim wrappers/prompts before the head: leading K=V assignments and
        # the Windows ``set``/``call``/``&`` prefixes.
        while tokens and ("=" in tokens[0] or tokens[0] in ("set", "call", "&")):
            tokens = tokens[1:]
        if not tokens:
            continue
        head = tokens[0].lower()
        if head not in _INVOCATION_HEADS:
            continue
        if head in ("pytest", "py.test"):
            body = tokens[1:]
        elif len(tokens) > 2 and tokens[1] in ("-m", "-M"):
            module = tokens[2].strip("\"'").lower()
            if module not in _TEST_MODULES:
                continue
            body = tokens[3:]
        else:
            continue
        collected: list[str] = []
        skip_next = False
        for token in body:
            if skip_next:
                skip_next = False
                continue
            if token in _PYTEST_SUBSET_FLAGS:
                continue
            if token in _PYTEST_VALUE_FLAGS:
                skip_next = True  # the value is an option value, not a path
                continue
            if token.startswith("-") or "=" in token:
                continue  # option, `--opt=value` or a K=V assignment
            if re.match(r"^\d*>&?\S*$", token):  # 2>&1, >nul
                continue
            collected.append(token)
        if collected:
            targets.append(collected)
    return targets


def pytest_non_python_target(command: str | Sequence[str]) -> str:
    """First non-Python pytest TARGET in *command*, or ``""``.

    A target is a positional argument of a real pytest/unittest invocation -
    never an option value (``-k``/``-m``/``-p``/``--ignore``), never a ``K=V``
    assignment and never part of another command in a compound line.
    """
    for target_list in _pytest_target_tokens(command):
        for token in target_list:
            if _is_non_python_test_target(token):
                return token
    return ""


def non_python_pytest_target_refusal(command: str | Sequence[str]) -> str:
    """Refusal message for a pytest invocation aimed at a non-Python file."""
    target = pytest_non_python_target(command)
    if not target:
        return ""
    return NON_PYTHON_TARGET_REFUSAL.format(target=target)


def non_python_path_refusal(path: str) -> str:
    """Refusal message for a test runner aimed at a bare PATH, else ``""``.

    Path-level twin of :func:`non_python_pytest_target_refusal`, for callers
    that already hold a resolved path instead of a command line (the ``tests``
    tool and the ``self_heal`` command).  Same predicate, same message, so the
    rule cannot drift between entry points (decision #124).
    """
    if not path or not _is_non_python_test_target(str(path)):
        return ""
    return NON_PYTHON_TARGET_REFUSAL.format(target=path)


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
