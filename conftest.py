"""Root pytest configuration.

Trace capture in agent.py chat_nlp is opt-out (AGENT_NO_TRACE=1 disables it)
so a real agent session produces a trace corpus by default.  Test runs must
not write reports/traces/ artifacts, so the whole suite runs with tracing
disabled unless a test explicitly enables it.
"""

import os
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("AGENT_NO_TRACE", "1")


# ---------------------------------------------------------------------------
# Suppress the noisy ``PermissionError: [WinError 5]`` traceback that pytest
# emits at exit on Windows.  During teardown pytest's ``cleanup_numbered_dir``
# → ``cleanup_dead_symlinks`` tries to ``stat()`` the ``pytest-current``
# symlink inside ``%TEMP%/pytest-of-<user>/``; this fails when another process
# still holds a handle on that directory.  We monkey-patch the cleanup
# function so it silently ignores ``PermissionError`` / ``OSError``.
# ---------------------------------------------------------------------------
try:
    import _pytest.pathlib as _ptplib

    _orig_cleanup = _ptplib.cleanup_dead_symlinks

    def _safe_cleanup_dead_symlinks(directory: Path) -> None:  # type: ignore[override]
        try:
            _orig_cleanup(directory)
        except (PermissionError, OSError):
            pass

    _ptplib.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks  # type: ignore[assignment]
except Exception:  # noqa: BLE001
    pass


@pytest.fixture(autouse=True)
def _isolate_from_real_tree_and_beacons(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every test from mutating tracked source or the live beacons.

    Two classes of test previously touched the real repo / live dashboard:

    * The harnessfix repair modules (``stuck_repeat``, ``tool_interface``,
      ``abandonment_resume``) apply/revert via a module-level
      ``_TARGET = Path("agent_core/llm/tool_loop.py")`` resolved relative to
      CWD.  Tests that call ``apply()``/``revert()`` (directly, or through
      ``run_loop`` / ``_reset_repairs``) therefore edit the REAL file.  When
      such a test runs inside an autonomous driver's post-repair ``pytest``
      gate, it reverts the very repair the loop just applied, so the loop
      records ``accepted`` but has nothing to commit and silently spins every
      remaining iteration re-applying a repair that can never persist.
    * ``harnessfix.progress`` writes the live beacon (``run_status.json``) and
      history (``run_history.jsonl``) to ``reports/harnessfix``.  Driver tests
      that call ``main()`` (e.g. ``test_autonomous_driver.py``) and the
      issue-loop test therefore pollute those live files, corrupting the
      dashboard during a real run.

    Both are redirected to a per-test temp dir here, once, for the whole
    suite.  Tests that already sandbox their own edits (e.g.
    ``test_repairs_stuck_repeat.py``) stay compatible: their own monkeypatch
    simply overrides this one for the duration of the test.
    """
    import harnessfix.progress as progress
    import harnessfix.repairs.abandonment_resume as abandonment_resume
    import harnessfix.repairs.stuck_repeat as stuck_repeat
    import harnessfix.repairs.tool_interface as tool_interface

    # Modules that edit a real source file via a module-level ``_TARGET``.
    # The redirected copies live in a SEPARATE temp dir (sibling of
    # ``tmp_path``) — never inside the per-test workspace — so the
    # workspace walkers in ``_suggest_consumers`` and ``_unwired_closure``
    # (``agent_core/commands/implement_cmd.py``) never see the redirected
    # files as "existing modules" or "reference sources".  Tests treat
    # ``tmp_path`` as the user's project; leaking harnessfix internals in
    # there caused false ``b``/``c``-substring pin matches against
    # ``tool_loop.py``'s 900 lines of Python source (every other file is
    # full of those letters) and false ``tool``-token matches in
    # ``_suggest_consumers``.  The regressions are pinned by
    # ``TestSuggestConsumers`` and ``TestUnwiredClosure`` in
    # ``tests/test_implement_safety.py``.
    targets_dir = tmp_path_factory.mktemp("harnessfix_targets_") / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    for mod in (stuck_repeat, tool_interface, abandonment_resume):
        real = mod._TARGET
        if not real.exists():
            continue
        local = targets_dir / f"{mod.__name__.split('.')[-1]}_{real.name}"
        local.write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setattr(mod, "_TARGET", local)
        # Revert any repair already applied in the real tree so the
        # sandboxed copy starts from the UN-applied (original) state.
        # All three repair modules need this: tool_interface's loop tests
        # read the target to assert whether the repair was applied;
        # stuck_repeat's apply/revert roundtrip tests copy the REAL file
        # and need it in the un-applied state; and the harnessfix_loop
        # tests expect stuck_repeat to be selectable (not already-applied).
        try:
            mod.revert()
        except Exception:
            pass

    # Redirect the live autonomous-run beacons to temp so driver/loop tests
    # never write into reports/harnessfix/*.
    beacon_dir = tmp_path / "reports" / "harnessfix"
    monkeypatch.setattr(progress, "OUTPUT_DIR", beacon_dir)
    monkeypatch.setattr(progress, "STATUS_PATH", beacon_dir / "run_status.json")
    monkeypatch.setattr(progress, "HISTORY_PATH", beacon_dir / "run_history.jsonl")


# ---------------------------------------------------------------------------
# Full-suite time budget.
#
# The real elapsed time of a FULL pytest run (bare `pytest`, no explicit test
# paths) is recorded in the repo `.env` as PYTEST_LAST_FULL_RUN_SECONDS.  The
# NEXT full run enforces a ceiling of max(PYTEST_FULL_SUITE_TIMEOUT, last
# recorded duration * (1 + margin)) — i.e. the saved time is reused with 20%
# headroom.  Targeted runs (`pytest tests/test_x.py`) are never measured or
# killed, so single-file debugging is safe.
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).resolve().parent / ".env"
_FULL_SUITE_TIMEOUT_KEY = "PYTEST_FULL_SUITE_TIMEOUT"
_LAST_FULL_RUN_KEY = "PYTEST_LAST_FULL_RUN_SECONDS"
_DEFAULT_FULL_SUITE_TIMEOUT = 600.0
_FULL_SUITE_MARGIN = 0.20  # 20% headroom over the last recorded duration

_watchdog_stop = threading.Event()


def _load_env() -> dict[str, str]:
    """Read the repo .env into a flat key->value dict (comments skipped)."""
    values: dict[str, str] = {}
    if not _ENV_FILE.exists():
        return values
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _, val = stripped.partition("=")
            values[key.strip()] = val.strip()
    return values


def _save_env(updates: dict[str, str]) -> None:
    """Update the repo .env in place, preserving comments and other keys."""
    values = _load_env()
    values.update(updates)
    if not _ENV_FILE.exists():
        _ENV_FILE.write_text("", encoding="utf-8")
    lines = _ENV_FILE.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        key = None
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
        if key in values:
            out.append(f"{key}={values[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, val in values.items():
        if key not in seen:
            out.append(f"{key}={val}")
    _ENV_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")


def _is_full_run(config: pytest.Config) -> bool:
    """A full run = no explicit test paths on the command line."""
    raw = getattr(config, "invocation_params", None)
    args = list(getattr(raw, "args", None) or [])
    return not any(a for a in args if not a.startswith("-"))


def _get_env_value(key: str) -> str:
    """Budget value for a key: process env wins, else repo .env."""
    if key in os.environ:
        return os.environ[key].strip()
    return _load_env().get(key, "").strip()


def _resolve_budget() -> float:
    """max(configured floor, last recorded full-run duration * (1 + margin))."""
    try:
        floor = float(_get_env_value(_FULL_SUITE_TIMEOUT_KEY) or _DEFAULT_FULL_SUITE_TIMEOUT)
    except ValueError:
        floor = _DEFAULT_FULL_SUITE_TIMEOUT
    try:
        last = float(_get_env_value(_LAST_FULL_RUN_KEY) or 0.0)
    except ValueError:
        last = 0.0
    return max(floor, last * (1.0 + _FULL_SUITE_MARGIN))


def _watchdog_loop(started: float, budget: float, stop: threading.Event) -> None:
    """Abort the process (exit 124) once `budget` seconds have elapsed."""
    while not stop.wait(1.0):
        if time.monotonic() - started > budget:
            print(
                f"\n[conftest] full pytest run exceeded {budget:.0f}s budget — aborting "
                f"(see PYTEST_FULL_SUITE_TIMEOUT / PYTEST_LAST_FULL_RUN_SECONDS in .env).",
                flush=True,
            )
            os._exit(124)  # 124 = timeout, same convention as `timeout(1)`


def pytest_sessionstart(session: pytest.Session) -> None:
    session._full_run_started = time.monotonic()
    session._is_full_run = _is_full_run(session.config)
    if not session._is_full_run:
        return
    budget = _resolve_budget()
    session._full_run_budget = budget
    print(
        f"\n[conftest] full pytest run budget: {budget:.0f}s "
        f"(floor {_get_env_value(_FULL_SUITE_TIMEOUT_KEY) or _DEFAULT_FULL_SUITE_TIMEOUT}s, "
        f"last run {_get_env_value(_LAST_FULL_RUN_KEY) or 'n/a'}s + {_FULL_SUITE_MARGIN:.0%} margin)",
        flush=True,
    )
    _watchdog_stop.clear()
    threading.Thread(
        target=_watchdog_loop,
        args=(session._full_run_started, budget, _watchdog_stop),
        daemon=True,
        name="pytest-full-run-watchdog",
    ).start()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not getattr(session, "_is_full_run", False):
        return
    elapsed = time.monotonic() - session._full_run_started
    _save_env({_LAST_FULL_RUN_KEY: f"{elapsed:.1f}"})
    _watchdog_stop.set()
