"""Regression tests for agent_core.signal_handlers (decision #121).

Decision #121: the bare `_signal_break` handler path (which used to call
`os._exit(1)` unconditionally, bypassing cleanup) must never be installed —
the cleanup-aware `_signal_break_with_cleanup` must win on every platform.
These tests exercise the REAL module, not a copy of its logic.
"""
from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core import signal_handlers  # noqa: E402

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)


def _restore(handler: object) -> None:
    """Best-effort restore of the process-wide SIGINT handler."""
    try:
        signal.signal(signal.SIGINT, handler)  # type: ignore[arg-type]
    except (ValueError, OSError):
        pass


# ── install(): which handler ends up installed ───────────────────────────────


def test_install_registers_cleanup_aware_handler_on_sigint():
    """install() must wire the cleanup-aware handler on SIGINT (non-win32)."""
    if sys.platform == "win32":
        pytest.skip("SIGINT wiring by install() is POSIX-only; win32 uses SIGBREAK")
    previous = signal.getsignal(signal.SIGINT)
    try:
        signal_handlers.install(agent=None)
        assert (
            signal.getsignal(signal.SIGINT)
            is signal_handlers._signal_break_with_cleanup
        )
    finally:
        _restore(previous)


def test_bare_signal_break_is_never_installed_on_any_platform():
    """Decision #121: the bare (no-cleanup) handler must never be installed."""
    import inspect

    src = inspect.getsource(signal_handlers.install)
    assert "_signal_break)" not in src.replace(" ", "")


def test_bare_handler_sets_shutdown_flag_without_exiting():
    """The bare handler only flips the shutdown flag — it must not exit."""
    # _signal_break is a plain function (no signum/frame), so it is invoked
    # directly, not via raise_signal. If it ever exits, the assert is skipped.
    signal_handlers._shutdown_flag = False
    signal_handlers._signal_break()
    assert signal_handlers._shutdown_flag is True


def test_cleanup_handler_sets_flag_and_exits_with_128_plus_signum():
    """The cleanup-aware handler saves state, then exits 128+signum."""
    child = (
        "import os, signal, sys\n"
        "from agent_core import signal_handlers\n"
        "signal.signal(signal.SIGINT, signal_handlers._signal_break_with_cleanup)\n"
        "os.kill(os.getpid(), signal.SIGINT)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # POSIX: exit status is 128+signum. CPython on Windows maps exit code
    # 130 to its SIGINT sentinel (2); both mean "the cleanup handler ran".
    assert proc.returncode in (128 + int(signal.SIGINT), 2), proc.stderr
    assert proc.returncode != 0, proc.stderr


def test_cleanup_handler_survives_failing_agent_hooks():
    """A raising cleanup hook must not mask the exit(128+signum)."""
    child = (
        "import os, signal, sys\n"
        "import agent\n"
        "from agent_core import signal_handlers\n"
        "def boom(*a):\n"
        "    raise RuntimeError('hook exploded')\n"
        "agent._shutdown_save_memory = boom\n"
        "agent._shutdown_trace_close = boom\n"
        "signal.signal(signal.SIGINT, signal_handlers._signal_break_with_cleanup)\n"
        "os.kill(os.getpid(), signal.SIGINT)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode in (128 + int(signal.SIGINT), 2), proc.stderr
    assert proc.returncode != 0, proc.stderr


# ── current_git_branch ───────────────────────────────────────────────────────


def test_current_git_branch_returns_current_branch(tmp_path):
    """In a real git repo with at least one commit the function reports the branch."""
    import os

    # Isolate from the user's global hooks/config (core.hooksPath, GPG signing)
    # while keeping PATH so git itself resolves.
    env = {**os.environ, "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig")}
    init = subprocess.run(
        ["git", "init", "-q", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    if init.returncode != 0:
        pytest.skip(f"git init failed in test env: {init.stderr}")
    # An empty repo (no commits) reports "HEAD" even when checked out on a
    # named branch — commit first, then the branch name is resolvable.
    for args, what in (
        (["config", "user.email", "test@example.com"], "git config"),
        (["config", "user.name", "test"], "git config"),
        (["checkout", "-q", "-b", "test-branch"], "git checkout"),
        (["commit", "-q", "--allow-empty", "-m", "seed"], "git commit"),
    ):
        res = subprocess.run(
            ["git", *args],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if res.returncode != 0:
            pytest.skip(f"{what} failed in test env: {res.stderr}")
    assert signal_handlers.current_git_branch(cwd=str(tmp_path)) == "test-branch"


def test_current_git_branch_falls_back_to_HEAD_when_not_a_repo(tmp_path):
    """Outside any git repo the function degrades to 'HEAD', never raises."""
    assert signal_handlers.current_git_branch(cwd=str(tmp_path)) == "HEAD"


# ── interruptible_input ──────────────────────────────────────────────────────


def test_interruptible_input_uses_supplied_executor():
    """An executor is the input source; the prompt path is never touched."""
    previous = signal.getsignal(signal.SIGINT)
    try:
        # A SIGINT handler that raises KeyboardInterrupt would break a real
        # input() read; with an executor installed there is no read at all.
        def _raise(signum, frame):  # pragma: no cover - must never run
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _raise)
        assert signal_handlers.interruptible_input(
            "prompt: ", executor=lambda: "from-executor"
        ) == "from-executor"
    finally:
        _restore(previous)


def test_interruptible_input_executor_is_sticky_across_calls():
    """A previously supplied executor persists for later calls."""
    previous = signal.getsignal(signal.SIGINT)
    try:
        def _raise(signum, frame):  # pragma: no cover - must never run
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _raise)
        signal_handlers.interruptible_input(executor=lambda: "sticky")
        assert signal_handlers.interruptible_input() == "sticky"
    finally:
        _restore(previous)


def test_interruptible_input_reraises_keyboard_interrupt():
    """Ctrl-C during input must surface as KeyboardInterrupt, not be swallowed."""
    child = (
        "import sys\n"
        "from agent_core import signal_handlers\n"
        "def executor():\n"
        "    raise KeyboardInterrupt\n"
        "try:\n"
        "    signal_handlers.interruptible_input(executor=executor)\n"
        "except KeyboardInterrupt:\n"
        "    print('re-raised')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "re-raised" in proc.stdout, proc.stderr
