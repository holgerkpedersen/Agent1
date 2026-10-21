"""Regression: Ctrl-Z (SIGTSTP) must not suspend the agent REPL on POSIX.

Before the fix, SIGTSTP kept its default action, so a Ctrl-Z (e.g. pasted
text containing ^Z) stopped the Python process and the shell dropped the
user out of the agent chat.
"""
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="SIGTSTP exists only on POSIX"
)

_CHILD = r"""
import os, signal, sys
from agent_core import signal_handlers
signal_handlers.install(agent=None)
os.kill(os.getpid(), signal.SIGTSTP)   # simulate Ctrl-Z
print("still-running", flush=True)
"""


def test_sigtstp_handler_installed_and_process_survives(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        timeout=20,
    )
    # A stopped process would never reach print(); a default-action SIGTSTP
    # would stop it (no returncode, the test would hang or time out).
    assert "still-running" in proc.stdout, proc.stderr
    assert proc.returncode == 0, proc.stderr


def test_install_sets_non_default_sigtstp_handler():
    import signal

    from agent_core import signal_handlers

    previous = signal.getsignal(signal.SIGTSTP)
    try:
        signal_handlers.install(agent=None)
        handler = signal.getsignal(signal.SIGTSTP)
        assert handler is not signal.SIG_DFL
        assert handler is signal_handlers._ignore_signal
    finally:
        signal.signal(signal.SIGTSTP, previous)
        signal.signal(signal.SIGINT, signal.default_int_handler)


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent
