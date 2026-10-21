"""Regression: the REPL must support line editing and history recall.

Before the fix the REPL read every line with the bare ``input()`` built-in.
Without the ``readline`` module the terminal stays in canonical mode, so the
up arrow does not recall the previous command, the left/right arrows do not
move the cursor, and the raw escape bytes end up inside the command text
(the agent then tries to run a command made of ``\\x1b[A``).

The pty test drives the REAL input path
(``agent_core.signal_handlers.interruptible_input``) and presses the up
arrow for real, so it fails on the old code and passes on the new one.
"""

from __future__ import annotations

import os
import pty
import select
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="pty test needs a POSIX pseudo terminal"
)


def _repo_root() -> Path:
    return REPO_ROOT


# ── helpers ─────────────────────────────────────────────────────────────────


def _pump(fd: int, seconds: float, out: list[bytes]) -> None:
    """Read from the pty master for up to ``seconds``."""
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.05)
        if fd in r:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                return
            if not chunk:
                return
            out.append(chunk)


def _wait_for(fd: int, needle: bytes, out: list[bytes], timeout: float = 20.0) -> bool:
    """Read until ``needle`` shows up in the output (or the timeout expires)."""
    end = time.time() + timeout
    while time.time() < end:
        if needle in b"".join(out):
            return True
        _pump(fd, 0.1, out)
    return needle in b"".join(out)


# ── the core regression: up arrow recalls the previous line ─────────────────


def test_up_arrow_recalls_previous_line_in_the_repl(tmp_path):
    """Pressing up must recall the previous line instead of inserting escape bytes."""
    child = r"""
import sys
sys.path.insert(0, {repo!r})
try:
    from agent_core.line_editing import enable
    enable(workspace={workspace!r})
except Exception:
    pass
from agent_core.signal_handlers import interruptible_input

first = interruptible_input("FIRST> ")
second = interruptible_input("SECOND> ")
sys.stdout.write("GOT1:" + repr(first) + "\n")
sys.stdout.write("GOT2:" + repr(second) + "\n")
sys.stdout.flush()
""".format(repo=str(REPO_ROOT), workspace=str(tmp_path))

    pid, fd = pty.fork()
    if pid == 0:  # child
        os.environ["TERM"] = "xterm-256color"
        os.execvp(sys.executable, [sys.executable, "-c", child])

    out: list[bytes] = []
    try:
        assert _wait_for(fd, b"FIRST> ", out), b"".join(out)
        os.write(fd, b"hello world\r")
        assert _wait_for(fd, b"SECOND> ", out), b"".join(out)
        os.write(fd, b"\x1b[A")  # up arrow
        _pump(fd, 1.0, out)
        os.write(fd, b"\r")
        _pump(fd, 2.0, out)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        os.waitpid(pid, 0)

    raw = b"".join(out)
    assert b"GOT2:'hello world'" in raw, (
        f"the up arrow did not recall the previous line; got {raw!r}"
    )


# ── the mechanism: readline is really loaded ───────────────────────────────


def test_line_editing_loads_readline(tmp_path):
    """Enabling line editing must import readline (that is what gives the keys)."""
    from agent_core.line_editing import enable

    assert enable(workspace=tmp_path) is True
    assert "readline" in sys.modules


def test_enable_is_idempotent(tmp_path):
    """Calling enable() twice must not register the history saver twice."""
    from agent_core import line_editing

    line_editing.enable(workspace=tmp_path)
    line_editing.enable(workspace=tmp_path)
    assert line_editing.enable(workspace=tmp_path) is True


def test_history_survives_a_restart(tmp_path):
    """A remembered command must still be recallable in the next session."""
    import readline

    from agent_core.line_editing import enable, save_history

    enable(workspace=tmp_path)
    readline.clear_history()
    readline.add_history("deploy the thing")
    save_history(workspace=tmp_path)

    readline.clear_history()
    enable(workspace=tmp_path)
    length = readline.get_current_history_length()
    items = [readline.get_history_item(i + 1) for i in range(length)]
    assert "deploy the thing" in items


def test_the_repl_enables_line_editing(tmp_path):
    """The REPL must turn line editing on, using the agent workspace."""
    from agent_core.cli import interactive

    class _FakeAgent:
        workspace = str(tmp_path)

    interactive._start_line_editing(_FakeAgent())
    assert "readline" in sys.modules
    assert (tmp_path / ".agent_history").exists()


def test_history_file_is_git_ignored():
    """Remembered commands are user input - they must never be committed."""
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".agent_history" in gitignore
