"""Terminal line editing and command history for the interactive REPL.

Why this module exists
----------------------
The REPL used to read every line with the bare ``input()`` builtin.  Without
the ``readline`` module the terminal stays in canonical mode, so:

* the up arrow does NOT recall the previous command,
* the left/right arrows do NOT move the cursor,
* the raw escape bytes end up inside the command text - the agent then tries
  to run a command made of ``\\x1b[A``.

Importing ``readline`` is what fixes all three: the module hooks the
built-in ``input()`` so every reader in the process (the REPL, the command
prompts, the paste reader) gets line editing.

``readline`` is part of the standard library on POSIX.  On Windows it is
usually missing, so every entry point degrades gracefully and the REPL keeps
working with plain ``input()``.

History storage
---------------
The recall history lives in ``<workspace>/.agent_history`` - one command per
line.  It holds the user's own commands (which may mention file names or
tokens), so it is git-ignored and never committed.

This module lives at the ``agent_core`` root (not under ``agent_core.cli``)
because both the CLI layer and the command layer use it: the CLI layer starts
it, and the command layer suspends it while reading pasted text.  Keeping it
neutral avoids a ``cli`` <-> ``commands`` import cycle.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

#: File name (inside the workspace) that holds the recall history.
HISTORY_FILENAME = ".agent_history"

#: Upper bound for the in-memory history (oldest entries are dropped).
MAX_HISTORY_ENTRIES = 1000

# Module state: whether line editing is on, and where the history lives.
_state: dict[str, Any] = {"enabled": False, "workspace": None, "path": None}


def _readline() -> Any | None:
    """Return the ``readline`` module, or None when it is unavailable."""
    try:
        import readline
    except ImportError:
        return None
    return readline


def history_path(workspace: str | Path | None = None) -> Path | None:
    """Return the history file path for ``workspace`` (None when unknown)."""
    if workspace is None:
        workspace = _state.get("workspace")
    if not workspace:
        return None
    return Path(workspace) / HISTORY_FILENAME


def is_enabled() -> bool:
    """True when line editing is active in this process."""
    return bool(_state.get("enabled"))


def _history_items(readline: Any) -> list[str]:
    """Return the current history entries as a list of strings."""
    try:
        length = readline.get_current_history_length()
    except Exception:
        return []
    items: list[str] = []
    for index in range(length):
        try:
            item = readline.get_history_item(index + 1)
        except Exception:
            continue
        if item is None:
            continue
        items.append(item if isinstance(item, str) else str(item))
    return items


def _load_history(path: Path) -> None:
    """Add the persisted history entries that are not in memory yet."""
    readline = _readline()
    if readline is None:
        return
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    known = set(_history_items(readline))
    for line in raw.splitlines():
        if not line.strip() or line in known:
            continue
        known.add(line)
        try:
            readline.add_history(line)
        except Exception:
            pass


def enable(workspace: str | Path | None = None) -> bool:
    """Turn on terminal line editing and load the saved history.

    Returns True when line editing is active, False when ``readline`` is not
    available (the REPL then keeps working with plain ``input()``).
    """
    readline = _readline()
    if readline is None:
        return False

    _state["enabled"] = True
    if workspace is not None:
        _state["workspace"] = str(workspace)

    path = history_path(workspace)
    if path is not None:
        _state["path"] = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()
        except OSError:
            pass
        _load_history(path)

    try:
        readline.set_history_length(MAX_HISTORY_ENTRIES)
    except Exception:
        pass
    return True


def save_history(workspace: str | Path | None = None) -> bool:
    """Persist the recall history to ``<workspace>/.agent_history``.

    Only single-line entries are stored: a multi-line entry (a pasted block)
    would be written unescaped and read back as several separate commands.
    Returns True when the history was written.
    """
    readline = _readline()
    if readline is None:
        return False
    path = history_path(workspace)
    if path is None:
        return False
    entries = [
        item for item in _history_items(readline) if item.strip() and "\n" not in item
    ]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(entry + "\n" for entry in entries),
            encoding="utf-8",
        )
    except OSError:
        return False
    return True


@contextmanager
def suspend_history() -> Iterator[bool]:
    """Stop ``input()`` from recording lines while reading pasted text.

    The paste reader consumes one line per ``input()`` call, so a pasted
    block would otherwise fill the recall history with its lines.  Yields
    True when the suspension is in effect.
    """
    readline = _readline()
    if readline is None or not hasattr(readline, "set_auto_history"):
        yield False
        return
    try:
        readline.set_auto_history(False)
    except Exception:
        yield False
        return
    try:
        yield True
    finally:
        try:
            readline.set_auto_history(True)
        except Exception:
            pass
