"""Command base class and registry for agent interactive mode."""
import asyncio
import difflib
import os
import re
import shutil
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent import Agent

_DIFF_HEADER_RE = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_RE_LINE_STARTS_WITH_MINUS = re.compile(r"^-")
_RE_LINE_STARTS_WITH_PLUS = re.compile(r"^\+")


def _diff_terminal_width() -> int:
    """Terminal width to cap the side-by-side diff viewer at.

    Uses ``shutil.get_terminal_size`` when available; falls back to 120 when
    not available or implausibly narrow, so non-tty/piped output (including
    tests) stays readable and deterministic.
    """
    try:
        width = shutil.get_terminal_size().columns
    except (OSError, ValueError):
        width = 0
    return width if width >= 60 else 120


def _wrap_line(text: str, width: int) -> list[str]:
    """Wrap a single line to ``width`` columns.

    Prefers word boundaries; hard-slices any fragment that is still too long
    (common for long single-token code lines).  An empty ``width`` is treated
    as unbounded and returns the line unchanged.  Leading whitespace is
    preserved so indentation remains visible in diff output.
    """
    if width <= 0 or len(text) <= width:
        return [text]
    leading = ""
    i = 0
    while i < len(text) and text[i] in (" ", "\t"):
        leading += text[i]
        i += 1
    rest = text[i:]
    if not rest:
        return [text]
    words = rest.split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur = " ".join([cur, w])
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    lines[0] = leading + lines[0]
    # Hard-slice any fragment still exceeding the width (long tokens).
    out: list[str] = []
    for ln in lines:
        while len(ln) > width:
            out.append(ln[:width])
            ln = ln[width:]
        out.append(ln)
    return out


def read_stdin(prompt: str = "Paste text. Type --- on its own line when done, or Ctrl+Z to finish:") -> str:
    """Read multi-line text from stdin until a sentinel line or EOF.

    Blank lines are preserved.  The sentinel line ``---`` (three dashes,
    alone on a line) terminates input.  Ctrl+Z / Ctrl+D also works.
    """
    print(prompt)
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "---":
            break
        lines.append(line)
    return "\n".join(lines)


# ── Flow-stop control ────────────────────────────────────────────────────
#
# A run (fix / optimize / implement / decide / workflow) may process many
# files in a loop.  Every interactive confirm answers through read_choice(),
# and the loops check stop_requested() so a single
# "stop/quit/abort" answer or Ctrl+C winds the whole run down instead of
# merely declining the current item.

_FLOW_STOP = False


class FlowStopped(BaseException):
    """Raised when the user interrupts an in-flight LLM request.

    Deliberately a BaseException: LLM calls sit inside ``except Exception``
    blocks that must NOT swallow it — the interrupt has to unwind the current
    command so the REPL can prompt again.
    """


def request_stop() -> None:
    """Ask every run loop in the current command to stop."""
    global _FLOW_STOP
    _FLOW_STOP = True


def stop_requested() -> bool:
    """True once the user asked (or Ctrl+C'd) the current run to stop."""
    return _FLOW_STOP


def clear_stop() -> None:
    """Reset the stop flag — called before each new command starts."""
    global _FLOW_STOP
    _FLOW_STOP = False


def read_input(prompt: str = "") -> str:
    """Read one line from stdin.

    EOFError returns "" without requesting a stop; KeyboardInterrupt requests
    a flow stop and returns "" so callers wind down gracefully.
    """
    try:
        return input(prompt)
    except EOFError:
        return ""
    except KeyboardInterrupt:
        request_stop()
        print("\n  Stopping the flow — no further changes will be applied.")
        return ""


_STOP_TOKENS = frozenset({"s", "stop", "q", "quit", "abort", "x"})


def read_choice(prompt: str) -> bool:
    """y/N confirm.  Returns True only for y/yes.

    The stop tokens (s/stop/q/quit/abort/x) and Ctrl+C request a flow stop
    (stop_requested()) and decline the current item; any other answer merely
    declines it.
    """
    resp = read_input(prompt).strip().lower()
    if resp in _STOP_TOKENS:
        request_stop()
        print("  Stopping the flow — no further changes will be applied.")
        return False
    return resp in ("y", "yes")


def chat_stoppable(chat: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap an async chat callable so Ctrl+C mid-request turns into a
    graceful FlowStopped + request_stop() instead of tearing down the run.

    Because the REPL runs under asyncio.run(), a KeyboardInterrupt during an
    ``await`` surfaces as a CancelledError thrown at that await point; the
    wrapper converts it back into an orderly stop.
    """
    async def wrapped(msgs: Any, **kw: Any) -> Any:
        task = asyncio.current_task()
        try:
            return await chat(msgs, **kw)
        except asyncio.CancelledError:
            if task is not None and hasattr(task, "uncancel"):
                task.uncancel()
            request_stop()
            print("\n  Stopped by user (Ctrl+C) — the current run was aborted.")
            raise FlowStopped() from None
    return wrapped


# ---------------------------------------------------------------------------
# Autonomous mode
# ---------------------------------------------------------------------------
#: Explicit override set by a per-command ``--auto`` flag; None = follow the
#: AGENT_AUTONOMOUS env var (read at call time, so runtime changes apply).
_AUTONOMOUS_OVERRIDE: bool | None = None


def is_autonomous() -> bool:
    """True when interactive prompts should auto-select their safe defaults."""
    if _AUTONOMOUS_OVERRIDE is not None:
        return _AUTONOMOUS_OVERRIDE
    return os.environ.get("AGENT_AUTONOMOUS", "").strip().lower() in ("1", "true", "yes")


def set_autonomous(flag: "bool | None") -> None:
    """Enable/disable autonomous mode for the current invocation (``--auto``).

    An explicit flag overrides the environment variable; pass ``None`` to
    clear the override and follow the env var again.
    """
    global _AUTONOMOUS_OVERRIDE
    _AUTONOMOUS_OVERRIDE = flag


def auto_choice(prompt: str, default: str, auto_default: str | None = None) -> str:
    """Read a choice from the user — or pick the safe default autonomously.

    Interactive mode: behaves exactly like ``read_input`` (EOF → ``""``,
    Ctrl+C → flow stop + ``""``).
    Autonomous mode: returns ``auto_default or default`` WITHOUT prompting and
    prints the choice, so runs can proceed headless.  ``auto_default`` must
    always be the SAFE option (decline/halt) for safety gates — autonomous
    mode never auto-approves.
    """
    if is_autonomous():
        choice = auto_default if auto_default is not None else default
        print(f"  {prompt.strip()} (auto: {choice})")
        return choice
    return read_input(prompt)


def show_file_diff(basename: str, original: str, new: str) -> None:
    """Display a side-by-side diff with aligned columns, line numbers, 5 lines of context, and colors."""
    original_lines = original.splitlines()
    new_lines = new.splitlines()
    diff = list(difflib.unified_diff(
        original_lines,
        new_lines,
        n=5,
    ))
    if not diff:
        print(f"\n  [PATCH: {basename}] (no changes)")
        return

    RED = "\033[41m"
    GREEN = "\033[42m"
    RESET = "\033[0m"
    old_lineno = 0
    new_lineno = 0
    added = removed = 0

    # Collect all rows first
    rows: list[tuple[str, str, str, str]] = []  # (old_no, old_text, new_no, new_text)
    pending_removes: list[tuple[int, str]] = []
    pending_adds: list[tuple[int, str]] = []

    def flush_pair() -> None:
        while pending_removes or pending_adds:
            old_no, old_text = pending_removes.pop(0) if pending_removes else (None, None)
            new_no, new_text = pending_adds.pop(0) if pending_adds else (None, None)
            rows.append((
                str(old_no) if old_no is not None else "",
                old_text if old_text is not None else "",
                str(new_no) if new_no is not None else "",
                new_text if new_text is not None else "",
            ))

    for line in diff:
        stripped = line.rstrip("\n")
        if stripped.startswith("+++") or stripped.startswith("---"):
            continue
        if stripped.startswith("@@"):
            flush_pair()
            m = _DIFF_HEADER_RE.search(stripped)
            if m:
                old_lineno = int(m.group(1)) - 1
                new_lineno = int(m.group(2)) - 1
            rows.append(("__HEADER__", stripped, "", ""))
        elif _RE_LINE_STARTS_WITH_MINUS.match(stripped):
            old_lineno += 1
            pending_removes.append((old_lineno, stripped[1:]))
        elif _RE_LINE_STARTS_WITH_PLUS.match(stripped):
            new_lineno += 1
            pending_adds.append((new_lineno, stripped[1:]))
        else:
            flush_pair()
            old_lineno += 1
            new_lineno += 1
            content = stripped[1:] if len(stripped) > 1 else ""
            rows.append((str(old_lineno), content, str(new_lineno), content))
    flush_pair()

    # Calculate column widths (exclude headers)
    data_rows = [r for r in rows if r[0] != "__HEADER__"]
    max_old_no = max(max((len(r[0]) for r in data_rows if r[0]), default=4), 4)
    max_new_no = max(max((len(r[2]) for r in data_rows if r[2]), default=4), 4)
    max_old_text = max((len(r[1]) for r in data_rows if r[1]), default=10)
    max_new_text = max((len(r[3]) for r in data_rows if r[3]), default=10)

    # Cap total width so long lines wrap inside the side-by-side view.
    # Row layout: "  {old_no} | {old_text}  {new_no} | {new_text}"
    # Fixed overhead: 2 indent + " | " (3) + 2 gap + " | " (3) = 10, plus
    # the two line-number columns.
    max_width = _diff_terminal_width()
    overhead = 10 + max_old_no + max_new_no
    avail = max(20, max_width - overhead)
    old_col = min(max_old_text, max(10, avail // 2))
    new_col = min(max_new_text, max(10, avail - old_col))

    old_blank_no = " " * max_old_no
    new_blank_no = " " * max_new_no

    # Print
    print(f"\n  [PATCH: {basename}]")
    for old_no, old_text, new_no, new_text in rows:
        if old_no == "__HEADER__":
            print(f"  {old_text}")
            continue

        has_old = bool(old_no)
        has_new = bool(new_no)

        if has_old and has_new:
            if old_text == new_text:
                # Context line
                old_parts = _wrap_line(old_text, old_col)
                new_parts = _wrap_line(new_text, new_col)
                for i in range(max(len(old_parts), len(new_parts))):
                    onum = f"{old_no:>{max_old_no}}" if i == 0 else old_blank_no
                    nnum = f"{new_no:>{max_new_no}}" if i == 0 else new_blank_no
                    o = old_parts[i] if i < len(old_parts) else ""
                    n = new_parts[i] if i < len(new_parts) else ""
                    print(f"  {onum} | {o:<{old_col}}  {nnum} | {n}")
            else:
                # Paired remove+add
                old_parts = _wrap_line(old_text, old_col)
                new_parts = _wrap_line(new_text, new_col)
                for i in range(max(len(old_parts), len(new_parts))):
                    onum = f"{old_no:>{max_old_no}}" if i == 0 else old_blank_no
                    nnum = f"{new_no:>{max_new_no}}" if i == 0 else new_blank_no
                    o = old_parts[i] if i < len(old_parts) else ""
                    n = new_parts[i] if i < len(new_parts) else ""
                    print(f"  {onum} | {RED}{o:<{old_col}}{RESET}  {nnum} | {GREEN}{n}{RESET}")
                removed += 1
                added += 1
        elif has_old:
            # Removed only
            old_parts = _wrap_line(old_text, old_col) if old_text else [""]
            for i, o in enumerate(old_parts):
                onum = f"{old_no:>{max_old_no}}" if i == 0 else old_blank_no
                print(f"  {onum} | {RED}{o:<{old_col}}{RESET}  {new_blank_no} |")
            removed += 1
        elif has_new:
            # Added only
            new_parts = _wrap_line(new_text, new_col) if new_text else [""]
            for i, n in enumerate(new_parts):
                nnum = f"{new_no:>{max_new_no}}" if i == 0 else new_blank_no
                print(f"  {old_blank_no} | {'':<{old_col}}  {nnum} | {GREEN}{n}{RESET}")
            added += 1
    print(f"  ({removed} lines removed, {added} lines added)")


def save_file_py(fpath: str, content: str, auto_yes: bool = True) -> bool:
    """Write ``content`` to ``fpath`` with a unified diff preview.

    With ``auto_yes=True`` the write happens immediately; otherwise a y/N
    confirmation is prompted after the diff.  Returns True only if the file
    was actually written (False on identical content or user declining).
    """
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            current = f.read()
    except FileNotFoundError:
        current = ""
    except (OSError, UnicodeDecodeError) as e:
        print(f"  Warning: could not read {fpath}: {e}")
        current = ""

    if current == content:
        return False

    show_file_diff(os.path.basename(fpath), current, content)
    if not auto_yes:
        if not read_choice(f"  Apply to {fpath}? [y/N] "):
            print(f"  Skipped: {fpath}")
            return False

    parent = os.path.dirname(fpath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    return True


class Command(ABC):
    """Abstract base class for interactive commands.
    
    All commands must implement execute() and provide a help text.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Command name used in REPL."""
        ...
    
    @property
    @abstractmethod
    def help_text(self) -> str:
        """Help text shown in commands list."""
        ...
    
    @abstractmethod
    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        """Execute the command.
        
        Args:
            args: Arguments after the command name
            agent: Agent instance for LLM/file operations
            
        Returns:
            True to continue REPL, False to exit
        """
        ...
    
    def error(self, msg: str) -> None:
        """Print error message."""
        print(f"Error: {msg}")
    
    def success(self, msg: str) -> None:
        """Print success message."""
        print(msg)
