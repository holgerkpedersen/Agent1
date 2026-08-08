"""Command base class and registry for agent interactive mode."""
import difflib
import re
import shutil
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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
    as unbounded and returns the line unchanged.
    """
    if width <= 0 or len(text) <= width:
        return [text]
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
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
    
    def error(self, msg: str):
        """Print error message."""
        print(f"Error: {msg}")
    
    def success(self, msg: str):
        """Print success message."""
        print(msg)