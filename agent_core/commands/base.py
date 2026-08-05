"""Command base class and registry for agent interactive mode."""
import difflib
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent


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
        fromfile=f"a/{basename}",
        tofile=f"b/{basename}",
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
            m = re.search(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", stripped)
            if m:
                old_lineno = int(m.group(1)) - 1
                new_lineno = int(m.group(2)) - 1
            rows.append(("__HEADER__", stripped, "", ""))
        elif stripped.startswith("-"):
            old_lineno += 1
            pending_removes.append((old_lineno, stripped[1:]))
        elif stripped.startswith("+"):
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
    max_old_no = max((len(r[0]) for r in data_rows if r[0]), default=4)
    max_old_no = max(max_old_no, 4)
    max_new_no = max((len(r[2]) for r in data_rows if r[2]), default=4)
    max_new_no = max(max_new_no, 4)
    max_old_text = max((len(r[1]) for r in data_rows if r[1]), default=10)
    max_new_text = max((len(r[3]) for r in data_rows if r[3]), default=10)

    # Print
    print(f"\n  [PATCH: {basename}]")
    for old_no, old_text, new_no, new_text in rows:
        if old_no == "__HEADER__":
            print(f"  {old_text}")
            continue
        if old_no and new_no:
            if old_text == new_text:
                # Context line
                print(f"  {old_no:>{max_old_no}} | {old_text:<{max_old_text}}  {new_no:>{max_new_no}} | {new_text}")
            else:
                # Paired remove+add
                print(f"  {old_no:>{max_old_no}} | {RED}{old_text:<{max_old_text}}{RESET}  {new_no:>{max_new_no}} | {GREEN}{new_text}{RESET}")
                removed += 1
                added += 1
        elif old_no:
            # Removed only
            if old_text:
                print(f"  {old_no:>{max_old_no}} | {RED}{old_text:<{max_old_text}}{RESET}  {'':>{max_new_no}} |")
            else:
                # Removed blank line
                print(f"  {old_no:>{max_old_no}} |{RED} {'':<{max_old_text-1}}{RESET}  {'':>{max_new_no}} |")
            removed += 1
        elif new_no:
            # Added only
            if new_text:
                print(f"  {'':>{max_old_no}} | {'':<{max_old_text}}  {new_no:>{max_new_no}} | {GREEN}{new_text}{RESET}")
            else:
                # Added blank line
                print(f"  {'':>{max_old_no}} | {'':<{max_old_text}}  {new_no:>{max_new_no}} |{GREEN} {'':<{max_new_text-1}}{RESET}")
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
