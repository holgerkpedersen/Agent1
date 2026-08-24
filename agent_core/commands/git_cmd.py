"""Git command for agent interactive mode."""
import subprocess

from .base import Command

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


#: Hard cap so a chatty git output (e.g. `git log -p`) can't flood the REPL.
_MAX_OUTPUT_CHARS = 4096


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} more chars]"


class GitCommand(Command):
    """Run a git command in the workspace."""

    @property
    def name(self) -> str:
        return "git"

    @property
    def help_text(self) -> str:
        return "git <args>  - Run a git command in the workspace (e.g. `git status`, `git diff --stat`)"

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        if not args or not any(a.strip() for a in args):
            print("Usage: git <args>  (e.g. `git status`, `git log --oneline -10`)")
            return True

        try:
            r = subprocess.run(
                ["git", *args],
                capture_output=True, text=True, cwd=agent.workspace, timeout=60,
            )
        except FileNotFoundError:
            print("Error: git executable not found on PATH.")
            return False
        except subprocess.TimeoutExpired:
            print(f"Error: `git {' '.join(args)}` timed out after 60s.")
            return False

        if r.stdout.strip():
            print(_truncate(r.stdout.rstrip()))
        if r.stderr.strip():
            print(_truncate(r.stderr.rstrip()))
        if not r.stdout.strip() and not r.stderr.strip():
            print(f"(no output, exit code {r.returncode})")
        return r.returncode == 0
