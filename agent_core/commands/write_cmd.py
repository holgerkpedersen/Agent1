"""Write command for agent interactive mode.

Overwriting an existing file shows an interactive unified-diff review first:
each hunk is displayed and must be approved (y/N) before it is applied —
rejected hunks leave the file's original lines intact (plan feature item 31).
``--yes`` skips the review entirely.
"""
import difflib

from .base import Command, read_choice, stop_requested

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class WriteCommand(Command):
    """Write content to a file, with per-hunk diff review for existing files."""

    @property
    def name(self) -> str:
        return "write"

    @property
    def help_text(self) -> str:
        return "write <path> <content> [--yes] - Write content to file (per-hunk diff review for existing files)"

    def _review_hunks(self, old_text: str, new_text: str) -> str:
        """Interactive per-hunk approve/reject over the old->new diff.

        Returns the merged content (approved hunks only).  Stop tokens abort
        the whole write (returns the original content unchanged).
        """
        from agent_core.patch_utils import apply_patch, split_patch_hunks

        diff_text = "\n".join(
            difflib.unified_diff(old_text.splitlines(), new_text.splitlines(), lineterm="")
        )
        hunks = split_patch_hunks(diff_text)
        if not hunks:
            return new_text

        approved: list[tuple[int, list[tuple[str, str]]]] = []
        for start, chunks in hunks:
            if stop_requested():
                return old_text
            print(f"\n  {'─' * 50}")
            for op, text in chunks:
                print(f"  {op} {text}")
            print(f"  {'─' * 50}")
            if read_choice(f"  Apply hunk at line {start}? (y/N): "):
                approved.append((start, chunks))

        if stop_requested():
            print("  Stopping the flow — no further changes will be applied.")
            return old_text
        if not approved:
            print("  No hunks approved — file left unchanged.")
            return old_text

        patch_text = "\n".join(
            f"@@ -{start} @@\n" + "\n".join(op + text for op, text in chunks)
            for start, chunks in approved
        )
        ok, patched = apply_patch(patch_text, old_text.splitlines())
        if not ok:
            print(f"  Patch failed to apply: {str(patched)[:200]}")
            return old_text
        return str(patched)

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        yes_mode = "--yes" in args or "--force" in args
        filtered = [a for a in args if a not in ("--yes", "--force")]
        if len(filtered) < 2:
            self.error("Usage: write <path> <content> [--yes]")
            return True

        path = filtered[0]
        content = filtered[1]

        if not yes_mode:
            try:
                existing = await agent.read_file(path, track_read=False)
            except Exception:
                existing = ""
            if not existing.startswith(("File not found", "Error")):
                if existing != content:
                    content = self._review_hunks(existing, content)

        result = await agent.write_file(path, content)
        print(result)
        return True
