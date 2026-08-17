"""Self-healing command (plan feature item 29).

``self_heal`` proposes an LLM patch for the failing test(s) in the target
directory, runs the test suite, and AUTO-REVERTS the previous round when a
patch does not reduce the failure count, chaining fix attempts until the
suite is green or the round budget is exhausted.

Invocation::

    self_heal [<path>] [--rounds N] [--yes]

- Default target is the workspace root (``pytest``).
- ``--rounds`` caps the number of patch-and-test cycles (default 3).
- ``--yes`` applies patches without per-file confirmation.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .base import Command, read_choice, stop_requested

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent

_PATCH_RE = re.compile(
    r"\[PATCH:\s*([^\]]+)\]\s*\n*(?:```|)(.*?)(?:```|$)", re.DOTALL
)
_FAILED_RE = re.compile(r"FAILED\s+([\w./\\-]+)::")


class SelfHealCommand(Command):
    """Propose a patch, run the test suite, auto-revert or chain fixes until green."""

    @property
    def name(self) -> str:
        return "self_heal"

    @property
    def help_text(self) -> str:
        return "self_heal [path] [--rounds N] [--yes] - patch failing tests and re-run until green"

    @staticmethod
    def _run_pytest(target: str) -> tuple[int, str]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", target, "-q", "--tb=line"],
                capture_output=True,
                text=True,
                timeout=900,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return -1, "pytest timed out (15 min limit)"
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode, output

    @staticmethod
    def _failing_files(output: str) -> list[str]:
        """Derive failing test FILES from the pytest summary."""
        files: list[str] = []
        for match in _FAILED_RE.finditer(output):
            node = match.group(1).replace("\\", "/")
            if node not in files:
                files.append(node)
        return files

    async def _fix_file(self, agent: 'Agent', path: str, failure_info: str) -> bool:
        """Ask the LLM for a [PATCH:] fixing *path* and apply it (with backup)."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except OSError as exc:
            print(f"    cannot read {path}: {exc}")
            return False

        basename = os.path.basename(path)
        messages = [
            {
                "role": "system",
                "content": (
                    "You fix failing Python tests. Output EXACTLY one "
                    f"[PATCH: {basename}] block with a unified-diff hunk that "
                    "repairs the failing code. Never rewrite whole files; never "
                    "disable or delete the failing test. If the failure is in "
                    "production code, fix the production code."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"File: {path}\n\nFailure:\n{failure_info[:2000]}\n\n"
                    f"Current source:\n```\n{source}\n```\n"
                ),
            },
        ]
        try:
            response = await agent.llm.chat(messages, disable_thinking=True)
        except Exception as exc:
            print(f"    LLM call failed: {exc}")
            return False
        if response.startswith("[Error") or response.startswith("[LM Studio"):
            print(f"    LLM error: {response[:200]}")
            return False

        match = _PATCH_RE.search(response)
        if not match:
            print("    no [PATCH:] block in LLM response")
            return False
        patch_text = match.group(2).strip()
        if not patch_text:
            print("    empty patch")
            return False

        original_lines = source.splitlines()
        from agent_core.patch_utils import apply_patch, apply_anchored_patch
        ok, patched = apply_patch(patch_text, original_lines)
        if not ok:
            ok, patched = apply_anchored_patch(patch_text, original_lines)
        if not ok:
            print(f"    patch could not be applied: {str(patched)[:200]}")
            return False
        new_source = str(patched)

        try:
            compile(new_source, basename, "exec")
        except SyntaxError as exc:
            print(f"    patched {basename} does not compile: {exc}")
            return False

        if self.backups is None:
            self.backups = {}
        self.backups[path] = (path + ".heal_bak", source)
        try:
            shutil.copyfile(path, path + ".heal_bak")
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_source)
            print(f"    applied LLM patch to {basename}")
            return True
        except OSError as exc:
            print(f"    write failed: {exc}")
            return False

    def _revert_all(self) -> None:
        if not self.backups:
            return
        for path, (backup, _orig) in self.backups.items():
            if os.path.exists(backup):
                shutil.copyfile(backup, path)
                print(f"    reverted {path}")
        self.backups = {}

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        rounds = 3
        yes_mode = False
        target = "."
        i = 0
        while i < len(args):
            if args[i] == "--rounds" and i + 1 < len(args):
                try:
                    rounds = max(1, int(args[i + 1]))
                except ValueError:
                    pass
                i += 2
                continue
            if args[i] in ("--yes", "--force"):
                yes_mode = True
                i += 1
                continue
            target = args[i]
            i += 1

        print(f"[self_heal] Running pytest on {target or '.'} ...")
        rc, output = self._run_pytest(target)
        if rc == 0:
            print("[self_heal] Suite already green — nothing to heal.")
            return True

        self.backups: dict[str, tuple[str, str]] | None = {}
        baseline_failures = self._failing_files(output)
        print(f"[self_heal] {len(baseline_failures)} failing file(s): {', '.join(baseline_failures)}")

        for round_no in range(1, rounds + 1):
            if stop_requested():
                break
            print(f"\n[self_heal] Round {round_no}/{rounds}")
            failures = self._failing_files(output)
            if not failures:
                break

            # Group failure info per file for targeted prompts.
            by_file: dict[str, list[str]] = {}
            for line in output.splitlines():
                m = _FAILED_RE.search(line)
                if m:
                    by_file.setdefault(m.group(1).replace("\\", "/"), []).append(line[200:])
            for path in failures:
                file_failures = by_file.get(path, [])
                failure_info = "\n".join(file_failures[:5]) or output[-2000:]
                if yes_mode or read_choice(f"  Heal {path}? (y/N): "):
                    await self._fix_file(agent, path, failure_info or output[-2000:])

            print("[self_heal] Re-running pytest after patches ...")
            new_rc, output = self._run_pytest(target)
            if new_rc == 0:
                print("[self_heal] Suite GREEN after healing.")
                return True
            new_failures = self._failing_files(output)
            print(f"[self_heal] Still {len(new_failures)} failing file(s).")
            if len(new_failures) >= len(failures):
                print("[self_heal] No improvement — reverting this round's patches.")
                self._revert_all()
                break

        print(f"\n[self_heal] Not fully green after {rounds} round(s). "
              f"Remaining failing files: {', '.join(self._failing_files(output))}")
        return True
