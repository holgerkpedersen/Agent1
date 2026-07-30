"""Optimize command — static analysis + LLM suggestions for performance/memory/quality.

Usage:
    optimize <file>              Scan for issues, print suggestions (no changes)
    optimize <file> --apply      Scan, ask y/N per file before applying
    optimize <file> --yes        Scan, apply all suggestions without asking
    optimize <dir>               Scan all .py files in directory
"""

import os
from pathlib import Path

from .base import Command, read_stdin
from agent_core import workspace_path
from agent_core.patterns import analyze as static_analyze

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class OptimizeCommand(Command):
    """Find and optionally apply performance/memory/quality optimizations."""

    @property
    def name(self) -> str:
        return "optimize"

    @property
    def help_text(self) -> str:
        return "optimize <file|dir> [--apply] [--yes] — Find and apply optimizations"

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        parts = list(args)
        apply_mode = "--apply" in parts
        yes_mode = "--yes" in parts
        stdin_mode = "--stdin" in parts

        parts = [p for p in parts if p not in ("--apply", "--yes", "--stdin")]

        targets: list[str] = []

        if stdin_mode:
            content = read_stdin("Paste code to analyze. Type --- on its own line when done:")
            if not content.strip():
                self.error("No code provided.")
                return True
            targets = ["<stdin>"]
        elif not parts:
            self.error("Usage: optimize <file|dir> [--apply] [--yes] [--stdin]")
            return True
        else:
            ws = workspace_path(agent.workspace)
            for arg in parts:
                full = os.path.join(ws, arg) if not os.path.isabs(arg) else arg
                if os.path.isdir(full):
                    for root, dirs, files in os.walk(full):
                        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
                        for f in sorted(files):
                            if f.endswith(".py"):
                                targets.append(os.path.normpath(os.path.join(root, f)))
                elif os.path.isfile(full) and full.endswith(".py"):
                    targets.append(os.path.normpath(full))
                else:
                    print(f"  Skipping {arg} (not a .py file or directory)")

        if not targets:
            self.error("No .py files found to analyze.")
            return True

        all_findings: list[dict] = []
        file_contents: dict[str, str] = {}

        for fpath in targets:
            if stdin_mode:
                content_val = content
            else:
                try:
                    content_val = Path(fpath).read_text(encoding="utf-8")
                except Exception:
                    continue
            file_contents[fpath] = content_val
            findings = static_analyze(content_val)
            if findings:
                all_findings.extend({"file": fpath, **f} for f in findings)

        if not all_findings:
            print(f"  Scanned {len(targets)} file(s) — nothing to optimize.")
            return True

        print(f"\n  Static analysis found {len(all_findings)} issue(s) in {len(file_contents)} file(s):\n")
        by_file: dict[str, list[dict]] = {}
        for f in all_findings:
            by_file.setdefault(f["file"], []).append(f)

        for fpath, findings in sorted(by_file.items()):
            rel = os.path.relpath(fpath, os.getcwd()) if not stdin_mode else fpath
            print(f"  {rel}:")
            for f in findings:
                print(f"    line {f['line']:>4}: [{f['pattern']}] {f['suggestion']}")
            print()

        # Phase 2 — LLM deep analysis
        print(f"  Sending to LLM for deeper analysis...")
        context = "\n\n".join(
            f"## {os.path.basename(fp)}\n```python\n{content_val}\n```"
            for fp, content_val in file_contents.items()
        )
        static_findings = "\n".join(
            f"- {f['file']}:{f['line']} [{f['pattern']}] {f['suggestion']}"
            for f in all_findings
        )
        llm_response = await agent.llm.chat([
            {"role": "system", "content": (
                "You are an expert Python optimizer. The static analyzer found these issues:\n\n"
                f"{static_findings}\n\n"
                "Confirm, refine, or add optimizations. Focus on: speed (CPU/loops/caching), "
                "memory (large allocations, leaks), dead code (unused/unreachable). "
                "Be concise. One suggestion per line. Format:\n"
                "[OPTIMIZE: file.py:10-15] Category: description → suggested fix"
            )},
            {"role": "user", "content": f"## Code\n\n{context}\n\nReview for additional optimizations beyond the static findings above."},
        ])

        if llm_response.startswith("[Error") or llm_response.startswith("[LM Studio"):
            print(f"  LLM error: {llm_response[:200]}")
            return True

        print(f"  LLM analysis:")
        for line in llm_response.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "```")):
                print(f"    {stripped}")

        if not apply_mode or stdin_mode:
            return True

        # Phase 3 — Apply (with confirmation)
        for fpath, content_val in file_contents.items():
            print(f"\n  Requesting fix for {os.path.basename(fpath)}...")
            fix_msg = [
                {"role": "system", "content": (
                    "Apply ALL the optimizations discussed. Output the complete corrected file.\n"
                    "Output as: [FILE: filename.py]\n```python\n# complete fixed code\n```\n"
                    "No explanations. No duplicate functions."
                )},
                {"role": "user", "content": (
                    f"Apply these optimizations to the file:\n\n"
                    f"## Static findings:\n{static_findings}\n\n"
                    f"## LLM analysis:\n{llm_response}\n\n"
                    f"## Current code:\n```python\n{content_val}\n```"
                )},
            ]
            fixed = await agent.llm.chat(fix_msg)
            if fixed.startswith("[Error") or fixed.startswith("[LM Studio"):
                continue

            import re as _re
            match = _re.search(r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```', fixed, _re.DOTALL)
            if not match:
                print(f"  Could not parse fix for {os.path.basename(fpath)}")
                continue

            new_code = match.group(2).strip()
            if not new_code or "import" not in new_code:
                print(f"  Skipping {os.path.basename(fpath)} — invalid fix")
                continue

            if not yes_mode:
                print(f"  Apply optimized version of {os.path.basename(fpath)}? ({len(content_val)} → {len(new_code)} bytes) (y/N): ", end="")
                try:
                    if input().strip().lower() != "y":
                        print(f"  Skipped.")
                        continue
                except (EOFError, KeyboardInterrupt):
                    print("  Cancelled.")
                    return True

            with open(fpath, "w", encoding="utf-8") as f:
                f.write(new_code)
            print(f"  Optimized: {os.path.basename(fpath)} ({len(new_code)} bytes)")

        return True
