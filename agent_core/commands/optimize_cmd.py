"""Optimize command — batched static analysis + LLM suggestions.

Usage:
    optimize <file>              Scan for issues, print suggestions (no changes)
    optimize <file> --list       Quick list of files with issues (no LLM)
    optimize <file> --apply      Scan, ask y/N per file before applying
    optimize <file> --yes        Scan, apply all suggestions without asking
    optimize <dir>               Scan all .py files in directory

Batching:
    Files are grouped into batches to stay within LLM token limits.
    Each batch is processed independently with its own context window.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Command, read_stdin, show_file_diff
from agent_core import workspace_path
from agent_core.patterns import analyze as static_analyze

if TYPE_CHECKING:
    from agent import Agent

# Token estimation: ~4 chars per token, with overhead for message framing
CHARS_PER_TOKEN = 4
SYSTEM_OVERHEAD_TOKENS = 200  # System prompt, formatting, etc.
MAX_BATCH_TOKENS = 25000     # Leave room for output within 32k limit
SAFETY_MARGIN = 0.8          # Use 80% of budget to be safe


def estimate_tokens(text: str) -> int:
    """Rough token estimation based on character count."""
    return len(text) // CHARS_PER_TOKEN


def create_batches(
    file_contents: dict[str, str],
    findings_by_file: dict[str, list[dict]],
    max_tokens: int = MAX_BATCH_TOKENS,
) -> list[dict]:
    """Group files into batches that fit within token budget.

    Returns list of dicts with keys: files, contents, findings, total_tokens
    """
    batches: list[dict] = []
    current_batch: dict = {
        "files": [],
        "contents": {},
        "findings": {},
        "total_tokens": SYSTEM_OVERHEAD_TOKENS,
    }

    # Sort files by size (smallest first) for better packing
    sorted_files = sorted(file_contents.keys(), key=lambda f: len(file_contents[f]))

    for fpath in sorted_files:
        content = file_contents[fpath]
        file_tokens = estimate_tokens(content)

        # Add findings text to estimate
        file_findings = findings_by_file.get(fpath, [])
        findings_text = "\n".join(
            f"  line {f['line']}: [{f['pattern']}] {f['suggestion']}"
            for f in file_findings
        )
        findings_tokens = estimate_tokens(findings_text)
        item_tokens = file_tokens + findings_tokens + 50  # 50 for formatting

        # Check if adding this file would exceed budget
        if (current_batch["total_tokens"] + item_tokens) * SAFETY_MARGIN > max_tokens:
            if current_batch["files"]:  # Don't create empty batches
                batches.append(current_batch)
                current_batch = {
                    "files": [],
                    "contents": {},
                    "findings": {},
                    "total_tokens": SYSTEM_OVERHEAD_TOKENS,
                }

        current_batch["files"].append(fpath)
        current_batch["contents"][fpath] = content
        current_batch["findings"][fpath] = file_findings
        current_batch["total_tokens"] += item_tokens

    # Add final batch if non-empty
    if current_batch["files"]:
        batches.append(current_batch)

    return batches


def format_batch_context(batch: dict) -> str:
    """Format batch contents and findings for LLM prompt."""
    parts = []
    for fpath in batch["files"]:
        basename = os.path.basename(fpath)
        content = batch["contents"][fpath]
        findings = batch["findings"][fpath]

        findings_text = "\n".join(
            f"    line {f['line']}: [{f['pattern']}] {f['suggestion']}"
            for f in findings
        ) if findings else "    (none)"

        parts.append(f"## {basename}\nFindings:\n{findings_text}\n```python\n{content}\n```")

    return "\n\n".join(parts)


def parse_llm_fixes(response: str, batch_files: list[str]) -> dict[str, str]:
    """Parse LLM response to extract fixed files.

    Returns dict mapping filename to new code content.
    """
    import re
    fixes: dict[str, str] = {}

    # Try to find [FILE: name.py]...``` blocks
    pattern = r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```'
    for match in re.finditer(pattern, response, re.DOTALL):
        filename = match.group(1).strip()
        code = match.group(2).strip()
        if code and "import" in code:
            fixes[filename] = code

    # Fallback: look for ```python blocks if no [FILE:] tags found
    if not fixes:
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if len(code_blocks) == len(batch_files):
            for fpath, code in zip(batch_files, code_blocks):
                if code.strip() and "import" in code:
                    fixes[os.path.basename(fpath)] = code.strip()

    return fixes


class OptimizeCommand(Command):
    """Find and optionally apply performance/memory/quality optimizations."""

    @property
    def name(self) -> str:
        return "optimize"

    @property
    def help_text(self) -> str:
        return "optimize <file|dir> [--apply] [--yes] [--list] — Find and apply optimizations (batched)"

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        parts = list(args)
        apply_mode = "--apply" in parts
        yes_mode = "--yes" in parts
        stdin_mode = "--stdin" in parts
        list_mode = "--list" in parts or "-l" in parts
        verbose = "--verbose" in parts or "-v" in parts

        parts = [p for p in parts if p not in ("--apply", "--yes", "--stdin", "--verbose", "-v", "--list", "-l")]

        targets: list[str] = []

        if stdin_mode:
            content = read_stdin("Paste code to analyze. Type --- on its own line when done:")
            if not content.strip():
                self.error("No code provided.")
                return True
            targets = ["<stdin>"]
        elif not parts:
            self.error("Usage: optimize <file|dir> [--apply] [--yes] [--stdin] [--list] [--verbose]")
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

        # Phase 1: Static analysis
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

        # Group findings by file
        by_file: dict[str, list[dict]] = {}
        for f in all_findings:
            by_file.setdefault(f["file"], []).append(f)

        # List mode: show compact summary and exit
        if list_mode:
            print(f"\n  {len(by_file)} file(s) with {len(all_findings)} issue(s):\n")
            for fpath, findings in sorted(by_file.items()):
                rel = os.path.relpath(fpath, os.getcwd()) if not stdin_mode else fpath
                patterns = ", ".join(sorted(set(f["pattern"] for f in findings)))
                print(f"  {rel} ({len(findings)}): {patterns}")
            print(f"\n  Run with --apply to fix these issues.")
            return True

        # Print static findings
        print(f"\n  Static analysis found {len(all_findings)} issue(s) in {len(file_contents)} file(s):\n")
        for fpath, findings in sorted(by_file.items()):
            rel = os.path.relpath(fpath, os.getcwd()) if not stdin_mode else fpath
            print(f"  {rel}:")
            for f in findings:
                print(f"    line {f['line']:>4}: [{f['pattern']}] {f['suggestion']}")
            print()

        if not apply_mode or stdin_mode:
            return True

        # Phase 2: Create batches and process
        batches = create_batches(file_contents, by_file)
        total_files = sum(len(b["files"]) for b in batches)
        print(f"  Processing {total_files} file(s) in {len(batches)} batch(es)...\n")

        all_fixes: dict[str, str] = {}

        for batch_idx, batch in enumerate(batches, 1):
            batch_str = f"Batch {batch_idx}/{len(batches)}"
            file_list = ", ".join(os.path.basename(f) for f in batch["files"])
            print(f"  {batch_str}: {file_list}")
            print(f"    Estimated tokens: {batch['total_tokens']}")

            context = format_batch_context(batch)
            static_findings = "\n".join(
                f"- {os.path.basename(fpath)}:{f['line']} [{f['pattern']}] {f['suggestion']}"
                for fpath, findings in batch["findings"].items()
                for f in findings
            )

            # Retry logic
            max_retries = 2
            for attempt in range(max_retries + 1):
                if attempt > 0:
                    print(f"    Retry {attempt}/{max_retries}...")

                try:
                    llm_response = await agent.llm.chat([
                        {"role": "system", "content": (
                            "You are an expert Python optimizer. Apply ALL optimizations to each file.\n\n"
                            "For EACH file, output:\n"
                            "[FILE: filename.py]\n```python\n# complete fixed code\n```\n\n"
                            "Rules:\n"
                            "- Output EVERY file from the input, even if unchanged\n"
                            "- No explanations between files\n"
                            "- No duplicate functions or classes\n"
                            "- Preserve all functionality"
                        )},
                        {"role": "user", "content": (
                            f"## Static findings:\n{static_findings}\n\n"
                            f"## Code to optimize:\n\n{context}"
                        )},
                    ])

                    if llm_response.startswith("[Error") or llm_response.startswith("[LM Studio"):
                        print(f"    LLM error: {llm_response[:100]}")
                        if attempt < max_retries:
                            continue
                        break

                    fixes = parse_llm_fixes(llm_response, batch["files"])
                    all_fixes.update(fixes)
                    print(f"    Got fixes for {len(fixes)} file(s)")
                    break

                except Exception as e:
                    print(f"    Error: {e}")
                    if attempt < max_retries:
                        continue
                    break

        if not all_fixes:
            print("\n  No fixes were generated.")
            return True

        # Phase 3: Apply fixes
        print(f"\n  Applying fixes...")
        applied = 0

        for fpath in batch["files"]:
            basename = os.path.basename(fpath)
            if basename not in all_fixes:
                continue

            new_code = all_fixes[basename]
            original_size = len(file_contents.get(fpath, ""))
            new_size = len(new_code)

            # Sanity check: don't apply if code shrank by >50% or grew by >200%
            if original_size > 0:
                ratio = new_size / original_size
                if ratio < 0.5 or ratio > 2.0:
                    print(f"  Skipping {basename} — suspicious size change ({original_size} → {new_size} bytes)")
                    continue

            show_file_diff(basename, file_contents.get(fpath, ""), new_code)

            if not yes_mode:
                print(f"  Apply {basename}? ({original_size} → {new_size} bytes) (y/N): ", end="")
                try:
                    if input().strip().lower() != "y":
                        print(f"    Skipped.")
                        continue
                except (EOFError, KeyboardInterrupt):
                    print("  Cancelled.")
                    return True

            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new_code)
                print(f"  Applied: {basename} ({new_size} bytes)")
                applied += 1
            except Exception as e:
                print(f"  Error writing {basename}: {e}")

        print(f"\n  Done. Applied {applied}/{len(all_fixes)} fix(es).")
        return True
