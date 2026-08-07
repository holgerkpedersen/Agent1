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

import ast
import logging
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Command, read_stdin, show_file_diff
from agent_core import workspace_path
from agent_core.patterns import analyze as static_analyze

if TYPE_CHECKING:
    from agent import Agent

logger = logging.getLogger(__name__)

# Stdlib module names (Python 3.10+).  Used to allow rewrite import additions
# only when the module is guaranteed to exist; relative/project/third-party
# additions stay rejected because their names cannot be validated cheaply.
_STDLIB_MODULES = frozenset(getattr(sys, "stdlib_module_names", ()))

# Rules appended to the optimizer's system prompt.  Additive only (see
# .opencode/memory.md — never remove requirements).  Kept as a module constant
# so the prompt contract is testable.
OPTIMIZE_RULES = (
    "- Output EVERY file from the input, even if unchanged\n"
    "- No explanations between files\n"
    "- No duplicate functions or classes\n"
    "- Preserve all functionality\n"
    "- Minimal, surgical changes only: fix exactly what the findings list points at. "
    "Do NOT rename, move, or restructure existing variables (e.g. keep `combined += x` "
    "accumulation as-is unless a finding directly targets it)\n"
    "- NEVER use the walrus operator `:=`, especially inside comprehensions — it binds "
    "the name into the enclosing function scope (PEP 572) and can silently leak/shadow; "
    "assign the value to a named local on its own line first\n"
    "- Preserve the original control-flow style: an imperative `for` loop stays a loop. "
    "The ONLY authorized loop-to-comprehension/generator conversion is when a "
    "list_append_join finding is listed for that loop — and never with `:=`. "
    "Do NOT fold several statements into a comprehension or compact one-liner in any "
    "other case\n"
    "- Decide and output immediately: do not deliberate at length in reasoning. "
    "If a change is not unambiguously required by a finding, skip it and keep the original\n"
    "- NEVER leave a dead assignment — every variable you assign must be read by later "
    "code; if an assignment becomes unused, remove it entirely\n"
    "- For a silent_except finding, replace `pass` with a print/log warning only — never "
    "convert `except ...: pass` to `raise`. Those handlers are usually intentional "
    "fallbacks; re-raising changes control flow and crashes otherwise-normal paths\n"
    "- Only change code inside a loop when a finding targets that exact line; never "
    "rewrite surrounding control flow the findings do not request\n"
    "- Hoist `regex_in_loop` matches to MODULE-LEVEL `_RE_*` constants at the top of the "
    "file (one definition). A `re.compile` inside a function body is NOT a fix — it still "
    "re-creates the pattern on every call\n"
    "- Never ADD an import statement that already exists earlier in the file — duplicate "
    "imports are a new defect (duplicate_import). Region output must not re-import modules "
    "the file header already imports\n"
    "- Do NOT introduce new helper variables, caches, or functions that the findings "
    "do not require\n"
)

# Token estimation: ~4 chars per token, with overhead for message framing
CHARS_PER_TOKEN = 4
SYSTEM_OVERHEAD_TOKENS = 200  # System prompt, formatting, etc.
MAX_BATCH_TOKENS = 25000     # Leave room for output within 32k limit
SAFETY_MARGIN = 0.8          # Use 80% of budget to be safe

# Divide & conquer: files above this input estimate are split into contiguous
# line regions so each LLM call only rewrites a slice of the file (avoids
# runaway reasoning + truncated output on large one-shot rewrites).
REGION_SPLIT_TOKENS = 4000   # per-file input estimate that triggers splitting
REGION_MAX_TOKENS = 2000     # target input tokens per region
REGION_MAX_OUTPUT_FACTOR = 4  # per-call max_tokens = factor x input tokens
REGION_MIN_MAX_TOKENS = 8192  # floor for the per-call max_tokens override
REGION_HARD_MAX_TOKENS = 50000  # ceiling for the override


def _request_max_tokens(input_tokens: int) -> int:
    """Cap the output budget so ``prompt + max_tokens`` fits the model context.

    Without this, a large prompt plus the profile's 32k max_tokens exceeds the
    context window and LM Studio rejects the request (HTTP 400) before any
    generation happens.  Output for an optimize rewrite is proportional to the
    input, so 4x input (with a floor) is a safe budget.  The ceiling
    (``REGION_HARD_MAX_TOKENS``) is intentionally generous so a model that
    spends a little of its budget on reasoning still has room to emit the full
    fixed code.
    """
    return min(REGION_HARD_MAX_TOKENS,
               max(REGION_MIN_MAX_TOKENS, input_tokens * REGION_MAX_OUTPUT_FACTOR))


_STMT_BOUNDARY_RE = re.compile(r'^(?:def |class |async def |@|if __name__|from |import )')


def _class_member_boundaries(source: str) -> set[int]:
    """Line starts (0-based) of statements directly inside class bodies.

    Methods, decorators and nested classes inside a class are statement
    starts even though they are indented, so they are safe regions boundaries
    — a big class body no longer has to be rewritten in one shot.  Falls back
    to an empty set if the source does not parse (boundaries stay modal to
    column-0 statements only).
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    starts: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # assignments/expressions are not cut points
            if stmt.decorator_list:
                first_dec = stmt.decorator_list[0]
                if hasattr(first_dec, "lineno"):
                    starts.add(first_dec.lineno - 1)
            elif stmt.lineno:
                starts.add(stmt.lineno - 1)
    return starts


def _leading_whitespace(line: str) -> str:
    """Leading whitespace of *line* ('' for blank/dedented lines)."""
    return line[: len(line) - len(line.lstrip())]


def split_into_regions(
    content: str, max_region_tokens: int = REGION_MAX_TOKENS
) -> list[tuple[int, int]]:
    """Split *content* at statement boundaries into contiguous ``(start, end)``
    0-based line ranges, each within the token budget.

    Boundaries are top-level ``def``/``class``/decorator/import lines plus
    the starts of statements directly inside class bodies (methods, nested
    classes, decorators), so a region is a sequence of complete statements —
    never a cut mid-function.  A single statement larger than the budget stays
    in its own region (we never split a statement).  Region slices keep their
    original indentation; that indentation is enforced on the returned code
    before the region is accepted.
    """
    lines = content.split("\n")
    boundaries = [0]
    member_starts = _class_member_boundaries(content)
    for i, line in enumerate(lines):
        if i in member_starts:
            boundaries.append(i)
            continue
        if not line or not line[:1].strip():
            continue  # blank or indented (inside a statement)
        if _STMT_BOUNDARY_RE.match(line.strip()):
            boundaries.append(i)
    boundaries = sorted(set(boundaries + [len(lines)]))

    regions: list[tuple[int, int]] = []
    seg_start = boundaries[0]
    for b in boundaries[1:]:
        if (b > seg_start
                and estimate_tokens("\n".join(lines[seg_start:b])) > max_region_tokens):
            regions.append((seg_start, b))
            seg_start = b
    if seg_start < len(lines):
        regions.append((seg_start, len(lines)))
    return regions


def _region_syntax_issues(code: str, slice_indent: str = "") -> list[dict]:
    """Syntax-only gate for a region slice.

    Full detectors need whole-file context (e.g. an import in region 0 used in
    region 3), so regions are validated for syntax here and the assembled file
    is validated in full after all regions are merged.

    *slice_indent* is the leading whitespace of the original slice's first
    non-blank line; the returned code must preserve it (indentation is a
    Python-level concern and must be right before the region is patched back).
    """
    if slice_indent and _leading_whitespace(code) != slice_indent:
        return [{
            "line": 1,
            "pattern": "region_indent_changed",
            "suggestion": (
                f"Region output changed the first line's leading whitespace "
                f"from {slice_indent!r} to {_leading_whitespace(code)!r}. "
                "Preserve the region's original indentation exactly."
            ),
        }]
    try:
        compile(textwrap.dedent(code), "<optimize:region>", "exec")
    except SyntaxError as e:
        return [{
            "line": e.lineno or 0,
            "pattern": "syntax_error",
            "suggestion": f"Region output is not valid Python: {e.msg}",
        }]
    return []


def _merge_regions(original: str, regions: dict[int, tuple[int, int, str]]) -> str:
    """Splice fixed region codes back into the original file lines.

    *regions* maps region index to ``(start, end, code)`` (0-based, end
    exclusive).  Regions are applied in order; any uncovered original lines
    are preserved as-is.
    """
    lines = original.split("\n")
    out: list[str] = []
    prev = 0
    for _, (start, end, code) in sorted(regions.items()):
        out.extend(lines[prev:start])
        out.extend(code.split("\n"))
        prev = end
    out.extend(lines[prev:])
    return "\n".join(out)


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


def validate_llm_code(code: str) -> list[dict]:
    """Validate LLM-generated code for quality issues.

    Returns a list of blocking issues that should prevent the code from being
    applied.  This always runs before any code is written to disk.

    Blocking checks:
    * ``syntax_error`` — the code does not even compile (never trust a
      non-compiling rewrite).
    * ``silent_except``, ``dead_assignment``, ``unreachable_code``,
      ``unused_import`` and ``walrus_in_comprehension`` covers scope-leaking
      ``:`` expressions and dead ``+=`` stores.
    """
    from agent_core.patterns import analyze
    blocking_patterns = {
        "silent_except",
        "dead_assignment",
        "unreachable_code",
        "unused_import",
        "walrus_in_comprehension",
    }
    # 1) Must parse / compile — this is the floor; a broken file must never be
    #    applied regardless of what the style detectors say.
    try:
        compile(code, "<optimize:llm>", "exec")
    except SyntaxError as e:
        return [{
            "line": e.lineno or 0,
            "pattern": "syntax_error",
            "suggestion": f"LLM output is not valid Python: {e.msg}"
            + (f" (line {e.lineno})" if e.lineno else ""),
        }]

    # 2) Style/quality detectors.
    findings = analyze(code)
    return [f for f in findings if f["pattern"] in blocking_patterns]


def parse_llm_fixes(
    response: str,
    batch_files: list[str],
    validate: bool = True,
    preserve_indent: bool = False,
) -> tuple[dict[str, str], dict[str, list[dict]]]:
    """Parse an LLM response into accepted fixes and rejected failures.

    Returns ``(fixes, failures)`` where *fixes* maps a file's basename to
    accepted code, and *failures* maps a basename to the list of blocking
    issues that caused it to be skipped.  Files that neither produce accepted
    code nor a parseable block are simply absent from both.

    A file is considered for acceptance iff its extracted code is non-empty.
    The previous ``import`` in code requirement is dropped: small modules with
    no imports would otherwise be silently skipped forever.

    With ``validate=False`` the extracted code is accepted without running
    ``validate_llm_code`` — used for region slices, which cannot be validated
    in isolation (an import in region 0 may be used in region 3).  The
    assembled file is validated in full after merging.

    With ``preserve_indent=True`` the leading whitespace of the extracted code
    is kept (region slices carry their original indentation into the merged
    file; stripping it would corrupt the indentation of indented methods).
    """
    fixes: dict[str, str] = {}
    failures: dict[str, list[dict]] = {}

    def _consider(basename: str, code: str) -> None:
        if preserve_indent:
            code = code.strip("\n").rstrip()
        else:
            code = code.strip()
        if not code:
            return
        if validate:
            issues = validate_llm_code(code)
            if issues:
                failures.setdefault(basename, []).extend(issues)
                return
        fixes[basename] = code

    # Primary format: [FILE: name.py] ... ```python ... ```
    pattern = r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```'
    seen_names: set[str] = set()
    for match in re.finditer(pattern, response, re.DOTALL):
        filename = match.group(1).strip()
        basename = os.path.basename(filename)
        code = match.group(2)
        if basename in seen_names:
            continue
        seen_names.add(basename)
        _consider(basename, code)

    # Fallback: bare ```python blocks, assumed 1:1 with batch_files in order.
    if not fixes and not failures:
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if len(code_blocks) == len(batch_files):
            for fpath, code in zip(batch_files, code_blocks):
                basename = os.path.basename(fpath)
                _consider(basename, code)

    _report_failures(failures)
    return fixes, failures


def _report_failures(failures: dict[str, list[dict]]) -> None:
    """Pretty-print per-file validation failures (deduped lines)."""
    for basename, issues in failures.items():
        print(f"  Skipping {basename} — LLM code quality issues:")
        seen: set[tuple] = set()
        for i in issues:
            key = (i["line"], i["pattern"], i["suggestion"])
            if key in seen:
                continue
            seen.add(key)
            print(f"    line {i['line']:>4}: [{i['pattern']}] {i['suggestion']}")


def _format_failures_for_prompt(failures: dict[str, list[dict]]) -> str:
    """Render validation failures as a retry hint for the LLM."""
    if not failures:
        return ""
    blocks: list[str] = []
    for basename, issues in failures.items():
        lines = [f"{basename}:"]
        for i in issues:
            lines.append(f"  - line {i['line']}: [{i['pattern']}] {i['suggestion']}")
        blocks.append("\n".join(lines))
    return (
        "The previous attempt produced invalid output for some files. "
        "Regenerate fixing these:\n"
        + "\n\n".join(blocks)
    )


def _blocked_regressions(original: str, candidate: str) -> list[dict]:
    """Blocking issues in *candidate* that make it worse than *original*.

    The apply gate for rewrites is "no regression", not "zero issues": a merge
    that reduces pre-existing problems (e.g. 6 ``silent_except`` -> 1) is a
    strict improvement and must be applied even though it is not perfectly
    clean — discarding it because a residual remains throws away the progress.

    Comparison is *per pattern-count*, not per line, because line numbers
    shift across any rewrite.  A pattern whose count grows beyond the
    original's (or a brand-new pattern type, ``syntax_error`` included) is a
    regression and is returned; the *excess* occurrences are reported so the
    repair pass has an exact target.
    """
    from collections import Counter

    orig_counts = Counter(i["pattern"] for i in validate_llm_code(original))
    cand = validate_llm_code(candidate)
    cand_counts = Counter(i["pattern"] for i in cand)

    regressions: list[dict] = []
    for pat, cnt in cand_counts.items():
        excess = cnt - orig_counts.get(pat, 0)
        if excess > 0:
            surplus = [i for i in cand if i["pattern"] == pat][:excess]
            regressions.extend(surplus)
    return regressions


async def _repair_merged_file(
    agent: "Agent",
    fpath: str,
    original: str,
    merged: str,
    issues: list[dict],
    max_retries: int,
) -> tuple[str | None, list[dict]]:
    """Retry-fix a merged region rewrite that *regressed* the original.

    The merged file is re-sent as a whole with the remaining regression issues
    listed as findings, up to ``max_retries`` repair passes.  A pass is
    accepted as soon as it is *no worse than the original* — it does not need
    to be perfectly clean (pre-existing residuals that merely shrank do not
    block).  Returns ``(accepted_code, final_regressions)``;
    *accepted_code* is ``None`` when every repair pass still regressed
    (caller skips the file).
    """
    import os

    basename = os.path.basename(fpath)
    feedback = ""
    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"    Repair pass {attempt}/{max_retries} for merged {basename}...")
        try:
            repair_batch = {
                "files": [fpath],
                "contents": {fpath: merged},
                "findings": {fpath: issues},
            }
            context = format_batch_context(repair_batch)
            static_findings = "\n".join(
                f"- {basename}:{i['line']} [{i['pattern']}] {i['suggestion']}"
                for i in issues
            )
            user_content = f"## Static findings:\n{static_findings}\n\n## Code to optimize:\n\n{context}"
            if feedback:
                user_content = feedback + "\n\n" + user_content
            llm_response = await agent.llm.chat(
                [
                    {"role": "system", "content": (
                        "You are an expert Python optimizer. Apply ALL optimizations to each file.\n\n"
                        "For EACH file, output:\n"
                        "[FILE: filename.py]\n```python\n# complete fixed code\n```\n\n"
                        "Rules:\n"
                        f"{OPTIMIZE_RULES}"
                    )},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=_request_max_tokens(
                    estimate_tokens(merged) + len(static_findings)
                ),
                disable_thinking=True,
            )
            if llm_response.startswith("[Error") or llm_response.startswith("[LM Studio"):
                if attempt < max_retries:
                    continue
                break

            fixes, failures = parse_llm_fixes(llm_response, [basename])
            if basename not in fixes:
                if failures:
                    feedback = _format_failures_for_prompt(failures)
                else:
                    feedback = (
                        "Your previous response contained no usable code blocks. "
                        "Regenerate: output exactly\n"
                        "[FILE: name.py]\n```python\n<complete fixed code>\n```\n"
                        "Do not return reasoning or prose alone."
                    )
                continue

            candidate = fixes[basename]
            new_issues = validate_llm_code(candidate)
            added = _blocked_added_imports(candidate, original)
            if added:
                new_issues = new_issues + [{
                    "line": 1,
                    "pattern": "changed_imports",
                    "suggestion": (
                        "Rewrite added import statement(s) not present in the "
                        "original: " + ", ".join(sorted(added))[:140] + ". "
                        "Restore the original import lines."
                    ),
                }]
            # Accept when no worse than the original, even if residual
            # pre-existing issues remain.
            regressions = _blocked_regressions(original, candidate)
            if added or regressions:
                issues = list(regressions) + [
                    i for i in new_issues if i["pattern"] == "changed_imports"
                ]
                merged = candidate
                feedback = _format_failures_for_prompt({basename: issues})
                continue
            return candidate, []
        except Exception as e:
            print(f"    Repair error: {e}")
            if attempt < max_retries:
                continue
            break
    return None, issues


def _code_def_class_names(code: str) -> set[str]:
    """Top-level (module + class-body) def/class names in *code*."""
    return {
        m.group(1)
        for m in re.finditer(r"^(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", code, re.MULTILINE)
    }


def _import_entries(code: str) -> set[str]:
    """Normalized import statements in *code*, e.g. ``from .base import Command``.

    Used to reject LLM rewrites that *add* or *relocate* imports relative to the
    original file — a compile-only validator cannot detect a rewrite that pulls
    names from a module that never exported them.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    entries: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                entries.add(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            module = prefix + (node.module or "") or prefix
            names = ", ".join(sorted(a.name for a in node.names))
            entries.add(f"from {module} import {names if names else '*'}")
    return entries


def _import_entry_counts(code: str) -> dict[str, int]:
    """Count of each normalized import statement in *code*.

    Unlike ``_import_entries``, this preserves multiplicity so a rewrite that
    *duplicates* an import already present in the file is detectable.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                entry = f"import {alias.name}"
                counts[entry] = counts.get(entry, 0) + 1
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            module = prefix + (node.module or "") or prefix
            names = ", ".join(sorted(a.name for a in node.names))
            entry = f"from {module} import {names if names else '*'}"
            counts[entry] = counts.get(entry, 0) + 1
    return counts


def _blocked_added_imports(candidate: str, original: str) -> set[str]:
    """Added or duplicated imports in *candidate* that make it unsafe to apply.

    A rewrite may add imports only if they are stdlib: a stdlib module is
    guaranteed to exist and export the requested names, so ``import logging``
    added to implement a ``silent_except`` fix is always safe.  Relative
    imports (``from .base import Command``) and third-party additions are
    rejected because the name's existence cannot be validated cheaply —
    ``compile()`` cannot catch a name a module never exported.

    A rewrite may never add a *second* copy of an import the file already
    has: duplicate imports are a new ``duplicate_import`` defect (region
    splices frequently re-inject ``import os``/``import re`` etc.), so those
    are rejected even though the module is stdlib.
    """
    cand_counts = _import_entry_counts(candidate)
    orig_counts = _import_entry_counts(original)
    blocked: set[str] = set()
    for entry, count in cand_counts.items():
        excess = count - orig_counts.get(entry, 0)
        if excess <= 0:
            continue
        blocked.add(entry)
        if entry in orig_counts:
            # Already imported elsewhere in the file — a duplicate, always bad.
            continue
        if entry.startswith("from ."):
            continue
        match = re.match(r"^(?:import|from)\s+([A-Za-z_]\w*)", entry)
        top = match.group(1) if match else None
        if top is not None and top in _STDLIB_MODULES:
            blocked.discard(entry)
    return blocked


def _post_apply_verify(basename: str, fpath: str, original: str, new_code: str) -> list[str]:
    """Run lightweight post-apply sanity checks; return warning strings.

    Never raises — this is a best-effort safety net, not a gate.
    Checks: compiles, and no def/class names were dropped relative to original.
    """
    warnings: list[str] = []
    try:
        compile(new_code, f"<optimize:{basename}>", "exec")
    except SyntaxError as e:
        warnings.append(f"  WARNING: written file {basename} has a syntax error: {e.msg}")
    old_names = _code_def_class_names(original)
    new_names = _code_def_class_names(new_code)
    dropped = old_names - new_names
    if dropped:
        warnings.append(
            f"  WARNING: {basename} dropped definition(s): {', '.join(sorted(dropped))}"
        )
    return warnings


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
            print("\n  Run with --apply to fix these issues.")
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
        max_retries = 2

        for batch_idx, batch in enumerate(batches, 1):
            batch_str = f"Batch {batch_idx}/{len(batches)}"
            file_list = ", ".join(os.path.basename(f) for f in batch["files"])
            print(f"  {batch_str}: {file_list}")
            print(f"    Estimated tokens: {batch['total_tokens']}")

            # Divide & conquer: files above the input budget are split into
            # contiguous line regions; each region gets its own LLM call so a
            # single response never has to re-emit a huge file (which burned
            # 15k reasoning tokens and truncated on an 834-line file).
            units: list[dict] = []
            for fpath in batch["files"]:
                content = batch["contents"][fpath]
                findings = batch["findings"][fpath]
                regions = split_into_regions(content)
                if len(regions) > 1:
                    print(f"    {os.path.basename(fpath)}: split into "
                          f"{len(regions)} region(s) — divide & conquer")
                    for start, end in regions:
                        region_code = "\n".join(content.split("\n")[start:end])
                        units.append({
                            "file": fpath,
                            "basename": os.path.basename(fpath),
                            "code": region_code,
                            "findings": [
                                {**f, "line": f["line"] - start}
                                for f in findings if start + 1 <= f["line"] <= end
                            ],
                            "is_region": True,
                            "start": start,
                            "end": end,
                        })
                else:
                    units.append({
                        "file": fpath,
                        "basename": os.path.basename(fpath),
                        "code": content,
                        "findings": findings,
                        "is_region": False,
                        "start": 0,
                        "end": len(content.split("\n")),
                    })

            region_fixes: dict[str, dict[int, tuple[int, int, str]]] = {}

            for unit in units:
                basename = unit["basename"]
                if unit["is_region"]:
                    print(f"    Region of {basename} (lines {unit['start'] + 1}-"
                          f"{unit['end']}, ~{estimate_tokens(unit['code'])} tokens)")
                feedback = ""
                for attempt in range(max_retries + 1):
                    if attempt > 0:
                        print(f"    Retry {attempt}/{max_retries}...")
                    try:
                        unit_batch = {
                            "files": [unit["file"]],
                            "contents": {unit["file"]: unit["code"]},
                            "findings": {unit["file"]: unit["findings"]},
                        }
                        context = format_batch_context(unit_batch)
                        static_findings = "\n".join(
                            f"- {basename}:{f['line']} [{f['pattern']}] {f['suggestion']}"
                            for f in unit["findings"]
                        ) if unit["findings"] else "- (none)"
                        user_content = (
                            f"## Static findings:\n{static_findings}\n\n"
                            f"## Code to optimize:\n\n{context}"
                        )
                        if unit["is_region"]:
                            user_content += (
                                f"\n\nThis is a REGION of a larger file (original lines "
                                f"{unit['start'] + 1}-{unit['end']}). Finding line numbers "
                                "are relative to this region's first line. Output the fixed "
                                f"region in [FILE: {basename}] — do not include other parts "
                                "of the file. Preserve the original indentation of the "
                                "region's first line exactly (it may start inside a class "
                                "body); the region will be spliced back into the file "
                                "verbatim."
                            )
                        if feedback:
                            user_content = feedback + "\n\n" + user_content
                        llm_response = await agent.llm.chat(
                            [
                                {"role": "system", "content": (
                                    "You are an expert Python optimizer. Apply ALL optimizations to each file.\n\n"
                                    "For EACH file, output:\n"
                                    "[FILE: filename.py]\n```python\n# complete fixed code\n```\n\n"
                                    "Rules:\n"
                                    f"{OPTIMIZE_RULES}"
                                )},
                                {"role": "user", "content": user_content},
                            ],
                            max_tokens=_request_max_tokens(
                                estimate_tokens(unit["code"]) + len(static_findings)
                            ),
                            disable_thinking=True,
                        )

                        if llm_response.startswith("[Error") or llm_response.startswith("[LM Studio"):
                            print(f"    LLM error: {llm_response[:100]}")
                            if attempt < max_retries:
                                continue
                            break

                        if unit["is_region"]:
                            # Regions cannot be fully validated in isolation
                            # (imports may be used in other regions) — syntax
                            # gate only; the merged file is validated after all
                            # regions return.  The returned region must keep
                            # the original slice's leading indentation so the
                            # splice does not corrupt indented class members.
                            fixes, failures = parse_llm_fixes(
                                llm_response, [basename], validate=False,
                                preserve_indent=True,
                            )
                            if basename in fixes:
                                slice_indent = ""
                                for line in unit["code"].split("\n"):
                                    if line.strip():
                                        slice_indent = _leading_whitespace(line)
                                        break
                                issues = _region_syntax_issues(
                                    fixes[basename], slice_indent
                                )
                                if issues:
                                    failures.setdefault(basename, []).extend(issues)
                                    del fixes[basename]
                        else:
                            fixes, failures = parse_llm_fixes(
                                llm_response, [basename]
                            )

                            # Reject rewrites whose import set is not a subset
                            # of the original's: compile() cannot catch an LLM
                            # relocating or adding imports to names a module
                            # never exported.  Stdlib additions (e.g. logging
                            # to fix a silent_except) are safe and allowed.
                            orig = batch["contents"][unit["file"]]
                            if basename in fixes:
                                added = _blocked_added_imports(
                                    fixes[basename], orig
                                )
                                if added:
                                    del fixes[basename]
                                    failures.setdefault(basename, []).append({
                                        "line": 1,
                                        "pattern": "changed_imports",
                                        "suggestion": (
                                            "Rewrite added import statement(s) not present "
                                            "in the original: "
                                            f"{', '.join(sorted(added))[:140]}. "
                                            "Restore the original import lines."
                                        ),
                                    })
                                    print(f"    Rejected 1 fix(es) for changed "
                                          f"imports: {basename}")
                                    for issue in failures[basename]:
                                        if issue["pattern"] == "changed_imports":
                                            print(f"    Skipping {basename} — "
                                                  f"[changed_imports] {issue['suggestion']}")

                        if unit["is_region"]:
                            if basename in fixes:
                                region_fixes.setdefault(basename, {})[unit["start"]] = (
                                    unit["start"], unit["end"], fixes[basename]
                                )
                                print(f"    Got fix for region (lines {unit['start'] + 1}-"
                                      f"{unit['end']})")
                                break
                        else:
                            if fixes:
                                all_fixes.update(fixes)
                                print(f"    Got fixes for {len(fixes)} file(s)")
                                break

                        if not fixes and not failures:
                            # Model returned no usable code (e.g. emptied into
                            # reasoning).  Do not silently drop the unit.
                            print(f"    No code blocks parsed from LLM response "
                                  f"({len(llm_response)} chars) — retrying...")
                            if attempt < max_retries:
                                feedback = (
                                    "Your previous response contained no usable code blocks. "
                                    "Regenerate: for EACH file output exactly:\n"
                                    "[FILE: name.py]\n```python\n<complete fixed code>\n```\n"
                                    "Do not return reasoning or prose alone."
                                )
                                continue
                            break

                        if failures and attempt < max_retries:
                            feedback = _format_failures_for_prompt(failures)
                            continue
                        break

                    except Exception as e:
                        print(f"    Error: {e}")
                        if attempt < max_retries:
                            continue
                        break

            # Assemble region-split files and validate the merged result.
            for basename, regions in region_fixes.items():
                fpath = next(f for f in batch["files"] if os.path.basename(f) == basename)
                original = batch["contents"][fpath]
                merged = _merge_regions(original, regions)
                # Gate on *no regression* relative to the original: a merge that
                # reduces pre-existing findings (e.g. 6 silent_except -> 1) is an
                # improvement and is applied, even with residuals left.  Only a
                # brand-new pattern type or growth above the original's count
                # triggers repair / skip.
                issues = _blocked_regressions(original, merged)
                if issues:
                    print(f"  Merged rewrite of {basename} regresses "
                          f"{len(issues)} finding(s) vs the original — running repair passes...")
                    repaired, issues = await _repair_merged_file(
                        agent, fpath, original, merged, issues, max_retries
                    )
                    if repaired is None:
                        print(f"  Skipping {basename} — merged rewrite regresses the original:")
                        _report_failures({basename: issues})
                        continue
                    merged = repaired
                added = _blocked_added_imports(merged, original)
                if added:
                    print(f"  Skipping {basename} — merged rewrite added import(s): "
                          f"{', '.join(sorted(added))[:140]}")
                    continue
                all_fixes[basename] = merged
                print(f"  Merged {len(regions)} region(s) -> {basename}")

        if not all_fixes:
            print("\n  No fixes were generated.")
            return True

        # Phase 3: Apply fixes
        print("\n  Applying fixes...")
        applied = 0

        for fpath in targets:
            basename = os.path.basename(fpath)
            if basename not in all_fixes:
                continue

            new_code = all_fixes[basename]
            original = file_contents.get(fpath, "")
            original_size = len(original)
            new_size = len(new_code)

            # Sanity check: don't apply if code shrank by >50% or grew by >200%
            if original_size > 0:
                ratio = new_size / original_size
                if ratio < 0.5 or ratio > 2.0:
                    print(f"  Skipping {basename} — suspicious size change ({original_size} -> {new_size} bytes)")
                    continue

            show_file_diff(basename, original, new_code)

            if not yes_mode:
                print(f"  Apply {basename}? ({original_size} -> {new_size} bytes) (y/N): ", end="")
                try:
                    if input().strip().lower() != "y":
                        print("    Skipped.")
                        continue
                except (EOFError, KeyboardInterrupt):
                    print("  Cancelled.")
                    return True

            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new_code)
                print(f"  Applied: {basename} ({new_size} bytes)")
                applied += 1
                for w in _post_apply_verify(basename, fpath, original, new_code):
                    print(w)
            except Exception as e:
                print(f"  Error writing {basename}: {e}")

        # Report residual findings on the files we actually changed.
        changed = [os.path.basename(fp) for fp in targets if os.path.basename(fp) in all_fixes]
        if changed:
            print("\n  Post-apply residual static findings (non-blocking):")
            for basename in changed:
                for fp in targets:
                    if os.path.basename(fp) == basename:
                        new_src = file_contents.get(fp, "")
                        try:
                            with open(fp, encoding="utf-8") as f:
                                new_src = f.read()
                        except Exception as exc:
                            logger.debug("Could not re-read %s: %s", fp, exc)
                        res = static_analyze(new_src)
                        if res:
                            for r in res:
                                print(f"    {basename}:{r['line']} [{r['pattern']}]")
                        break

        print(f"\n  Done. Applied {applied}/{len(all_fixes)} fix(es).")
        return True
