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
from agent_core.patch_utils import apply_anchored_patch, apply_patch, split_patch_hunks
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
    "The fix must replace the whole loop accumulation with a single comprehension — "
    "swapping .append(x) for .extend([x]) / += [x] is NOT a fix (same per-iteration cost). "
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
    "- Address ONLY the listed findings. Any edit not required by a finding — comments, "
    "whitespace, docstrings, unrelated lines — is a failure; change exactly the code the "
    "findings point at and nothing else\n"
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
REGION_CONTEXT_RESERVE = 900    # tokens kept between the prompt and the model window
# Input tokens beyond which a region switches to patch mode: re-emitting a
# ~18k-token single statement (e.g. a 1350-line method) as a [FILE:] region
# overflows the window and burns server time for no gain; hunks are small.
REGION_PATCH_MODE_TOKENS = 6000
REGION_PATCH_MAX_TOKENS = 8192   # hunks only — output budget for one targeted patch.
                              # Raised from 4096 so a reasoning model that briefly
                              # thinks before emitting the hunk still has room for
                              # the patch to fit without truncation (was the cause
                              # of "No valid hunks in patch" retries on Laguna).

# @@ -start,count +start,count @@ (lenient:  +start part may be missing)
_PATCH_HUNK_RE = re.compile(
    r"@@\s*-(\d+)(?:,(\d+))?(?:\s*\+(\d+)(?:,(\d+))?)?\s*@@"
)


def _strip_markdown_fence(text: str) -> str:
    """Remove surrounding ```lang / ``` fences AND prose before the first @@.
    Returns '' when no hunk header exists."""
    stripped = text.strip()
    fence = re.match(r"^```[a-zA-Z0-9]*\s*\n", stripped)
    if fence:
        stripped = stripped[fence.end():]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    first = stripped.find("@@")
    if first == -1:
        return ""
    return stripped[first:].strip("\n")


def _shift_hunk_starts(patch_text: str, delta: int) -> str:
    """Re-anchor every ``@@`` hunk start in *patch_text* by *delta* lines.

    Models sometimes report line numbers relative to the WHOLE file instead of
    the region slice; shifting by ``-region_start`` maps those hunks back onto
    the region.  Starts are clamped to >= 1.
    """
    def _repl(m: re.Match) -> str:
        old = max(1, int(m.group(1)) + delta)
        count_old = f",{int(m.group(2)) + delta}" if m.group(2) else ""
        new = max(1, int(m.group(3)) + delta) if m.group(3) else m.group(3)
        count_new = f",{int(m.group(4)) + delta}" if m.group(4) else ""
        header = f"@@ -{old}{count_old}"
        if new is not None:
            header += f" +{new}{count_new}"
        return header + " @@"

    return _PATCH_HUNK_RE.sub(_repl, patch_text)


def _request_max_tokens(input_tokens: int, context_tokens: int | None = None) -> int:
    """Cap the output budget so ``prompt + max_tokens`` fits the model context.

    The known failure mode: a large prompt plus the profile's 32k max_tokens
    exceeds the context window and LM Studio rejects the request (HTTP 400)
    before any generation happens.  When the loaded model's ``context_length``
    is known (see ``_loaded_model_context``), the budget is
    ``context - prompt - REGION_CONTEXT_RESERVE`` so the request always fits.
    Without that information, output for a whole-region rewrite is
    proportional to the input, so 4x input (with a floor) is a safe budget.
    The ceiling (``REGION_HARD_MAX_TOKENS``) is intentionally generous so a
    model that spends a little of its budget on reasoning still has room to
    emit the full fixed code.
    """
    if context_tokens:
        budget = context_tokens - input_tokens - REGION_CONTEXT_RESERVE
        return max(1, min(REGION_HARD_MAX_TOKENS, budget))
    return min(REGION_HARD_MAX_TOKENS,
               max(REGION_MIN_MAX_TOKENS, input_tokens * REGION_MAX_OUTPUT_FACTOR))


# One probe per process: every unit test / CLI call goes through execute(),
# and the loaded-model context window does not change mid-run.
_PROBE_CACHED: list[int | None] = [None]
_PROBE_DONE = False


def _loaded_model_context() -> int | None:
    """Context window (tokens) of the currently loaded LM Studio model.

    Returns None when the server is unreachable or no loaded model exposes a
    context length — callers then fall back to the heuristic in
    ``_request_max_tokens``.  When multiple models are loaded, the minimum
    context wins so the budget is safe for whichever instance serves the
    request.  Result is cached per process (probe once).
    """
    global _PROBE_DONE
    if _PROBE_DONE:
        return _PROBE_CACHED[0]
    _PROBE_DONE = True
    try:
        from agent_core.llm.lmstudio import get_models_status
        lengths = {
            m["context_length"]
            for m in get_models_status()
            if m["loaded"] and m.get("context_length")
        }
        _PROBE_CACHED[0] = min(lengths) if lengths else None
    except Exception:
        _PROBE_CACHED[0] = None
    return _PROBE_CACHED[0]


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


def _ast_statement_seams(source: str) -> set[int]:
    """Line indexes (0-based) between CONSECUTIVE COMPLETE statements in any
    statement list of the AST: module, class and function bodies, ``if``/
    ``else``/``elif`` blocks, ``for``/``while`` bodies, ``with``/``try``/
    ``except``/``finally`` blocks, etc.

    Each seam is the start line of a statement that has an immediately
    preceding sibling, so a seam can never land inside a single statement —
    cutting between siblings preserves statement completeness on both sides
    (the dedent+splice mechanism then reassembles the blocks verbatim).  We
    only pair statements from the SAME attribute list of a node: the parts of
    one compound statement (an ``if`` and its ``else``, a ``try`` and its
    handlers) live in different lists, so they are never separated by a seam.

    The point is to stop huge single statements such as a 1000+ line method —
    previously one unsplittable region that forced the LLM into (unreliable)
    hunk mode — from monopolizing a region.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    seams: set[int] = set()
    for node in ast.walk(tree):
        for attr in ("body", "orelse", "finalbody"):
            body = getattr(node, attr, None)
            if not isinstance(body, list) or len(body) < 2:
                continue
            if not all(isinstance(x, ast.stmt) for x in body):
                continue
            for nxt in body[1:]:
                first = nxt.decorator_list[0] if getattr(nxt, "decorator_list", None) else nxt
                if getattr(first, "lineno", None) is not None:
                    seams.add(first.lineno - 1)
    return seams


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
    classes, decorators) plus AST statement seams at ANY nesting level (see
    ``_ast_statement_seams``), so a region is a sequence of complete
    statements — never a cut mid-function or mid-compound.  A single
    statement larger than the budget stays in its own region (we never split
    a statement).  Region slices keep their original indentation; that
    indentation is enforced on the returned code before the region is
    accepted.
    """
    lines = content.split("\n")
    boundaries = [0]
    cut_points = _class_member_boundaries(content) | _ast_statement_seams(content)
    for i, line in enumerate(lines):
        if i in cut_points:
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
                "Region output changed the first line's leading whitespace "
                f"from {slice_indent!r} to {_leading_whitespace(code)!r}. "
                "Preserve the region's original indentation exactly."
            ),
        }]
    if slice_indent:
        # Slice starts mid-body (a seam inside a function).  Its statements
        # implicitly belong to the enclosing header ABOVE the slice, so the
        # slice alone cannot be compiled in isolation — e.g. it may contain
        # column-0 defs whose bodies are deeper-indented relative to them,
        # not to the slice.  The full file is compiled after all regions are
        # merged (that is the authority); skip the isolated check here.
        return []
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



CONTEXT_MAX_LINES = 120
FINDING_MAX_ATTEMPTS = 3
SCOPE_TOLERANCE = 20  # max lines a hunk start may deviate from the finding line


def _finding_context(source: str, line_no: int) -> str | None:
    """Numbered window (absolute file line numbers) around *line_no*.

    Uses the innermost enclosing AST statement, widened to its enclosing
    function/method/class when that keeps the window useful, and caps the
    window at ``CONTEXT_MAX_LINES`` lines.  Falls back to a plain +/- window
    when the source does not parse.  Returns None for out-of-range lines.
    """
    lines = source.split("\n")
    if line_no < 1 or line_no > len(lines):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    start = end = None
    if tree is not None:
        best = None
        best_ln = best_end = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.stmt):
                continue
            ln = getattr(node, "lineno", None)
            en = getattr(node, "end_lineno", None)
            if ln is None or en is None or not (ln <= line_no <= en):
                continue
            if best is None or (en - ln) < (best_end - best_ln):
                best, best_ln, best_end = node, ln, en
        if best is not None:
            scope = best
            # Widen to the smallest enclosing function/method/class so the LLM
            # sees enough surrounding code (init, setup, teardown) to produce a
            # correct multi-line fix.  The innermost statement (e.g. a single
            # AugAssign) is often too narrow.
            fns = [
                n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.lineno <= line_no <= n.end_lineno
            ]
            if fns:
                smallest = min(fns, key=lambda n: n.end_lineno - n.lineno)
                if (smallest.end_lineno - smallest.lineno
                        >= scope.end_lineno - scope.lineno):
                    scope = smallest
            start, end = scope.lineno, scope.end_lineno
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) and scope.decorator_list:
                first = scope.decorator_list[0]
                if getattr(first, "lineno", None) is not None and first.lineno < start:
                    start = first.lineno
    if start is None:
        start, end = max(1, line_no - 10), min(len(lines), line_no + 20)
    if end - start + 1 > CONTEXT_MAX_LINES:
        start = max(1, line_no - CONTEXT_MAX_LINES // 2)
        end = min(len(lines), start + CONTEXT_MAX_LINES - 1)
        start = max(1, end - CONTEXT_MAX_LINES + 1)
    out = [f"{i:>6} | {lines[i - 1]}" for i in range(start, end + 1)]
    return "\n".join(out)


def _cosmetic_only(old_lines: list[str], new_lines: list[str]) -> bool:
    """True when the only difference between the two line lists is
    whitespace (indentation / trailing blanks / blank-line formatting) —
    LLMs routinely "fix" regions by stripping whitespace while the actual
    finding line stands untouched."""
    if old_lines == new_lines:
        return True
    import difflib
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if [l.strip() for l in old_lines[i1:i2]] != [l.strip() for l in new_lines[j1:j2]]:
            return False
    return True


_MECH_PASS_RE = re.compile(r"^\s+pass\s*$")
_UNRESOLVED_RE = re.compile(r"\[UNRESOLVED:\s*[^\]]+\]\s*(.*)", re.DOTALL)
_NONE_EQ_RE = re.compile(r"\b(\w+)\s*(==|!=)\s*None\b")
_TYPE_EQ_RE = re.compile(r"type\((\w+)\)\s*==\s*(\w+)")
_TYPE_NE_RE = re.compile(r"type\((\w+)\)\s*!=\s*(\w+)")
_TYPE_IN_RE = re.compile(r"type\((\w+)\)\s+in\s+\(([^)]+)\)")
_PLACEHOLDER_PHRASES = ("unchanged line", "rest of the", "remaining code", "...", "code unchanged")


# ── Mechanical fixers (one per pattern) ──────────────────────────

def _fix_unused_import(wl, idx, line, basename, finding):
    src_line = wl[idx]
    stripped = src_line.strip()
    if stripped.startswith("from __future__ import"):
        return None
    target_m = re.search(r"Imported\s+['\"]([\w.]+)['\"]", finding.get("suggestion", ""))
    if not target_m:
        return None
    target = target_m.group(1)
    imp_single_m = re.match(r"^import\s+([A-Za-z_]\w*)(?:\s+as\s+\w+)?$", stripped)
    if imp_single_m:
        if imp_single_m.group(1) == target:
            return f"@@ -{line},1 +{line},0 @@\n-{src_line}"
        return None
    from_single_m = re.match(r"^from\s+\S+\s+import\s+([A-Za-z_]\w*)$", stripped)
    if from_single_m:
        if from_single_m.group(1) == target:
            return f"@@ -{line},1 +{line},0 @@\n-{src_line}"
        return None
    import_m = re.match(r"^import\s+([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)$", stripped)
    if import_m:
        names = [n.strip() for n in import_m.group(1).split(",")]
        if len(names) == 1:
            return None
        remaining = [n for n in names if n != target]
        if len(remaining) == len(names):
            return None
        return _multi_name_import_hunk(src_line, "import", remaining, line)
    from_m = re.match(r"^from\s+(\S+)\s+import\s+([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)$", stripped)
    if from_m:
        module = from_m.group(1)
        names = [n.strip() for n in from_m.group(2).split(",")]
        if len(names) == 1:
            return None
        remaining = [n for n in names if n != target]
        if len(remaining) == len(names):
            return None
        return _multi_name_import_hunk(src_line, f"from {module}", remaining, line)
    return None


def _multi_name_import_hunk(
    src_line: str, prefix: str, remaining: list[str], line: int
) -> str | None:
    """Build a hunk that removes a name from a multi-name import line."""
    if prefix.startswith("from "):
        new_import = f"{prefix} import {', '.join(remaining)}"
    else:
        new_import = f"{prefix} {', '.join(remaining)}"
    if new_import.strip() == src_line.strip():
        return None
    return f"@@ -{line},1 +{line},1 @@\n-{src_line}\n+{new_import}"


def _fix_dead_assignment(wl, idx, line, basename, finding):
    if "immediately overwritten" in finding.get("suggestion", ""):
        return f"@@ -{line},1 +{line},0 @@\n-{wl[idx]}"
    if "never used after" in finding.get("suggestion", ""):
        code = "\n".join(wl)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        assign_name = None
        for node in ast.walk(tree):
            if getattr(node, 'lineno', -1) == idx + 1 and isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assign_name = target.id
                        break
        if assign_name is None:
            return None
        for enclosing in ast.walk(tree):
            if isinstance(enclosing, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if enclosing.lineno <= idx + 1 <= enclosing.end_lineno:
                    for sub in ast.walk(enclosing):
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id == assign_name:
                            if sub.lineno > idx + 1:
                                return None
                    return f"@@ -{line},1 +{line},0 @@\n-{wl[idx]}"
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id == assign_name:
                if sub.lineno > idx + 1:
                    return None
        return f"@@ -{line},1 +{line},0 @@\n-{wl[idx]}"
    return None


def _fix_silent_except(wl, idx, line, basename, finding):
    ei = idx
    e_indent = len(wl[ei]) - len(wl[ei].lstrip())
    for j in range(ei + 1, min(ei + 20, len(wl))):
        nxt = wl[j].rstrip("\r")
        if nxt.strip() == "" or nxt.lstrip().startswith("#"):
            continue
        nxt_indent = len(nxt) - len(nxt.lstrip())
        if nxt_indent <= e_indent:
            break
        if _MECH_PASS_RE.match(nxt):
            indent = " " * nxt_indent
            return (f"@@ -{j + 1},1 +{j + 1},1 @@\n"
                    f"-{nxt}\n"
                    f"+{indent}print(\"Warning: silenced exception in "
                    f"{basename}:{line}\")")
    return None


def _fix_none_eq(wl, idx, line, basename, finding):
    old_line = wl[idx]
    new_line = old_line.replace("== None", "is None").replace("!= None", "is not None")
    if new_line == old_line:
        return None
    return f"@@ -{line},1 +{line},1 @@\n-{old_line}\n+{new_line}"


def _fix_fstring_literal(wl, idx, line, basename, finding):
    old_line = wl[idx]
    from agent_core.patterns import _fstring_lines_without_interpolation
    hits = _fstring_lines_without_interpolation(old_line)
    if not hits:
        return None
    ln, col = hits[0]
    new_line = old_line[:col] + old_line[col + 1:]
    return f"@@ -{line},1 +{line},1 @@\n-{old_line}\n+{new_line}"


def _fix_iter_dict_keys(wl, idx, line, basename, finding):
    old_line = wl[idx]
    new_line = old_line.replace(".keys()", "")
    if new_line == old_line:
        return None
    return f"@@ -{line},1 +{line},1 @@\n-{old_line}\n+{new_line}"


def _fix_type_comparison(wl, idx, line, basename, finding):
    old_line = wl[idx]
    stripped = old_line.lstrip()
    m = _TYPE_EQ_RE.search(stripped)
    if m:
        new_line = old_line.replace(
            m.group(0), f"isinstance({m.group(1)}, {m.group(2)})"
        )
        return f"@@ -{line},1 +{line},1 @@\n-{old_line}\n+{new_line}"
    m = _TYPE_NE_RE.search(stripped)
    if m:
        new_line = old_line.replace(
            m.group(0), f"not isinstance({m.group(1)}, {m.group(2)})"
        )
        return f"@@ -{line},1 +{line},1 @@\n-{old_line}\n+{new_line}"
    m = _TYPE_IN_RE.search(stripped)
    if m:
        new_line = old_line.replace(
            m.group(0), f"isinstance({m.group(1)}, ({m.group(2)}))"
        )
        return f"@@ -{line},1 +{line},1 @@\n-{old_line}\n+{new_line}"
    return None


def _fix_regex_in_loop(wl, idx, line, basename, finding):
    from agent_core.patterns import _loop_spans
    import io, tokenize

    spans = _loop_spans(wl)
    spanning = [sp for sp in spans if sp[0] < idx < sp[1]]
    if not spanning:
        return None
    s, e, indent = min(spanning, key=lambda sp: sp[0])
    loop_start = s
    loop_indent = indent

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(wl[idx]).readline))
    except tokenize.TokenError:
        return None

    call_i = None
    func_name = ""
    for i, (typ, s, _, _, _) in enumerate(toks):
        if typ == tokenize.NAME and s == "re":
            if i + 1 < len(toks) and toks[i + 1][1] == "." and i + 2 < len(toks):
                if toks[i + 2][0] == tokenize.NAME and toks[i + 2][1] in ("compile", "match", "search", "sub", "findall"):
                    if i + 3 < len(toks) and toks[i + 3][1] == "(":
                        call_i = i
                        func_name = toks[i + 2][1]
                        break
    if call_i is None:
        return None

    j = call_i + 4
    while j < len(toks) and toks[j][0] in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE):
        j += 1
    if j >= len(toks) or toks[j][0] != tokenize.STRING:
        return None
    arg_tok = toks[j]
    first_arg_src = arg_tok[1]
    arg_end = arg_tok[3][1]

    if "{" in first_arg_src and "f" in first_arg_src[:4].lower():
        return None

    k = j + 1
    while k < len(toks) and toks[k][0] in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE):
        k += 1
    if k < len(toks) and toks[k][1] == "+":
        return None

    existing = set()
    _RE_1 = re.compile(r"\b_RE_(\d+)\s*=")
    for l in wl:
        m = _RE_1.search(l)
        if m:
            existing.add(int(m.group(1)))
    n = 1
    while n in existing:
        n += 1
    name = f"_RE_{n}"

    orig_line = wl[idx]
    re_pos = toks[call_i][2][1]

    after_arg = orig_line[arg_end:]
    _rest = after_arg.lstrip()
    if _rest.startswith(","):
        after_clean = _rest.split(",", 1)[1].lstrip()
    else:
        after_clean = _rest

    new_line = f"{orig_line[:re_pos]}{name}.{func_name}({after_clean}"

    compile_line = f"{' ' * loop_indent}{name} = re.compile({first_arg_src})"

    loop_header = wl[loop_start]
    context = wl[loop_start + 1 : idx]

    hunk_lines = []
    hunk_lines.append(f"+{compile_line}")
    hunk_lines.append(f" {loop_header.rstrip()}")
    hunk_lines += [f" {cl.rstrip()}" for cl in context]
    hunk_lines.append(f"-{orig_line.rstrip()}")
    hunk_lines.append(f"+{new_line.rstrip()}")

    old_count = 1 + len(context) + 1
    new_count = 2 + len(context) + 1
    first_line = loop_start + 1
    hunk = f"@@ -{first_line},{old_count} +{first_line},{new_count} @@\n" + "\n".join(hunk_lines)
    return hunk


def _fix_list_append_join(wl, idx, line, basename, finding):
    """Mechanical: replace ``out = []`` + ``for x in y: out.append(expr)``
    with ``out = [expr for x in y]``.  Only handles stateless, single-statement
    loops whose list is built from scratch (empty initializer)."""
    from agent_core.patterns import _loop_spans

    spans = _loop_spans(wl)
    spanning = [sp for sp in spans if sp[0] < idx < sp[1]]
    if not spanning:
        return None
    s, e, indent = min(spanning, key=lambda sp: sp[0])

    header = wl[s]
    fm = re.match(r"^\s*for\s+(.+?)\s+in\s+(.+?)\s*:\s*$", header)
    if not fm:
        return None
    for_vars = fm.group(1).strip()
    iterable = fm.group(2).strip()

    al = wl[idx]
    m = re.search(r"\b([A-Za-z_]\w*)\s*\.(?:append|extend)\((.+)\)\s*$", al)
    if not m:
        return None
    target_var = m.group(1)
    expr = m.group(2).strip()

    for j in range(s + 1, e):
        stripped_j = wl[j].strip()
        if stripped_j and not stripped_j.startswith("#") and j != idx:
            return None

    init_idx = None
    for j in range(s - 1, max(0, s - 6), -1):
        if re.match(rf"^\s*{re.escape(target_var)}\s*(?::[^=]*)?\s*=\s*\[\]", wl[j]):
            init_idx = j
            break
    if init_idx is None:
        return None

    init_line = wl[init_idx]
    init_indent = len(init_line) - len(init_line.lstrip())
    comp_line = f"{' ' * init_indent}{target_var} = [{expr} for {for_vars} in {iterable}]"

    ctx_before = wl[max(0, init_idx - 3) : init_idx]
    ctx_after = wl[e : e + 3]
    removed = wl[init_idx : e]

    hunk_lines = []
    hunk_lines += [f" {cl.rstrip()}" for cl in ctx_before]
    hunk_lines += [f"-{rl.rstrip()}" for rl in removed]
    hunk_lines.append(f"+{comp_line}")
    hunk_lines += [f" {cl.rstrip()}" for cl in ctx_after]

    old_c = len(removed) + len(ctx_before) + len(ctx_after)
    new_c = 1 + len(ctx_before) + len(ctx_after)
    first = init_idx + 1
    return f"@@ -{first},{old_c} +{first},{new_c} @@\n" + "\n".join(hunk_lines)


_MECH_FIXERS: dict[str, object] = {
    "unused_import": _fix_unused_import,
    "dead_assignment": _fix_dead_assignment,
    "silent_except": _fix_silent_except,
    "none_eq": _fix_none_eq,
    "fstring_without_placeholder": _fix_fstring_literal,
    "iter_dict_keys": _fix_iter_dict_keys,
    "type_comparison": _fix_type_comparison,
    "duplicate_import": _fix_unused_import,
    "regex_in_loop": _fix_regex_in_loop,
    "list_append_join": _fix_list_append_join,
}
_MECH_PATTERNS: frozenset[str] = frozenset(_MECH_FIXERS)


_PATTERN_GUIDANCE: dict[str, str] = {
    "mutable_default_arg": (
        "- For **mutable_default_arg**: replace the mutable default ([] / {}) "
        "with ``None`` and add a guard inside the function body, e.g. "
        "``if x is None: x = []``. Do NOT keep the mutable literal — it "
        "persists across function calls."
    ),
    "redundant_bool_expr": (
        "- For **redundant_bool_expr**: replace ``return True if cond else False`` "
        "with ``return bool(cond)``. Replace ``return False if cond else True`` "
        "with ``return not cond`` (or ``return not bool(cond)`` when the "
        "condition may return a non-boolean)."
    ),
    "regex_in_loop": (
        "- For **regex_in_loop**: hoist the ``re.compile/s/match/search`` outside the "
        "loop. STATIC patterns → module‑level ``_X_RE = re.compile('...')``. DYNAMIC "
        "patterns (``rf'...{param}'`` where param is constant per call) → hoist to "
        "the top of the enclosing function, one line above the loop at the LOOP's "
        "indentation level (NOT inside the loop body).\n"
        "- Put the compile definition AND the usage replacement in the **same hunk**. "
        "If hoisting to module level, use ONE hunk anchored inside the function that "
        "removes the inline ``re.`` call, NOT two hunks — otherwise the line numbers "
        "shift and the second hunk cannot anchor.  Example:\n"
        "[PATCH: file.py]\n"
        "@@ -42,2 +42,4 @@\n"
        " def scan(files):\n"
        "+    _PAT_RE = re.compile(r'...(config)..')\n"
        "     for f in files:\n"
        "-        m = re.search(r'...(config)..', f)\n"
        "+        m = _PAT_RE.search(f)\n"
    ),
}


def _tri_mech_fix(work_lines: list[str], line: int, pattern: str,
                  basename: str, finding: dict) -> str | None:
    """Return a synthetic ``@@`` hunk for a deterministic fix, or None."""
    fixer = _MECH_FIXERS.get(pattern)
    if fixer is None:
        return None
    idx = line - 1
    if not (0 <= idx < len(work_lines)):
        return None
    return fixer(work_lines, idx, line, basename, finding)


def _count_pattern(code: str, pattern: str) -> int:
    """Number of occurrences of *pattern* in *code* according to the static analyser."""
    from agent_core.patterns import analyze
    return sum(1 for f in analyze(code) if f["pattern"] == pattern)


def _count_any_increased(old_code: str, new_code: str, *, exclude: str) -> list[str]:
    """Return pattern names whose count *increased* from *old_code* to *new_code*."""
    from agent_core.patterns import analyze
    from collections import Counter
    old_counts = Counter(f["pattern"] for f in analyze(old_code))
    new_counts = Counter(f["pattern"] for f in analyze(new_code))
    increased: list[str] = []
    for p in sorted(set(old_counts) | set(new_counts)):
        if p == exclude:
            continue
        if new_counts.get(p, 0) > old_counts.get(p, 0):
            increased.append(p)
    return increased


def _patch_system_prompt(basename: str) -> str:
    """System prompt for the per-finding patch loop: hunk-only output with
    absolute file line numbers, minimal edits, honest [UNRESOLVED:]."""
    return (
        "You are an expert Python optimizer. Fix the finding in the request.\n\n"
        "Output EXACTLY one block in this format:\n"
        f"[PATCH: {basename}]\n"
        "@@ -10,3 +10,3 @@\n"
        "    unchanged line\n"
        "-    old line\n"
        "+    new line\n\n"
        "For PURE-REMOVAL findings (unused_import, dead_assignment, dead_code), "
        "the fix is just a `-` line — NO `+` replacement and NO `pass`: just delete "
        "the offending line(s). The hunk becomes `@@ -10,1 +10,0 @@` followed by "
        "the `-` line and zero or more context lines.\n\n"
        "Rules:\n"
        "- @@ line numbers are ABSOLUTE, 1-based positions in the whole file — "
        "the number after '-' is the line number of the FIRST line in your hunk "
        "body (the first unchanged/changed line you emit), matching the numbers "
        "printed left of '|' in the numbered context I provide.\n"
        "- In the hunk BODY, every line is the RAW source line (indentation only) "
        "— NOT prefixed with '46 |'. The '46 |' numbers exist only in the numbered "
        "context you are reading for reference.\n"
        "- After the '-' / '+' marker put the ENTIRE line text directly (the "
        "line's own indentation included, no extra padding space).\n"
        "- Context lines MUST be copied VERBATIM from the numbered context. "
        "NEVER use \"unchanged line\", \"...\", or any other placeholder — copy the "
        "actual source line exactly.\n"
        "- The '-' line (old code) is REMOVED and the '+' line REPLACES it — "
        "do NOT keep the old buggy line as context and add a new line next to it. "
        "For single-line replacements keep hunk line counts neutral (as many removed "
        "as changed).  When REMOVING a loop entirely (list_append_join → comprehension), "
        "a net line reduction is expected — the comprehension replaces the whole loop.\n"
        "- When fixing **list_append_join**: the finding is that a list is built "
        "incrementally with .append/.extend/+= per-iteration then consumed by .join(). "
        "The fix must REMOVE the per-iteration build: replace the loop (or loop body) "
        "with a single list comprehension/generator that builds the WHOLE list at once. "
        "Swapping .append(x) for .extend([x]), .extend((x,)), += [x], or any other "
        "single-element incremental call is NOT a fix — it is the same O(n) building "
        "cost.  The fix must eliminate the per-iteration call entirely by restructuring "
        "to a comprehension/generator.\n"
        "- Touch ONLY the code required by the finding. Do NOT reformat "
        "whitespace, docstrings, or unrelated lines. Do NOT rename or move "
        "existing variables.\n"
        "- NEVER modify docstring lines (lines starting with `\"\"\"` or `'''`), "
        "their indentation, or their contents — even if they look misplaced.\n"
        "- Never add or remove import statements.  Exception: adding a stdlib import "
        "such as ``import logging`` is allowed when the finding needs it "
        "(e.g. ``silent_except`` fixes using ``logger.warning(...)``).\n"
        "- When fixing **silent_except**: REPLACE the ``-`` line (the ``pass``) with "
        "a ``+`` line — never add a new line before ``pass`` and leave the dead "
        "``pass`` in place.  Use ``logger.warning(...)`` if logging is already "
        "imported, ``print(f\"...\")`` otherwise.\n"
        + "".join(_PATTERN_GUIDANCE.values())
        + "\n- Never re-emit whole functions or files; never use [FILE:] blocks.\n"
        "- If the finding cannot be fixed without breaking behavior, reply "
        f"with exactly: [UNRESOLVED: {basename}] <one-sentence reason>\n\n"
        "Example (replacement — string_concat_in_loop, the '-' line is "
        "the buggy `cur += \" \" + w` and must be REMOVED; the '+' line replaces "
        "it; the leading @@ number is the first body line, line 45):\n"
        "[PATCH: base.py]\n"
        "@@ -45,3 +45,3 @@\n"
        "        elif len(cur) + 1 + len(w) <= width:\n"
        "-            cur += \" \" + w\n"
        "+            cur = \" \".join([cur, w])\n"
        "        else:\n\n"
        "Example (pure removal — unused_import, dead_assignment: NO + replacement, "
        "NO pass, just delete.  The `@@` second side count is 0):\n"
        "[PATCH: model_cmd.py]\n"
        "@@ -3,1 +3,0 @@\n"
        "-import os\n\n"
        "Example (silent_except — REPLACE the pass: the `-` line is `pass`, "
        "the `+` line is the print/log statement; do NOT keep `pass`):\n"
        "[PATCH: tool.py]\n"
        "@@ -42,2 +42,2 @@\n"
        "            except Exception:\n"
        "-                pass\n"
        "+                logger.warning(\"Failed to scan %s\", path)\n\n"
        "Example (list_append_join — the for-loop + .append() are removed, a single "
        "comprehension line replaces both; line count shrinks because the loop is gone):\n"
        "[PATCH: taskplan_cmd.py]\n"
        "@@ -49,2 +49,1 @@\n"
        "-    for d, names in sorted(taken.items())[:12]:\n"
        "-        lines.append(f\"  {d or 'root'}: {', '.join(names[:12])}\")\n"
        "+    lines += [f\"  {d or 'root'}: {', '.join(names[:12])}\" for d, names in sorted(taken.items())[:12]]\n"
    )



def _format_failures_for_prompt(failures: dict[str, list[dict]]) -> str:
    """Render validation failures as a retry hint for the LLM."""
    if not failures:
        return ""
    blocks: list[str] = []
    for basename, issues in failures.items():
        lines = [f"{basename}:", *(f"  - line {i['line']}: [{i['pattern']}] {i['suggestion']}" for i in issues)]
        blocks.append("\n".join(lines))
    return (
        "The previous attempt produced invalid output for some files. "
        "Regenerate fixing these:\n"
        + "\n\n".join(blocks)
    )


def _affected_original_lines(original: str, candidate: str) -> set[int]:
    """0-based line numbers in *original* affected by the rewrite (diff).

    Insertions/deletions/replacements mark the corresponding original lines as
    affected; aligning via diff keeps the set meaningful even when the LLM
    shift lines around.
    """
    import difflib

    matcher = difflib.SequenceMatcher(None, original.split("\n"), candidate.split("\n"))
    affected: set[int] = set()
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag != "equal":
            affected.update(range(i1, i2))
    return affected


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

    Additionally, a rewrite that changed lines but never touches any line
    listed in the original findings (comment/whitespace-only edits while the
    findings stand) is reported as a ``noop_rewrite`` regression: it is
    evidence the LLM dodged the findings instead of fixing them.
    """
    from collections import Counter

    orig_findings = validate_llm_code(original)
    cand_findings = validate_llm_code(candidate)
    orig_counts = Counter(i["pattern"] for i in orig_findings)
    cand_counts = Counter(i["pattern"] for i in cand_findings)

    regressions: list[dict] = []
    for pat, cnt in cand_counts.items():
        excess = cnt - orig_counts.get(pat, 0)
        if excess > 0:
            surplus = [i for i in cand_findings if i["pattern"] == pat][:excess]
            regressions.extend(surplus)

    if not regressions and orig_findings:
        affected = _affected_original_lines(original, candidate)
        if affected and not (affected & {f["line"] for f in orig_findings}):
            regressions.append({
                "line": sorted(affected)[0],
                "pattern": "noop_rewrite",
                "suggestion": (
                    "The rewrite changed only lines that no listed finding "
                    "targets (comments/whitespace/unrelated edits). Change "
                    "exactly the code the findings point at and nothing else."
                ),
            })
    return regressions


def _surgically_revert_regressions(
    original: str, candidate: str, max_rounds: int = 5
) -> tuple[str, list[dict]]:
    """Deterministically excise regressing lines from a merged rewrite.

    Repair passes re-send the whole file to the LLM, which at low temperature
    tends to reintroduce the same mistakes (run9: a walrus-in-comprehension
    and an added unused import survived 2 repair passes and the merge was
    discarded).  Instead of asking the LLM again, revert *exactly* the lines
    that regress: each candidate line is mapped back to the original line via
    ``difflib`` opcodes (identity for ``equal``, positional for ``replace``)
    and the original content is restored.  A regression on a brand-new
    inserted line is removed outright when the file still compiles without it.

    Returns ``(cleaned_code, residual_regressions)``; lines that cannot be
    reverted safely are left untouched and reported as residual issues for
    the LLM repair pass (or a skip) to handle.
    """
    import difflib

    cand_lines = candidate.splitlines()
    orig_lines = original.splitlines()

    def line_map() -> dict[int, int | None]:
        sm = difflib.SequenceMatcher(None, orig_lines, cand_lines, autojunk=False)
        mapping: dict[int, int | None] = {}
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    mapping[j1 + k] = i1 + k
            elif tag == "replace":
                olen = i2 - i1
                clen = j2 - j1
                if clen <= olen:
                    for k in range(clen):
                        mapping[j1 + k] = i1 + k
        return mapping

    def compiles(text: str) -> bool:
        try:
            compile(text, "<surgical>", "exec")
            return True
        except SyntaxError:
            return False

    reverted: list[int] = []
    removed: list[int] = []
    for _ in range(max_rounds):
        regressions = _blocked_regressions(original, "\n".join(cand_lines))
        if not regressions:
            break
        mapping = line_map()
        changed = False
        for finding in list(regressions):
            line = finding["line"] - 1
            if line >= len(cand_lines):
                continue
            orig_line = mapping.get(line)
            if orig_line is not None and orig_line < len(orig_lines):
                cand_lines[line] = orig_lines[orig_line]
                reverted.append(finding["line"])
                changed = True
                continue
            candidate_without = cand_lines[:line] + cand_lines[line + 1:]
            if compiles("\n".join(candidate_without)):
                cand_lines = candidate_without
                removed.append(finding["line"])
                changed = True
        if not changed:
            break

    cleaned = "\n".join(cand_lines)
    residual = _blocked_regressions(original, cleaned)
    if reverted or removed:
        print(
            f"    Surgically reverted {len(reverted)} regression line(s), "
            f"removed {len(removed)} inserted line(s)"
            + (f"; {len(residual)} left for repair" if residual else " — merge clean")
        )
    return cleaned, residual


async def _repair_merged_file(
    agent: "Agent",
    fpath: str,
    original: str,
    merged: str,
    issues: list[dict],
    max_retries: int,
    context_tokens: int | None = None,
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
                    estimate_tokens(user_content) + 300, context_tokens
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
            for a in sorted(node.names, key=lambda a: a.name):
                if a.name == "*":
                    entries.add(f"from {module} import *")
                else:
                    entries.add(f"from {module} import {a.name}")
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
            for a in sorted(node.names, key=lambda a: a.name):
                if a.name == "*":
                    entry = f"from {module} import *"
                else:
                    entry = f"from {module} import {a.name}"
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
    _RE_2 = re.compile(r"^(?:import|from)\s+([A-Za-z_]\w*)")
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
        match = _RE_2.match(entry)
        top = match.group(1) if match else None
        if top is not None and top in _STDLIB_MODULES:
            blocked.discard(entry)
    return blocked


def _undefined_hoisted_names(code: str) -> list[str]:
    """Return undefined underscore names (hoisted regex/constants) that would
    cause NameError at runtime. Avoids common false positives: imports,
    function/class defs, and module-level dunders (__name__, __file__, …)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            defined.update(node.names)
    undefined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id.startswith("_") and node.id not in defined and not (node.id.startswith("__") and node.id.endswith("__")):
                undefined.add(node.id)
    return sorted(undefined)


def _regresses_defined_names(original: str, patched: str) -> list[str]:
    """Return non-underscore names loaded in *patched* but no longer stored,
    yet stored in *original* — these cause NameError at runtime."""
    try:
        o_tree = ast.parse(original)
        p_tree = ast.parse(patched)
    except SyntaxError:
        return []
    o_stored: set[str] = set()
    for node in ast.walk(o_tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            o_stored.add(node.id)
        elif isinstance(node, ast.arg):
            o_stored.add(node.arg)
    p_stored: set[str] = set()
    p_loaded: set[str] = set()
    for node in ast.walk(p_tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                p_stored.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                p_loaded.add(node.id)
        elif isinstance(node, ast.arg):
            p_stored.add(node.arg)
    regressed = p_loaded - p_stored
    regressed &= o_stored
    regressed -= {"__name__", "__file__", "__doc__"}
    return sorted(regressed)


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

        # Probe LM Studio once per run before spending any LLM time: gives the
        # loaded model's context window (for token budgeting) and an early
        # heads-up when the server is down (a dead server used to eat retry
        # loops).  The probe is advisory — hard-aborting here would break
        # offline unit runs that mock the LLM, so chat errors still carry the
        # failure themselves.
        model_ctx = _loaded_model_context()
        if model_ctx is None:
            print("  ⚠ LM Studio unreachable / no loaded model with a context length — "
                  "heuristic token budgets are in effect (large regions may fail).")

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

            # One targeted LLM call per finding: the model only ever sees the
            # enclosing block (absolute numbered lines) and must answer with
            # hunks, so untargeted code cannot drift and no-op "rewrites" are
            # impossible by construction.
            for fpath in batch["files"]:
                basename = os.path.basename(fpath)
                if not batch["findings"][fpath]:
                    continue
                original = batch["contents"][fpath]
                print(f"  {basename}: {len(batch['findings'][fpath])} finding(s) — targeted patches")

                work_lines = original.split("\n")
                pending = list(batch["findings"][fpath])
                while pending:
                    finding = pending.pop(0)
                    line, pattern = finding["line"], finding["pattern"]
                    context = _finding_context("\n".join(work_lines), line)
                    if context is None:
                        print(f"    line {line} [{pattern}] — context unavailable (line out of range), skipped")
                        continue
                    feedback = ""
                    resolved = False
                    # ── deterministic mechanical fix (no LLM) ────────────
                    mech_raw = _tri_mech_fix(work_lines, line, pattern, basename, finding)
                    if mech_raw is not None:
                        ok, patched = apply_patch(mech_raw, work_lines)
                        if not ok:
                            ok, patched = apply_anchored_patch(mech_raw, work_lines)
                        if ok:
                            try:
                                compile(patched, "<mechanical>", "exec")
                            except SyntaxError:
                                ok = False
                            if ok and _undefined_hoisted_names(patched):
                                ok = False
                            if ok and _count_any_increased("\n".join(work_lines), patched, exclude=pattern):
                                ok = False
                            if ok:
                                before_cnt = _count_pattern("\n".join(work_lines), pattern)
                                after_cnt = _count_pattern(patched, pattern)
                                if after_cnt < before_cnt:
                                    new_lines = patched.split("\n")
                                    while new_lines and new_lines[-1] == "":
                                        new_lines.pop()
                                    line_delta = patched.count("\n") - len(work_lines)
                                    work_lines = new_lines
                                    if line_delta:
                                        for pf in pending:
                                            if pf["line"] > line:
                                                pf["line"] += line_delta
                                    resolved = True
                                    print(f"    Fixed line {line} [{pattern}] (mechanical, {before_cnt} -> {after_cnt} remaining)")
                    if resolved:
                        continue
                    _resync = static_analyze("\n".join(work_lines))
                    if not any(f["line"] == line and f["pattern"] == pattern for f in _resync):
                        continue
                    # ── LLM patch loop ──────────────────────────────────
                    _PATCH_RE = re.compile(
                        rf"\[PATCH:\s*{re.escape(basename)}\s*\](.*?)(?=\[PATCH:|\Z)", re.DOTALL
                    )
                    for attempt in range(FINDING_MAX_ATTEMPTS):
                        if attempt > 0:
                            print(f"    Retry {attempt}/{FINDING_MAX_ATTEMPTS - 1} for line {line}...")
                        user_content = (
                            "## Finding to resolve\n"
                            f"{basename}:{line} [{pattern}] {finding['suggestion']}\n\n"
                            "## Numbered context (numbers ARE the absolute file line numbers)\n"
                            f"```\n{context}\n```\n\n"
                            "Output exactly one [PATCH: " + basename + "] block per the system rules."
                        )
                        if feedback:
                            user_content = feedback + "\n\n" + user_content
                        try:
                            llm_response = await agent.llm.chat(
                                [
                                    {"role": "system", "content": _patch_system_prompt(basename)},
                                    {"role": "user", "content": user_content},
                                ],
                                max_tokens=min(
                                    REGION_PATCH_MAX_TOKENS,
                                    _request_max_tokens(
                                        estimate_tokens(user_content) + 300, model_ctx
                                    ),
                                ),
                                disable_thinking=True,
                            )
                        except Exception as exc:
                            print(f"    Error: {exc}")
                            resolved = False
                            break
                        if llm_response.startswith("[Error") or llm_response.startswith("[LM Studio"):
                            print(f"    LLM error: {llm_response[:100]}")
                            resolved = False
                            break
                        unresolved = _UNRESOLVED_RE.search(llm_response)
                        if unresolved:
                            print(f"    Unresolved line {line} [{pattern}]: "
                                  f"{unresolved.group(1).strip() or 'model reported un-resolvable'}")
                            resolved = False
                            break
                        patch_block = _PATCH_RE.search(llm_response)
                        raw = patch_block.group(1) if patch_block else ""
                        raw = _strip_markdown_fence(raw)
                        if not raw.strip():
                            # Model may have produced a hunk without the
                            # [PATCH:] tag — treat the whole response body.
                            raw = _strip_markdown_fence(llm_response)
                        if not raw.strip():
                            feedback = (
                                "Your response contained no [PATCH: " + basename +
                                "] block. Output ONLY the hunk block; no prose, "
                                "no [FILE:] blocks."
                            )
                            print(f"    Feedback: {feedback[:240]}")
                            continue
                        placeholder_warn = any(p in raw.lower() for p in _PLACEHOLDER_PHRASES)
                        if placeholder_warn:
                            feedback = ("NEVER use placeholder text. Copy actual source lines from the numbered context. "
                                        + (feedback if feedback else ""))
                        ok, patched = apply_patch(raw, work_lines)
                        if not ok:
                            ok, patched = apply_anchored_patch(raw, work_lines)
                        if not ok:
                            # Debug: show the model output that couldn't be parsed.
                            if attempt == 0:
                                print(f"    Raw model output: {raw[:300]}")
                            snippet = "\n".join(
                                f"  {i + 1:4d} | {l}"
                                for i, l in enumerate(work_lines[max(0, line - 4):line + 3], start=max(0, line - 4))
                            )
                            feedback = (
                                f"Your patch could not be applied: {patched[:200]}\n"
                                f"Here is the actual numbered context:\n{snippet}\n"
                                "Copy these EXACTLY for context/'-' lines. Use correct @@ numbers."
                            )
                            if placeholder_warn or any(p in raw.lower() for p in _PLACEHOLDER_PHRASES):
                                feedback = "NEVER use placeholder text like \"unchanged line\". Copy actual source lines from the numbered context. " + feedback
                            if attempt < FINDING_MAX_ATTEMPTS - 1:
                                print(f"    Feedback: {feedback[:600]}")
                                continue
                            resolved = False
                            break
                        new_lines = patched.split("\n")
                        if _cosmetic_only(work_lines, new_lines):
                            feedback = (
                                "Your patch only changed indentation/whitespace (or nothing). "
                                f"The finding at line {line} [{pattern}] must actually change. "
                                "Output hunks that modify the code the finding targets."
                            )
                            if attempt < FINDING_MAX_ATTEMPTS - 1:
                                print(f"    Feedback: {feedback[:240]}")
                                continue
                            resolved = False
                            break
                        # Nullification guard: replacing functional code with pass
                        idx = line - 1
                        if any(
                            0 <= idx + off < len(work_lines)
                            and 0 <= idx + off < len(new_lines)
                            and any(kw in work_lines[idx + off] for kw in (".append(", ".write(", ".setdefault("))
                            and _MECH_PASS_RE.match(new_lines[idx + off])
                            for off in range(-3, 4)
                        ):
                            feedback = (
                                "You replaced a functional line (e.g. .append/.write) with `pass` — "
                                "this destroys the code. Convert it properly (list comprehension, "
                                "generator, etc.), do NOT nullify it with `pass`."
                            )
                            if attempt < FINDING_MAX_ATTEMPTS - 1:
                                print(f"    Feedback: {feedback[:240]}")
                                continue
                            resolved = False
                            break
                        try:
                            compile(patched, "<optimize:patched>", "exec")
                        except SyntaxError as exc:
                            feedback = (
                                f"Your patched file does not compile: {exc.msg} (line {exc.lineno}). "
                                "Regenerate valid Python."
                            )
                            if attempt < FINDING_MAX_ATTEMPTS - 1:
                                print(f"    Feedback: {feedback[:240]}")
                                continue
                            resolved = False
                            break
                        undefined = _undefined_hoisted_names(patched)
                        if undefined:
                            feedback = (
                                f"Your patch references undefined names: {', '.join(undefined)}. "
                                "Define these names or restore the original code."
                            )
                            if attempt < FINDING_MAX_ATTEMPTS - 1:
                                print(f"    Feedback: {feedback[:240]}")
                                continue
                            resolved = False
                            break
                        dropped_names = _regresses_defined_names(original, patched)
                        if dropped_names:
                            feedback = (
                                f"Your patch removed the definition of: {', '.join(dropped_names)}. "
                                "Restore the original definitions or keep the old code unchanged."
                            )
                            if attempt < FINDING_MAX_ATTEMPTS - 1:
                                print(f"    Feedback: {feedback[:240]}")
                                continue
                            resolved = False
                            break
                        added = _blocked_added_imports(patched, "\n".join(work_lines))
                        if added:
                            feedback = (
                                f"Your patch changed the import set: {', '.join(sorted(added))[:160]}. "
                                "NEVER add or remove import statements — keep the original imports untouched."
                            )
                            if attempt < FINDING_MAX_ATTEMPTS - 1:
                                print(f"    Feedback: {feedback[:240]}")
                                continue
                            resolved = False
                            break
                        # Acceptance: the pattern's footprint must shrink.  Count
                        # occurrences before/after; a finding is resolved only when
                        # the patched file reports fewer instances of the pattern.
                        before_cnt = _count_pattern("\n".join(work_lines), pattern)
                        after_cnt = _count_pattern(patched, pattern)
                        if after_cnt >= before_cnt:
                            feedback = (
                                f"After your patch the file still contains [{pattern}] "
                                f"({after_cnt} occurrence(s), {before_cnt} before). The patch must "
                                "actually remove the finding - change exactly the code it targets."
                            )
                            if attempt < FINDING_MAX_ATTEMPTS - 1:
                                print(f"    Feedback: {feedback[:240]}")
                                continue
                            resolved = False
                            break
                        regressed = _count_any_increased("\n".join(work_lines), patched, exclude=pattern)
                        if regressed:
                            feedback = (
                                f"Your patch reduced [{pattern}] but introduced or increased: "
                                f"{', '.join(sorted(regressed))[:160]}. "
                                "Refactor without introducing new static-analysis issues."
                            )
                            if attempt < FINDING_MAX_ATTEMPTS - 1:
                                print(f"    Feedback: {feedback[:240]}")
                                continue
                            resolved = False
                            break
                        # Scope gate: reject patches whose hunks wander outside
                        # the finding's vicinity (e.g. adding trailing blank lines).
                        out_of_scope: list[int] = []
                        for hunk_start, _ in split_patch_hunks(raw):
                            if abs(hunk_start - line) > SCOPE_TOLERANCE:
                                out_of_scope.append(hunk_start)
                        if out_of_scope:
                            feedback = (
                                f"Your patch changed line(s) near {out_of_scope} which are far "
                                f"from the finding at line {line}. Only modify the target area "
                                f"(±{SCOPE_TOLERANCE} lines). Strip unrelated hunks and re-emit only "
                                "the fix for this finding."
                            )
                            if attempt < FINDING_MAX_ATTEMPTS - 1:
                                print(f"    Feedback: {feedback[:240]}")
                                continue
                            resolved = False
                            break
                        new_lines = patched.split("\n")
                        line_delta = patched.count("\n") - len(work_lines)
                        # apply_patch always appends \n to every line, so if the
                        # original file has a trailing newline the patched result
                        # ends with \n\n.  splitlines() (used by show_file_diff)
                        # sees the inner \n\n as an extra empty "line".  Strip all
                        # trailing empties, then add exactly one back if the
                        # original file had a trailing newline.
                        while new_lines and new_lines[-1] == "":
                            new_lines.pop()
                        if original.endswith("\n"):
                            new_lines.append("")
                        work_lines = new_lines
                        if line_delta:
                            for pf in pending:
                                if pf["line"] > line:
                                    pf["line"] += line_delta
                        resolved = True
                        print(f"    Fixed line {line} [{pattern}] (hunk applied, "
                              f"{before_cnt} -> {after_cnt} remaining)")
                        break
                    if resolved:
                        continue
                    print(f"  Unresolved: {basename}:{line} [{pattern}]")

                new_full = "\n".join(work_lines)
                if new_full != original:
                    all_fixes[basename] = new_full
                    print(f"  {basename}: patched ({len(original)} -> {len(new_full)} bytes)")
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