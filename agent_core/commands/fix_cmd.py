"""Fix command for agent interactive mode."""
import ast
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .base import Command, show_file_diff, save_file_py, read_choice, stop_requested
from agent_core.decisions import decisions_as_system_prompt, extract_from_changes, add_decision
from .implement_cmd import (
    _apply_patch as _impl_apply_patch,
    _classify_error,
    _extract_window,
    _parse_line_number,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


def _is_stdlib_path(p: str) -> bool:
    """True when *p* lives under the Python installation (stdlib or site-packages)."""
    prefixes = [sys.prefix, sys.base_prefix]
    _base = getattr(sys, "_base_executable", None)
    if _base:
        prefixes.append(_base)
    norm = os.path.normpath(p).lower()
    return any(norm.startswith(os.path.normpath(pr).lower()) for pr in prefixes if pr)


def _is_trackable_file(p: str) -> bool:
    """A real file that we are allowed to modify (not frozen, not stdlib)."""
    if p.startswith("<"):
        return False
    if not os.path.exists(p):
        return False
    if _is_stdlib_path(p):
        return False
    return True


def _max_identical_run(text: str) -> int:
    """Longest run of IDENTICAL consecutive non-blank lines in *text*.

    Blank lines are excluded so a patch that legitimately adds spacing does
    not look like corruption.  Pure comment lines ARE counted — a run of 45
    identical ``# TODO`` comments is exactly the corruption signature we
    want to catch.
    """
    longest = 0
    prev: str | None = None
    run = 0
    for raw in text.split('\n'):
        line = raw.rstrip('\r').strip()
        if not line:
            run = 0
            prev = None
            continue
        if line == prev:
            run += 1
            if run > longest:
                longest = run
        else:
            run = 1
            prev = line
            if run > longest:
                longest = run
    return longest


def _looks_corrupted(original: str, result: str) -> str | None:
    """Return a human reason when *result* looks like a mis-applied patch.

    Two signatures of anchor/positional mis-application that the
    syntax-only gate (``compile``) cannot catch:

    * **duplicate-run explosion** — a comment or line was repeated many
      times (e.g. the 45x-TODO corruption).  Reject when the longest
      identical run in *result* is both >8 and >2x the longest run in
      *original*.
    * **runaway growth** — a [PATCH:] that doubles the file is almost
      never a legitimate mypy fix.  Reject when *result* has >2x the
      non-blank line count of *original*.
    """
    res_run = _max_identical_run(result)
    orig_run = _max_identical_run(original)
    if res_run > 8 and res_run > 2 * max(orig_run, 1):
        return (f"result introduces a {res_run}-long identical-line run "
                f"(original max {orig_run}) — likely a mis-anchored patch")
    orig_n = sum(1 for ln in original.split('\n') if ln.strip())
    res_n = sum(1 for ln in result.split('\n') if ln.strip())
    if orig_n > 20 and res_n > 2 * orig_n:
        return (f"result grew {orig_n}->{res_n} non-blank lines (>2x) — "
                f"rejecting runaway patch")
    return None


# ---------------------------------------------------------------------------
# Deterministic (non-LLM) mypy fixers
# ---------------------------------------------------------------------------
# Each of these transforms is provably safe for the error subclass it targets.
# They are run BEFORE the LLM ladder so the flaky patch path only has to deal
# with the errors that genuinely need judgment (missing return statement,
# untyped-call into third-party code, element types for bare tuple/dict, ...).
# Every transform re-validates with ``compile``; the caller re-runs mypy
# afterwards, so a wrong guess is rolled back by the normal verification.

_TYPE_IGNORE_RE = re.compile(r'\s*#\s*type:\s*ignore(?:\[[^\]]*\])?\s*$')


def _collapse_duplicate_runs(text: str, threshold: int = 5) -> tuple[str, int]:
    """Collapse runs of > *threshold* identical consecutive lines to one.

    Returns ``(new_text, collapsed_runs)``.  Blank-line runs are left alone so
    we never normalise spacing.  This is the mechanical repair for the
    corruption seen in the wild (a ``# TODO`` comment copy-pasted 45x by a
    mis-anchored patch).
    """
    lines = text.split('\n')
    out: list[str] = []
    i = 0
    runs = 0
    n = len(lines)
    while i < n:
        j = i + 1
        while j < n and lines[j] == lines[i]:
            j += 1
        run_len = j - i
        if run_len > threshold and lines[i].strip():
            out.append(lines[i])
            runs += 1
        else:
            out.extend(lines[i:j])
        i = j
    return '\n'.join(out), runs


def _syntax_ok(source: str) -> bool:
    try:
        compile(source, "<mechanical>", "exec")
        return True
    except SyntaxError:
        return False


def _fix_unused_ignore(lines: list[str], lineno: int) -> list[str] | None:
    """Drop a trailing ``# type: ignore`` directive on *lineno* (1-indexed)."""
    idx = lineno - 1
    if idx < 0 or idx >= len(lines):
        return None
    m = _TYPE_IGNORE_RE.search(lines[idx])
    if not m:
        return None
    new_line = lines[idx][:m.start()].rstrip()
    if not new_line.strip():
        return lines[:idx] + lines[idx + 1:]  # whole line was the comment
    return lines[:idx] + [new_line] + lines[idx + 1:]


def _fix_redundant_cast(lines: list[str], lineno: int, typ: str) -> list[str] | None:
    """Replace ``cast(typ, expr)`` with ``expr`` on *lineno* (1-indexed)."""
    idx = lineno - 1
    if idx < 0 or idx >= len(lines):
        return None
    line = lines[idx]
    m = re.search(r'\bcast\s*\(\s*', line)
    if not m:
        return None
    rest = line[m.end():]
    depth = 1
    i = 0
    type_end: int | None = None
    while i < len(rest):
        c = rest[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                break
        elif c == ',' and depth == 1:
            type_end = i
            break
        i += 1
    if type_end is None:
        return None
    if rest[:type_end].strip() != typ:
        return None
    arg_start = type_end + 1
    depth = 1
    j = arg_start
    arg_end: int | None = None
    while j < len(rest):
        c = rest[j]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                arg_end = j
                break
        j += 1
    if arg_end is None:
        return None
    arg = rest[arg_start:arg_end].strip()
    new_line = line[:m.start()] + arg + rest[arg_end + 1:]
    return lines[:idx] + [new_line] + lines[idx + 1:]


def _fix_implicit_optional(lines: list[str], lineno: int) -> list[str] | None:
    """``name: T = None`` -> ``name: T | None = None`` on *lineno* (1-indexed).

    Only when *T* is a plain annotation without an existing ``| None`` or
    ``Optional[...]`` wrapper (mypy would not have flagged those).
    """
    idx = lineno - 1
    if idx < 0 or idx >= len(lines):
        return None
    line = lines[idx]
    m = re.search(
        r'(\b\w+\s*:\s*)([A-Za-z_][\w.]*(?:\[[^\]]*\])?)\s*=\s*None\b',
        line,
    )
    if not m:
        return None
    type_str = m.group(2)
    if type_str.endswith('None') or 'Optional' in type_str:
        return None
    new_line = line[:m.end(2)] + ' | None' + line[m.end(2):]
    return lines[:idx] + [new_line] + lines[idx + 1:]


def _enclosing_function_name(source: str, lineno: int) -> str | None:
    """Name of the function whose body contains *lineno*, or None."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    enclosing: str | None = None
    best_start: int = -1
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = node.end_lineno or start
            if start <= lineno <= end and start > best_start:
                best_start = start
                enclosing = node.name
    return enclosing


def _fix_attr_defined_rename(
    lines: list[str], lineno: int, full_source: str, bad: str, suggested: str
) -> list[str] | None:
    """Rename ``<obj>.bad`` to ``<obj>.suggested`` on *lineno* when safe.

    Only applied when *suggested* is actually a ``def`` in *full_source* AND
    the rename would not turn a call inside the ``def suggested`` body into a
    recursive call (the mypy "maybe '_tool_x'?" hint fires for delegation
    methods whose target was mis-spelled — renaming there just creates
    infinite recursion, so leave those for the LLM).
    """
    idx = lineno - 1
    if idx < 0 or idx >= len(lines):
        return None
    if not re.search(r'\bdef\s+' + re.escape(suggested) + r'\s*\(', full_source):
        return None
    if _enclosing_function_name(full_source, lineno) == suggested:
        return None  # would be self-recursion — skip
    if not re.search(r'\.' + re.escape(bad) + r'\b', lines[idx]):
        return None
    new_line = re.sub(
        r'(\.)' + re.escape(bad) + r'\b', r'\1' + suggested, lines[idx]
    )
    return lines[:idx] + [new_line] + lines[idx + 1:]


def _function_returns_value(source: str, def_lineno: int) -> bool:
    """True if the function starting on *def_lineno* has a ``return <expr>``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True  # do not risk adding a wrong ``-> None``

    def returns_value(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        def _scan(body: list[ast.stmt]) -> bool:
            for stmt in body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    if not (isinstance(stmt.value, ast.Constant) and stmt.value.value is None):
                        return True
                body_attr = getattr(stmt, 'body', None)
                if isinstance(body_attr, list) and _scan(body_attr):
                    return True
            return False
        return _scan(node.body)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno == def_lineno:
            return returns_value(node)
    return True


def _fix_missing_return_none(lines: list[str], def_lineno: int) -> list[str] | None:
    """Append ``-> None`` to the function signature on *def_lineno* when the
    body has no value-returning ``return``.
    """
    idx = def_lineno - 1
    if idx < 0 or idx >= len(lines):
        return None
    def_line = lines[idx]
    stripped = def_line.lstrip()
    if not (stripped.startswith('def ') or stripped.startswith('async def ')):
        return None
    indent = len(def_line) - len(def_line.lstrip())
    if def_line.rstrip().endswith(':'):
        close_idx = idx
    else:
        close_idx = -1
        for j in range(idx + 1, len(lines)):
            cand = lines[j]
            cstrip = cand.strip()
            cindent = len(cand) - len(cand.lstrip())
            if cstrip.endswith(':') and (cindent <= indent or cstrip[:1] in ')]}'):
                close_idx = j
                break
        if close_idx < 0:
            return None
    sig_text = '\n'.join(lines[idx:close_idx + 1])
    if '->' in sig_text:
        return None
    source_for_check = '\n'.join(lines)
    if _function_returns_value(source_for_check, def_lineno):
        return None
    close_line = lines[close_idx]
    colon = close_line.rfind(':')
    new_close = close_line[:colon] + ' -> None' + close_line[colon:]
    return lines[:close_idx] + [new_close] + lines[close_idx + 1:]


def _parse_mypy_error(err: str) -> tuple[int, str] | None:
    """Extract ``(line_number, error_code)`` from a mypy error line."""
    m = re.match(r'^(.*?):(\d+): error: (.*?)\s*\[([a-z-]+)\]\s*$', err.strip())
    if not m:
        return None
    return int(m.group(2)), m.group(4)


def extract_signatures(source: str) -> dict:
    """Extract function/class signatures from Python source."""
    sigs = {}
    for m in re.finditer(r'^class\s+(\w+)\s*(?:\((.*?)\))?\s*:', source, re.MULTILINE):
        cls_name = m.group(1)
        bases = m.group(2).strip() if m.group(2) else ""
        sigs[cls_name] = f"class {cls_name}({bases})" if bases else f"class {cls_name}"

    for m in re.finditer(r'^\s+def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*(.+?))?\s*:', source, re.MULTILINE):
        func_name = m.group(1)
        params = m.group(2).strip() if m.group(2) else ""
        returns = m.group(3).strip() if m.group(3) else ""
        sig = f"{func_name}({params})"
        if returns:
            sig += f" -> {returns}"
        sigs[func_name] = sig
    return sigs


class FixCommand(Command):
    """Fix code errors from traceback or description."""

    @property
    def name(self) -> str:
        return "fix"

    @property
    def help_text(self) -> str:
        return 'fix "<traceback>" | <file> --desc "issue" [--full] | --mypy [path...] - Fix code errors'

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        parts = args
        
        # Handle --stdin flag
        stdin_mode = "--stdin" in parts
        if stdin_mode:
            parts = [p for p in parts if p != "--stdin"]
            from .base import read_stdin
            stdin_text = read_stdin("Paste problem description or traceback. Type --- on its own line when done, or Ctrl+Z to finish:")
            if not stdin_text.strip():
                self.error("No input provided")
                return True
            # If --desc not in parts, use stdin as description
            if "--desc" not in parts:
                parts.extend(["--desc", stdin_text.strip()])

        if "--desc" in parts:
            di = parts.index("--desc")
            desc_text = parts[di + 1].strip('"') if di + 1 < len(parts) else ""

            target_file = None
            for p in parts:
                if not p.startswith("--") and p != desc_text:
                    target_file = p
                    break

            if not desc_text:
                self.error('Usage: fix <file> --desc "describe what\'s wrong"')
                return True

            if target_file:
                target_file = os.path.abspath(target_file)
                if not os.path.exists(target_file):
                    self.error(f"Target not found: {target_file}")
                    return True
                if os.path.isdir(target_file):
                    ws_dir = target_file
                    target_file = None
                else:
                    ws_dir = str(Path(target_file).parent)
            else:
                ws_dir = os.path.abspath(".")

            print(f"\nAnalyzing project in {ws_dir}...")
            print(f"Problem: {desc_text[:120]}...")

            candidate_files = set()

            if target_file and os.path.isfile(target_file):
                candidate_files.add(target_file)
            elif not target_file:
                for f in os.listdir(ws_dir):
                    fp = os.path.join(ws_dir, f)
                    if f.endswith(".py") and os.path.isfile(fp):
                        candidate_files.add(fp)
                        print(f"  Seed: {f}")

            _IMPORT_FROM_RE = re.compile(r'from\s+(\S+)\s+import\s+')
            def get_imported_files(filepath):
                result = set()
                try:
                    with open(filepath, "r") as f:
                        content = f.read()
                    for match in _IMPORT_FROM_RE.finditer(content):
                        module = match.group(1)
                        path = module.replace('.', os.sep) + '.py'
                        for search_dir in [ws_dir, str(Path(filepath).parent)]:
                            full = os.path.join(search_dir, path)
                            if os.path.isfile(full):
                                result.add(os.path.normpath(full))
                                break
                except Exception as exc:
                    print(f"  Warning: failed to parse imports in {filepath}: {exc}")
                return result

            for f in list(candidate_files):
                candidate_files |= get_imported_files(f)
            for f in list(candidate_files):
                candidate_files |= get_imported_files(f)

            keywords = {w.lower() for w in re.findall(r'\w+', desc_text) if len(w) > 3} - {'this', 'that', 'with', 'from', 'they', 'have', 'what', 'when', 'then', 'than', 'show', 'just', 'like'}

            _TASK_FILE_RE = re.compile(r'`([^`]+\.py)`')
            resp_matched = set()
            tasks_md_path = os.path.join(ws_dir, "project_tasks.md")
            if os.path.exists(tasks_md_path):
                current_file = None
                with open(tasks_md_path, "r", encoding="utf-8") as tf:
                    for line in tf:
                        m = _TASK_FILE_RE.search(line)
                        if m:
                            current_file = m.group(1)
                        elif current_file and line.strip().startswith('-'):
                            task_text = line.strip('- ').strip().lower()
                            if any(kw in task_text for kw in keywords):
                                fp = os.path.normpath(os.path.join(ws_dir, current_file))
                                if os.path.isfile(fp):
                                    resp_matched.add(fp)
                                    print(f"  Responsibility match: {current_file} -> '{task_text[:80]}'")
                if resp_matched:
                    candidate_files |= resp_matched
                    print(f"  + {len(resp_matched)} files from project_tasks.md responsibility matching")

            plan_md_path = os.path.join(ws_dir, "project_plan.md")
            if os.path.exists(plan_md_path):
                plan_matched = set()
                with open(plan_md_path, "r", encoding="utf-8") as pf:
                    plan_text = pf.read()
                for root, dirs, files in os.walk(ws_dir):
                    for f in files:
                        if f.endswith(".py") and f != "__init__.py":
                            fp = os.path.normpath(os.path.join(root, f))
                            rel = os.path.relpath(fp, ws_dir).replace("\\", "/")
                            idx = plan_text.find(rel)
                            if idx > 0:
                                snippet = plan_text[max(0, idx-100):idx+200].lower()
                                if any(kw in snippet for kw in keywords):
                                    plan_matched.add(fp)
                                    print(f"  Plan match: {rel}")
                if plan_matched:
                    candidate_files |= plan_matched

            print(f"  Tracing imports: {len(candidate_files)} relevant files")

            full_mode = "--full" in parts

            if full_mode:
                all_source = "## Project structure\n\n"
                py_files = []
                sig_map = {}
                for root, dirs, files in os.walk(ws_dir):
                    if ".git" in root or "__pycache__" in root:
                        continue
                    for f in files:
                        if not f.endswith(".py") or f == "__init__.py":
                            continue
                        fp = os.path.normpath(os.path.join(root, f))
                        rel = os.path.relpath(fp, ws_dir).replace("\\", "/")
                        py_files.append(fp)
                        with open(fp, "r", encoding="utf-8") as sf:
                            content = sf.read()
                        if fp in candidate_files:
                            all_source += f"\n\n# === {fp} ===\n{content}"
                        else:
                            try:
                                sigs = extract_signatures(content)
                                if sigs:
                                    sig_map[rel] = ", ".join(f"{n}" for n in sorted(sigs.keys())[:8])
                            except Exception as exc:
                                print(f"  Warning: failed to extract signatures from {fp}: {exc}")
                all_source += f"\n\n## Other project files (signatures only, {len(sig_map)} total)\n\n"
                for rel, sigs in sorted(sig_map.items()):
                    all_source += f"  {rel}: {sigs}\n"
                print(f"  Collected {len(py_files)} Python files ({len(all_source)} bytes)")
                msgs = [
                    {"role": "system", "content": "You are an expert Python debugger. Analyze the codebase below. Fix ALL files needed. Keep code concise. NEVER create duplicate functions or classes (_v1, _v2, _clean, _final variants). One implementation per concept.\n\nPrefer [PATCH:] format (minimal diff — only the lines that change):\n[PATCH: path/to/file.py]\n@@ -10,3 +10,2 @@\n- old line\n+ new line\n- old line\n\nOnly use [FILE:] for new files or when the entire file must be rewritten:\n[FILE: absolute/path/to/file.py]\n```python\n# complete fixed code\n```"},
                    {"role": "user", "content": f"The user reports this issue:\n\n{desc_text}\n\nFull project codebase:\n\n{all_source}\n\nAnalyze the issue, find the root cause, and fix ALL affected files. Output each fixed file with its full path."}
                ]
                print("Sending to LLM for deep analysis...")
                response = await agent.llm.chat(msgs)
                self._apply_fix_response(response, ws_dir, desc_text)
                return True

            # ---- On-demand path (default) ----

            # Score candidate files by keyword relevance
            scored = []
            for fp in candidate_files:
                if not os.path.isfile(fp) or not fp.endswith(".py"):
                    continue
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    continue
                score = sum(1 for kw in keywords if kw in content.lower())
                scored.append((fp, score, content))
            scored.sort(key=lambda x: -x[1])

            top_count = min(5, len(scored))
            top_files = scored[:top_count]
            rest_files = scored[top_count:]

            # Build signature map for ALL project files (for reference)
            sig_map = {}
            for root, dirs, files in os.walk(ws_dir):
                if ".git" in root or "__pycache__" in root:
                    continue
                for f in files:
                    if not f.endswith(".py") or f == "__init__.py":
                        continue
                    fp = os.path.normpath(os.path.join(root, f))
                    rel = os.path.relpath(fp, ws_dir).replace("\\", "/")
                    # Skip files already in top_files (we have full source)
                    if fp in {t[0] for t in top_files}:
                        continue
                    with open(fp, "r", encoding="utf-8") as sf:
                        content = sf.read()
                    try:
                        sigs = extract_signatures(content)
                        if sigs:
                            sig_map[rel] = ", ".join(f"{n}" for n in sorted(sigs.keys())[:8])
                    except Exception as exc:
                        print(f"  Warning: failed to extract signatures from {fp}: {exc}")

            # Build initial context: full source for top-N, signatures for rest
            context = f"## Issue\n{desc_text}\n\n"
            context += "## Relevant files (full source — highest keyword match)\n"
            for fp, score, content in top_files:
                rel = os.path.relpath(fp, ws_dir).replace("\\", "/")
                context += f"\n# === {rel} (relevance: {score} keywords) ===\n{content}\n"

            if rest_files:
                context += f"\n## Other candidate files ({len(rest_files)} more — signatures only)\n"
                for fp, score, content in rest_files[:15]:
                    rel = os.path.relpath(fp, ws_dir).replace("\\", "/")
                    sigs = extract_signatures(content)
                    names = ", ".join(sorted(sigs.keys())[:8]) if sigs else "(empty)"
                    context += f"  {rel}  [{names}]\n"

            context += f"\n## Other project files ({len(sig_map)} total — signatures only)\n"
            for rel, names in sorted(sig_map.items())[:20]:
                context += f"  {rel}: {names}\n"
            if len(sig_map) > 20:
                context += f"  ... and {len(sig_map) - 20} more files.\n"

            print(f"  On-demand: {len(top_files)} full files + {len(rest_files)} candidate sigs + {len(sig_map)} other sigs ({len(context)} bytes)")
            print(f"  Full source: {', '.join(os.path.basename(fp) for fp, _, _ in top_files)}")
            if rest_files:
                print(f"  Candidates (sigs only): {', '.join(os.path.basename(fp) for fp, _, _ in rest_files[:8])}", end="")
                if len(rest_files) > 8:
                    print(f" ... +{len(rest_files) - 8} more", end="")
                print()

            read_paths: set[str] = {fp for fp, _, _ in top_files}
            system = ("You are an expert Python debugger.\n\n"
                      f"WORKSPACE: {ws_dir}\n"
                      "Files below use paths RELATIVE to the workspace.\n\n"
                      "FORMAT (plain text only — NO XML or <tool_call> tags):\n"
                      "  To view a file:  [READ: <relative_path>]\n"
                      "  To submit a fix: [FILE: <relative_path>]\n"
                      "    ```python\n    # complete corrected code here\n    ```\n\n"
                      "Use EXACTLY the relative filenames shown in the 'Relevant files' section above.\n"
                      "Do NOT add directory prefixes that aren't already shown.\n"
                      "Do NOT wrap commands in <tool_call> or any XML tags.\n"
                      "If you cannot determine the exact file, explain without [READ:] or [FILE:] tags.")

            _TOOL_CALL_RE = re.compile(r'</?tool_call>')
            _READ_DIRECTIVE_RE = re.compile(r'\[READ:\s*([^\]]+)\]')
            response = ""
            for round_num in range(1, 4):
                user = (f"Issue: {desc_text}\n\n## Context\n{context}\n\n"
                        "Find the root cause. Request files with [READ: path] or provide fixes with [FILE: path].")

                print(f"  Round {round_num} ({len(context)} bytes)...")
                response = await agent.llm.chat([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])

                if response.startswith("[Error") or response.startswith("[LM Studio"):
                    self.error(f"LLM error: {response[:200]}")
                    return True

                # Check for [READ:] directives — handle both plain and <tool_call> wrapped
                raw = response
                # Strip <tool_call> / </tool_call> wrappers if present
                raw = _TOOL_CALL_RE.sub('', raw)
                raw_reads = _READ_DIRECTIVE_RE.findall(raw)
                read_requests = [
                    r.strip() for r in raw_reads
                    if ".py" in r and "\\" not in r and "$" not in r
                    and "^" not in r and "(" not in r and not r.startswith("\\")
                ]
                if read_requests:
                    new_files = []
                    bad_reads = []
                    for req_path in read_requests:
                        req_path = req_path.strip()
                        full = os.path.normpath(os.path.join(ws_dir, req_path))
                        if full in read_paths:
                            continue
                        if os.path.isfile(full) and full.endswith(".py"):
                            try:
                                with open(full, "r", encoding="utf-8") as f:
                                    fcontent = f.read()
                                rel = os.path.relpath(full, ws_dir).replace("\\", "/")
                                context += f"\n\n# === {rel} (requested by LLM) ===\n{fcontent}\n"
                                read_paths.add(full)
                                new_files.append(rel)
                            except Exception as exc:
                                print(f"  Warning: failed to read requested file {full}: {exc}")
                        else:
                            bad_reads.append(req_path)
                    if new_files:
                        print(f"    Read: {', '.join(new_files)}")
                        continue
                    elif bad_reads:
                        print(f"    [READ] could not resolve: {', '.join(bad_reads[:3])} — stopping")
                        break
                    else:
                        break

                # No [READ:] — check for [FILE:] or show raw response
                break

            if not response:
                return True

            # Display response if it's informational (no file fixes)
            clean = _TOOL_CALL_RE.sub('', response)

            # Parse [PATCH:] blocks
            patches = re.findall(r'\[PATCH:\s*([^\]]+)\]\s*\n(.*?)(?=\[PATCH:|\[FILE:|\Z)', clean, re.DOTALL)
            if patches:
                for fpath, patch_text in patches:
                    self._apply_patch(patch_text.strip(), fpath.strip(), ws_dir)

            fixes = re.findall(r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```', clean, re.DOTALL)
            valid_fixes = []
            for fpath, new_code in fixes:
                full = os.path.normpath(os.path.join(ws_dir, fpath.strip()))
                if os.path.exists(full):
                    valid_fixes.append((full, new_code.strip()))

            if not valid_fixes:
                if fixes and not patches:
                    print(f"  (ignored {len(fixes)} [FILE:] blocks — no matching files on disk)")
                elif not patches:
                    print(response)
                return True

            # Apply fixes
            fixed_count = 0
            for full, new_code in valid_fixes:
                fpath = os.path.relpath(full, ws_dir).replace("\\", "/")
                if not os.path.exists(full):
                    print(f"  Skipping {fpath} (not found)")
                    continue
                if len(new_code) < 50 or "import" not in new_code:
                    print(f"  Skipping {fpath} (invalid content)")
                    continue
                if _is_stdlib_path(full):
                    print(f"  Skipping {fpath} (stdlib — cannot modify)")
                    continue

                with open(full, "w", encoding="utf-8") as f:
                    f.write(new_code)
                fixed_count += 1
                print(f"  Fixed: {fpath} ({len(new_code)} bytes)")

                changelog_path = os.path.join(ws_dir, "CHANGES.md")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                entry = f"\n## {timestamp} — fix --desc\n\n**Change**: Modified `{os.path.basename(fpath)}`\n**Reason**: {desc_text[:200]}\n"
                with open(changelog_path, "a", encoding="utf-8") as cl:
                    cl.write(entry)

            print(f"\nFixed {fixed_count}/{len(fixes)} files.")
            return True

        if "--mypy" in parts:
            return await self._fix_mypy(parts, agent)

        else:
            return await self._fix_traceback(parts, agent)

    async def _fix_mypy(self, parts: list[str], agent: 'Agent') -> bool:
        """Run mypy over the workspace and LLM-fix errors grouped by owning file.

        Usage: fix --mypy [path...] [--limit N] [--rounds N]
        Default targets: agent_core/, agent1/ and agent.py in the current dir.
        Each LLM attempt fixes a slice of ~4 errors per file; --rounds is the
        maximum number of attempts per file.
        """
        limit = 5
        rounds = 2
        yes_mode = "--yes" in parts
        ws_dir = os.path.abspath(".")
        targets: list[str] = []
        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "--yes":
                i += 1
                continue
            if p == "--limit" and i + 1 < len(parts):
                limit = int(parts[i + 1])
                i += 2
                continue
            if p == "--rounds" and i + 1 < len(parts):
                rounds = int(parts[i + 1])
                i += 2
                continue
            if p == "--mypy":
                i += 1
                continue
            targets.append(p)
            i += 1
        if not targets:
            for t in ("agent_core", "agent1"):
                if os.path.isdir(os.path.join(ws_dir, t)):
                    targets.append(os.path.join(ws_dir, t))
            if os.path.isfile(os.path.join(ws_dir, "agent.py")):
                targets.append(os.path.join(ws_dir, "agent.py"))

        def run_mypy(args: list[str]) -> str:
            r = subprocess.run(
                [sys.executable, "-m", "mypy", *args, "--ignore-missing-imports", "--follow-imports=skip"],
                capture_output=True, text=True, cwd=ws_dir,
            )
            return r.stdout

        def parse_errors(stdout: str) -> dict[str, list[str]]:
            by_file: dict[str, list[str]] = {}
            for line in stdout.split('\n'):
                m = re.match(r'^(.*?):(\d+): error: (.*)$', line.strip())
                if m:
                    fpath = m.group(1).replace('\\', '/')
                    by_file.setdefault(fpath, []).append(line.strip())
            return by_file

        errors_by_file = parse_errors(run_mypy(targets))
        files = sorted(errors_by_file.items(), key=lambda kv: -len(kv[1]))
        print(f"[fix --mypy] {sum(len(v) for v in errors_by_file.values())} error(s) in {len(files)} file(s)")
        if not files:
            print("Workspace is mypy-clean. Nothing to fix.")
            return True

        for rel_file, errs in files[:limit]:
            if stop_requested():
                print("  Stopped by user — remaining files skipped.")
                break
            full = os.path.normpath(os.path.join(ws_dir, rel_file))
            if not os.path.isfile(full):
                print(f"  Skipping {rel_file} (not found)")
                continue
            print(f"\n[fix --mypy] {rel_file}: {len(errs)} error(s)")
            remaining = list(errs)
            remaining = self._repair_and_mechanical(full, rel_file, remaining, yes_mode)
            for attempt in range(max(rounds, 1)):
                if stop_requested():
                    break
                if not remaining:
                    break
                slice_errs = remaining[:4]
                try:
                    with open(full, "r", encoding="utf-8") as f:
                        current = f.read()
                except OSError as e:
                    print(f"  Cannot read {rel_file}: {e}")
                    break
                lines = current.split('\n')
                user_sections = []
                for err in slice_errs:
                    error_line = _parse_line_number(err)
                    before, after, instruction = _classify_error(err)
                    window = _extract_window(lines, error_line, before, after)
                    user_sections.append(
                        f"### Error\n{err}\n\nInstruction: {instruction}\n\n"
                        f"Relevant code:\n```python\n{window}\n```\n"
                    )
                msgs = [
                    {"role": "system", "content": f"Fix the mypy errors in {rel_file}. Make the smallest possible targeted changes. Prefer [PATCH:]; if the file is over ~200 lines use [PATCH:] exclusively. Output the fix as ONE of: [PATCH: {rel_file}] with unified hunks, or [FILE: {rel_file}] with the complete corrected file. Do NOT output anything else."},
                    {"role": "user", "content": "\n".join(user_sections) + f"\n\nOutput format:\n[PATCH: {rel_file}]\n@@ -line,count +line,count @@\n unchanged line\n-removed line\n+added line\n\nOR if a larger rewrite is needed:\n[FILE: {rel_file}]\n```python\n# complete corrected file\n```"},
                ]
                print(f"  Attempt {attempt + 1} on {len(slice_errs)} error(s)...")
                response = await agent.llm.chat(msgs, max_tokens=8000, disable_thinking=True)
                if not response or response.startswith("[Error") or response.startswith("[LM Studio"):
                    print(f"  LLM error: {response}")
                    print("  Hint: the model burned its whole output budget on reasoning. Load a non-thinking model (`model load <name>`) or a faster coder model for fix --mypy.")
                    break
                prev = current
                before_errs = list(remaining)
                count_before = len(before_errs)
                changed = self._apply_fix_blocks(response, ws_dir, rel_file, auto_yes=yes_mode, context_errors=slice_errs)
                if not changed:
                    print("  Fix did not apply — retrying once with verbatim-copy instructions")
                    retry_msgs = [
                        msgs[0],
                        {"role": "user", "content": "Your previous [PATCH: ...] hunks did not match the file: the context lines were not exact. Reproduce the hunks using ONLY the exact lines shown in the 'Relevant code' windows above — copy them character-for-character, including indentation. Output a focused patch that touches only the listed lines. Do NOT rewrite whole functions or files.\n\nOutput format (strictly a patch):\n[PATCH: " + rel_file + "]\n@@ -line,count +line,count @@\n unchanged line\n-removed line\n+added line\n"},
                    ]
                    response = await agent.llm.chat(retry_msgs, max_tokens=5000, disable_thinking=True)
                    if not response or response.startswith("[Error") or response.startswith("[LM Studio"):
                        print(f"  LLM error: {response}")
                        break
                    changed = self._apply_fix_blocks(response, ws_dir, rel_file, auto_yes=yes_mode)
                if not changed:
                    print("  Slice failed — retrying the first error alone")
                    single_msgs = [
                        msgs[0],
                        {"role": "user", "content": user_sections[0] +
                            "\n\nOutput ONLY a minimal [PATCH:] fixing this one error. "
                            "Copy the context lines character-for-character from the window above."},
                    ]
                    response = await agent.llm.chat(single_msgs, max_tokens=4000, disable_thinking=True)
                    if response and not response.startswith("[Error") and not response.startswith("[LM Studio"):
                        changed = self._apply_fix_blocks(response, ws_dir, rel_file, auto_yes=yes_mode, context_errors=slice_errs[:1])
                    if changed:
                        remaining = parse_errors(run_mypy([rel_file])).get(rel_file, [])
                        if len(remaining) > count_before:
                            print(f"  Single-error fix regressed: {count_before} -> {len(remaining)} errors — restoring previous version")
                            with open(full, "w", encoding="utf-8") as f:
                                f.write(prev)
                            remaining = before_errs[4:]
                            continue
                        print(f"  Single-error fallback applied: {len(remaining)} error(s) remaining")
                        if not remaining:
                            break
                        continue
                    remaining = remaining[4:]
                    print(f"  No usable fix for these {len(slice_errs)} error(s) — skipping them")
                    continue
                r = subprocess.run([sys.executable, "-m", "py_compile", full], capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"  Fix introduced a compile error — restoring previous version\n  {r.stderr.strip()[:200]}")
                    with open(full, "w", encoding="utf-8") as f:
                        f.write(prev)
                    remaining = before_errs[4:]
                    continue
                remaining = parse_errors(run_mypy([rel_file])).get(rel_file, [])
                if len(remaining) > count_before:
                    print(f"  Fix regressed: {count_before} -> {len(remaining)} errors — restoring previous version")
                    with open(full, "w", encoding="utf-8") as f:
                        f.write(prev)
                    remaining = before_errs[4:]
                    continue
                print(f"  Attempt {attempt + 1}: {len(remaining)} error(s) remaining in {rel_file}")
                if not remaining:
                    break

        final = parse_errors(run_mypy(targets))
        total = sum(len(v) for v in final.values())
        print(f"\n[fix --mypy] Done. {total} error(s) remain in {len(final)} file(s).")
        for fp, cnt in sorted(final.items(), key=lambda kv: -len(kv[1]))[:10]:
            print(f"  {fp}: {len(cnt)}")
        return True

    def _repair_and_mechanical(
        self, full: str, rel_file: str, errs: list[str], auto_yes: bool
    ) -> list[str]:
        """Pre-flight corruption repair + deterministic mypy fixes.

        Returns the still-unfixed error list (re-queried from mypy so the
        caller's attempt loop only sees what genuinely needs the LLM).
        """
        try:
            with open(full, "r", encoding="utf-8") as f:
                source = f.read()
        except OSError as e:
            print(f"  Cannot read {rel_file}: {e}")
            return errs

        # 1) Collapse duplicated-line corruption left by a prior bad apply.
        collapsed_source, runs = _collapse_duplicate_runs(source)
        if runs:
            print(f"  [repair] collapsed {runs} duplicated-line run(s) in {rel_file}")
            if save_file_py(full, collapsed_source, auto_yes=auto_yes):
                source = collapsed_source
            errs = self._rerun_mypy_errors(rel_file, full, errs)

        # 2) Deterministic fixes — process bottom-up so a dropped line never
        #    invalidates the line numbers of errors yet to be handled.
        parsed: list[tuple[int, str, str]] = []
        for err in errs:
            info = _parse_mypy_error(err)
            if info is None:
                continue
            parsed.append((info[0], info[1], err))
        parsed.sort(key=lambda t: -t[0])

        new_lines = source.split('\n')
        changed = False
        for lineno, code, err in parsed:
            before = list(new_lines)
            after: list[str] | None = None
            if code == "unused-ignore":
                after = _fix_unused_ignore(before, lineno)
                label = "unused-ignore"
            elif code == "redundant-cast":
                m = re.search(r'Redundant cast to "([^"]+)"', err)
                if m:
                    after = _fix_redundant_cast(before, lineno, m.group(1))
                label = "redundant-cast"
            elif code == "assignment" and "default has type \"None\"" in err:
                after = _fix_implicit_optional(before, lineno)
                label = "implicit-optional"
            elif code == "attr-defined":
                m = re.search(r'has no attribute "([^"]+)"; maybe "([^"]+)"', err)
                if m:
                    after = _fix_attr_defined_rename(
                        before, lineno, '\n'.join(new_lines), m.group(1), m.group(2)
                    )
                label = "attr-defined-rename"
            elif code == "no-untyped-def":
                after = _fix_missing_return_none(before, lineno)
                label = "return-None"
            else:
                continue
            if after is None or after == before:
                continue
            joined = '\n'.join(after)
            if not _syntax_ok(joined):
                print(f"  mechanical {label} rejected (syntax) — {err[:80]}")
                continue
            print(f"  mechanical {label} @ {rel_file}:{lineno}")
            if save_file_py(full, joined, auto_yes=auto_yes):
                new_lines = after
                changed = True

        if changed:
            return self._rerun_mypy_errors(rel_file, full, errs)
        return errs

    def _rerun_mypy_errors(
        self, rel_file: str, full: str, prev_errs: list[str]
    ) -> list[str]:
        """Re-run mypy on *full* and return its current error list for it."""
        ws_dir = os.path.dirname(full)
        r = subprocess.run(
            [sys.executable, "-m", "mypy", full, "--ignore-missing-imports",
             "--follow-imports=skip"],
            capture_output=True, text=True, cwd=ws_dir,
        )
        by_file: dict[str, list[str]] = {}
        for line in r.stdout.split('\n'):
            m = re.match(r'^(.*?):(\d+): error: (.*)$', line.strip())
            if m:
                fpath = m.group(1).replace('\\', '/')
                by_file.setdefault(fpath, []).append(line.strip())
        match_key = rel_file.replace('\\', '/')
        # mypy prints the path as it was given; try both rel and basename.
        for key in (match_key, os.path.basename(rel_file), full.replace('\\', '/')):
            if key in by_file:
                return by_file[key]
        return []

    def _apply_fix_blocks(self, response: str, ws_dir: str, rel_file: str,
                          auto_yes: bool = False,
                          context_errors: list[str] | None = None) -> int:
        """Apply [PATCH:]/[FILE:] blocks targeting rel_file with diff+confirm.

        Returns the number of blocks applied. Writes go through save_file_py
        (y/N confirm unless auto_yes); the caller re-compiles and restores on
        failure.  When *context_errors* is given, the issues the model was
        asked to fix are printed above each presented patch.
        """
        full = os.path.normpath(os.path.join(ws_dir, rel_file))
        if not os.path.isfile(full):
            return 0
        clean = re.sub(r'</?tool_call>', '', response)
        try:
            with open(full, "r", encoding="utf-8") as f:
                current = f.read()
        except OSError as e:
            print(f"  Cannot read {rel_file}: {e}")
            return 0

        def _targets(block_path: str) -> bool:
            bp = block_path.replace('\\', '/')
            return bp == rel_file or bp == os.path.basename(rel_file)

        applied = 0
        from agent_core.patch_utils import split_source_lines

        def _show_targeted_issues() -> None:
            if context_errors:
                print(f"  Issues this patch targets ({len(context_errors)}):")
                for e in context_errors:
                    print(f"    {e}")

        lines = split_source_lines(current)
        for m in re.finditer(r'\[PATCH:\s*([^\]]+)\]\s*\n?(.*?)(?=\[PATCH:|\[FILE:|\Z)', clean, re.DOTALL):
            if not _targets(m.group(1).strip()):
                continue
            patch_text = m.group(2).strip()
            ok, result = _impl_apply_patch(patch_text, full, lines)
            if not ok:
                from agent_core.patch_utils import apply_anchored_patch
                ok, result = apply_anchored_patch(patch_text, lines)
            if not ok:
                print(f"  Patch apply failed: {result[:150]}")
                continue
            reason = _looks_corrupted(current, result)
            if reason:
                print(f"  Rejected patch — corruption guard: {reason}")
                continue
            _show_targeted_issues()
            if save_file_py(full, result, auto_yes=auto_yes):
                current = result
                applied += 1
        for m in re.finditer(r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)(?:\n```|$)', clean, re.DOTALL):
            if not _targets(m.group(1).strip()):
                continue
            new_code = m.group(2).strip()
            if len(new_code) < 10 or not re.search(r'\b(import|def |class )\b', new_code):
                print(f"  WARNING: [FILE:] content for {rel_file} looks invalid, skipping")
                continue
            if len(current.splitlines()) > 200 and len(new_code) > len(current) * 1.5:
                print(f"  WARNING: [FILE:] content for {rel_file} is an oversized rewrite, skipping (use [PATCH:] instead)")
                continue
            _show_targeted_issues()
            if save_file_py(full, new_code, auto_yes=auto_yes):
                current = new_code
                applied += 1

        if applied:
            changelog_path = os.path.join(ws_dir, "CHANGES.md")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            entry = f"\n## {timestamp} — fix --mypy\n\n**Change**: Modified `{os.path.basename(full)}`\n**Reason**: mypy error fixes\n"
            try:
                with open(changelog_path, "a", encoding="utf-8") as cl:
                    cl.write(entry)
            except OSError:
                pass
        return applied

    def _apply_fix_response(self, response: str, ws_dir: str, desc_text: str) -> None:
        """Parse [FILE:] and [PATCH:] blocks from *response* and apply them to disk."""
        if response.startswith("[Error") or response.startswith("[LM Studio"):
            print(f"LLM error: {response[:200]}")
            return
        clean = re.sub(r'</?tool_call>', '', response)

        # Parse [PATCH:] blocks
        patches = re.findall(r'\[PATCH:\s*([^\]]+)\]\s*\n(.*?)(?=\[PATCH:|\[FILE:|\Z)', clean, re.DOTALL)
        if patches:
            for fpath, patch_text in patches:
                fpath = fpath.strip()
                self._apply_patch(patch_text.strip(), fpath, ws_dir, reason=desc_text)

        # Parse [FILE:] blocks (full file rewrite)
        fixes = re.findall(r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```', clean, re.DOTALL)
        if not fixes:
            if not patches:
                print("Could not parse fixes.")
                print(response[:1000])
            return
        fixed_count = 0
        for fpath, new_code in fixes:
            fpath = fpath.strip()
            new_code = new_code.strip()
            full = os.path.normpath(os.path.join(ws_dir, fpath)) if not os.path.isabs(fpath) else fpath
            if not os.path.exists(full):
                continue
            if len(new_code) < 50 or "import" not in new_code:
                print(f"  Skipping {fpath} (invalid content)")
                continue
            if _is_stdlib_path(full):
                print(f"  Skipping {fpath} (stdlib — cannot modify)")
                continue
            with open(full, "w", encoding="utf-8") as f:
                f.write(new_code)
            fixed_count += 1
            print(f"  Fixed: {fpath} ({len(new_code)} bytes)")
        print(f"\nFixed {fixed_count}/{len(fixes)} files.")

    def _apply_patch(self, patch_text: str, fpath: str, ws_dir: str,
                     reason: str | None = None) -> bool:
        """Apply a unified-diff-style patch to a file. Returns True on success.

        *reason* (the issue the patch is meant to fix) is shown above the
        diff so the user can judge the patch against it.
        """
        full = os.path.normpath(os.path.join(ws_dir, fpath)) if not os.path.isabs(fpath) else fpath
        if not os.path.exists(full):
            print(f"  Skipping {fpath} (not found)")
            return False
        if _is_stdlib_path(full):
            print(f"  Skipping {fpath} (stdlib)")
            return False

        try:
            with open(full, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"  Cannot read {fpath}: {e}")
            return False

        # Parse hunks: @@ -start,count @@ ... @@
        hunks = []
        for m in re.finditer(r'@@\s*-(\d+)(?:,\d+)?\s*\+(\d+)(?:,\d+)?\s*@@(.*?)(?=@@|\Z)', patch_text, re.DOTALL):
            start = int(m.group(1))
            body = m.group(3).rstrip()
            chunks: list[tuple[str, str | None]] = []  # ('-', line) or ('+', line) or (' ', line)
            for line in body.split('\n'):
                line = line.rstrip('\r')
                if line.startswith('-'):
                    chunks.append(('-', line[1:]))
                elif line.startswith('+'):
                    chunks.append(('+', line[1:]))
                elif line.startswith(' '):
                    chunks.append((' ', line[1:]))
            if chunks:
                hunks.append((start, chunks))

        if not hunks:
            print(f"  Could not parse patch for {fpath}")
            return False

        # Filter out broken hunks (incomplete: has - but no +, empty + lines, or incomplete lines)
        incomplete_ops = ('=', '+', '-', '*', '/', ',', '(', '[', '{')
        valid_hunks = []
        for start, chunks in hunks:
            has_minus = any(op == '-' for op, _ in chunks)
            has_plus = any(op == '+' for op, _ in chunks)
            if has_minus and not has_plus:
                continue  # Removal only — skip
            if any(op == '+' and not text.strip() for op, text in chunks):
                continue  # Empty replacement — skip
            # Filter incomplete lines (trailing operators like =, +, -, etc.)
            if any(op == '+' and text.rstrip().endswith(incomplete_ops) for op, text in chunks):
                continue  # Incomplete replacement — skip
            valid_hunks.append((start, chunks))

        if not valid_hunks:
            print(f"  No valid hunks in patch for {fpath}")
            return False

        hunks = valid_hunks

        # Verify old lines exist before applying (whitespace-tolerant)
        # Record, per hunk, whether its +/- lines carry an LLM padding space
        # (git-style puts the whole line right after the marker; LLM-style adds
        # one padding space).  Detect by comparing a '-' line's leading
        # whitespace against the matching file line.
        hunk_padding = {}
        for start, chunks in hunks:
            idx = start - 1
            padded = False
            for op, text in chunks:
                if op in ('-', ' '):
                    if idx < 0 or idx >= len(lines):
                        print(f"  Patch mismatch at line {idx+1}: line out of range")
                        return False
                    actual = lines[idx].rstrip('\r\n')
                    if op == '-':
                        patch_lead = len(text) - len(text.lstrip())
                        file_lead = len(actual) - len(actual.lstrip())
                        if not padded and patch_lead != file_lead and text.strip() == actual.strip():
                            padded = True
                    if actual.strip() != text.strip():
                        print(f"  Patch mismatch at line {idx+1}: expected '{text[:60]}', got '{actual[:60]}'")
                        return False
                    idx += 1
                elif op == '+':
                    pass  # new lines don't need verification
            hunk_padding[start] = padded

        def _render_plus(start: int, text: str) -> str:
            if hunk_padding.get(start, False) and text.startswith(' '):
                return text[1:]
            return text

        # Apply hunks (reverse order to preserve line numbers).  Content is
        # applied verbatim (only the marker padding is stripped) — indentation
        # is never rewritten here; the syntax check rejects broken indentation.
        result = lines[:]
        for start, chunks in reversed(hunks):
            old_lines = []
            new_lines = []
            i = 0
            while i < len(chunks):
                op, text = chunks[i]
                if op == '-':
                    # Check if next chunk is a + line (same logical change)
                    if i + 1 < len(chunks) and chunks[i + 1][0] == '+':
                        # The + line is applied verbatim.
                        new_lines.append(_render_plus(start, chunks[i + 1][1]))
                        old_lines.append(text)
                        i += 2  # Skip both - and + lines
                    else:
                        new_lines.append(_render_plus(start, text))
                        old_lines.append(text)
                        i += 1
                elif op == '+':
                    new_lines.append(_render_plus(start, text))
                    i += 1
                elif op == ' ':
                    old_lines.append(text)
                    new_lines.append(text)
                    i += 1
            
            idx = start - 1
            if idx + len(old_lines) <= len(result):
                del result[idx:idx + len(old_lines)]
                for i, text in enumerate(new_lines):
                    result.insert(idx + i, text + '\n')

        # Syntax check
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(''.join(result))
            tmp = tf.name
        r = subprocess.run(["python", "-m", "py_compile", tmp], capture_output=True, text=True)
        os.unlink(tmp)
        if r.returncode != 0:
            print(f"  Patch would break syntax: {r.stderr[:200]}")
            return False

        # Show diff
        if reason:
            print(f"  Reason: {reason[:400]}")
        show_file_diff(fpath, ''.join(lines), ''.join(result))

        # Apply
        if not read_choice("  Apply this patch? (y/N): "):
            print("  Skipped.")
            return False

        with open(full, "w", encoding="utf-8") as f:
            f.write(''.join(result))
        print(f"  Patched: {fpath}")
        return True
    async def _fix_traceback(self, parts: list[str], agent: 'Agent') -> bool:
        traceback_text = ""
        if parts:
            traceback_text = " ".join(parts)

        if traceback_text.startswith('"') and traceback_text.endswith('"'):
            traceback_text = traceback_text[1:-1]

        if not traceback_text or "File \"" not in traceback_text:
            print("Paste the full traceback, then press Enter on an empty line:")
            lines = []
            if traceback_text:
                lines.append(traceback_text)
            while True:
                try:
                    line = input()
                    if not line.strip():
                        break
                    lines.append(line)
                except EOFError:
                    break
            traceback_text = "\n".join(lines)

        if not traceback_text or "File \"" not in traceback_text:
            self.error("Could not find file references in traceback.")
            return True

        print(f"\nParsing traceback ({len(traceback_text)} chars)...")

        file_pattern = r'File "(.*?)", line (\d+)'
        matches = re.findall(file_pattern, traceback_text)

        if not matches:
            self.error("Could not find file/line references in traceback.")
            return True

        fpath, line_num = matches[-1]

        if fpath.startswith("<"):
            self.error(f"Cannot fix built-in module: {fpath}")
            return True

        if not os.path.exists(fpath):
            self.error(f"File not found: {fpath}")
            return True

        error_lines = traceback_text.strip().split('\n')
        error_msg = error_lines[-1] if error_lines else "Unknown error"

        print(f"\nError in {fpath}:{line_num}")
        print(f"  {error_msg}")

        is_import_error = "ImportError" in error_msg or "ModuleNotFoundError" in error_msg or "cannot import" in error_msg

        if is_import_error:
            all_files_in_trace = matches
            if len(all_files_in_trace) > 1:
                print(f"\n  Cascade detected! {len(all_files_in_trace)} files in trace:")
                # Find the first trackable file to mark as ROOT in the listing.
                root_idx: int | None = None
                for idx, (fp, _ln) in enumerate(all_files_in_trace):
                    if _is_trackable_file(fp):
                        root_idx = idx
                        break
                for i, (fp, ln) in enumerate(all_files_in_trace):
                    marker = " -> ROOT" if i == root_idx else ""
                    print(f"    {i+1}. {fp}:{ln}{marker}")

                # Walk from the start to find the first *user* file that actually
                # exists and isn't frozen or under the Python installation.
                for root_file, root_ln in all_files_in_trace:
                    if not _is_trackable_file(root_file):
                        continue
                    if root_file != fpath:
                        print(f"\n  Root cause is in {root_file}:{root_ln}, not in {fpath}")
                        print(f"  Fixing {root_file} instead...")
                        fpath = root_file
                        line_num = root_ln
                        error_msg = f"Cascading ImportError from {fpath}"
                    break

                # Shadowed stdlib module detection (e.g. local types.py shadowing
                # the stdlib types module).
                shadow_match = re.search(
                    r"partially initialized module ['\"](\S+?)['\"]",
                    error_msg,
                )
                if shadow_match:
                    shadowed = shadow_match.group(1)
                    print(f"\n  Shadow warning: local file is conflicting with stdlib module '{shadowed}'")
                    ws = os.path.dirname(os.path.abspath(fpath))
                    for root, _dirs, files in os.walk(ws):
                        for fn in files:
                            if fn == f"{shadowed}.py":
                                candidate = os.path.normpath(os.path.join(root, fn))
                                print(f"  The local file {candidate} shadows '{shadowed}' from the Python stdlib.")
                                print(f"  Fix: rename or move it (e.g. {shadowed}_defs.py or put it inside a package).")
                        _dirs[:] = []
                        break
                    print("  Skipping LLM fix — this is a naming conflict, not a code error.")
                    return True

        if _is_stdlib_path(fpath):
            print(f"\n  Skipping: {fpath} is a stdlib file (under Python installation).")
            print("  The root cause is likely a local file shadowing a stdlib module name.")
            return True

        with open(fpath, "r", encoding="utf-8") as f:
            current_code = f.read()

        lines_list = current_code.split('\n')
        line_idx = int(line_num) - 1
        start = max(0, line_idx - 3)
        end = min(len(lines_list), line_idx + 4)
        print(f"\n  Context (lines {start+1}-{end}):")
        for i in range(start, end):
            marker = ">>>" if i == line_idx else "   "
            print(f"  {marker} {i+1}: {lines_list[i][:120]}")

        project_dir = str(Path(fpath).parent)
        export_map = {}
        for root, dirs, files in os.walk(project_dir):
            if ".git" in root or "__pycache__" in root:
                continue
            for fp_name in files:
                if fp_name.endswith(".py"):
                    full = os.path.join(root, fp_name)
                    try:
                        with open(full, "r") as pf:
                            src = pf.read()
                        exports = set()
                        for m in re.finditer(r'^(?:class|def)\s+(\w+)', src, re.MULTILINE):
                            exports.add(m.group(1))
                        if exports:
                            rel = os.path.relpath(full, project_dir).replace("\\", "/")
                            export_map[rel] = exports
                    except Exception as e:
                        print(f"  Warning: failed to parse {full}: {e}")

        all_broken = []
        for match in re.finditer(r'from\s+(\S+)\s+import\s+(.+?)(?:\s*#|\s*$)', current_code):
            src_module = match.group(1)
            imported_names = [n.strip().split(' as ')[0].strip() for n in match.group(2).strip('()').split(',')]
            src_file = src_module.replace('.', '/') + '.py'
            if src_file not in export_map:
                continue
            src_exports = export_map.get(src_file, set())
            for name in imported_names:
                if name.isupper() or name.startswith('_'):
                    continue
                if name not in src_exports:
                    all_broken.append((src_module, name, src_file, sorted(src_exports)))

        if all_broken:
            print(f"\n  Found {len(all_broken)} broken imports in {fpath}:")
            for mod, name, src_file, avail in all_broken:
                print(f"    '{name}' from '{mod}' not found. Available: {', '.join(avail[:5])}")

        print("\nSending to LLM for fix...")
        fix_system = "Fix ALL broken imports in this file. Use ONLY imports that exist in the project. Keep stdlib/third-party imports unchanged. No duplicate functions. No _v1/_v2 variants.\n\nWhen fixing type errors (arg-type, incompatible type), search the ENTIRE file for where the variable is defined/initialized, not just where the error occurs. Fix the initialization to use the correct type. Do NOT change function signatures.\n\nOutput the fix using ONE of these formats:\n[PATCH: filename.py] — for small fixes near the error line\n[FILE: filename.py] — when the fix is far from the error or needs full context\n\n[PATCH:] example:\n[PATCH: filename.py]\n@@ -10,3 +10,2 @@\n- old line\n+ new line\n\n[FILE:] example:\n[FILE: filename.py]\n```python\n# complete corrected file\n```"
        # Inject past decisions as constraints
        try:
            basename = os.path.basename(fpath) if fpath else ""
            constraints = decisions_as_system_prompt(
                str(Path(fpath).parent.resolve()) if fpath else "",
                [basename] if basename else []
            )
            if constraints:
                fix_system += constraints
        except Exception:
            pass
        fix_msgs = [
            {"role": "system", "content": fix_system},
            {"role": "user", "content": f"Fix ALL errors in {fpath}:\n\nError from traceback at line {line_num}:\n{error_msg}\n\nAll broken imports in this file (must fix ALL):\n" + "\n".join([f"  import '{n}' from '{m}' — not found. Available in {s}: {', '.join(a[:8])}" for m, n, s, a in all_broken]) + f"\n\nFull traceback:\n{traceback_text}\n\nCurrent code:\n```python\n{current_code}\n```"}
        ]
        fixed = await agent.llm.chat(fix_msgs)

        if fixed.startswith("[Error") or fixed.startswith("[LM Studio"):
            self.error(f"LLM error: {fixed[:200]}")
            return True

        match = re.search(r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```', fixed, re.DOTALL)
        patch_match = re.search(r'\[PATCH:\s*([^\]]+)\]\s*\n(.*?)(?=\[FILE:|\Z)', fixed, re.DOTALL)

        if patch_match:
            fpath_patch = patch_match.group(1).strip()
            patch_text = patch_match.group(2).strip()
            ws_dir = str(Path(fpath).parent)
            ok = self._apply_patch(patch_text, fpath_patch, ws_dir, reason=error_msg)
            if ok:
                print(f"\nFixed: {fpath_patch} (patch applied)")
                result = subprocess.run(
                    ["python", "-m", "py_compile", fpath],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print("Compiled OK!")
                else:
                    print(f"Still has errors:\n{result.stderr[:300]}")
            else:
                print(f"Patch failed for {fpath_patch}")
        elif match:
            new_code = match.group(2).strip()
            if len(new_code) > len(current_code) * 0.1 and 'import' in new_code:
                if _is_stdlib_path(fpath):
                    print(f"\nSkipping: {fpath} is a stdlib file.")
                    return True
                print(f"  Reason: {error_msg[:400]}")
                if save_file_py(fpath, new_code, auto_yes=False):
                    print(f"\nFixed: {fpath} ({len(new_code)} bytes)")
                else:
                    print(f"\nSkipped: {fpath} (no changes)")

                changelog_path = os.path.join(str(Path(fpath).parent), "CHANGES.md")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                entry = f"\n## {timestamp} — fix\n\n**Change**: Modified `{os.path.basename(fpath)}`\n**Reason**: {error_msg[:200]}\n"
                with open(changelog_path, "a", encoding="utf-8") as cl:
                    cl.write(entry)

                result = subprocess.run(
                    ["python", "-m", "py_compile", fpath],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print("Compiled OK!")
                else:
                    print(f"Still has errors:\n{result.stderr[:300]}")
            else:
                print("LLM returned invalid fix (too short or not code)")
        else:
            print("Could not parse fix from LLM response")
            print(f"Raw: {fixed[:300]}")

        # Auto-extract design decisions from this fix
        if fpath and os.path.exists(fpath):
            try:
                basename = os.path.basename(fpath)
                candidates = await extract_from_changes(
                    agent, [basename],
                    context=f"Fixed {fpath}: {error_msg[:300] if error_msg else 'unknown error'}"
                )
                if candidates:
                    print(f"\n[decide] Extracted {len(candidates)} decision candidates from this fix:")
                    for i, c in enumerate(candidates, 1):
                        print(f"  {i}. {c.get('title', 'Untitled')}")
                    print("  Record? (1/all/N, press Enter to skip): ", end="")
                    try:
                        choice = input().strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        choice = ""
                    if choice and choice != "n":
                        ws_str = str(Path(fpath).parent.resolve())
                        if choice == "all":
                            selected = range(len(candidates))
                        else:
                            selected = []
                            for part in choice.replace(" ", "").split(","):
                                try:
                                    selected.append(int(part) - 1)
                                except ValueError:
                                    pass
                        for idx in selected:
                            if 0 <= idx < len(candidates):
                                c = candidates[idx]
                                record = add_decision(
                                    ws_str,
                                    c.get("title", "Untitled"),
                                    context=c.get("context", ""),
                                    decision=c.get("decision", ""),
                                    rationale=c.get("rationale", ""),
                                    affected_files=c.get("affected_files", []),
                                    tags=c.get("tags", []),
                                )
                                print(f"  Recorded #{record['id']}: {record['title']}")
            except Exception:
                pass

        return True