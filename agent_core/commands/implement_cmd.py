"""Implement command for agent interactive mode."""
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from .base import Command, auto_choice, show_file_diff, read_input, stop_requested
from .doc_paths import find_input
from agent_core import to_windows_path, workspace_path
from agent_core.decisions import decisions_as_system_prompt, extract_from_changes, add_decision

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from agent import Agent


_STDLIB_COMMON = frozenset({
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
    "asyncore", "atexit", "audioop", "base64", "bdb", "binascii", "binhex",
    "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath",
    "cmd", "code", "codecs", "codeop", "collections", "colorsys", "compileall",
    "concurrent", "config", "configparser", "contextlib", "contextvars",
    "copy", "copyreg", "cProfile", "crypt", "csv", "ctypes", "curses",
    "dataclasses", "datetime", "dbm", "decimal", "difflib", "dis",
    "distutils", "doctest", "email", "encodings", "enum", "errno", "faulthandler",
    "fcntl", "filecmp", "fileinput", "fnmatch", "formatter", "fractions",
    "ftplib", "functools", "gc", "getopt", "getpass", "gettext", "glob",
    "graphlib", "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http",
    "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect", "io",
    "ipaddress", "itertools", "json", "keyword", "lib2to3", "linecache",
    "locale", "logging", "lzma", "mailbox", "mailcap", "marshal",
    "math", "mimetypes", "mmap", "modulefinder", "multiprocessing", "netrc",
    "nis", "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev",
    "parser", "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil",
    "platform", "plistlib", "poplib", "posix", "posixpath", "pprint",
    "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr", "pydoc",
    "queue", "quopri", "random", "re", "readline", "reprlib", "resource",
    "rlcompleter", "runpy", "sched", "secrets", "select", "selectors",
    "shelve", "shlex", "shutil", "signal", "site", "smtpd", "smtplib",
    "sndhdr", "socket", "socketserver", "sqlite3", "ssl", "stat",
    "statistics", "string", "stringprep", "struct", "subprocess",
    "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap",
    "threading", "time", "timeit", "tkinter", "token", "tokenize",
    "trace", "traceback", "tracemalloc", "tty", "turtle", "turtledemo",
    "types", "typing", "unicodedata", "unittest", "urllib", "uu",
    "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
    "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipfile", "zipimport", "zlib",
    # Non-stdlib but very common collision targets
    "logger", "utils", "helpers", "common", "base", "core",
    "memory", "cache", "settings", "constants",
})

_EXISTING_ROOT_PACKAGES = frozenset({
    "agent_core", "agent1", "agent",
})

_SAFE_SUBPACKAGE_CANDIDATES = ("agent1", "src/agent1")


def _find_safe_subpackage(workspace: Path) -> str:
    """Return an existing or newly created safe sub-package path under *workspace*."""
    for cand in _SAFE_SUBPACKAGE_CANDIDATES:
        d = workspace / cand
        if d.is_dir():
            return cand
    # If none exist, automatically create agent1/
    if not (workspace / "agent1").exists():
        (workspace / "agent1").mkdir(parents=True, exist_ok=True)
        (workspace / "agent1" / "__init__.py").touch(exist_ok=True)
    return "agent1"


def _is_dangerous_filename(filename: str, workspace: Path) -> tuple[bool, str]:
    """Check whether *filename* would shadow a stdlib module, collide with an
    existing package, or create a package at the workspace root.

    Returns (is_dangerous, reason).
    """
    name = filename.removesuffix(".py")
    if not name or name == filename or name.endswith("/"):
        return True, f"invalid target name: {filename!r}"

    # __init__.py at the workspace root turns the entire repo into a package
    # and breaks relative imports for all top-level modules.
    dest = (workspace / filename).resolve()
    if name == "__init__" and dest.parent.resolve() == workspace.resolve():
        return True, "__init__.py at workspace root would turn the entire repo into a package"

    # Shallow names (no directory prefix) at workspace root are always
    # dangerous — they shadow stdlib, collide with packages, or pollute the
    # repo root.  Auto-repair will prefix them with a safe sub-package.
    if "/" not in filename and "\\" not in filename:
        return True, f"bare workspace-root file {filename!r} — needs sub-package prefix"

    # Check if any directory in the path shadows a stdlib module, ignoring
    # existing project packages (e.g. `agent_core/utils/`).
    shadow = _shadowing_stdlib_dir(filename, workspace)
    if shadow:
        return True, f"directory '{shadow}' shadows stdlib module"

    return False, ""


def _shadowing_stdlib_dir(filename: str, workspace: Path) -> str:
    """Return the first path segment that would shadow a stdlib/common module.

    A segment only "shadows" when the corresponding directory does NOT yet
    exist in the workspace.  An existing package such as ``agent_core/utils/``
    is a deliberate project choice and never shadows — planned paths under it
    must not be renamed.
    """
    parts = filename.replace("\\", "/").split("/")
    for idx, part in enumerate(parts[:-1]):
        if part in _STDLIB_COMMON and not (workspace / "/".join(parts[: idx + 1])).exists():
            return part
    return ""


def _extract_file_context(source: str, filename: str, radius: int = 400) -> str:
    """Extract paragraphs mentioning a file from analysis/plan text."""
    if not source or not filename:
        return ""
    name = os.path.basename(filename)
    parts = []
    for match in re.finditer(re.escape(name), source):
        start = max(0, match.start() - radius)
        end = min(len(source), match.end() + radius)
        snippet = source[start:end].strip()
        if snippet and snippet not in parts:
            parts.append(snippet)
    if not parts:
        return ""
    return "\n\n...\n\n".join(parts[:3])


def _extract_task_line(taskplan: str, filename: str) -> str:
    """Extract the task line for a specific file from the task plan.

    Uses backtick-bounded matching — `` `filename` `` — to avoid false
    positives from substring matches (e.g. ``agent.py`` in ``agent_core/agent.py``).
    """
    name = os.path.basename(filename)
    for line in taskplan.split('\n'):
        if f"`{filename}`" in line:
            return line.strip()
    for line in taskplan.split('\n'):
        if f"`{name}`" in line:
            return line.strip()
    for line in taskplan.split('\n'):
        stripped = line.strip()
        if name in stripped and re.match(r'^\d+\.?\s+`', stripped):
            return stripped
    return ""


_PATH_LINE_RE = re.compile(r'(?:[A-Za-z]:)?[\\/]?[\w./\\+-]*\.py:(\d+):')
_FILE_LINE_RE = re.compile(r'File "([^"]+)", line (\d+)')
_WORD_LINE_RE = re.compile(r'\bline (\d+)\b', re.IGNORECASE)


def _parse_line_number(err: str) -> int:
    """Extract the target line number from an error string.

    Anchors on structured locations — mypy's ``path.py:LINE:`` prefix,
    tracebacks' ``File "...", line N``, plain ``line N`` — instead of
    scanning for the first digit.  Bare digit scanning grabbed version
    fragments and centred the fix window on the wrong lines (observed:
    ``agent_core/llm/v2/client.py:88:`` yielded 2 from ``v2``).
    """
    m = _PATH_LINE_RE.search(err)
    if m:
        n = int(m.group(1))
        if 0 < n <= 1_000_000:
            return n
    m = _FILE_LINE_RE.search(err)
    if m:
        n = int(m.group(2))
        if 0 < n <= 1_000_000:
            return n
    m = _WORD_LINE_RE.search(err)
    if m:
        n = int(m.group(1))
        if 0 < n <= 1_000_000:
            return n
    if "CROSS:" in err:
        return 1
    return 0


def _classify_error(err: str) -> tuple[int, int, str]:
    """Return (window_before, window_after, instruction) for an error pattern."""
    if "union-attr" in err or ("has no attribute" in err and "CROSS:" not in err):
        return (30, 15,
            "A variable might be None. Check for an 'is not None' guard ABOVE the error line. "
            "REUSE the guarded local variable — do NOT re-read from a list/dict after the guard. "
            "Minimal change: move assignment before the guard, or add a local variable binding.")
    if "arg-type" in err or ("incompatible type" in err and "expected" in err):
        return (120, 10,
            "A variable has the wrong type. Search the ENTIRE file for where it's DEFINED/INITIALIZED. "
            "The error says a value has the wrong type — find the line that initializes the collection/variable. "
            "Fix the initialization to use the correct type. Do NOT change function signatures. "
            "If the fix is far from the error, use [FILE:] to show the full corrected file.")
    if "No overload variant" in err or ("incompatible type" in err and "expected" not in err):
        return (20, 10,
            "The type hint is too broad. Add explicit type parameters: List[T], dict[K,V], Optional[T]. "
            "Declare the list/dict with its full generic type at the variable definition site.")
    if "ROOT_CAUSE:" in err:
        return (50, 20,
            "This class is missing attributes needed by downstream files. "
            "Add the missing fields to the class definition (as @dataclass fields or class attributes). "
            "Show the complete corrected class definition with all fields.")
    if "COMPILE:" in err:
        return (10, 10,
            "Fix the syntax error at the indicated line. Show ONLY the corrected line(s). "
            "Keep surrounding code unchanged.")
    if "IMPORT:" in err:
        return (5, 5,
            "Fix the import path. Use bare imports for same-directory modules. "
            "Replace absolute 'from src.agent1.' prefixes with bare 'from ' imports.")
    if "CROSS:" in err:
        return (40, 30,
            "Add the missing attribute as a @property or class attribute in the referenced class. "
            "Show the complete class definition with the new attribute.")
    if "SMOKE:" in err:
        return (30, 20,
            "The class failed to instantiate. Check __init__ parameters and types. "
            "The test attempted to create an instance — ensure __init__ accepts the right arguments.")
    if "is not defined" in err or "Name" in err or "name-defined" in err:
        return (1, 40,
            "Missing import. Add 'from module import Name' at the top of the file. "
            "Look at the available exports in other project files — do NOT define the class yourself. "
            "IMPORT it from the file that already defines it.")
    if "Missing return statement" in err or "[return]" in err:
        return (-1, -1,
            "The function does not return on all code paths. The window below shows "
            "the ENTIRE enclosing function. Find where control can fall off the end "
            "without returning (e.g. an unmatched if/elif chain, or a try/except "
            "missing an else). Add a return for that unhandled path.")
    return (40, 20,
        "Fix this error in the code. Keep changes minimal — do NOT rewrite the entire file. "
        "Output ONLY the corrected file.")


def _extract_window(lines: list[str], error_line: int, before: int, after: int) -> str:
    """Extract a code window around *error_line* (1-indexed)."""
    idx = error_line - 1
    start = max(0, idx - before)
    end = min(len(lines), idx + after)
    result = []
    for i, line in enumerate(lines[start:end], start=start + 1):
        marker = ">>" if i == error_line else "  "
        result.append(f"{i:>4} {marker} {line.rstrip()}")
    return "\n".join(result)


def _trace_variable_source(err: str, lines: list[str], error_line: int) -> str:
    """For arg-type errors, trace the variable back to its initialization.

    Returns a code block showing the root cause (where the variable is initialized)
    with line numbers, or empty string if tracing fails.
    """
    if "arg-type" not in err and ("incompatible type" not in err or "expected" not in err):
        return ""

    # Extract function name and argument position from error
    # Pattern: Argument N to "func_name" has incompatible type "X"; expected "Y"
    func_match = re.search(r'Argument (\d+) to "(\w+)"', err)
    if not func_match:
        return ""
    arg_num = int(func_match.group(1))
    func_name = func_match.group(2)

    # Find the function definition and extract parameter name
    param_name = ""
    for i, line in enumerate(lines):
        if f"def {func_name}(" in line:
            params_match = _DEF_PARAMS_RE.search(line)
            if params_match:
                params = [p.strip().split(':')[0].strip().split('=')[0].strip()
                          for p in params_match.group(1).split(',')]
                if arg_num <= len(params):
                    param_name = params[arg_num - 1]
            break

    if not param_name or param_name == "self":
        return ""

    # Find where the parameter is assigned at the error site
    assigned_from = ""
    _ASSIGN_RE = re.compile(rf'{param_name}\s*=\s*(.+)')
    for i in range(error_line - 1, max(0, error_line - 20), -1):
        line = lines[i].strip()
        if f"{param_name} = " in line or f"{param_name}=" in line:
            rhs_match = _ASSIGN_RE.search(line)
            if rhs_match:
                assigned_from = rhs_match.group(1).strip()
            break

    if not assigned_from:
        return ""

    # Find where the source collection is initialized
    init_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(rf'self\.\w+\s*=\s*\[', stripped) or re.match(rf'self\.\w+\s*=\s*\{{', stripped):
            init_lines.append((i + 1, line.rstrip()))

    if not init_lines:
        return ""

    # Build the root cause section
    result_parts = []
    result_parts.append(f"Variable `{param_name}` is assigned from `{assigned_from}` (see error site):")
    result_parts.append("")
    for i in range(error_line - 1, max(0, error_line - 10), -1):
        if param_name in lines[i]:
            result_parts.append(f"  {i + 1}: {lines[i].rstrip()}")
            break

    result_parts.append("")
    result_parts.append("The source collection is initialized here:")
    result_parts.append("")
    result_parts += [f"  {line_num}: {line_text}" for line_num, line_text in init_lines[-3:]]

    result_parts.append("")
    result_parts.append("Fix: Change the initialization to use the correct type (e.g. 0 instead of None).")

    return "\n".join(result_parts)


def _find_class_definition_file(class_name: str, ws_dir: str, exclude_file: str = "") -> tuple[str, str] | None:
    """Find the file that defines a class. Returns (file_path, source_content) or None.

    When multiple files define the same class, returns the one with the most
    attributes (most-derived / most complete definition).
    """
    matches: list[tuple[str, str, int]] = []
    for root, dirs, files in os.walk(ws_dir):
        if ".git" in root or "__pycache__" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            fp = os.path.normpath(os.path.join(root, f))
            if fp == exclude_file:
                continue
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as sf:
                    src = sf.read()
                if re.search(rf'^class\s+{re.escape(class_name)}\b', src, re.MULTILINE):
                    # Count attributes to prefer the most-complete definition
                    attr_count = len(re.findall(r'^\s+\w+\s*[:=]', src, re.MULTILINE))
                    matches.append((fp, src, attr_count))
            except Exception:
                print("WARNING: failed to read file during signature extraction:", f)  # silent_except fix

    if not matches:
        return None
    # Return the most-complete definition (most attributes)
    best = max(matches, key=lambda m: m[2])
    return (best[0], best[1])


_ATTR_NO_ATTR_RE = re.compile(r'"(\w+)" has no attribute "(\w+)"')
_FOR_NAME_RE = re.compile(r'for "(\w+)"')
_IMPORT_FROM_RE = re.compile(r"cannot import name '(\w+)' from '(\w+)'")
_ASSIGN_TYPE_RE = re.compile(r'Incompatible types in assignment.*variable has type "([^"]+)"')
_NAME_NOT_DEF_RE = re.compile(r'Name "(\w+)" is not defined')


def _group_related_errors(errors: list[tuple[str, str, str]], ws_dir: str) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Group errors by root cause. Returns [(root_cause_file, [errors])].

    Phase 1: Categorize errors into class/import/assignment/name/other buckets.
    Phase 2: Find definition files and create root-cause entries.
    Phase 3: Sort groups by file dependency (topological order).
    """
    class_errors: dict[str, list[tuple[str, str, str]]] = {}
    import_errors: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    assign_errors: dict[str, list[tuple[str, str, str]]] = {}
    name_errors: dict[str, list[tuple[str, str, str]]] = {}
    other_errors: list[tuple[str, str, str]] = []

    for fname, fpath, err in errors:
        # Pattern: "MouseEventData" has no attribute "button"
        cls_match = _ATTR_NO_ATTR_RE.search(err)
        if cls_match:
            class_errors.setdefault(cls_match.group(1), []).append((fname, fpath, err))
            continue
        # Pattern: Unexpected keyword argument "timestamp" for "MouseEventData"
        cls_match = _FOR_NAME_RE.search(err)
        if cls_match and "keyword argument" in err:
            class_errors.setdefault(cls_match.group(1), []).append((fname, fpath, err))
            continue
        # Pattern: cannot import name 'X' from 'Y'
        import_match = _IMPORT_FROM_RE.search(err)
        if import_match:
            key = (import_match.group(2), import_match.group(1))  # (source_module, missing_name)
            import_errors.setdefault(key, []).append((fname, fpath, err))
            continue
        # Pattern: Incompatible types in assignment
        assign_match = _ASSIGN_TYPE_RE.search(err)
        if assign_match:
            # Extract the variable name from the error (e.g. "agent.py:683: error: Incompatible types ...")
            var_match = re.search(r'Incompatible types in assignment.*"(\w+)"', err)
            if var_match:
                assign_errors.setdefault(var_match.group(1), []).append((fname, fpath, err))
                continue
        # Pattern: Name "X" is not defined
        name_match = _NAME_NOT_DEF_RE.search(err)
        if name_match:
            name_errors.setdefault(name_match.group(1), []).append((fname, fpath, err))
            continue
        other_errors.append((fname, fpath, err))

    result: list[tuple[str, list[tuple[str, str, str]]]] = []

    # Class-definition groups: root cause = file defining the class
    for cls_name, cls_errs in class_errors.items():
        defn = _find_class_definition_file(cls_name, ws_dir)
        if defn:
            defn_path, _ = defn
            root_err = (os.path.basename(defn_path), defn_path,
                        f"ROOT_CAUSE: {cls_name} is defined here but missing attributes needed by downstream files")
            result.append((defn_path, [root_err] + cls_errs))
        else:
            result.append(("", cls_errs))

    # Import-error groups: root cause = source module
    for (source_module, missing_name), imp_errs in import_errors.items():
        defn = _find_class_definition_file(missing_name, ws_dir)
        if defn:
            defn_path, _ = defn
            root_err = (os.path.basename(defn_path), defn_path,
                        f"ROOT_CAUSE: {missing_name} should be exported from this module")
            result.append((defn_path, [root_err] + imp_errs))
        else:
            result.append(("", imp_errs))

    # Assignment-type groups: root cause = file where variable is initialized
    for var_name, var_errs in assign_errors.items():
        # Try to find where the variable is defined/initialized
        defn_path = _find_variable_definition(var_name, var_errs, ws_dir)
        if defn_path:
            root_err = (os.path.basename(defn_path), defn_path,
                        f"ROOT_CAUSE: {var_name} has wrong type at definition site")
            result.append((defn_path, [root_err] + var_errs))
        else:
            result.append(("", var_errs))

    # Name-not-defined groups stay consumer-side: the name exists somewhere
    # (that is how mypy resolved it), so the minimal fix is an IMPORT in the
    # file reporting the error — not an edit to the defining module. Editing
    # the definer sent the LLM to "fix" a file that was never broken
    # (observed 2026-08-23: MetricsCollector group rewrote metrics_collector.py).
    for name_name, name_errs in name_errors.items():
        result.append(("", name_errs))

    # Ungrouped errors (no root cause detected)
    if other_errors:
        result.append(("", other_errors))

    return result


def _find_variable_definition(var_name: str, errors: list[tuple[str, str, str]], ws_dir: str) -> str | None:
    """Find where a variable is defined/initialized based on error context.

    Searches the files containing errors for assignment patterns like
    ``var = Something(...)`` or ``var: SomeType = ...``.
    """
    # Collect unique files from the errors
    error_files = {fpath for _, fpath, _ in errors if fpath}
    for fpath in error_files:
        if not fpath or not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as sf:
                src = sf.read()
        except Exception:
            continue
        # Look for initialization patterns
        patterns = [
            rf'^\s*{re.escape(var_name)}\s*=\s*\w+',
            rf'^\s*{re.escape(var_name)}\s*:\s*\w+.*=',
        ]
        for pat in patterns:
            if re.search(pat, src, re.MULTILINE):
                return fpath
    return None


def _topological_sort_groups(groups: list[tuple[str, list[tuple[str, str, str]]]], ws_dir: str) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Sort error groups by file dependency order.

    If root-cause file A imports from root-cause file B, then B is fixed
    first (its exports need to exist before A can reference them).

    Groups with empty root-cause paths (unknown root) are placed at the end
    since they cannot participate in the dependency graph.
    """
    if len(groups) <= 1:
        return groups

    # Build dependency graph: for each root-cause file, find its imports
    # from other root-cause files in this batch
    root_files = {}  # path -> index in groups
    for i, (group_path, _) in enumerate(groups):
        if group_path:
            root_files[group_path] = i

    # Parse imports in each root-cause file
    deps: dict[int, set[int]] = {i: set() for i in range(len(groups))}
    for group_path, idx in root_files.items():
        if not os.path.isfile(group_path):
            continue
        try:
            with open(group_path, "r", encoding="utf-8", errors="replace") as sf:
                src = sf.read()
        except Exception:
            continue
        # Find imports from project modules
        for m in re.finditer(r'from\s+([\w.]+)\s+import', src):
            mod = m.group(1)
            imported_file = os.path.normpath(os.path.join(ws_dir, mod.replace('.', '/') + '.py'))
            if imported_file in root_files and imported_file != group_path:
                deps[idx].add(root_files[imported_file])

    # DFS topological sort (Kahn's algorithm variant)
    visited = set()
    order: list[int] = []

    def dfs(node: int) -> None:
        if node in visited:
            return
        visited.add(node)
        for dep in deps.get(node, set()):
            dfs(dep)
        order.append(node)

    # Start with groups that have no dependencies on other groups
    for i in range(len(groups)):
        dfs(i)

    # Reorder: groups with empty path go to the end
    has_path = [(i, groups[i]) for i in order if groups[i][0]]
    no_path = [(i, groups[i]) for i in order if not groups[i][0]]

    return [g for _, g in has_path + no_path]


def _build_fix_prompt(err: str, current_code: str, fname: str, prefer_file: bool = False) -> list[dict[str, str]]:
    """Build a strategy-specific fix prompt with relevant code window only.
    
    If prefer_file is True (e.g. after a patch failure), tell the LLM to use [FILE:] format.
    """
    lines = current_code.split("\n")
    error_line = _parse_line_number(err)
    before, after, instruction = _classify_error(err)
    window = _extract_window(lines, error_line, before, after)

    # For arg-type errors, trace the variable back to its initialization
    root_cause = _trace_variable_source(err, lines, error_line)
    root_section = ""
    if root_cause:
        root_section = f"## Root cause — READ THIS FIRST\n{root_cause}\n\n"

    # For import-related errors, also show the file header (first 40 lines)
    header = ""
    if "import" in instruction.lower() or "is not defined" in instruction or "Name" in instruction:
        header_end = min(40, len(lines))
        header_lines = [f"{i:>4}    {line.rstrip()}" for i, line in enumerate(lines[:header_end], start=1)]
        header = "File header (imports):\n```python\n" + "\n".join(header_lines) + "\n```\n\n"

    sys_msg = f"Fix this specific issue. {instruction}"
    if prefer_file:
        sys_msg += " Patches failed before. Use [FILE:] format — output the complete corrected file."
        # Include the FULL file so the LLM can generate a valid [FILE:] block
        full_code_lines = [f"{i:>4}    {line.rstrip()}" for i, line in enumerate(lines[:300], start=1)]
        if len(lines) > 300:
            full_code_lines.append(f"  ... ({len(lines) - 300} more lines)")
        full_code = "\n".join(full_code_lines)
        user_msg = (
            f"Error in {fname}, line {error_line}:\n{err}\n\n"
            f"{root_section}"
            f"Full file (first 300 lines):\n```python\n{full_code}\n```\n\n"
            f"[FILE: {fname}]\n"
            "```python\n# complete corrected file\n```\n\n"
            "Output the complete corrected file using [FILE:] format. Do NOT use [PATCH:]."
        )
    else:
        user_msg = (
            f"Error in {fname}, line {error_line}:\n{err}\n\n"
            f"{root_section}"
            f"{header}"
            f"Relevant code:\n```python\n{window}\n```\n\n"
            "Output the fix using ONE of these formats:\n\n"
            "Option A — [PATCH:] for small fixes near the error:\n"
            f"[PATCH: {fname}]\n"
            "@@ -line,count +line,count @@\n"
            " unchanged context line\n"
            "-removed line\n"
            "+added line\n"
            " unchanged context line\n\n"
            "Option B — [FILE:] when the fix is far from the error or needs full context:\n"
            f"[FILE: {fname}]\n"
            "```python\n# complete corrected file\n```\n\n"
            "Choose the format that makes the fix clearest. Do NOT output anything else."
        )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]


def _build_root_cause_prompt(class_name: str, class_src: str, downstream_errors: list[tuple[str, str, str]], prefer_file: bool = False) -> list[dict[str, str]]:
    """Build a prompt for fixing a root cause class with downstream error context.
    
    Shows the LLM:
    1. The class definition that needs fixing
    2. The downstream errors that will be fixed by this change
    3. Explicit instruction to fix the class first
    """
    # Extract the class definition from the source
    lines = class_src.split("\n")
    class_start = -1
    class_end = len(lines)
    for i, line in enumerate(lines):
        if re.match(rf'^class\s+{re.escape(class_name)}\b', line):
            class_start = i
            break
    if class_start == -1:
        class_start = 0
    
    # Find class end (next class or end of file)
    _RE_2 = re.compile(r'^class\s+\w+')
    for i in range(class_start + 1, len(lines)):
        if _RE_2.match(lines[i]):
            class_end = i
            break
    
    class_window = "\n".join(f"{i+1:>4}    {line.rstrip()}" for i, line in enumerate(lines[class_start:class_end], start=class_start))
    
    # Build downstream error summary
    error_summary: list[str] = []
    for fname, fpath, err in downstream_errors:
        error_summary.append(f"- {fname}: {err[:200]}")
    
    downstream_section = ""
    if error_summary:
        downstream_section = "## Downstream errors that this fix will resolve\n" + "\n".join(error_summary) + "\n\n"
    
    sys_msg = f"Fix the {class_name} definition. Add all missing fields/attributes."
    if prefer_file:
        sys_msg += " Patches failed before. Use [FILE:] format — output the complete corrected file."
        user_msg = (
            f"The class `{class_name}` is missing attributes needed by downstream files.\n\n"
            f"{downstream_section}"
            f"## Current class definition\n```python\n{class_window}\n```\n\n"
            "Add the missing fields to this class. For @dataclass, add new fields with types. "
            "For regular classes, add them in __init__.\n\n"
            f"Output the complete corrected file:\n[FILE: ...]\n```python\n# complete corrected file\n```"
        )
    else:
        user_msg = (
            f"The class `{class_name}` is missing attributes needed by downstream files.\n\n"
            f"{downstream_section}"
            f"## Current class definition\n```python\n{class_window}\n```\n\n"
            "Add the missing fields to this class. For @dataclass, add new fields with types. "
            "For regular classes, add them in __init__.\n\n"
            "Output the fix using ONE of these formats:\n\n"
            "Option A — [PATCH:] for small changes near the class definition:\n"
            "[PATCH: filename.py]\n"
            "@@ -line,count +line,count @@\n"
            " unchanged context line\n"
            "-removed line\n"
            "+added line\n"
            " unchanged context line\n\n"
            "Option B — [FILE:] when the change is large or spans multiple sections:\n"
            "[FILE: filename.py]\n"
            "```python\n# complete corrected file\n```\n\n"
            "Choose the format that makes the fix clearest. Do NOT output anything else."
        )

    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]


_FIX_PATCH_RE = re.compile(r'\[PATCH:\s*([^\]]+)\]\s*\n?(.*?)(?=\[PATCH:|\Z)', re.DOTALL)
_FIX_FILE_RE = re.compile(r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)(?:\n```|$)', re.DOTALL)
_CLASS_DEF_RE = re.compile(r'^class\s+(\w+)', re.MULTILINE)
_DEF_PARAMS_RE = re.compile(r'def\s+\w+\s*\((.*?)\)', re.DOTALL)


def _find_patch_block(response: str) -> re.Match[str] | None:
    """Extract the first ``[PATCH:]`` block from an LLM response."""
    return _FIX_PATCH_RE.search(response)


def _find_file_block(response: str) -> re.Match[str] | None:
    """Extract the first ``[FILE:]`` block from an LLM response."""
    return _FIX_FILE_RE.search(response)


def _block_target_matches(captured_name: str, fname: str) -> bool:
    """True when an LLM-emitted block name refers to *fname* (invariant #3).

    Normalizes separators and case; an exact relative-path match or a
    basename match is accepted (LLMs often emit just ``file.py``). An empty
    capture is treated as matching.
    """
    cap = captured_name.strip().replace("\\", "/").strip("./").lower()
    tgt = fname.strip().replace("\\", "/").strip("./").lower()
    if not cap:
        return True
    if cap == tgt:
        return True
    return os.path.basename(cap) == os.path.basename(tgt)


_FENCE_TAIL_RE = re.compile(r'```\s*$')
_CODE_HINT_RE = re.compile(r'\b(import|def |class )\b')


def _announce(text: str) -> None:
    print(f"  [llm] {text}", flush=True)


def _tool_call_retry_msgs(fname: str, err: str) -> list[dict[str, str]]:
    """Strict retry prompt when the model answered with tool-call XML."""
    return [
        {"role": "system", "content": "You are fixing a Python type error. Output ONLY a [PATCH: filename.py] or [FILE: filename.py] block. No tool calls, no prose."},
        {"role": "user", "content": f"Fix: {err[:300]}\n\nOutput:\n[PATCH: {fname}]\n@@ -line,count +line,count @@\n-removed\n+added"},
    ]


def _format_reminder_msgs(fname: str, err: str) -> list[dict[str, str]]:
    """Strict retry prompt when the answer contained no code block."""
    return [
        {"role": "system", "content": "You must output ONLY a [PATCH: filename.py] or [FILE: filename.py] block. No prose, no explanations."},
        {"role": "user", "content": f"Fix the error in {fname}.\n\nError: {err[:300]}\n\nOutput format:\n[PATCH: {fname}]\n@@ -line,count +line,count @@\n-removed\n+added\n\nOR:\n[FILE: {fname}]\n```python\n# complete corrected file\n```"},
    ]


async def _chat_fix_text(agent: Any, msgs: list[dict[str, str]], fname: str, err: str) -> str:
    """One fix request incl. retries: tool-call rejection and one format
    reminder when no code block is present. ``[Error ...]`` strings pass through.
    """
    fixed = str(await agent.llm.chat(msgs, disable_thinking=True))
    if fixed.startswith("[Error") or fixed.startswith("[LM Studio"):
        return fixed
    if "<tool_call" in fixed:
        _announce("retrying (tool-call response rejected)")
        fixed = str(await agent.llm.chat(_tool_call_retry_msgs(fname, err), disable_thinking=True))
        if fixed.startswith("[Error") or fixed.startswith("[LM Studio"):
            return fixed
    if "[PATCH:" not in fixed and "[FILE:" not in fixed:
        print(f"  [debug] no code block in LLM response (first 500 chars):\n{fixed[:500]}\n  ---")
        _announce("retrying with format reminder")
        fixed = str(await agent.llm.chat(_format_reminder_msgs(fname, err), disable_thinking=True))
    return fixed


def _apply_file_block(raw_code: str, fpath: str, backup: str) -> tuple[bool, str]:
    """Write an [FILE:] body to *fpath*, py_compile it; roll back on failure."""
    new_code = _FENCE_TAIL_RE.sub('', raw_code.strip()).strip()
    if not _CODE_HINT_RE.search(new_code):
        return False, "content is not valid Python code"
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(new_code)
    r = subprocess.run(["python", "-m", "py_compile", fpath], capture_output=True, text=True)
    if r.returncode == 0:
        return True, f"{len(new_code)} bytes"
    tail = r.stderr.strip().splitlines()[-1][:150] if r.stderr.strip() else ""
    truncated = "truncated" in tail.lower() or "EOF" in tail
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(backup)
    label = "compile error (truncated output)" if truncated else "compile error"
    return False, f"{label}, rolled back: {tail}"


def _try_patch_block(patch_text: str, fpath: str, backup: str) -> tuple[bool, str]:
    """Apply a [PATCH:] hunk with diff preview; write only on success."""
    from agent_core.patch_utils import split_source_lines

    ok, new_text = _apply_patch(patch_text, fpath, split_source_lines(backup))
    if not ok:
        return False, new_text[:200]
    show_file_diff(fpath, backup, new_text)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(new_text)
    return True, "patch applied"


def _consume_fix_blocks(
    fixed: str, fname: str, fpath: str, current_code: str, file_first: bool
) -> tuple[bool, str]:
    """Parse, guard (invariant #3) and apply code blocks from an LLM response.

    Tries both formats in *file_first* order, falling back to the other when
    the preferred one is absent or fails. Returns (applied, note).
    """
    blocks = [("FILE", _find_file_block(fixed)), ("PATCH", _find_patch_block(fixed))]
    if not file_first:
        blocks.reverse()
    saw_block = False
    note = ""
    for kind, m in blocks:
        if m is None:
            continue
        saw_block = True
        if not _block_target_matches(m.group(1), fname):
            print(f"  WARNING: ignoring [{kind}: {m.group(1)}] — targets another file (invariant #3)")
            continue
        if kind == "FILE":
            ok, note = _apply_file_block(m.group(2), fpath, current_code)
        else:
            ok, note = _try_patch_block(m.group(2), fpath, current_code)
        if ok:
            return True, note
        print(f"  [{kind}] failed: {note}")
    if not saw_block:
        return False, "no [PATCH:] or [FILE:] found in LLM response"
    return False, note or "all code blocks failed to apply"


def _apply_patch(patch_text: str, fpath: str, original_lines: list[str]) -> tuple[bool, str]:
    """Apply a unified-diff patch. Returns (success, result or error message).

    Delegates to the shared ``agent_core.patch_utils.apply_patch`` used by the
    optimize command's patch mode; *fpath* is kept for call-site compatibility.
    """
    from agent_core.patch_utils import apply_patch, split_source_lines
    return apply_patch(patch_text, original_lines)


def _unwired_closure(py_new: list[str], ws: str, initial: set[str]) -> set[str]:
    """Expand an initial delete set to its transitive closure.

    A generated file that is only referenced by other to-be-deleted generated
    files (or by nothing at all) joins the set, so answering 'y' removes the
    whole orphaned component — no dead module survives as a leftover. Files
    referenced by surviving generated files or by real project code are
    pinned and never added.

    Uses the same substring heuristic as ``detect_unwired_modules`` but
    operates on the LOGICAL set (files still on disk are ignored once they
    are in the delete set), so the cascade works before any file is removed.
    """
    import os as _os

    def _referencers(mod_name: str, self_file: str) -> set[str]:
        refs: set[str] = set()
        for root, dirs, files in _os.walk(ws):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
            for f in files:
                if not f.endswith(".py"):
                    continue
                fp = _os.path.join(root, f)
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
                        content = fh.read()
                except OSError:
                    continue
                if mod_name in content:
                    rel = _os.path.relpath(fp, ws).replace(_os.sep, "/")
                    if rel != self_file:
                        refs.add(rel)
        return refs

    delete_set = set(initial)
    changed = True
    while changed:
        changed = False
        for f in py_new:
            if f in delete_set:
                continue
            mod_name = f.replace("/", ".").replace(".py", "")
            refs = _referencers(mod_name, f)
            # Pinned while any reference comes from a file that survives.
            if refs - delete_set:
                continue
            delete_set.add(f)
            changed = True
    return delete_set


def _filter_duplicate_planned(all_files: list[str], dup_reasons: list[str]) -> tuple[list[str], list[str]]:
    """Split *all_files* into (remaining, blocked_duplicates).

    Files named in *dup_reasons* are dropped; everything else is kept so the
    run can continue with the genuinely new / [MODIFY] entries.
    """
    blocked = {reason.split(" — ", 1)[0] for reason in dup_reasons}
    remaining = [f for f in all_files if f not in blocked]
    return remaining, sorted(blocked)


_CONSUMER_CONCEPTS: dict[str, list[str]] = {
    "chain_limiter": ["agent.py"],
    "self_mod_guard": ["agent.py"],
    "output_sanitizer": ["agent_core/security/sanitizer.py"],
    "tool_schema": ["agent_core/tool_schemas.py"],
    "shell_allowlist": ["agent_core/security/allowlist.py"],
    "path_guard": ["agent_core/path_utils.py", "agent_core/security/path_utils.py"],
    "normalizer": ["agent_core/path_utils.py"],
    "sanitizer_fix": ["agent_core/security/sanitizer.py"],
}


def _suggest_consumers(unwired_files: list[str], ws: str) -> dict[str, list[str]]:
    """Deterministic consumer suggestions for unwired modules.

    Uses shared-name-token matching against existing modules, a small concept
    map for known cases, and ``agent.py`` as the final fallback.
    """
    from agent_core.patterns import _shared_name_tokens

    existing: list[str] = []
    for root, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache", "backups")]
        for f in files:
            if f.endswith(".py") and not f.endswith("__init__.py"):
                existing.append(os.path.relpath(os.path.join(root, f), ws).replace("\\", "/"))

    suggestions: dict[str, list[str]] = {}
    for fname in unwired_files:
        stem = fname.rsplit("/", 1)[-1].replace(".py", "")
        hits: list[str] = []
        for other in existing:
            if other == fname:
                continue
            other_stem = other.rsplit("/", 1)[-1].replace(".py", "")
            if _shared_name_tokens(stem, other_stem):
                hits.append(other)
        hits.extend(_CONSUMER_CONCEPTS.get(stem, []))
        seen: set[str] = set()
        ordered: list[str] = []
        for h in hits:
            if h not in seen:
                seen.add(h)
                ordered.append(h)
        suggestions[fname] = ordered[:3] or ["agent.py"]
    return suggestions


async def _wire_in_modules(agent: "Agent", files: list[str], suggestions: dict[str, list[str]], ws: str) -> None:
    """One LLM pass that imports + wires each module into its suggested
    consumer; every patch is applied and py_compile-verified.  Unwirable
    modules are reported honestly instead of being marked wired."""
    for fname in files:
        if stop_requested():
            break
        path = Path(ws) / fname
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"    Could not read {fname}: {exc}")
            continue
        consumers = suggestions.get(fname, ["agent.py"])[:1]
        consumer_path = Path(ws) / consumers[0]
        if not consumer_path.exists():
            print(f"    Consumer {consumers[0]} missing — skipping {fname}")
            continue
        consumer_src = consumer_path.read_text(encoding="utf-8")
        prompt = (
            "Wire the module below into the consumer file. Produce ONLY [PATCH: <consumer>] "
            "unified-diff blocks that add the import and ONE sensible usage call site. "
            "Do not rewrite the consumer.\n\n"
            f"## Module to wire: {fname}\n{content[:4000]}\n\n"
            f"## Consumer: {consumers[0]}\n{consumer_src[:6000]}"
        )
        r = await agent.llm.chat([
            {"role": "system", "content": (
                "You are a Python integrator. Output ONLY [PATCH: file] blocks in unified-diff "
                "format. No prose, no [FILE:], no XML tags."
            )},
            {"role": "user", "content": prompt},
        ], disable_thinking=True)
        if r.startswith("[Error") or r.startswith("[LM Studio"):
            print(f"    Wire-in failed for {fname}: {r[:120]}")
            continue
        ok_count = 0
        for m in re.finditer(r"\[PATCH:\s*([^\]]+)\]\s*\n(.*?)(?=\[PATCH:|\Z)", r, re.DOTALL):
            patch_file = m.group(1).strip().strip("`")
            patch_text = m.group(2)
            target = Path(ws) / patch_file
            if not target.exists():
                print(f"    Wire-in: patch target {patch_file} does not exist — skipped")
                continue
            original = target.read_text(encoding="utf-8").splitlines()
            ok, result = _apply_patch(patch_text, str(target), original)
            if ok:
                verify = subprocess.run(
                    ["python", "-m", "py_compile", str(target)],
                    capture_output=True, text=True,
                )
                if verify.returncode == 0:
                    print(f"    Wired: {patch_file} (imports {fname})")
                    ok_count += 1
                else:
                    detail = verify.stderr.strip().splitlines()[-1][:120] if verify.stderr else ""
                    print(f"    Patch applied but py_compile failed for {patch_file} — manual check needed: {detail}")
            else:
                print(f"    Wire-in patch failed for {patch_file}: {result[:120]}")
        if ok_count == 0:
            print(f"    Could not wire {fname} automatically — left for manual integration.")


def _is_planned_test_file(fname: str) -> bool:
    """A planned TEST file is never a near-duplicate by design: it
    intentionally mirrors the module it covers (e.g. ``test_agent_chat_nlp.py``
    vs ``agent.py``), so the semantic gate must not flag or drop it."""
    f = fname.replace("\\", "/")
    base = f.rsplit("/", 1)[-1]
    return (
        base.startswith("test_")
        or base.startswith("conftest.")
        or f.startswith("tests/")
        or "/tests/" in "/" + f
    )


def _ensure_package_inits(start: Path, ws_root: Path) -> None:
    """Ensure ``__init__.py`` exists in every package directory of *start*
    up to *ws_root* so ``from src.agent1.xxx import`` works regardless of
    import style.  Never overwrites an existing ``__init__.py`` that has
    content.

    Directories under a ``tests/`` tree and under ``src/`` are skipped:
    packageifying ``tests/`` breaks implicit-sibling test imports (pytest
    collection failure), and ``src`` is intentionally a PEP 420 namespace
    root (no committed ``src/__init__.py``) — touching it would turn
    namespace-only ``src.*`` imports into regular-package artifacts.
    """
    curr = start
    while curr != ws_root and curr != curr.parent:
        try:
            rel = curr.relative_to(ws_root).as_posix()
        except ValueError:
            break
        if not (
            rel == "tests"
            or rel.startswith("tests/")
            or rel == "src"
            or rel.startswith("src/")
        ):
            init = curr / "__init__.py"
            if not init.exists() or init.stat().st_size == 0:
                init.touch()
        curr = curr.parent


def _check_planned_duplicates(planned_new: list[str], ws: str, taskplan_content: str = "") -> list[str]:
    """Return human-readable reasons why *planned_new* modules duplicate
    existing project modules.

    Combines the deterministic name gates (:func:`detect_module_collisions`)
    with the precision-first semantic layer (:class:`ModuleSimilarity`): TF-IDF
    cosine over docstrings + task descriptions, and an optional embeddings
    backend.  Each reason carries the evidence that fired.  Planned test files
    are exempt (:func:`_is_planned_test_file`).
    """
    from agent_core.patterns import detect_module_collisions
    from agent_core.utils.module_similarity import ModuleSimilarity, PlannedModule

    planned_new = [f for f in planned_new if not _is_planned_test_file(f)]

    existing: list[str] = []
    for root, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache", "backups")]
        for f in files:
            if f.endswith(".py"):
                existing.append(os.path.relpath(os.path.join(root, f), ws).replace("\\", "/"))

    reasons: list[str] = []
    for finding in detect_module_collisions(planned_new, existing_files=existing):
        fname = finding["file"]
        suggestion = finding["suggestion"]
        hits = [e for e in existing if e.rsplit("/", 1)[-1] in suggestion]
        reasons.append(f"{fname} — duplicates existing module(s): {', '.join(hits[:3])}")

    # Semantic (geometric) layer — task descriptions sharpen the features.
    descriptions = {}
    if taskplan_content:
        for fname in planned_new:
            descriptions[fname] = _extract_task_line(taskplan_content, fname)
    planned = [
        PlannedModule(fname, descriptions.get(fname, "")) for fname in planned_new
    ]
    if planned:
        sim = ModuleSimilarity(ws)
        for finding in sim.find_duplicates(planned):
            reasons.append(
                f"{finding.file} — {finding.evidence} -> {finding.existing}"
            )
    return reasons


def _prune_empty_dirs(ws: str, deleted_files: set[str]) -> None:
    """Remove package __init__.py files and directories left empty after
    deletion (e.g. a generated package whose only module was deleted)."""
    dirs: set[str] = set()
    for f in deleted_files:
        parent = os.path.dirname(f)
        if parent:
            dirs.add(parent)
    for d in sorted(dirs, key=lambda p: p.count(os.sep), reverse=True):
        full = os.path.join(ws, d.replace("/", os.sep))
        try:
            entries = list(Path(full).iterdir())
        except OSError:
            continue
        if entries and all(e.name == "__init__.py" for e in entries):
            for e in entries:
                e.unlink()
                print(f"    Removed empty package marker: {e}")
        try:
            Path(full).rmdir()
            print(f"    Removed empty directory: {full}")
        except OSError:
            pass


def _run_python_snippet(ws: str, extra_paths: list[str], code_lines: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a temp file with workspace and extra paths in sys.path.

    Returns the CompletedProcess from subprocess.run.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(f"import sys; sys.path.insert(0, {repr(ws)})\n")
        for p in extra_paths:
            tf.write(f"sys.path.insert(0, {repr(p)})\n")
        for line in code_lines:
            tf.write(line + "\n")
        tfpath = tf.name
    r = subprocess.run(["python", tfpath], capture_output=True, text=True, cwd=str(Path(ws)))
    os.unlink(tfpath)
    return r


class ImplementCommand(Command):
    """Implement files from task plan using LLM."""

    @property
    def name(self) -> str:
        return "implement"

    @property
    def help_text(self) -> str:
        return (
            "implement <taskplan.md> [analysis.md] [plan.md] [entities.md] [--workspace <path>] "
            "- Implement files from a task plan\n"
            "  Flags:\n"
            "    --modify         BROWNFIELD MODE: existing .py files get a reviewed diff-apply\n"
            "                     instead of being skipped as \"already exists\".\n"
            "    --force          Overwrite existing files wholesale (DANGEROUS on brownfield).\n"
            "    --allow-rewrite  With --modify: permit a diff that is a wholesale rewrite\n"
            "                     (similarity < 0.5) instead of rejecting it.\n"
            "    --fix            Auto-fix loop: repair syntax/mypy failures and retry.\n"
            "    --review         Deep LLM review pass of the generated code (slow).\n"
            "    --retry          Retry failed generation batches.\n"
            "    --keep           Resume from a matching implement cache (true resume).\n"
            "    --refresh        With --keep: ignore the cached file list and rebuild it\n"
            "                     from the taskplan before resuming.\n"
            "    --status         Read-only plan progress report (exists / compiles /\n"
            "                     stdlib-shadow state per file). Makes NO LLM calls.\n"
            "  Brownfield tip: for an existing repo use --modify so [MODIFY] tasks are applied\n"
            "  as reviewed diffs; without it existing files are skipped. Add --fix for the\n"
            "  auto-fix loop, and --allow-rewrite only when a rewrite is truly wanted."
        )

    def _apply_modify_diff(self, filename: str, filepath: Path, content: str, allow_rewrite: bool) -> tuple[bool, str]:
        """--modify: merge *content* into the existing *filepath* as a reviewed
        unified diff instead of skipping it (default) or overwriting it
        wholesale (--force).

        Returns (applied, note).  The diff is generated from the two buffers,
        applied through the shared tolerant ``patch_utils`` machinery, compiled,
        shown with ``show_file_diff``, and applied only after approval (safe
        auto-default: decline in autonomous mode).
        """
        try:
            prev = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return False, f"skipped — cannot read existing file: {exc}"
        if not filename.endswith(".py"):
            return False, "skipped — --modify applies to .py files only"
        if prev.strip() == content.strip():
            return False, "unchanged — generated content matches the file"

        similarity = difflib.SequenceMatcher(None, prev, content).ratio()
        if similarity < 0.5 and not allow_rewrite:
            return False, (
                f"rejected — wholesale rewrite of existing file "
                f"(similarity {similarity:.2f}); use --allow-rewrite to force"
            )

        diff_text = "\n".join(
            difflib.unified_diff(prev.splitlines(), content.splitlines(), lineterm="")
        )
        if not diff_text.strip():
            return False, "unchanged — no diff"

        from agent_core.patch_utils import apply_patch, apply_anchored_patch
        ok, patched = apply_patch(diff_text, prev.splitlines())
        if not ok:
            ok, patched = apply_anchored_patch(diff_text, prev.splitlines())
        if not ok:
            return False, f"rejected — could not apply generated diff: {str(patched)[:200]}"
        patched_text = str(patched)
        if patched_text == prev:
            return False, "unchanged — diff produced no change"

        try:
            compile(patched_text, filename, "exec")
        except SyntaxError as exc:
            return False, f"rejected — patched file does not compile: {exc}"

        show_file_diff(filename, prev, patched_text)
        choice = auto_choice(f"  Apply {filename}? (y/N): ", default="n", auto_default="n").strip().lower()
        if choice not in ("y", "yes"):
            if stop_requested():
                print("  Stopping the flow — no further changes will be applied.")
            return False, "skipped by user"
        try:
            filepath.write_text(patched_text, encoding="utf-8")
        except OSError as exc:
            return False, f"skipped — write failed: {exc}"
        return True, f"modified ({diff_text.count(chr(10)) + 1} diff lines)"

    @staticmethod
    def _status_report(all_files: list[str], target_workspace: str) -> None:
        """``implement --status``: read-only plan progress, zero LLM calls.

        For every file in the cached/planned list it reports exists/compiles/
        shadow state, so a resumed run can be sanity-checked before spending
        generation budget.  Mirrors the exact predicates the real run uses
        (:func:`file_needs_generation` + ``_shadowing_stdlib_dir``).
        """
        from agent_core.commands.implement_cmd import _shadowing_stdlib_dir

        ws_path = Path(workspace_path(target_workspace))
        ready: list[str] = []
        needs_gen: list[tuple[str, str]] = []
        shadowed: list[str] = []
        for fname in all_files:
            fpath = ws_path / fname
            # Shadow check FIRST: a planned file under a NOT-yet-existing
            # stdlib-shadowing directory (e.g. `logging/formatter.py` when no
            # logging/ package exists) must be flagged even though it does
            # not exist yet — that is precisely the dangerous case.
            shadow = _shadowing_stdlib_dir(fname, ws_path)
            if shadow:
                shadowed.append(f"{fname} (dir '{shadow}' shadows stdlib)")
                continue
            if not fpath.exists():
                needs_gen.append((fname, "not found"))
                continue
            if fpath.stat().st_size == 0:
                needs_gen.append((fname, "empty file"))
                continue
            if fname.endswith(".py"):
                result = subprocess.run(
                    ["python", "-m", "py_compile", os.path.realpath(fpath)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    tail = result.stderr.strip().splitlines()[-1][:120] if result.stderr.strip() else ""
                    needs_gen.append((fname, f"compile failed: {tail}"))
                    continue
            ready.append(fname)

        print(f"\n  [status] {len(ready)} ready / {len(needs_gen)} need work / "
              f"{len(shadowed)} stdlib-shadowing  —  {len(all_files)} planned")
        print("  Plan state is read from disk only; no LLM calls are made.\n")
        if ready:
            print(f"  Ready ({len(ready)}):")
            for fname in ready:
                print(f"    ✓ {fname}")
        if needs_gen:
            print(f"\n  Needs generation ({len(needs_gen)}):")
            for fname, why in needs_gen:
                print(f"    ✗ {fname} — {why}")
        if shadowed:
            print(f"\n  Stdlib-shadowing paths ({len(shadowed)}) — will be redirected:")
            for entry in shadowed:
                print(f"    ! {entry}")
        print()

    @staticmethod
    def _auto_review(py_new: list[str], ws: str) -> None:
        """Run fast static checks on new files (no LLM). Safe to call always."""
        print(f"\n{'─'*40}")
        found = 0
        from agent_core.patterns import detect_module_collisions, detect_class_conflicts, detect_unwired_modules

        collisions = detect_module_collisions(py_new)
        if collisions:
            print("  ⚠ Module name near existing files:")
            for c in collisions:
                print(f"    {c['file']}: {c['suggestion']}")
                found += 1

        conflicts = detect_class_conflicts(py_new, ws)
        if conflicts:
            print("  ⚠ Class/function name conflicts:")
            for cc in conflicts:
                print(f"    {cc['file']}:{cc['line']}: {cc['suggestion']}")
                found += 1

        unwired = detect_unwired_modules(py_new, ws)
        if unwired:
            print("  ⚠ New modules not imported by any code:")
            for uw in unwired:
                print(f"    {uw['file']}: {uw['suggestion']}")
                found += 1

        if not found:
            print("  ✓ Quick review: no conflicts or wiring issues.")
        print(f"{'─'*40}")

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        from agent_core.patch_utils import split_source_lines
        parts = args

        if len(parts) < 1:
            self.error("Usage: implement <taskplan.md> [analysis.md] [plan.md] [entities.md] [--keep] [--refresh] [--force] [--modify] [--fix] [--retry] [--review] [--allow-rewrite] [--no-history] [--status] [--workspace <path>]")
            return True

        keep_mode = "--keep" in parts
        refresh_cache = "--refresh" in parts
        force_mode = "--force" in parts
        modify_mode = "--modify" in parts
        fix_mode = "--fix" in parts
        retry_mode = "--retry" in parts
        review_mode = "--review" in parts
        allow_rewrite = "--allow-rewrite" in parts
        history_mode = "--no-history" not in parts
        status_mode = "--status" in parts

        target_workspace = agent.workspace
        if "--workspace" in parts:
            ws_idx = parts.index("--workspace")
            if ws_idx + 1 < len(parts):
                target_workspace = parts[ws_idx + 1].strip('"')

        skip_tokens = ["--keep", "--refresh", "--force", "--modify", "--fix", "--retry", "--review", "--workspace", "--allow-rewrite", "--no-history", "--status", target_workspace]
        filtered_parts = [p for p in parts if p not in skip_tokens]

        taskplan_file = filtered_parts[0] if filtered_parts else ""
        analysis_file = filtered_parts[1] if len(filtered_parts) > 1 else "analysis.md"
        plan_file = filtered_parts[2] if len(filtered_parts) > 2 else "plan.md"
        entities_file = filtered_parts[3] if len(filtered_parts) > 3 else "entities.md"

        # Resolve relative paths against the target workspace
        if not os.path.isabs(taskplan_file):
            taskplan_file = os.path.join(target_workspace, taskplan_file)
        if not os.path.isabs(analysis_file):
            analysis_file = os.path.join(target_workspace, analysis_file)
        if not os.path.isabs(plan_file):
            plan_file = os.path.join(target_workspace, plan_file)
        if not os.path.isabs(entities_file):
            entities_file = os.path.join(target_workspace, entities_file)

        # Docs now live in .docs/<timestamp>/ — when a name is not found as
        # given, fall back to the newest run folder (then the workspace root).
        resolved_taskplan = find_input(target_workspace, taskplan_file)
        if resolved_taskplan != taskplan_file:
            print(f"  Resolved taskplan: {resolved_taskplan}")
            taskplan_file = resolved_taskplan
        analysis_file = find_input(target_workspace, analysis_file)
        plan_file = find_input(target_workspace, plan_file)
        entities_file = find_input(target_workspace, entities_file)

        cache_file = os.path.join(os.path.dirname(os.path.realpath(taskplan_file)), ".implement_cache.json")

        analysis_content = ""
        plan_content = ""
        entities_content = ""
        taskplan_content = ""

        try:
            with open(taskplan_file, "r", encoding="utf-8") as f:
                taskplan_content = f.read()
        except FileNotFoundError:
            self.error(f"File not found: {taskplan_file}")
            return True

        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                analysis_content = f.read()
        except FileNotFoundError:
            print(f"Warning: {analysis_file} not found")

        try:
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_content = f.read()
        except FileNotFoundError:
            print(f"Warning: {plan_file} not found")

        try:
            with open(entities_file, "r", encoding="utf-8") as f:
                entities_content = f.read()
        except FileNotFoundError:
            print(f"Warning: {entities_file} not found")

        all_files = None

        if keep_mode and not refresh_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                # Check taskplan content hasn't changed (stale cache = wrong filenames)
                cached_hash = cache_data.get("taskplan_hash", "")
                current_hash = hashlib.md5(taskplan_content.encode()).hexdigest()[:8] if taskplan_content else ""
                if cache_data.get("taskplan") == taskplan_file and cached_hash == current_hash:
                    all_files = cache_data.get("files", [])
                    print(f"Using cached file list ({len(all_files)} files): {', '.join(all_files)}")
                else:
                    print("Taskplan changed — refreshing file list")
            except Exception:
                print("Warning: failed to load cache")

        if all_files is None:
            # Parse filenames directly from taskplan — the source of truth.
            # LLM is only a fallback when the taskplan format is non-standard.
            all_files = re.findall(r'`([^`]+\.py)`', taskplan_content)
            if not all_files:
                all_files = re.findall(r'`([^`\s]+\.[a-z]{2,4})`', taskplan_content)

            if not all_files:
                print("Analyzing task plan to identify all files...")
                list_messages = [
                    {"role": "system", "content": "List ALL files that need to be implemented from the task plan. Reply with ONLY filenames, one per line. No explanations.\n\nUse EXACTLY the filenames from the task plan. Do not rename or invent new filenames. Every file MUST use a sub-package prefix — never bare root-level names."},
                    {"role": "user", "content": f"List every file that needs to be created or modified from this task plan:\n\n## Task Plan:\n{taskplan_content}\n\n## Analysis:\n{analysis_content if analysis_content else 'N/A'}\n\n## Plan:\n{plan_content if plan_content else 'N/A'}\n\n## Entities:\n{entities_content if entities_content else 'N/A'}"}
                ]
                file_list_response = await agent.llm.chat(list_messages, disable_thinking=True)
                if not file_list_response or file_list_response.startswith("[Error") or file_list_response.startswith("[LM Studio"):
                    self.error(f"LM Studio API not responding or returned an error: {file_list_response}")
                    return True
                file_lines = [line.strip() for line in file_list_response.strip().split('\n') if line.strip() and not line.startswith('#')]
                all_files = [f for f in file_lines if f.endswith(('.py', '.json', '.yaml', '.yml', '.env', '.md', '.txt', '.cfg', '.ini', '.toml'))]
                if not all_files:
                    all_files = re.findall(r'`([^`]+\.(?:py|json|yaml|yml|env|txt|cfg|ini|toml))`', file_list_response)
                if not all_files:
                    all_files = re.findall(r'`([^`]+\.py)`', taskplan_content)

            # Deduplicate while preserving order
            all_files = list(dict.fromkeys(all_files))

            cache_data = {"taskplan": taskplan_file, "files": all_files, "taskplan_hash": hashlib.md5(taskplan_content.encode()).hexdigest()[:8] if taskplan_content else ""}
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f)
                print(f"Cached file list to {cache_file}")
            except Exception:
                print("Warning: failed to save cache")

        print(f"Found {len(all_files)} files to implement: {', '.join(all_files)}")

        # ---- Read-only status report: exit before any LLM call ----
        if status_mode:
            self._status_report(all_files, target_workspace)
            return True

        # ---- Layer 1: planned-new-file gate (before any generation) ----
        # A plan that proposes modules duplicating existing ones (e.g.
        # shell_allowlist.py vs allowlist.py) is almost always wrong — offer to
        # drop the duplicates and continue with the rest (default), force the
        # generation anyway, or abort.
        if not force_mode:
            planned_new = [
                f for f in all_files
                if f.endswith(".py")
                and not (Path(workspace_path(target_workspace)) / f).exists()
            ]
            if planned_new:
                dup_reasons = _check_planned_duplicates(
                    planned_new, str(target_workspace),
                    taskplan_content if taskplan_content else "",
                )
                if dup_reasons:
                    print("\n  [implement] Planned files duplicate existing modules:")
                    for reason in dup_reasons:
                        print(f"    {reason}")
                    choice = auto_choice(
                        "  Options: [m]odify-existing (drop duplicates, continue with the rest) "
                        "[f]orce-generate anyway [a]bort (default m): ",
                        default="m",
                        auto_default="m",
                    ).strip().lower()
                    if stop_requested():
                        return True
                    if choice in ("", "m", "modify", "modify-existing"):
                        remaining, blocked = _filter_duplicate_planned(all_files, dup_reasons)
                        dropped = set(all_files) - set(remaining)
                        if dropped:
                            print(
                                f"  Dropped {len(dropped)} planned duplicate(s) — "
                                "extend the existing module(s) instead:"
                            )
                            for b in blocked:
                                print(f"    - {b}")
                            all_files = remaining
                            try:
                                cache_data["files"] = all_files
                                with open(cache_file, "w", encoding="utf-8") as f:
                                    json.dump(cache_data, f)
                            except Exception:
                                print("Warning: failed to update cache after filtering")
                    elif choice in ("f", "force"):
                        force_mode = True
                        print("  --force semantics: generating the duplicate modules anyway.")
                    else:
                        print("  Aborted.")
                        return True

        def file_needs_generation(fname: str) -> tuple[bool, str]:
            raw_ws = workspace_path(target_workspace)
            fpath = Path(raw_ws) / fname
            if not fpath.exists():
                return True, "not found"
            if fpath.stat().st_size == 0:
                return True, "empty"
            if fname.endswith(".py"):
                result = subprocess.run(
                    ["python", "-m", "py_compile", os.path.realpath(fpath)],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    return True, f"compile failed: {result.stderr.strip()}"
                return False, "OK"
            return False, "OK"

        if retry_mode:
            missing = []
            for fname in all_files:
                shadow_name = _shadowing_stdlib_dir(fname, Path(workspace_path(target_workspace)))
                if shadow_name:
                    missing.append(fname)
                    print(f"  SHADOW: {fname} shadows stdlib '{shadow_name}'")
                elif file_needs_generation(fname)[0]:
                    missing.append(fname)
                else:
                    print(f"  OK: {fname}")
            if not missing:
                print("  All files present and compile OK — nothing to retry.")
                return True
            print(f"\n  Retrying {len(missing)} missing file(s): {', '.join(missing)}\n")
            all_files = missing
            force_mode = True  # Overwrite anything that exists but doesn't compile

        protected_files = set()
        if os.path.exists(".protected"):
            with open(".protected", "r", encoding="utf-8") as pf:
                for line in pf:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        protected_files.add(line)
        print(f"Protected files: {protected_files}")

        analyzed_file = ""
        match = re.search(r'# Analysis of (\S+)', analysis_content)
        if match:
            analyzed_file = match.group(1)
            print(f"Analyzed file from analysis.md: {analyzed_file}")

        implemented = []
        file_outcomes: dict[str, str] = {}  # filename -> reason/status
        if keep_mode and not force_mode:
            files_to_skip = []
            for fname in all_files:
                is_analyzed = analyzed_file and fname == analyzed_file
                if is_analyzed:
                    files_to_skip.append(f"{fname}: force-regenerate (was analyzed)")
                    continue
                needs_gen, reason = file_needs_generation(fname)
                if needs_gen:
                    files_to_skip.append(f"{fname}: {reason}")
                    file_outcomes[fname] = f"needs generation — {reason}"
                else:
                    shadow_name = _shadowing_stdlib_dir(fname, Path(workspace_path(target_workspace)))
                    if shadow_name:
                        files_to_skip.append(
                            f"{fname}: COMPILED BUT SHADOWS stdlib '{shadow_name}' — must be renamed"
                        )
                        file_outcomes[fname] = f"shadows stdlib '{shadow_name}'"
                    else:
                        if modify_mode and fname.endswith(".py"):
                            # BROWNFIELD: with --modify an existing compile-OK
                            # file is a diff-apply target, NOT a skip — leave it
                            # out of `implemented` so it flows into
                            # files_to_generate and reaches the modify branch.
                            files_to_skip.append(f"{fname}: modify target (diff-apply)")
                        else:
                            files_to_skip.append(f"{fname}: already exists, compile OK")
                            file_outcomes[fname] = "already exists and compiles OK"
                            if fname not in implemented:
                                implemented.append(fname)

            print(f"\nFiles to skip (already exist and compile): {len(files_to_skip)}")
            for fs in files_to_skip:
                print(f"  - {fs}")

            files_to_generate = [fname for fname in all_files if fname not in implemented]
            print(f"\nFiles to generate: {len(files_to_generate)}: {', '.join(files_to_generate)}")

            if not files_to_generate:
                print("All files already exist and compile. Nothing to do.")
                if not fix_mode:
                    return True
                print("\n[fix] Running validation on existing files...")
                implemented = [f for f in all_files if f.endswith(".py")]
                # all_files = implemented  # dead assignment removed

            all_files = files_to_generate

        errors = []
        batch_size = 1

        pre_snapshot = set()
        ws = target_workspace
        ws = to_windows_path(ws)
        for fp in Path(ws).rglob("*"):
            if fp.is_file() and ".git" not in str(fp) and "__pycache__" not in str(fp):
                pre_snapshot.add(str(fp.relative_to(Path(ws))).replace("\\", "/"))

        def extract_signatures(source: str) -> dict[str, str]:
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

            for td_match in re.finditer(r'class\s+(\w+)\s*\(\s*(?:TypedDict|Protocol)\b', source):
                td_name = td_match.group(1)
                pos = td_match.end()
                paren_depth = 0
                while pos < len(source) and source[pos] != ':':
                    if source[pos] == '(': paren_depth += 1
                    elif source[pos] == ')': paren_depth -= 1
                    pos += 1
                if pos < len(source) and source[pos] == ':':
                    body_start = pos + 1
                    fields = []
                    lines = source[body_start:].split('\n')
                    for line in lines:
                        stripped = line.strip()
                        if not stripped or stripped.startswith('#') or stripped.startswith('"""'):
                            continue
                        if not line.startswith('    ') and stripped:
                            break
                        if ':' in stripped and not stripped.startswith('def ') and not stripped.startswith('class '):
                            field_name = stripped.split(':')[0].strip()
                            field_type = ':'.join(stripped.split(':')[1:]).strip()
                            fields.append(f"{field_name}: {field_type}")
                    if fields:
                        sigs[td_name] = f"{td_name} fields: {{{', '.join(fields[:8])}}}"

            return sigs

        export_map = {}
        for fname in list(all_files):
            fp = Path(ws) / fname
            if fp.exists():
                try:
                    existing = fp.read_text(encoding="utf-8")
                    sigs = extract_signatures(existing)
                    if sigs:
                        export_map[fname] = sigs
                except Exception:
                    print("WARNING: failed to read file during signature extraction:", fname)  # silent_except fix

        if export_map:
            total_exports = sum(len(v) for v in export_map.values())
            print(f"Initial export map: {total_exports} signatures from {len(export_map)} existing files")

            broken_existing = []
            for fname in export_map:
                fp = Path(ws) / fname
                if not fp.exists() or not fname.endswith(".py"):
                    continue
                r = subprocess.run(
                    ["python", "-c", f"import py_compile; py_compile.compile(r'{os.path.realpath(fp)}', doraise=True)"],
                    capture_output=True, text=True, cwd=str(Path(ws))
                )
                if r.returncode != 0:
                    broken_existing.append((fname, r.stderr.strip()[-150:]))

            if broken_existing:
                print(f"\n  WARNING: {len(broken_existing)} existing files fail py_compile:")
                for fname, err in broken_existing:
                    print(f"    {fname}: {err[:100]}")
                print("  These exports may be incomplete. Consider running --fix first.")

        generated_content = {}

        for i in range(0, len(all_files), batch_size):
            batch = all_files[i:i+batch_size]
            print(f"\nGenerating batch {i//batch_size + 1}/{(len(all_files) + batch_size - 1)//batch_size}: {batch}")

            # In --modify mode, classify each batch file as existing (needs
            # [PATCH:]) or new (needs [FILE:]) so the LLM generates the
            # right format — the wholesale-rewrite guard rejects [FILE:]
            # rewrites of existing files with similarity < 0.5.
            existing_files: set[str] = set()
            new_files: set[str] = set()
            if modify_mode:
                ws_path = Path(workspace_path(target_workspace))
                for bf in batch:
                    if (ws_path / bf).exists():
                        existing_files.add(bf)
                    else:
                        new_files.add(bf)

            batch_files_md = "\n".join([f"- {f}" for f in batch])
            target_file = batch[0]

            export_context = ""
            if export_map:
                export_lines = []
                for mod, sigs in sorted(export_map.items()):
                    if sigs:
                        sig_list = ", ".join(f"{name}: {sig}" for name, sig in sorted(sigs.items()))
                        export_lines.append(f"  {mod} -> {sig_list}")
                if export_lines:
                    export_context = "\n\nAvailable project modules (use only these names with these exact signatures):\n" + "\n".join(export_lines)

            task_context = _extract_task_line(taskplan_content, target_file)
            analysis_context = _extract_file_context(analysis_content, target_file)
            plan_context = _extract_file_context(plan_content, target_file)

            # Redirect stdlib-shadowing paths to safe alternatives. Only
            # redirect when the directory segment does NOT already exist in
            # the workspace — an existing package like `agent_core/utils/` is
            # a deliberate project choice and must not be rewritten.
            stdlib_shadow_warning = ""
            redirected_batch = []
            for planned_f in batch:
                shadow_part = _shadowing_stdlib_dir(planned_f, Path(workspace_path(target_workspace)))
                if shadow_part:
                    safe_name = shadow_part + "_utils"
                    new_f = planned_f.replace(shadow_part + "/", safe_name + "/", 1)
                    redirected_batch.append(new_f)
                    stdlib_shadow_warning = (
                        f"\n\nCRITICAL — The original path '{planned_f}' shadows stdlib module "
                        f"'{shadow_part}'. You MUST use '{new_f}' instead. "
                        f"Do NOT write to the shadowing path."
                    )
                else:
                    redirected_batch.append(planned_f)
            if redirected_batch != batch:
                batch = redirected_batch
                target_file = batch[0]
                batch_files_md = "\n".join([f"- {f}" for f in batch])

            # Build a collision warning: class/function names already in the target directory
            collision_warning = ""
            target_dir = os.path.dirname(target_file) if "/" in target_file or "\\" in target_file else ""
            if target_dir:
                taken: dict[str, list[str]] = {}
                for mod, sigs in export_map.items():
                    mod_dir = mod.split("/", 1)[0] if "/" in mod else ""
                    if mod_dir and (mod.startswith(target_dir + "/") or mod_dir == target_dir.split("/")[-1]):
                        for name in sigs:
                            if not name.startswith("__"):
                                taken.setdefault(name, []).append(mod)
                if taken:
                    taken_list = ", ".join(f"{n} (in {', '.join(fs[:2])})" for n, fs in sorted(taken.items())[:15])
                    collision_warning = f"\n\nCRITICAL — DO NOT create these names in the new file: {taken_list}. They already exist in the target directory. Modify the existing file instead if you need to extend them."

            user_context = f"Implement ONLY this file — no other files:\n{batch_files_md}\n{export_context}{collision_warning}{stdlib_shadow_warning}"
            if task_context:
                user_context += f"\n\nTask: {task_context}"
            if analysis_context:
                user_context += f"\n\nRelevant analysis:\n{analysis_context}"
            if plan_context:
                user_context += f"\n\nRelevant plan:\n{plan_context}"

            # Inject past decisions as design constraints
            try:
                constraints = decisions_as_system_prompt(target_workspace, batch)
                if constraints:
                    user_context += constraints
            except Exception:
                pass

            # Inject past executions that touched this batch (2026-08-19:
            # implement consulted decisions but never past tool results/
            # errors, even though the trace corpus and execution ledger hold
            # them).  Purely additive prompt context — cannot affect the
            # write/cascade invariants.
            if history_mode:
                try:
                    from harnessfix.history import format_batch_history

                    history_block = format_batch_history(batch, workspace_path(target_workspace))
                    if history_block:
                        user_context += history_block
                except Exception:
                    pass

            if modify_mode and existing_files:
                # Modify mode: existing files get [PATCH:] (minimal diff),
                # new files get [FILE:] (complete implementation).  This
                # aligns the generate phase with the wholesale-rewrite
                # guard that rejects [FILE:] rewrites of existing files.
                existing_list = ", ".join(sorted(existing_files))
                new_list = ", ".join(sorted(new_files)) if new_files else "(none)"
                system_prompt = (
                    "You are an expert Python developer. Implement the specified files concisely.\n\n"
                    "RULES:\n"
                    "0. NEVER use <tool_call>, <function_call>, or XML tags. Respond in plain text only.\n"
                    "1. All code MUST pass mypy strict type checking and py_compile.\n"
                    "2. You receive an export map listing every class/function/constant that already exists.\n"
                    "3. NEVER redefine a name that already exists in the export map — IMPORT it instead.\n"
                    "4. NEVER create duplicate functions or classes — check the export map first.\n\n"
                    "OUTPUT FORMAT:\n"
                    f"EXISTING files (modify with minimal diff): {existing_list}\n"
                    "  Use [PATCH: filename.py] with unified diff hunks:\n"
                    "  [PATCH: filename.py]\n"
                    "  @@ -line,count +line,count @@\n"
                    "   unchanged context line\n"
                    "  -removed line\n"
                    "  +added line\n"
                    "   unchanged context line\n\n"
                    f"NEW files (complete implementation): {new_list}\n"
                    "  Use [FILE: filename.py] with the complete code:\n"
                    "  [FILE: filename.py]\n"
                    "  ```python\n"
                    "  # complete code\n"
                    "  ```\n\n"
                    "Only change lines that need modification. Do NOT rewrite entire existing files."
                )
            else:
                system_prompt = (
                    "You are an expert Python developer. Implement the specified files concisely.\n\n"
                    "RULES:\n"
                    "0. NEVER use <tool_call>, <function_call>, or XML tags. Respond in plain text with [FILE:] blocks only.\n"
                    "1. All code MUST pass mypy strict type checking and py_compile.\n"
                    "2. You receive an export map listing every class/function/constant that already exists and which file defines it. IMPORT those names — NEVER redefine a name that already exists in the export map. If 'Grid' is listed under grid.py, write 'from grid import Grid', do NOT write 'class Grid' again.\n"
                    "3. Each file has ONE clear responsibility. Define ONLY the classes/functions assigned to that file. All other needed names come from imports.\n"
                    "4. NEW files: small and focused — max 150 lines.\n"
                    "5. MODIFYING existing files: add only the minimal change. DO NOT rewrite the entire file.\n"
                    "6. NEVER create duplicate functions or classes — check the export map before defining anything.\n"
                    "7. Prefer composition over inheritance. Inject dependencies via __init__.\n\n"
                    "Format each file as:\n"
                    "[FILE: filename.py]\n"
                    "```python\n"
                    "# code\n"
                    "```"
                )
            impl_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context}
            ]

            impl_response = None
            for attempt in range(3):
                try:
                    impl_response = await agent.llm.chat(impl_messages, max_tokens=12000, disable_thinking=True)
                    # "[Error: ...]" and "[LM Studio ...]" are transport/model
                    # failure sentinels, not content — retrying them is the
                    # whole point of this loop (observed: an "[LM Studio
                    # stream error]" broke out as if valid and the batch was
                    # silently dropped by the block parser).
                    if impl_response and not impl_response.startswith(("[Error:", "[LM Studio")):
                        break
                    print(f"  Attempt {attempt + 1} failed, retrying...")
                    if "reasoning" in str(impl_response).lower():
                        impl_messages.append({"role": "user", "content": "Answer immediately — no reasoning, output only [FILE:] blocks."})
                except Exception as e:
                    print(f"  Attempt {attempt + 1} error: {e}, retrying...")
                    if "reasoning" in str(e).lower():
                        impl_messages.append({"role": "user", "content": "Answer immediately — no reasoning, output only [FILE:] blocks."})
                    if attempt == 2:
                        impl_response = None

            if not impl_response or impl_response.startswith(("[Error:", "[LM Studio")):
                print("  Failed after 3 attempts, skipping batch")
                for bf in batch:
                    file_outcomes[bf] = f"generation failed — {str(impl_response)[:120] or 'no response'}"
                continue

            file_patterns = [
                r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```\s*$',
                r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```',
                r'\[FILE:\s*([^\]]+)\]\s*\n+(.*?)(?=\[FILE:|\[PATCH:|$)',
            ]
            patch_patterns = [
                r'\[PATCH:\s*([^\]]+)\]\s*\n(.*?)(?=\[PATCH:|\[FILE:|$)',
            ]

            # Try [FILE:] blocks first, then [PATCH:] blocks
            matches = []
            for pattern in file_patterns:
                matches = list(re.findall(pattern, impl_response, re.DOTALL))
                if matches:
                    break

            # In modify mode, also try [PATCH:] blocks for existing files
            patch_matches = []
            if modify_mode and not matches:
                for pattern in patch_patterns:
                    patch_matches = list(re.findall(pattern, impl_response, re.DOTALL))
                    if patch_matches:
                        break

            if not matches and not patch_matches and ("<tool_call" in impl_response or "<tool_call>" in impl_response):
                print("  Detected tool calls, retrying with plain text instruction...")
                impl_messages.append({"role": "user", "content": "Respond ONLY in [FILE: filename.py] or [PATCH: filename.py] format. No <tool_call> tags."})
                impl_response = await agent.llm.chat(impl_messages, disable_thinking=True)
                for pattern in file_patterns:
                    matches = list(re.findall(pattern, impl_response, re.DOTALL))
                    if matches:
                        break
                if modify_mode and not matches:
                    for pattern in patch_patterns:
                        patch_matches = list(re.findall(pattern, impl_response, re.DOTALL))
                        if patch_matches:
                            break

            # Transient/empty LLM responses: re-ask for the batch instead of
            # silently dropping the file. Give up only after extra retries.
            parse_retries = 0
            while not matches and not patch_matches and parse_retries < 2:
                print(f"  No [FILE:] or [PATCH:] blocks parsed — retry {parse_retries + 1}/2...")
                impl_messages.append(
                    {"role": "user", "content": f"Output exactly one block for {batch[0]}: [PATCH: {batch[0]}] with diff hunks (for existing files) or [FILE: {batch[0]}] with complete code (for new files). No preamble, no prose, no tool calls."}
                )
                try:
                    impl_response = await agent.llm.chat(impl_messages, max_tokens=12000, disable_thinking=True)
                except Exception as e:
                    print(f"  Retry error: {e}")
                    impl_response = None
                for pattern in file_patterns:
                    matches = list(re.findall(pattern, impl_response or "", re.DOTALL))
                    if matches:
                        break
                if modify_mode and not matches:
                    for pattern in patch_patterns:
                        patch_matches = list(re.findall(pattern, impl_response or "", re.DOTALL))
                        if patch_matches:
                            break
                parse_retries += 1

            if not matches and not patch_matches:
                print("  Warning: Could not parse files from batch response")
                print(f"  Raw response: {str(impl_response)[:500]}")
                continue

            # Process [FILE:] blocks (new files or full rewrites)
            for filename, content in matches:
                content = content.strip()
                if filename not in batch:
                    # The LLM emitted a [FILE:] block for a name outside the
                    # planned batch.  Accepting it silently drops the planned
                    # file and can overwrite content of an unrelated file
                    # (observed: the secrets.py batch returned a
                    # [FILE: agent_core/security/sanitizer.py] block).  Treat
                    # it as a parse failure for the planned file instead.
                    print(f"  WARNING: [FILE: {filename}] is not in the planned batch {batch} — ignored")
                    continue
                generated_content[filename] = content
                print(f"  Generated: {filename} ({len(content)} bytes)")

                sigs = extract_signatures(content)
                if sigs:
                    export_map[filename] = sigs

            # Process [PATCH:] blocks (modify mode — apply diff to existing file)
            for filename, patch_text in patch_matches:
                filename = filename.strip()
                patch_text = patch_text.strip()
                if filename not in batch:
                    print(f"  WARNING: [PATCH: {filename}] is not in the planned batch {batch} — ignored")
                    continue
                ws_path = Path(workspace_path(target_workspace))
                filepath = ws_path / filename
                if not filepath.exists():
                    print(f"  WARNING: [PATCH: {filename}] — file does not exist, skipping patch")
                    continue
                try:
                    existing_text = filepath.read_text(encoding="utf-8")
                except OSError as exc:
                    print(f"  WARNING: [PATCH: {filename}] — cannot read file: {exc}")
                    continue
                from agent_core.patch_utils import apply_patch, split_source_lines
                ok, patched = apply_patch(patch_text, split_source_lines(existing_text))
                if not ok:
                    print(f"  WARNING: [PATCH: {filename}] — patch did not apply: {str(patched)[:200]}")
                    continue
                patched_text = str(patched)
                if patched_text == existing_text:
                    print(f"  [PATCH: {filename}] — patch produced no change")
                    continue
                try:
                    compile(patched_text, filename, "exec")
                except SyntaxError as exc:
                    print(f"  WARNING: [PATCH: {filename}] — patched file does not compile: {exc}")
                    continue
                generated_content[filename] = patched_text
                print(f"  Generated (patch): {filename} ({len(patched_text)} bytes)")

        if generated_content:
            for fname in all_files:
                if fname not in export_map:
                    fp = Path(ws) / fname
                    if fp.exists():
                        try:
                            existing = fp.read_text(encoding="utf-8")
                            sigs = extract_signatures(existing)
                            if sigs:
                                export_map[fname] = sigs
                        except Exception:
                            print(f"  Warning: Could not read existing file {fname}")

            print(f"\nExport map: {sum(len(v) for v in export_map.values())} exports across {len(export_map)} modules")

            broken_imports = {}
            stdlib_modules = set()

            for fname, content in generated_content.items():
                missing = []
                for match in re.finditer(r'from\s+(\S+)\s+import\s+(.+?)(?:\s*#|\s*$)', content):
                    src_module = match.group(1)
                    imported_names = [n.strip().split(' as ')[0].strip() for n in match.group(2).strip('()').split(',')]
                    src_file = src_module.replace('.', '/') + '.py'

                    if src_file not in export_map:
                        stdlib_modules.add(src_module)
                        continue

                    for name in imported_names:
                        if name.isupper() or name.startswith('_'):
                            continue
                        src_exports = export_map.get(src_file, {})
                        if name not in src_exports:
                            missing.append((src_module, name))

                if missing:
                    broken_imports[fname] = missing

            if stdlib_modules:
                print(f"  Skipped stdlib/third-party: {', '.join(sorted(stdlib_modules))}")

            if broken_imports:
                print(f"\n  Found {sum(len(v) for v in broken_imports.values())} broken imports in {len(broken_imports)} files:")
                for fname, missing in broken_imports.items():
                    for mod, name in missing:
                        print(f"    {fname}: '{name}' from '{mod}' not found")
                    if fname in generated_content:
                        print(f"  REJECTED: {fname} — references names not defined in the project")
                        file_outcomes[fname] = f"rejected — {len(missing)} unresolved import(s)"
                        del generated_content[fname]
            else:
                print("  All imports verified!")
        else:
            print("No content generated.")

        # Safety net: back up every existing target BEFORE any write happens,
        # so a rejected or rolled-back generation can never destroy original
        # work (observed: the post-loop dependency cleanup deleted an
        # untouched existing agent.py that merely imported a rejected module).
        backup_ws = Path(to_windows_path(target_workspace))
        backup_dir = backup_ws / "backups"
        backup_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_map: dict[str, Path] = {}
        for bfile in list(generated_content):
            bfp = backup_ws / bfile
            if bfp.is_file():
                try:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    bname = bfile.replace("/", "_").replace("\\", "_")
                    bcopy = backup_dir / f"{bname}_{backup_ts}.py"
                    shutil.copy2(bfp, bcopy)
                    backup_map[bfile] = bcopy
                except OSError as exc:
                    print(f"  WARNING: could not back up existing {bfile}: {exc}")
        if backup_map:
            print(f"  Backed up {len(backup_map)} existing target file(s) to backups/ ({backup_ts})")
        written_files: set[str] = set()

        _RE_3 = re.compile(r'_\d+$|_v\d+$|_clean$|_final$')
        for filename, content in generated_content.items():
            raw_workspace = workspace_path(target_workspace)
            workspace = Path(raw_workspace)
            filepath = workspace / filename

            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Ensure __init__.py exists in every package directory so
            # `from src.agent1.xxx import` works regardless of import style.
            # Never overwrite an existing __init__.py that has content.
            # tests/ and src/ trees are skipped (see _ensure_package_inits).
            _ensure_package_inits(filepath.parent, workspace.resolve())

            skip_reason = None
            is_analyzed_file = analyzed_file and filename == analyzed_file
            #: --modify target: existing compile-OK module whose generated
            #: content is merged in as a reviewed diff instead of being
            #: skipped (default) or overwritten wholesale (--force).
            modify_target = False

            if not force_mode and not is_analyzed_file and filepath.exists() and filepath.stat().st_size > 0:
                if filename.endswith(".py"):
                    filepath_str = os.path.realpath(filepath)
                    result = subprocess.run(
                        ["python", "-m", "py_compile", filepath_str],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        if modify_mode:
                            modify_target = True
                        else:
                            skip_reason = "Already exists and compiles OK"

            # ---- Layer 2: near-duplicate module gate (pre-generation) ----
            # A NEW module whose name/concept duplicates an existing one is
            # skipped instead of generated; the existing module is the right
            # place for the change.
            if skip_reason is None and not force_mode and not is_analyzed_file:
                if filename.endswith(".py") and not filepath.exists() and not _is_planned_test_file(filename):
                    dup_reasons = _check_planned_duplicates(
                        [filename], str(target_workspace),
                        taskplan_content if taskplan_content else "",
                    )
                    if dup_reasons:
                        skip_reason = (
                            f"Near-duplicate of existing module — {dup_reasons[0].split(' — ', 1)[-1]} "
                            "(extend the existing module; use --force to generate anyway)"
                        )

            if skip_reason:
                print(f"  Skipping {filename}: {skip_reason}")
                file_outcomes.setdefault(filename, skip_reason)
                continue

            if modify_target:
                ok_modify, note = self._apply_modify_diff(
                    filename, filepath, content, allow_rewrite,
                )
                print(f"  {filename}: {note}")
                file_outcomes[filename] = note
                # Only a diff that was actually applied counts as implemented.
                # Appending rejected/skipped files here lets the post-loop
                # dependency cleanup delete untouched originals (observed:
                # a rejected modify target was unlinked because its imports
                # referenced another rejected file).
                if ok_modify:
                    implemented.append(filename)
                continue

            if filename.endswith(".py"):
                func_names = re.findall(r'def\s+(\w+)', content)
                if len(func_names) > 20:
                    # from collections import Counter  # unused_import removed
                    similar_prefixes: dict[str, int] = {}
                    for name in func_names:
                        prefix = _RE_3.sub('', name)
                        similar_prefixes[prefix] = similar_prefixes.get(prefix, 0) + 1
                    max_dupes = max(similar_prefixes.values()) if similar_prefixes else 1
                    if max_dupes > 10:
                        print(f"  REJECTED: {filename} has {max_dupes} near-duplicate functions")
                        file_outcomes[filename] = f"rejected — {max_dupes} near-duplicate functions"
                        continue
                if len(content) > 50000:
                    print(f"  REJECTED: {filename} is {len(content)} bytes (max 50KB)")
                    file_outcomes[filename] = f"rejected — {len(content)} bytes, max 50KB"
                    continue
                if len(content) < 10:
                    print(f"  REJECTED: {filename} is empty (0 bytes) — LLM returned no content")
                    file_outcomes[filename] = "rejected — empty response from LLM"
                    continue

            dangerous, reason = _is_dangerous_filename(filename, workspace)
            if dangerous:
                # Try auto-repair: prefix bare root-level filenames with a safe sub-package.
                if "/" not in filename and "\\" not in filename:
                    safe_dir = _find_safe_subpackage(workspace)
                    new_filename = f"{safe_dir}/{filename}"
                    new_filepath = workspace / new_filename
                    new_dangerous, _ = _is_dangerous_filename(new_filename, workspace)
                    if not new_dangerous:
                        if new_filepath.exists() and new_filepath.stat().st_size > 100:
                            print(f"  Skipped: {new_filename} already exists (avoiding overwrite)")
                            file_outcomes[filename] = f"skipped — {new_filename} already exists"
                            continue
                        print(f"  Auto-repaired: {filename} -> {new_filename}")
                        file_outcomes[filename] = f"auto-repaired -> {new_filename}"
                        filename = new_filename
                        filepath = new_filepath
                    else:
                        print(f"  REJECTED: {reason}")
                        file_outcomes[filename] = f"rejected — {reason}"
                elif "shadows stdlib" in reason:
                    # Redirect stdlib-shadowing path: logging -> logging_utils
                    f_parts = filename.replace("\\", "/").split("/")
                    for idx, part in enumerate(f_parts[:-1]):
                        if part in _STDLIB_COMMON:
                            safe_name = part + "_utils"
                            f_parts[idx] = safe_name
                            new_filename = "/".join(f_parts)
                            new_filepath = workspace / new_filename
                            print(f"  Redirected: {filename} -> {new_filename} (shadows stdlib '{part}')")
                            file_outcomes[filename] = f"redirected -> {new_filename}"
                            filename = new_filename
                            filepath = new_filepath
                            break
                        continue
                else:
                    print(f"  REJECTED: {reason}")
                    file_outcomes[filename] = f"rejected — {reason}"
                    continue

            # Write to temp → compile → rename on success, delete on failure
            prev_content = filepath.read_text(encoding="utf-8") if filepath.exists() else None
            tmp_path = str(filepath) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)

            if filename.endswith(".py"):
                r = subprocess.run(
                    ["python", "-m", "py_compile", tmp_path],
                    capture_output=True, text=True
                )

                if r.returncode != 0:
                    stripped = content.rstrip()
                    lines = stripped.splitlines() if stripped else []
                    last_line = lines[-1].strip() if lines else ""

                    ends_mid = last_line.endswith(('(', '[', '{', ':', ',', '+', '-', '*', '/', '=', '\\'))
                    incomplete = (last_line and last_line[-1].isalnum()
                        and not last_line.endswith(('pass', 'return', 'break', 'continue', 'True', 'False', 'None')))
                    opens = sum(last_line.count(c) for c in '({[')
                    closes = sum(last_line.count(c) for c in ')}]')
                    unbalanced = opens > closes

                    is_truncated = ends_mid or (incomplete and unbalanced) or (incomplete and len(content) < 500)

                    if is_truncated and (
                        "unterminated" in r.stderr or "unexpected EOF" in r.stderr
                        or "was never closed" in r.stderr or "invalid syntax" in r.stderr
                        or "SyntaxError" in r.stderr
                    ):
                        os.unlink(tmp_path)
                        print(f"  WARNING: {filename} appears truncated, re-requesting...")
                        # Switch to deep-analysis profile (50K tokens) for the retry
                        old_temp = agent.llm._provider.temperature
                        old_tok = agent.llm._provider.max_tokens
                        try:
                            from agent_core.llm.model_profiles import get_profile
                            dp = get_profile("deep-analysis")
                            agent.llm._provider.temperature = dp.temperature
                            agent.llm._provider.max_tokens = dp.max_tokens
                        except Exception:
                            print("Warning: failed to switch to deep-analysis profile")
                        retry_msgs = [
                            {"role": "system", "content": "Generate ONLY the complete code for this file. Output as:\n[FILE: filename.py]\n```python\n# complete code here\n```"},
                            {"role": "user", "content": f"Generate complete code for {filename}."}
                        ]
                        retry_content = await agent.llm.chat(retry_msgs, disable_thinking=True)
                        # Restore original profile
                        agent.llm._provider.temperature = old_temp
                        agent.llm._provider.max_tokens = old_tok
                        if not retry_content.startswith("[Error"):
                            match = re.search(r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```', retry_content, re.DOTALL)
                            if match:
                                new_content = match.group(2).strip()
                                if len(new_content) > len(content) * 0.5:
                                    if (
                                        filepath.exists()
                                        and filename.endswith(".py")
                                        and not allow_rewrite
                                    ):
                                        existing_text = filepath.read_text(encoding="utf-8")
                                        similarity = difflib.SequenceMatcher(
                                            None, existing_text, new_content
                                        ).ratio()
                                        if existing_text.strip() and similarity < 0.5:
                                            print(
                                                f"  REJECTED: {filename} retry would be a wholesale "
                                                f"rewrite (similarity {similarity:.2f}) — refusing."
                                            )
                                            file_outcomes[filename] = "rejected — wholesale rewrite of existing file"
                                            continue
                                    with open(filepath, "w", encoding="utf-8") as f:
                                        f.write(new_content)
                                    content = new_content
                                    written_files.add(filename)
                                    print(f"  Re-written: {filename} ({len(content)} bytes)")
                                    filepath_str = os.path.realpath(filepath)
                                    r = subprocess.run(
                                        ["python", "-m", "py_compile", filepath_str],
                                        capture_output=True, text=True
                                    )

                    if r.returncode != 0:
                        os.unlink(tmp_path)
                        errors.append(f"{filename}: {r.stderr}")
                        print(f"  Compile error in {filename}")
                    else:
                        # MODIFY-rewrite guard: never let a batch wholesale-replace
                        # an existing file.  Observed: a MODIFY task replaced
                        # tool_loop.py's 416 lines with a 4-line stub and gutted
                        # tool_schemas.py.  Only --allow-rewrite permits it.
                        if (
                            filepath.exists()
                            and filename.endswith(".py")
                            and not allow_rewrite
                        ):
                            try:
                                existing_text = filepath.read_text(encoding="utf-8")
                            except OSError:
                                existing_text = ""
                            if existing_text.strip():
                                similarity = difflib.SequenceMatcher(
                                    None, existing_text, content
                                ).ratio()
                                if similarity < 0.5:
                                    os.unlink(tmp_path)
                                    print(
                                        f"  REJECTED: {filename} would be a wholesale rewrite "
                                        f"(similarity {similarity:.2f}) — refusing to replace an "
                                        "existing file; use --allow-rewrite to replace it."
                                    )
                                    file_outcomes[filename] = "rejected — wholesale rewrite of existing file"
                                    continue
                        os.replace(tmp_path, filepath)
                        written_files.add(filename)
                        print(f"  Compiled OK: {filename}")
                else:
                    os.replace(tmp_path, filepath)
                    written_files.add(filename)
                    print(f"  Compiled OK: {filename}")

            # Post-write: reject files that create class-name conflicts in same directory
            if filepath.exists() and filename.endswith(".py"):
                from agent_core.patterns import detect_class_conflicts
                conflicts = detect_class_conflicts([filename], ws)
                if conflicts:
                    if prev_content is not None:
                        filepath.write_text(prev_content, encoding="utf-8")  # restore original
                    else:
                        filepath.unlink()  # new file, safe to delete
                    print(f"  REJECTED: class-name conflict — {conflicts[0]['suggestion']}")
                    file_outcomes[filename] = f"rejected — {conflicts[0]['suggestion']}"
                    continue
            implemented.append(filename)
            print(f"  Written: {filename}")

            manifest_file = Path(ws) / ".generated_manifest.json"
            manifest = {}
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
            manifest[filename] = {"date": str(datetime.now()), "size": len(content)}
            manifest_file.write_text(json.dumps(manifest, indent=2))

            if filename.endswith(".py"):
                filepath_str = os.path.realpath(filepath)
                result = subprocess.run(
                    ["python", "-m", "py_compile", filepath_str],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    errors.append(f"{filename}: {result.stderr}")
                    print(f"  Compile error in {filename}")
                else:
                    print(f"  Compiled OK: {filename}")

        # Post-loop: remove files that depend on rejected modules
        # Only cascade-reject for compilation errors, not name collisions.
        # A name collision means the class already exists elsewhere — importing
        # files are not broken.
        #
        # SAFETY: only files this run actually wrote may be touched.  A
        # pre-existing file that merely imports a rejected module is NEVER
        # deleted (observed: an untouched agent.py was unlinked because it
        # imported a wholesale-rewrite-rejected tool_loop.py).  Written files
        # that had a pre-run original are restored from backups/ instead of
        # being deleted outright.
        rejected_files = {k for k, v in file_outcomes.items()
                          if "rejected" in v and "class-name conflict" not in v}
        if rejected_files:
            for fname in list(implemented):
                fpath = Path(ws) / fname
                if not fpath.exists():
                    continue
                try:
                    content = fpath.read_text(encoding="utf-8")
                except Exception:
                    continue
                # Check imports against rejected files
                for m in re.finditer(r'^(?:from|import)\s+(\S+)', content, re.MULTILINE):
                    module = m.group(1)
                    mod_file = module.replace(".", "/") + ".py"
                    for rf in rejected_files:
                        rf_no_ext = rf.replace(".py", "").replace("/", ".")
                        if mod_file == rf or module == rf_no_ext or module.endswith("." + rf_no_ext.rsplit("/", 1)[-1]):
                            if fname not in written_files:
                                # Pre-existing file this run did NOT write:
                                # never delete it, report instead.
                                print(
                                    f"  KEPT: {fname} imports rejected module {rf} — "
                                    "existing file left untouched (only files written "
                                    "this run are removed)"
                                )
                                file_outcomes[fname] = f"kept — imports rejected {rf}; existing file preserved"
                                break
                            p = Path(ws) / fname
                            backup = backup_map.get(fname)
                            if p.exists():
                                if backup is not None and backup.exists():
                                    shutil.copy2(backup, p)
                                    print(f"  REJECTED: {fname} depends on rejected file {rf} — original restored from backups/")
                                else:
                                    p.unlink()
                                    print(f"  REJECTED: {fname} depends on rejected file {rf}")
                            file_outcomes[fname] = f"rejected — depends on rejected {rf}"
                            if fname in implemented:
                                implemented.remove(fname)
                            break

        print(f"\n{'='*50}")
        print(f"Implementation complete: {len(implemented)}/{len(all_files)} files")

        if implemented:
            print("\nImplemented files:")
            for f in implemented:
                print(f"  - {f}")

        if errors:
            print(f"\nCompilation errors ({len(errors)}):")
            for err in errors:
                print(f"  - {err}")
        else:
            print("\nAll Python files compiled successfully!")

        missing_files = set(all_files) - set(implemented)
        repaired = {k: v for k, v in file_outcomes.items() if "auto-repaired" in v}
        rejected = {k: v for k, v in file_outcomes.items() if "rejected" in v}
        skipped = {k: v for k, v in file_outcomes.items() if "skipped" in v}
        truly_missing = missing_files - set(repaired) - set(rejected) - set(skipped)

        if repaired:
            print(f"\nAuto-repaired ({len(repaired)}):")
            for fs_item, how in sorted(repaired.items()):
                print(f"  {fs_item}  — {how}")

        if truly_missing:
            print(f"\nMissing — could not generate ({len(truly_missing)}):")
            for fs in sorted(truly_missing):
                reason = file_outcomes.get(fs, "unknown")
                print(f"  - {fs}  ({reason})")

        if rejected:
            print(f"\nRejected ({len(rejected)}):")
            for fs, why in sorted(rejected.items()):
                print(f"  - {fs}: {why}")

        if skipped:
            print(f"\nSkipped — target exists ({len(skipped)}):")
            for fs, why in sorted(skipped.items()):
                print(f"  - {why}")

        post_snapshot = set()
        for fp in Path(ws).rglob("*"):
            if fp.is_file() and ".git" not in str(fp) and "__pycache__" not in str(fp):
                post_snapshot.add(str(fp.relative_to(Path(ws))).replace("\\", "/"))

        new_files = post_snapshot - pre_snapshot
        removed_files = pre_snapshot - post_snapshot
        if new_files:
            print(f"\n  New files created: {len(new_files)}")
            for fs in sorted(new_files):
                print(f"    + {fs}")

            # Auto-run static safety checks on new files (fast, no LLM)
            py_new = sorted(f for f in new_files if f.endswith(".py"))
            if py_new and len(py_new) <= 10:
                self._auto_review(py_new, ws)

            if not review_mode:
                print("\n  Tip: run 'implement --review' for deep LLM analysis.")
        if removed_files:
            print(f"\n  Files no longer present: {len(removed_files)}")
            for fs in sorted(removed_files):
                print(f"    - {fs}")

        if not implemented and all_files:
            print(
                f"\n  WARNING: 0 of {len(all_files)} planned file(s) were changed — existing "
                "targets were rejected by the wholesale-rewrite guard.  Rerun with "
                "--allow-rewrite or --force to deliberately replace existing files."
            )

        if fix_mode and implemented:
            print(f"\n{'='*50}")
            print("[fix] Deep validation: checking imports + class instantiation...")
            print(f"{'='*50}")

            prev_error_sigs: dict[str, str] = {}
            file_error_sigs: dict[str, str] = {}  # per-file last-seen error signature
            patch_failed: set[str] = set()  # files where [PATCH:] format failed — prefer [FILE:] next

            # ---- Pre-loop cross-file attribute check ----
            # Verify chained attribute accesses across module boundaries.
            # Runs once since attribute chains don't change between fix attempts.
            ws = target_workspace
            if ws.startswith("/c/") or ws.startswith("/C/"):
                ws = "C:" + ws[2:]
            cross_errors: list[tuple[str, str, str]] = []
            for fname in implemented:
                fp = Path(ws) / fname
                if not fp.exists():
                    continue
                with open(fp, "r", encoding="utf-8", errors="replace") as sf:
                    src = sf.read()
                src_dir = str(fp.parent.resolve())
                imports_by_class: dict[str, str] = {}
                for im in re.finditer(r'from\s+([\w.]+)\s+import\s+(.+?)(?:\s*$|\s*#)', src, re.MULTILINE):
                    mod_dotted = im.group(1)
                    mod_file = mod_dotted.replace('.', '/') + ".py"
                    if not (Path(ws) / mod_file).exists():
                        continue
                    for n in re.findall(r'(\w+)', im.group(2)):
                        if n[0].isupper():
                            imports_by_class[n] = mod_dotted
                var_types: dict[str, str] = {}
                for cls_name, mod_file in imports_by_class.items():
                    for va in re.finditer(rf'(\w+)\s*=\s*{cls_name}\s*\(', src):
                        var_types[va.group(1)] = cls_name
                for var, cls_name in var_types.items():
                    for chain_m in re.finditer(rf'\b{var}\.(\w+(?:\.\w+)*)\b', src):
                        chain = chain_m.group(1).split('.')
                        mod_py = imports_by_class[cls_name]
                        check_attrs = '[' + ', '.join(f"'{a}'" for a in chain) + ']'
                        code = [
                            f"from {mod_py} import {cls_name}",
                            f"try: obj = {cls_name}()",
                            "except: print('SKIP'); sys.exit(0)",
                        ]
                        for attr in chain:
                            code.append(f"if not hasattr(obj, '{attr}'): print('MISSING: {cls_name}.{attr}'); sys.exit(2)")
                            code.append(f"obj = obj.{attr}")
                        code.append(f"print('OK: {cls_name}.' + '.'.join({check_attrs}))")
                        r = _run_python_snippet(ws, [src_dir], code)
                        if r.returncode == 2:
                            err_line = ""
                            for line in (r.stdout or "").split('\n'):
                                if 'MISSING:' in line:
                                    err_line = line.strip()
                                    break
                            cross_errors.append((fname, str(fp), f"CROSS: {err_line or f'{cls_name}.{chain[0]}'} (accessed as {var}.{chain_m.group(1)} in {fname})"))
                        elif "OK:" in (r.stdout or ""):
                            print(f"  {r.stdout.strip()}")
            if cross_errors:
                print(f"\n[fix] {len(cross_errors)} cross-file attribute errors:")
                for fname, fpath, err in cross_errors:
                    print(f"  - {err}")
            else:
                print("[fix] Cross-file attributes all verified!")

            _RE_4 = re.compile(r'^class\s+(\w+)', re.MULTILINE)
            for fix_attempt in range(3):
                errors_found = []
                current_error_sigs: dict[str, str] = {}
                preexisting_sigs: set[str] = set()

                for fname in implemented:
                    if not fname.endswith(".py"):
                        continue
                    ws = target_workspace
                    if ws.startswith("/c/") or ws.startswith("/C/"):
                        ws = "C:" + ws[2:]
                    fp = Path(ws) / fname
                    fpath_str = os.path.realpath(fp)

                    r = subprocess.run(["python", "-m", "py_compile", fpath_str], capture_output=True, text=True)
                    if r.returncode != 0:
                        errors_found.append((fname, fpath_str, f"COMPILE: {r.stderr.strip()}"))
                        continue

                    mod_name = fname[:-3].replace('\\', '.').replace('/', '.')
                    r = _run_python_snippet(ws, [str(fp.parent.resolve())], [
                        f"import {mod_name}",
                        "print('OK')",
                    ])
                    if r.returncode != 0:
                        err_text = r.stderr.strip()
                        mm = re.search(r"No module named '([^']+)'", err_text)
                        if mm:
                            batch_mods = {
                                bm[:-3].replace('\\', '.').replace('/', '.')
                                for bm in list(implemented) + list(all_files)
                                if bm.endswith('.py')
                            }
                            if mm.group(1) not in batch_mods:
                                preexisting_sigs.add(f"{fname}: missing module '{mm.group(1)}'")
                                print(f"  (pre-existing, skipped) {fname}: missing module '{mm.group(1)}' is not produced by this batch")
                                continue
                        errors_found.append((fname, fpath_str, f"IMPORT: {err_text}"))
                        continue

                    r = subprocess.run(
                        ["python", "-m", "mypy", fpath_str, "--ignore-missing-imports"],
                        capture_output=True, text=True, cwd=str(Path(ws))
                    )
                    if r.returncode != 0 and "No module named" not in r.stderr:
                        norm_fname = fname.replace("\\", "/")
                        real_fname = fpath_str.replace("\\", "/")
                        type_errors = []
                        foreign_errors = []
                        for line in r.stdout.split('\n'):
                            line = line.strip()
                            if not line or ':' not in line or line.startswith('Found'):
                                continue
                            if (': note:' in line or 'annotation-unchecked' in line
                                    or '"list" is invariant' in line or '--check-untyped-defs' in line
                                    or 'No overload variant' in line or 'Possible overload variants' in line
                                    or 'no-untyped-def' in line or 'no-untyped-call' in line):
                                continue
                            err_path = line.split(':', 1)[0].replace("\\", "/")
                            if err_path == norm_fname or err_path == real_fname:
                                type_errors.append(line)
                            else:
                                foreign_errors.append(line)
                        if foreign_errors:
                            for fe in foreign_errors:
                                preexisting_sigs.add(fe)
                            print(f"  (pre-existing, skipped) {fname}: {len(foreign_errors)} error(s) in other files: {foreign_errors[0]}")
                        if type_errors:
                            for te in type_errors[:5]:
                                errors_found.append((fname, fpath_str, f"TYPE: {te}"))

                    with open(fpath_str, "r", encoding="utf-8") as f:
                        source = f.read()
                    class_names = _RE_4.findall(source)
                    for cn in class_names:
                        r = _run_python_snippet(ws, [str(fp.parent.resolve())], [
                            f"import {mod_name}",
                            f"c={mod_name}.{cn}",
                            "import inspect",
                            f"try:\n    sig=inspect.signature(c)\n    print(f'OK: {cn}'+str(list(sig.parameters.keys())))\n"
                            f"except (ValueError, TypeError):\n    print(f'OK: {cn} (builtin/Protocol/TypedDict)')",
                        ])
                        if r.returncode != 0:
                            errors_found.append((fname, fpath_str, f"CLASS {cn}: {r.stderr.strip()}"))
                        else:
                            print(f"  {fname}: {r.stdout.strip()}")

                if preexisting_sigs:
                    print(f"  (pre-existing, skipped) {len(preexisting_sigs)} unique error(s) outside this batch — reported, not LLM-fixed")

                if not errors_found:
                    print("\n[fix] All files pass deep validation!")
                    print("\n[fix] Running smoke tests (instantiate + call methods)...")

                    smoke_errors = []

                    for fname in implemented:
                        if not fname.endswith(".py"):
                            continue
                        fp = Path(ws) / fname
                        fpath_str = os.path.realpath(fp)

                        with open(fpath_str, "r", encoding="utf-8") as sf:
                            source = sf.read()

                        for cn_match in re.finditer(r'class\s+(\w+)\s*(?:\(.*?\))?\s*:', source):
                            cn = cn_match.group(1)
                            if cn.startswith('_') or 'Protocol' in source[cn_match.start():cn_match.start()+200]:
                                continue

                            mod_name = fname[:-3].replace('\\', '.').replace('/', '.')

                            # Enum classes — validate by accessing members, not calling ()
                            class_header = source[cn_match.start():cn_match.end()+50]
                            if 'Enum' in class_header:
                                smoke_lines = [
                                    f"import {mod_name}",
                                    f"print(f'Testing {cn}...')",
                                    "try:",
                                    f"    _ = list({mod_name}.{cn})[0]",
                                    f"    print(f'  OK: {cn} is valid Enum')",
                                    f"except Exception as e:\n    print(f'  FAIL: {cn}: {{type(e).__name__}}: {{e}}')",
                                ]
                                r = _run_python_snippet(ws, [str(fp.parent.resolve())], smoke_lines)
                                if r.returncode != 0:
                                    smoke_errors.append((fname, fpath_str, r.stderr.strip()[-300:]))
                                else:
                                    output = r.stdout.strip()
                                    if "FAIL:" in output:
                                        smoke_errors.append((fname, fpath_str, output))
                                        print(f"  {fname}: {output}")
                                    elif "OK:" in output:
                                        print(f"  {output}")
                                continue

                            init_match = re.search(
                                rf'class\s+{re.escape(cn)}\b(?s:.*?)def\s+__init__\s*\((.*?)\)\s*(?:->[^:]+)?:',
                                source[cn_match.start():cn_match.start() + 2000],
                            )
                            params = init_match.group(1) if init_match else ""

                            required = []
                            dc_m = re.search(r'@dataclass\b[^\n]*\n\s*class\s+' + re.escape(cn) + r'\b', source[:cn_match.end()])
                            if dc_m:
                                for dline in source[cn_match.end():].split('\n'):
                                    stripped = dline.strip()
                                    if not stripped or stripped.startswith(('#', '"""', '@', 'def ', 'class ')):
                                        continue
                                    if re.match(r'^\w+\s*:', stripped) and '=' not in stripped:
                                        required.append(stripped.split(':')[0].strip())
                                    elif not dline.startswith('    ') and stripped:
                                        break
                            else:
                                for p in params.split(','):
                                    p = p.strip()
                                    if not p or p == 'self':
                                        continue
                                    if '=' not in p:
                                        required.append(p.split(':')[0].strip())

                            mod_name = fname[:-3].replace('\\', '.').replace('/', '.')

                            smoke_lines = [
                                f"import {mod_name}",
                                "import inspect",
                                f"print(f'Testing {cn}...')",
                                "try:",
                                f"    if inspect.isabstract({mod_name}.{cn}):",
                                f"        print(f'  OK: {cn} is abstract by design')",
                                "        raise SystemExit(0)",
                            ]
                            if not required:
                                smoke_lines.append(f"    obj = {mod_name}.{cn}()")
                                smoke_lines.append(f"    print(f'  OK: {cn}() works')")
                            elif len(required) <= 2:
                                call_args = ", ".join(f'"mock_{r.split(":")[0].strip()}"' if ':' in r else f'"mock_{r.strip()}"' for r in required)
                                smoke_lines.append(f"    obj = {mod_name}.{cn}({call_args})")
                                smoke_lines.append(f"    print(f'  OK: {cn}() works')")
                            else:
                                smoke_lines.append(f"    print(f'  SKIP: {cn} needs {len(required)} args')")
                            smoke_lines.append(f"except Exception as e:\n    print(f'  FAIL: {cn}: {{type(e).__name__}}: {{e}}')")
                            r = _run_python_snippet(ws, [str(fp.parent.resolve())], smoke_lines)

                            if r.returncode != 0:
                                smoke_errors.append((fname, fpath_str, r.stderr.strip()[-300:]))
                            else:
                                output = r.stdout.strip()
                                if "FAIL:" in output:
                                    smoke_errors.append((fname, fpath_str, output))
                                    print(f"  {fname}: {output}")
                                elif "SKIP:" in output:
                                    pass
                                elif "OK:" in output:
                                    print(f"  {output}")

                    if not smoke_errors:
                        print("[fix] Smoke tests all passed!")
                        break

                    print(f"\n[fix] {len(smoke_errors)} smoke test failures:")
                    for fname, fpath, err in smoke_errors:
                        errors_found.append((fname, fpath, f"SMOKE: {err[:200]}"))
                        print(f"  - {fname}: {err[:100]}")

                # Fold cross-file errors into the first fix attempt
                if fix_attempt == 0 and cross_errors:
                    errors_found = cross_errors + errors_found

                # Group errors by root cause class
                error_groups = _group_related_errors(errors_found, ws)

                # Sort groups by dependency order so root causes that other
                # root causes depend on are fixed first
                error_groups = _topological_sort_groups(error_groups, ws)

                print(f"\n[fix] Attempt {fix_attempt + 1}: {len(errors_found)} errors in {len(error_groups)} groups")
                for fname, fpath, err in errors_found:
                    print(f"  - {fname}:")
                    print(f"    {err}")

                    # Build error signature per file for deduplication
                    sig = f"{fname}:{err[:200]}"
                    current_error_sigs[fname] = current_error_sigs.get(fname, "") + sig

                # Deduplicate: stop if same errors as previous attempt
                if fix_attempt > 0 and current_error_sigs == prev_error_sigs:
                    print("\n[fix] Same errors as previous attempt — stopping (no progress possible).")
                    break
                prev_error_sigs = dict(current_error_sigs)

                if fix_attempt >= 2:
                    print("[fix] Max attempts reached.")
                    break

                # Process error groups: fix root causes first
                fixed_files_this_round = set()
                for group_path, group_errors in error_groups:
                    # Check if this group has a root cause
                    root_err = None
                    downstream_errs = []
                    for fname, fpath, err in group_errors:
                        if "ROOT_CAUSE:" in err:
                            root_err = (fname, fpath, err)
                        else:
                            downstream_errs.append((fname, fpath, err))

                    if root_err:
                        # Fix root cause first
                        fname, fpath, err = root_err
                        if fname in fixed_files_this_round:
                            print(f"\n[fix] Skipping {fname} — already fixed this round.")
                            continue

                        prev_sig = file_error_sigs.get(fname, "")
                        cur_sig = f"{err[:200]}"
                        if prev_sig == cur_sig:
                            print(f"\n[fix] Skipping {fname} — same error as previous attempt.")
                            continue

                        print(f"\n[fix] Fixing root cause: {fname}...")
                        print(f"  Downstream files: {', '.join(e[0] for e in downstream_errs)}")

                        if not os.path.exists(fpath):
                            print(f"  Skipping {fname} — file not found")
                            file_error_sigs[fname] = cur_sig
                            continue
                        with open(fpath, "r", encoding="utf-8") as f:
                            current_code = f.read()

                        # Use root cause prompt with downstream context
                        _rc_m = re.search(r'ROOT_CAUSE: (\w+)', err)
                        _rc_class = _rc_m.group(1) if _rc_m else ""
                        fix_msgs = _build_root_cause_prompt(
                            _rc_class,
                            current_code,
                            downstream_errs,
                            prefer_file=(fname in patch_failed)
                        )
                        print(f"  [llm] requesting root-cause fix for {fname} ({_rc_class or 'class'}) ...", flush=True)
                        fixed = await _chat_fix_text(agent, fix_msgs, fname, err)
                        if fixed.startswith("[Error") or fixed.startswith("[LM Studio"):
                            print(f"  LLM error: {fixed}")
                            continue
                        applied, note = _consume_fix_blocks(fixed, fname, fpath, current_code, file_first=True)
                        if applied:
                            print(f"  Fixed root cause: {fname} ({note})")
                            file_error_sigs[fname] = ""
                            fixed_files_this_round.add(fname)
                        else:
                            print(f"  Root-cause fix failed for {fname}: {note}")
                            patch_failed.add(fname)
                            file_error_sigs[fname] = cur_sig

                    # Fix downstream errors (if root cause was fixed or no root cause)
                    root_fname = root_err[0] if root_err else None
                    root_fixed = root_fname and root_fname in fixed_files_this_round
                    if root_err and not root_fixed:
                        # Root cause fix failed — skip downstream to avoid
                        # wasting LLM calls on errors that won't resolve
                        print(f"\n  Skipping {len(downstream_errs)} downstream errors — root cause fix failed")
                        for fname, fpath, err in downstream_errs:
                            file_error_sigs[fname] = f"{err[:200]}"
                        continue
                    for fname, fpath, err in downstream_errs:
                        if fname in fixed_files_this_round:
                            continue
                        prev_sig = file_error_sigs.get(fname, "")
                        cur_sig = f"{err[:200]}"
                        if prev_sig == cur_sig:
                            print(f"\n[fix] Skipping {fname} — same error as previous attempt.")
                            continue
                        print(f"\n[fix] Fixing {fname}...")
                        if not os.path.exists(fpath):
                            print(f"  Skipping {fname} — file not found (may have been rejected)")
                            file_error_sigs[fname] = cur_sig
                            continue
                        with open(fpath, "r", encoding="utf-8") as f:
                            current_code = f.read()

                        fix_msgs = _build_fix_prompt(err, current_code, fname, prefer_file=(fname in patch_failed))
                        print(f"  [llm] requesting fix for {fname} ({err.split(':')[0].strip()[:60]}) ...", flush=True)
                        fixed = await _chat_fix_text(agent, fix_msgs, fname, err)
                        if fixed.startswith("[Error") or fixed.startswith("[LM Studio"):
                            print(f"  LLM error: {fixed}")
                            continue
                        applied, note = _consume_fix_blocks(fixed, fname, fpath, current_code, file_first=False)
                        if applied:
                            print(f"  Fixed: {fname} ({note})")
                            file_error_sigs[fname] = ""
                            fixed_files_this_round.add(fname)
                        else:
                            print(f"  Fix failed for {fname}: {note}")
                            patch_failed.add(fname)
                            file_error_sigs[fname] = cur_sig

            print("\n[fix] Complete")

        if review_mode and new_files:
            print(f"\n{'='*50}")
            print(f"[review] Reviewing {len(new_files)} new file(s): {', '.join(sorted(new_files)[:10])}")
            print(f"{'='*50}")

            py_new = sorted(f for f in new_files if f.endswith(".py"))
            all_content: dict[str, str] = {}
            for fname in py_new:
                fpath = Path(ws) / fname
                if fpath.exists() and fpath.stat().st_size > 0:
                    try:
                        all_content[fname] = fpath.read_text(encoding="utf-8")
                    except Exception:
                        print(f"WARNING: Failed to read {fname} — skipping review")

            # ---- Static cross-file analysis ----
            func_locations: dict[str, list[str]] = {}
            class_locations: dict[str, list[str]] = {}
            for fname, content in all_content.items():
                for m in re.finditer(r'^\s*def\s+(\w+)', content, re.MULTILINE):
                    name = m.group(1)
                    if not name.startswith("__"):  # skip dunders (__init__, __str__, etc.)
                        func_locations.setdefault(name, []).append(fname)
                for m in re.finditer(r'^\s*class\s+(\w+)', content, re.MULTILINE):
                    class_locations.setdefault(m.group(1), []).append(fname)

            dup_funcs = {n: fs for n, fs in func_locations.items() if len(fs) > 1}
            dup_classes = {n: fs for n, fs in class_locations.items() if len(fs) > 1}
            near_dupes: list[tuple[str, str]] = []
            filenames_list = list(all_content.items())
            for i in range(len(filenames_list)):
                for j in range(i + 1, len(filenames_list)):
                    fa, ca = filenames_list[i]
                    fb, cb = filenames_list[j]
                    if ca == cb and len(ca) > 100:
                        near_dupes.append((fa, fb))

            issues_found = []
            if dup_funcs:
                print("\n  [review] Duplicate functions across files:")
                for name, files in sorted(dup_funcs.items()):
                    print(f"    {name}() in: {', '.join(files)}")
                    issues_found.append(f"Duplicate function `{name}()` defined in {len(files)} files: {', '.join(files)}")
            if dup_classes:
                print("\n  [review] Duplicate classes across files:")
                for name, files in sorted(dup_classes.items()):
                    print(f"    {name} in: {', '.join(files)}")
                    issues_found.append(f"Duplicate class `{name}` defined in {len(files)} files: {', '.join(files)}")
            if near_dupes:
                print("\n  [review] Near-identical files:")
                for fa, fb in near_dupes:
                    print(f"    {fa} ≈ {fb} ({len(all_content[fa])} bytes each)")
                    issues_found.append(f"Nearly identical files: {fa} and {fb} — consider merging")

            # ---- Cross-file attribute & wiring checks ----
            from agent_core.patterns import detect_module_collisions, detect_attribute_errors, detect_unwired_modules, detect_class_conflicts

            collisions = detect_module_collisions(py_new, existing_files=list(all_content.keys()))
            if collisions:
                print("\n  [review] Module name collisions:")
                for c in collisions:
                    print(f"    {c['file']}: {c['suggestion']}")
                    issues_found.append(f"{c['file']}: {c['suggestion']}")

            attr_errors = detect_attribute_errors(all_content, ws)
            if attr_errors:
                print("\n  [review] Attribute errors (likely bugs):")
                for ae in attr_errors:
                    print(f"    {ae['file']}: {ae['suggestion']}")
                    issues_found.append(f"{ae['file']}: {ae['suggestion']}")

            unwired = detect_unwired_modules(py_new, ws)
            if unwired:
                print("\n  [review] Unwired modules (not imported by any code):")
                for uw in unwired:
                    print(f"    {uw['file']}: {uw['suggestion']}")
                    issues_found.append(f"{uw['file']}: {uw['suggestion']}")

            class_conflicts = detect_class_conflicts(py_new, ws)
            if class_conflicts:
                print("\n  [review] Class/function name conflicts with existing code:")
                for cc in class_conflicts:
                    print(f"    {cc['file']}:{cc['line']}: {cc['suggestion']}")
                    issues_found.append(f"{cc['file']}:{cc['line']}: {cc['suggestion']}")

            static_summary = ""
            if issues_found:
                static_summary = "\n".join(f"- {i}" for i in issues_found)
                static_summary = f"## Static analysis found {len(issues_found)} issue(s):\n\n{static_summary}\n\n"

            # ---- Actionable: offer to delete dangerous files ----
            dangerous_files: set[str] = set()
            dangerous_files.update(cc["file"] for cc in class_conflicts)
            dangerous_files.update(uw["file"] for uw in unwired)
            if dangerous_files:
                reasons = []
                if class_conflicts:
                    reasons.append(f"{sum(1 for cc in class_conflicts if cc['file'] in dangerous_files)} have class-name conflicts")
                if unwired:
                    reasons.append(f"{sum(1 for uw in unwired if uw['file'] in dangerous_files)} are unwired")
                print(f"\n  [review] {len(dangerous_files)} file(s) have issues ({', '.join(reasons)}):")
                for df in sorted(dangerous_files):
                    print(f"    {df}")
                choice = auto_choice(
                    "  Delete these files (and any generated files only they import)? (y/N): ",
                    default="n",
                    auto_default="n",
                ).strip().lower()
                if stop_requested():
                    return True
                if choice == "y":
                    # Delete the flagged files, then the TRANSITIVE CLOSURE:
                    # a generated file that is only imported by other deleted
                    # generated files becomes unwired itself and is removed
                    # too, so no orphaned dead code survives. Empty packages
                    # left behind are pruned as well.
                    delete_set = _unwired_closure(py_new, ws, dangerous_files)
                    for df in sorted(delete_set):
                        if stop_requested():
                            break
                        path = Path(ws) / df
                        if path.exists():
                            path.unlink()
                            print(f"    Deleted: {df}")
                    _prune_empty_dirs(ws, delete_set)

            # ---- Wire-in offer for kept unwired modules ----
            kept_unwired = [
                uw["file"] for uw in unwired
                if (Path(ws) / uw["file"]).exists()
            ]
            if kept_unwired:
                suggestions = _suggest_consumers(kept_unwired, ws)
                print(f"\n  [review] {len(kept_unwired)} new module(s) are not imported by any code:")
                for fname in kept_unwired:
                    print(f"    {fname}  → suggested consumer(s): {', '.join(suggestions[fname])}")
                wire_choice = auto_choice(
                    "  Wire them into consumers (LLM pass, max 3 files)? (y/N): ",
                    default="n",
                    auto_default="n",
                ).strip().lower()
                if stop_requested():
                    return True
                if wire_choice in ("y", "yes"):
                    await _wire_in_modules(agent, kept_unwired[:3], suggestions, ws)

            # ---- Per-file LLM review ----
            for fname in list(all_content.keys())[:8]:
                if stop_requested():
                    break
                content = all_content[fname]
                print(f"\n  Reviewing {fname} ({len(content)} bytes)...")
                review_msg = [
                    {"role": "system", "content": (
                        "Review this code for correctness. Check for:\n"
                        "1. Broken imports (importing from modules/paths that don't exist)\n"
                        "2. Missing __init__.py in new packages\n"
                        "3. Invalid API schemas (missing type:object, properties, required)\n"
                        "4. Off-by-one errors (e.g. 0-index vs 1-index line numbers)\n"
                        "5. Empty except blocks or silent error swallowing\n"
                        "6. Unused imports, missing imports, or type mismatches\n"
                        "7. Duplicated code — functions/classes redefined across files instead of imported\n"
                        "8. DRY violations — repeated logic that should be extracted into a shared utility\n\n"
                        "Be concise. List only actual bugs. Skip style concerns."
                    )},
                    {"role": "user", "content": f"{static_summary}Review this file for bugs:\n\n```python\n{content}\n```"},
                ]
                review = await agent.llm.chat(review_msg, disable_thinking=True)
                if review.startswith("[Error") or review.startswith("[LM Studio"):
                    continue
                if any(kw in review.lower() for kw in ("bug", "issue", "error", "broken", "missing", "invalid", "fix", "should", "incorrect", "fails", "dup", "duplicate", "dry", "repeat", "same as")):
                    print(f"  {review}")
                else:
                    print("  No bugs found.")

        # Auto-extract design decisions from this implementation
        if implemented and target_workspace:
            try:
                candidates = await extract_from_changes(
                    agent, implemented,
                    context=f"Task plan: {taskplan_content[:500] if taskplan_content else 'N/A'}"
                )
                if candidates:
                    print(f"\n[decide] Extracted {len(candidates)} decision candidates from this run:")
                    for i, c in enumerate(candidates, 1):
                        print(f"  {i}. {c.get('title', 'Untitled')}")
                        ctx = c.get('context', '')
                        if ctx:
                            print(f"     {ctx}")
                    print("\n  Record? (1,2/all/N, press Enter to skip): ", end="")
                    choice = read_input().strip().lower()
                    if stop_requested():
                        return True
                    if choice and choice != "n":
                        ws_str = str(Path(workspace_path(target_workspace)).resolve())
                        if choice == "all":
                            selected = list(range(len(candidates)))
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

        # Structured history record so FUTURE runs can reuse this execution
        # (read-only consumers; harmless when reports/ is unwritable).
        if target_workspace and all_files:
            try:
                from harnessfix.history import append_execution

                file_records = [
                    {"path": fname, "status": "written" if fname in implemented else "skipped"}
                    for fname in all_files
                ]
                append_execution(
                    workspace_path(target_workspace),
                    "implement",
                    file_records,
                    outcome="ok" if implemented else "noop",
                    note=f"implemented {len(implemented)}/{len(all_files)} files",
                )
            except Exception:
                pass

        return True