"""Filename-safety predicates and task-plan text extraction.

Extracted from ``agent_core/commands/implement_cmd.py`` (which was over
100 KB) with no behavioural change.  ``implement_cmd`` re-exports every
public name, so existing imports keep working verbatim:

* ``fix_cmd.py`` imports the error-analysis helpers from ``implement_cmd``;
* ``proposal_core.py`` imports the gate predicates from ``implement_cmd``;
* tests import them from ``implement_cmd`` too.

The module is deliberately dependency-light (stdlib only) so it can be
imported from anywhere in the package without circular-import risk.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_STDLIB_COMMON = frozenset(
    {
        "abc",
        "aifc",
        "argparse",
        "array",
        "ast",
        "asynchat",
        "asyncore",
        "atexit",
        "audioop",
        "base64",
        "bdb",
        "binascii",
        "binhex",
        "bisect",
        "builtins",
        "bz2",
        "calendar",
        "cgi",
        "cgitb",
        "chunk",
        "cmath",
        "cmd",
        "code",
        "codecs",
        "codeop",
        "collections",
        "colorsys",
        "compileall",
        "concurrent",
        "config",
        "configparser",
        "contextlib",
        "contextvars",
        "copy",
        "copyreg",
        "cProfile",
        "crypt",
        "csv",
        "ctypes",
        "curses",
        "dataclasses",
        "datetime",
        "dbm",
        "decimal",
        "difflib",
        "dis",
        "distutils",
        "doctest",
        "email",
        "encodings",
        "enum",
        "errno",
        "faulthandler",
        "fcntl",
        "filecmp",
        "fileinput",
        "fnmatch",
        "formatter",
        "fractions",
        "ftplib",
        "functools",
        "gc",
        "getopt",
        "getpass",
        "gettext",
        "glob",
        "graphlib",
        "grp",
        "gzip",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "http",
        "idlelib",
        "imaplib",
        "imghdr",
        "imp",
        "importlib",
        "inspect",
        "io",
        "ipaddress",
        "itertools",
        "json",
        "keyword",
        "lib2to3",
        "linecache",
        "locale",
        "logging",
        "lzma",
        "mailbox",
        "mailcap",
        "marshal",
        "math",
        "mimetypes",
        "mmap",
        "modulefinder",
        "multiprocessing",
        "netrc",
        "nis",
        "nntplib",
        "numbers",
        "operator",
        "optparse",
        "os",
        "ossaudiodev",
        "parser",
        "pathlib",
        "pdb",
        "pickle",
        "pickletools",
        "pipes",
        "pkgutil",
        "platform",
        "plistlib",
        "poplib",
        "posix",
        "posixpath",
        "pprint",
        "profile",
        "pstats",
        "pty",
        "pwd",
        "py_compile",
        "pyclbr",
        "pydoc",
        "queue",
        "quopri",
        "random",
        "re",
        "readline",
        "reprlib",
        "resource",
        "rlcompleter",
        "runpy",
        "sched",
        "secrets",
        "select",
        "selectors",
        "shelve",
        "shlex",
        "shutil",
        "signal",
        "site",
        "smtpd",
        "smtplib",
        "sndhdr",
        "socket",
        "socketserver",
        "sqlite3",
        "stat",
        "statistics",
        "string",
        "stringprep",
        "struct",
        "subprocess",
        "sunau",
        "symtable",
        "sys",
        "sysconfig",
        "syslog",
        "tabnanny",
        "tarfile",
        "telnetlib",
        "tempfile",
        "termios",
        "test",
        "textwrap",
        "threading",
        "time",
        "timeit",
        "tkinter",
        "token",
        "tokenize",
        "trace",
        "traceback",
        "tracemalloc",
        "tty",
        "turtle",
        "turtledemo",
        "types",
        "typing",
        "unicodedata",
        "unittest",
        "urllib",
        "uu",
        "uuid",
        "venv",
        "warnings",
        "wave",
        "weakref",
        "webbrowser",
        "winreg",
        "winsound",
        "wsgiref",
        "xdrlib",
        "xml",
        "xmlrpc",
        "zipapp",
        "zipfile",
        "zipimport",
        "zlib",
        # Non-stdlib but very common collision targets
        "logger",
        "utils",
        "helpers",
        "common",
        "base",
        "core",
        "memory",
        "cache",
        "settings",
        "constants",
    }
)

_EXISTING_ROOT_PACKAGES = frozenset(
    {
        "agent_core",
        "agent1",
        "agent",
    }
)

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
        return (
            True,
            "__init__.py at workspace root would turn the entire repo into a package",
        )

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
        if (
            part in _STDLIB_COMMON
            and not (workspace / "/".join(parts[: idx + 1])).exists()
        ):
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
    _RE_1 = re.compile(r"^\d+\.?\s+`")
    for line in taskplan.split("\n"):
        if f"`{filename}`" in line:
            return line.strip()
    for line in taskplan.split("\n"):
        if f"`{name}`" in line:
            return line.strip()
    for line in taskplan.split("\n"):
        stripped = line.strip()
        if name in stripped and _RE_1.match(stripped):
            return stripped
    return ""
