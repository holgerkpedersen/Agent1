"""Safe shell command allow-list and validation logic.

This module provides an explicit allow-list of permitted shell commands to replace
the fragile blacklist approach previously used in tool execution.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Final, List, Set

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Allow-list of safe shell commands (binary names only).
#:
#: This is the TIER-2 sandbox list: a command must appear here to run under
#: :func:`agent_core.tools.shell_ops.run_command` (structural scan +
#: ``shell=False``).  It is deliberately small — every entry is a capability.
#: ``python`` is the widest entry by far (it can read and write any file the
#: process can reach), so this list is NOT the boundary for the interactive
#: ``run`` tool; see ``RUN_SHELL_COMMANDS`` for that.
SAFE_COMMANDS: Final[Set[str]] = {
    "python",
    "python3",
    "git",
    "ls",
    "dir",
    "cat",
    "type",
    "head",
    "tail",
    "echo",
    "pwd",
}

#: Commands the interactive ``run`` tool may execute in **strict** mode
#: (``AGENT_RUN_SANDBOX=strict``).  These run through the TIER-1 destructive
#: block-list plus a structural scan that ignores quoted text, and are still
#: launched in the advertised shell so pipelines keep working.
#:
#: The list is an *interpreter/cmdlet* list, not a program list: a PowerShell
#: cmdlet is only reachable through ``pwsh``/``powershell``, so blocking a
#: cmdlet name here buys nothing — what matters is that no **arbitrary**
#: executable is reachable, and that ``pwsh``/``powershell`` themselves are
#: gated by :data:`RUN_SHELL_INVOKE_PATTERNS`.
RUN_SHELL_COMMANDS: Final[Set[str]] = {
    # Interpreters the agent legitimately drives.
    "python",
    "python3",
    "py",
    "pytest",
    # Version control.
    "git",
    # POSIX file/text tools (PowerShell aliases these to cmdlets).
    "ls",
    "dir",
    "cat",
    "type",
    "head",
    "tail",
    "echo",
    "pwd",
    "find",
    "grep",
    "wc",
    "sort",
    "uniq",
    "tr",
    "cut",
    "sed",
    "tree",
    "which",
    "where",
    "basename",
    "dirname",
    "file",
    "stat",
    "du",
    "df",
    "touch",
    "mkdir",
    "mv",
    "cp",
    "rm",
    "md5sum",
    "sha256sum",
    # Windows console text tools.
    "findstr",
    "more",
    "set",
    # PowerShell builtins used for output shaping (see the Invoke- patterns).
    "write-output",
    "write-host",
    "select-string",
    "select-object",
    "foreach-object",
    "foreach",
    "get-childitem",
    "get-content",
    "get-location",
    "get-command",
    "measure-object",
    "format-table",
    # The shell itself, reachable only under the Invoke- gate below.
    "pwsh",
    "powershell",
}

#: Interpreter modules the agent may run with ``python -m <module>`` in
#: **strict** mode.  A module name is not attacker-chosen code only when the
#: module is one the agent is *supposed* to drive; ``python -m pip install``
#: downloads a package and **executes the package's ``setup.py`` locally**,
#: which is arbitrary code execution with a network fetch in front of it —
#: the supply-chain leg of the Black October payload-drop chain.  The
#: allow-list keeps the tooling the agent needs (pytest, mypy) and refuses
#: every package-install channel: pip, uv, poetry, npm, npx, dotnet.
#:
#: The entry is a bare module name, or a name with a trailing ``*`` meaning
#: "this name, and any ``name.submodule`` below it".
RUN_SHELL_MODULE_ALLOWLIST: Final[Set[str]] = {
    "pytest",
    "mypy",
    "coverage",
    "unittest",
    "pylint",
    "black",
    "ruff",
    "isort",
    "flake8",
    "pyright",
    "py_compile",
    "venv",  # creates a local venv; installing INTO it is gated by pip
    "pip.*",  # read-only pip is safe (see RUN_SHELL_MODULE_BLOCKED_SUBCOMMANDS)
}

#: Subcommands that are refused after an allow-listed module.  ``pip show``
#: and ``pip list`` read metadata; ``pip install``/``download``/``wheel``
#: fetch a package and run its build code on this machine.  Prefix-matched
#: (:data:`_module_subcommand_blocked`), so a flag is never mistaken for a
#: verb and ``install-something`` is caught by ``install``.
RUN_SHELL_MODULE_BLOCKED_SUBCOMMANDS: Final[Dict[str, Set[str]]] = {
    "pip": {
        "install",
        "uninstall",
        "download",
        "wheel",
        "cache",
        "config",
        "index",
        "mock",  # `python -m pip mock` is the classic typosquat lure name
    },
}

#: Names that are **never** a legal ``-m`` target in strict mode, whatever
#: the allow-list says (a package manager is a package manager).
RUN_SHELL_PACKAGE_MANAGERS: Final[Set[str]] = {
    "pip",
    "pip3",
    "easy_install",
    "setuptools",
    "distutils",
    "ensurepip",
    "uv",
    "uvx",
    "poetry",
    "pdm",
    "flit",
    "tox",
    "npm",
    "npx",
    "yarn",
    "pnpm",
    "dotnet",
    "nuget",
    "choco",
    "winget",
    "scoop",
}

#: Command shapes that are refused in **strict** mode even when the binary is
#: in :data:`RUN_SHELL_COMMANDS`.  Each entry closes a specific escape hatch
#: that turns an allowed name into arbitrary code execution.
#:
#: Deliberately NOT blocked: ``python -m <module>`` for a module in
#: :data:`RUN_SHELL_MODULE_ALLOWLIST` (the agent runs ``python -m pytest``),
#: and a leading absolute path (``/usr/bin/git``, which stays subject to the
#: binary allow-list).  ``-m`` targets outside that list — including every
#: package manager — are refused by :func:`_run_shell_module_violation`.
RUN_SHELL_INVOKE_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (
        re.compile(r"\b(?:invoke-expression|iex)\b", re.I),
        "dynamic code execution (Invoke-Expression / iex)",
    ),
    (
        # .NET interop types — the constructors behind the classic
        # download-and-run (`New-Object Net.WebClient` + `DownloadString`),
        # reflection-loaded assemblies, and direct P/Invoke.  Legit agent
        # work never needs them; the allow-list never reaches cmdlets, so
        # this closes the shape that survives the binary gate.
        re.compile(
            r"\bnew-object\b[\s\S]*\bnet\.(?:webclient|webrequest)"
            r"|\bnew-object\b[\s\S]*\bsystem\.(?:diagnostics\.process|net\.webclient"
            r"|reflection\.assembly|runtime\.interopservices\.marshal)"
            r"|\b(?:net\.webclient|net\.webrequest|system\.diagnostics\.process"
            r"|system\.reflection\.assembly|system\.runtime\.interopservices\.marshal)\b",
            re.I,
        ),
        ".NET interop type (download-and-run / process spawn / P/Invoke)",
    ),
    (
        re.compile(
            r"\bpwsh(?=[\s'\"-])[\s\S]*\s-(?:c|e|command|encodedcommand)\b", re.I
        ),
        "nested shell -c / -Command (arbitrary code via the shell)",
    ),
    (
        re.compile(
            r"\bpowershell(?=[\s'\"-])[\s\S]*\s-(?:c|e|command|encodedcommand)\b",
            re.I,
        ),
        "nested shell -c / -Command (arbitrary code via the shell)",
    ),
    (
        # `./prog`, `.\prog.ps1`, `& prog` — the call forms for local files.
        re.compile(r"^[\s'\"&]*(?:\.[/\\]|&\s)", re.I),
        "path-prefixed execution (./program, & program)",
    ),
    (
        re.compile(r"(?:^|[;&|]\s*)\.(?=\s|$)", re.I),
        "dot-source (`. file` loads and runs a script)",
    ),
    (
        # python/py -c, and combined short flags like -uc.  `python -3`
        # (the versioned launcher name) is accepted, so an optional numeric
        # flag may sit between the binary and -c.
        re.compile(r"\b(?:python\d*|py)(?:\s+-\d[a-z0-9]*)?\s+-[a-z]*c(?=\s|$)", re.I),
        "python -c (arbitrary code via the interpreter)",
    ),
    (
        # `py -3 -c` — the launcher takes a version flag before -c, so the
        # binary and the flag are not adjacent.
        re.compile(r"\bpy\s+-\d[a-z0-9]*\s+-[a-z]*c(?=\s|$)", re.I),
        "py -<ver> -c (arbitrary code via the interpreter)",
    ),
)

#: Structural shell patterns that must never appear in a command string —
#: pipes, redirection, chaining, command substitution.  Even with
#: ``shell=False`` these are rejected up front so no caller can smuggle a
#: chained command past the binary allow-list (plan OPS item 2).
_UNSAFE_SHELL_TOKENS: Final[tuple[tuple[str, str], ...]] = (
    ("&&", "command chaining (&&)"),
    ("||", "command chaining (||)"),
    (";", "command separator (;)"),
    ("|", "pipe (|)"),
    (">", "redirection (>)"),
    ("<", "redirection (<)"),
    ("`", "command substitution (`)"),
    ("$(", "command substitution ($()"),
    ("\n", "embedded newline"),
    ("\r", "embedded newline"),
)

#: Destructive shell patterns refused outright (word-boundary,
#: case-insensitive) — the command injection surface of the NLP loop.
#: Moved here from ``agent.py`` so the policy has ONE owner (plan #16);
#: ``agent._DANGEROUS_SHELL_PATTERNS`` is now an alias of this object.
DESTRUCTIVE_SHELL_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"rm\s+-r[f]?", re.I), "recursive file removal (rm -r/-rf)"),
    (re.compile(r"\bdeltree\b", re.I), "deltree"),
    (re.compile(r"\brd\s+/s", re.I), "rd /s"),
    (re.compile(r"\brmdir\s+/s", re.I), "rmdir /s"),
    (re.compile(r"\bdel\s+/[sqf]", re.I), "del /s /q /f"),
    (re.compile(r"\bformat\s+[a-z]:", re.I), "format <drive>:"),
    (re.compile(r"\bshutdown\b", re.I), "shutdown"),
    (re.compile(r"\breboot\b", re.I), "reboot"),
    (re.compile(r"restart-computer", re.I), "restart-computer"),
    (re.compile(r"stop-computer", re.I), "stop-computer"),
    (re.compile(r"\bdiskpart\b", re.I), "diskpart"),
    (re.compile(r"\bmkfs\b", re.I), "mkfs"),
    (re.compile(r"wipefs", re.I), "wipefs"),
    (re.compile(r"\bdd\s+of=", re.I), "dd of="),
    (re.compile(r"taskkill\s+/f", re.I), "taskkill /f"),
    (re.compile(r"\breg\s+delete", re.I), "reg delete"),
    (re.compile(r"remove-item\s+-recurse", re.I), "Remove-Item -Recurse"),
    (re.compile(r"clear-recyclebin", re.I), "Clear-RecycleBin"),
    (re.compile(r"format-volume", re.I), "Format-Volume"),
    (re.compile(r"invoke-expression", re.I), "Invoke-Expression"),
)

# ---------------------------------------------------------------------------
# Dynamic Allowlist State & Logger
# ---------------------------------------------------------------------------

_dynamic_allowlist: Set[str] = set()
_logger = logging.getLogger(__name__)


def _normalize(binary_name: str) -> str:
    """Normalize a binary name for allow-list comparison."""
    normalized = binary_name.strip().lower()
    for suffix in (".exe", ".bat", ".cmd"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _basename(invocation: str) -> str:
    """Reduce a command invocation to its bare binary name.

    ``/usr/bin/git``, ``C:\\Program Files\\Git\\bin\\git.exe`` and ``git``
    all name the same binary, so the allow-list must compare basenames after
    normalizing separators.  Quote characters are stripped so a quoted
    invocation (``"git"`` on Windows) resolves to the same name.
    """
    text = invocation.strip().strip("'\"")
    return text.replace("\\", "/").rsplit("/", 1)[-1]


def _resolve_invocation(masked: str, start: int, original: str) -> str:
    """The invocation text, with a masked quoted span resolved from *original*.

    *masked* is the :func:`_blanked_quote_mask` copy (quoted spans are runs
    of ``@``) and *start* is the offset where the invocation begins.  A
    quoted invocation (``"C:\\Program Files\\Git\\bin\\git.exe"``) survives
    the mask as a run of ``@``; the real name is taken from the same slice
    of the ORIGINAL text so the allow-list sees the binary, not the mask.
    """
    if not masked.startswith("@"):
        return masked
    quoted = original[start : start + len(masked)]
    # The mask blanks the quote characters too, so the original slice is the
    # full quoted token (`"path with spaces"`); strip the quotes.
    return quoted.strip().strip("'\"")


# A command invocation (the first token of a shell segment): a bare name, a
# masked quoted span, or an unquoted path.  A path keeps running across
# spaces as long as the *next* path component is followed by another
# separator, so `C:\Program Files\Git\bin\git.exe` is ONE invocation while
# the trailing ` status` argument is not swallowed.  Splitting on
# whitespace broke the path apart and invented a segment starting with
# `Program` — a false positive that blocked normal work.
_SEGMENT_INVOKE_RE = re.compile(
    r"""
    (?:
        @+                                        # quoted invocation (mask)
      | [\w.:\-]+(?:[/\\][\w.:\- ]*[/\\][\w.:\-]+)+  # path (spaces inside ok)
      | [\w.\-]+                                  # bare binary name
    )
    """,
    re.VERBOSE,
)



# A command invocation (the first token of a shell segment): a bare name, a
# quoted path, or an unquoted path.  A path keeps running across spaces as
# long as the *next* path component is followed by another separator, so
# `C:\Program Files\Git\bin\git.exe` is ONE invocation while the trailing
# ` status` argument is not swallowed.  Splitting on whitespace broke the
# path apart and invented a segment starting with `Program` — a false
# positive that blocked normal work.
_SEGMENT_INVOKE_RE = re.compile(
    r"""
    (?:
        @+                                   # quoted invocation (masked span)
      | (?:[\w.:\-]+[/\\][\w.:\- ]*[/\\])+[\w.:\-]+  # path (spaces inside ok)
      | [\w.\-]+                             # bare binary name
    )
    """,
    re.VERBOSE,
)


def _segment_invocations(command: str, original: str) -> List[str]:
    """The binary invocation (first token) of every shell segment.

    *command* must have its quoted spans masked (see
    :func:`_blanked_quote_mask`) so a quoted argument is never mistaken for
    a command, and its redirection operators dropped (see
    :data:`_REDIRECTION_RE`) so ``2>&1`` does not create a segment starting
    with ``1``.  *original* is the un-masked command text, used to recover
    the real name of a quoted invocation (see :func:`_resolve_invocation`).
    A segment that starts with an operator (``&&``, ``|``, ``;``) is a
    continuation of the previous segment, not a new command, and is skipped.

    The invocation is taken with :data:`_SEGMENT_INVOKE_RE` (not whitespace
    splitting) so a path with spaces — quoted
    (``"C:\\Program Files\\Git\\bin\\git.exe"``) or unquoted
    (``C:\\Program Files\\Git\\bin\\git.exe``) — is ONE invocation.
    """
    invocations: List[str] = []
    for segment in re.split(r"&&|\|\||[;|&\r\n]", command):
        seg = segment.strip()
        if not seg:
            continue
        # An operator-led segment (`&& git status`, `| head -20`) continues
        # the previous command; its first token is an argument, not a binary.
        if seg[0] in ("&", "|", ";", ">"):
            continue
        match = _SEGMENT_INVOKE_RE.match(seg)
        if match:
            # A masked quoted invocation (`@...`) is resolved against the
            # ORIGINAL text so the allow-list sees the real binary name.
            invocations.append(
                _resolve_invocation(match.group(0), match.start(), original)
            )
        else:
            parts = seg.split()
            if parts:
                invocations.append(parts[0])
    return invocations


def _blanked_quote_mask(command: str) -> str:
    """Equal-length copy of *command* with quoted spans replaced by ``@``.

    ``@`` is not a shell operator and not a quote, so a quoted span is inert
    for the structural scan: ``git commit -m "fix: a && b"`` stays ONE
    segment starting with ``git`` instead of being split at the ``&&``
    inside the quoted argument.  Replacing each quoted character with a
    single ``@`` keeps the length identical, so reported offsets stay valid.

    The invocation matcher (:data:`_SEGMENT_INVOKE_RE`) treats a leading
    run of ``@`` as a quoted invocation and recovers the real binary name
    from the ORIGINAL text, so masking never loses the name itself.
    """
    out: List[str] = []
    quote: str | None = None
    escaped = False
    for ch in command:
        if escaped:
            out.append("@" if quote else ch)
            escaped = False
            continue
        if ch == "\\" and quote:
            out.append("@")
            escaped = True
            continue
        if quote:
            out.append("@")
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append("@")
            continue
        out.append(ch)
    return "".join(out)



# Redirection operators, matched before the segment split.  `2>&1` contains
# `&`; splitting on it would invent a segment starting with `1` and refuse a
# command (``cat a.txt 2>&1 | head -20``) that is entirely allow-listed.
# Redirection targets are files, not commands, so dropping the operator
# leaves nothing that needs allow-listing.
_REDIRECTION_RE = re.compile(r"(?:\d*)\s*(?:>>|>|<|2>&1|<&|>&)")


# `python -m <module> [<sub>]` — the interpreter, its flags, the -m flag
# (short cluster allowed: -Bm, -qm), the module, and the module's first
# positional verb.  The verb is only checked when it is a bare word, so
# `python -m pip install -r requirements.txt` reports `install` while
# `python -m pytest -q` reports nothing.
_MODULE_INVOKE_RE = re.compile(
    r"\b(?:python\d*|py)\b(?P<flags>(?:\s+-[A-Za-z0-9:]+)*)"
    r"\s+(?:-[A-Za-z]*m[mc]?\b|--module\s)"
    r"\s*(?P<module>[A-Za-z_][\w.]*)"
    r"(?P<sub>\s+[A-Za-z][\w-]*)?",
    re.I,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_unsafe_shell_pattern(cmd_str: str) -> str | None:
    """Return a description of the first unsafe shell pattern in *cmd_str*.

    Rejects shell metacharacters, pipes, redirection operators, command
    chaining (``&&`` / ``||`` / ``;``), and command substitution.  Returns
    ``None`` when the command string is structurally safe — the binary
    allow-list then decides whether the command may run.

    This scanner looks at the **raw** string, so it also rejects
    metacharacters that merely appear inside a quoted argument.  The
    interactive ``run`` tool cannot use it for that reason (``python -c
    "import sys; ..."`` is legitimate); :func:`find_run_shell_pattern`
    quotes-aware scanner is the one for that path.
    """
    if not cmd_str:
        return None
    for token, description in _UNSAFE_SHELL_TOKENS:
        if token in cmd_str:
            return description
    return None


def find_run_shell_pattern(cmd_str: str) -> str | None:
    """Return the first **strict-mode** violation in *cmd_str*, or ``None``.

    This is the TIER-2 policy for the interactive ``run`` tool
    (``AGENT_RUN_SANDBOX=strict``).  Three checks, in order:

    1. :data:`RUN_SHELL_INVOKE_PATTERNS` — the escape hatches that turn an
       allowed name into arbitrary code (``Invoke-Expression``, a nested
       shell ``-Command``, ``python -c``, dot-sourcing, ``./prog``) — plus
       the ``python -m`` module gate (:func:`_run_shell_module_violation`).
    2. The binary name must be in :data:`RUN_SHELL_COMMANDS`.  Every shell
       segment is checked, so ``git status && curl ...`` is refused for
       ``curl`` even though the pipe itself is allowed.
    3. Unquoted command substitution (``$(`` / backtick) — pipes and
       redirection stay legal because every segment is allow-listed.

    Quoted spans are blanked before the structural check
    (:func:`_strip_quoted_segments`), so
    ``python -c "import sys; sys.exit(3)"`` is not rejected for its ``;``
    while ``git status && rm -rf /`` is refused for ``rm``.
    """
    if not cmd_str:
        return None

    # 1. Escape-hatch shapes, scanned on the quoted-blanked copy so a
    #    quoted argument containing the word "iex" is not a violation.
    stripped = _strip_quoted_segments(cmd_str)
    for pattern, description in RUN_SHELL_INVOKE_PATTERNS:
        if pattern.search(stripped):
            return description
    module_violation = _run_shell_module_violation(stripped)
    if module_violation:
        return module_violation

    # 2. Every shell segment must start with an allow-listed binary.
    #    Quoted spans are masked (not blanked) so a quoted argument is
    #    never mistaken for a command: `git commit -m "fix: a && b"` is
    #    ONE segment starting with `git`, not two segments starting with
    #    `b`.  Redirection is dropped before splitting: `2>&1` contains
    #    `&`, and splitting it would invent a segment starting with `1`.
    #    The target of a redirection is a file, not a command, so there is
    #    nothing to allow-list there.
    tokenizable = _REDIRECTION_RE.sub(" ", _blanked_quote_mask(cmd_str))
    for invocation in _segment_invocations(tokenizable, cmd_str):
        # `git` invoked as a path (`/usr/bin/git status`) is the same
        # binary; `./prog` was already refused by the Invoke- patterns.
        binary = _basename(invocation)
        if not is_run_shell_allowed(binary):
            return f"command '{binary}' is not in the run allow-list"

    # 3. Command substitution (same quoted-blanked copy as check 2).
    if "$(" in stripped:
        return "command substitution ($()"
    if "`" in stripped:
        return "command substitution (`)"
    return None


def _run_shell_module_violation(cmd_str: str) -> str | None:
    """Refuse ``python -m <module>`` when *module* is not agent tooling.

    The strict binary allow-list gates the interpreter, not what the
    interpreter loads.  ``python -m pytest`` is tooling; ``python -m pip
    install X`` is a network fetch that ends in **local execution of the
    package's build code** (``setup.py``) — the supply-chain leg of the
    Black October payload-drop chain, and reachable from the allow-list.

    Quoted spans are blanked first (see :func:`_strip_quoted_segments`), so
    the *arguments* of a command are never mistaken for the module name.
    """
    stripped = _strip_quoted_segments(cmd_str)
    for match in _MODULE_INVOKE_RE.finditer(stripped):
        module = match.group("module").lower()
        sub = (match.group("sub") or "").strip().lower()
        # Interpreter flags that change HOW code runs (-i runs an
        # interactive prompt) defeat the point of the sandbox, so the gate
        # applies to them too.
        loosening = [
            flag
            for flag in (match.group("flags") or "").split()
            if flag.lower() in ("-i", "-interactive")
        ]
        if loosening:
            return (
                f"interpreter flag '{loosening[0]}' with -m "
                "(changes how code runs; strict sandbox refuses it)"
            )
        allowed = _module_is_allowed(module)
        if sub in RUN_SHELL_PACKAGE_MANAGERS:
            return _package_manager_reason(module)
        if allowed:
            if sub and _module_subcommand_blocked(module, sub):
                return (
                    f"'{module} {sub}' fetches a package and runs its build "
                    "code (strict sandbox allows read-only metadata only)"
                )
            continue
        # A package manager outside the allow-list (pipx, poetry, …) gets
        # the supply-chain reason, not the generic one.
        if module in RUN_SHELL_PACKAGE_MANAGERS:
            return _package_manager_reason(module)
        return f"module '{module}' via -m is not in the run module allow-list"
    return None


def _package_manager_reason(module: str) -> str:
    """Why a package manager is refused as an ``-m`` target."""
    return (
        f"package manager via -m ({module}) "
        "(supply chain: runs the package's build code)"
    )


def _module_is_allowed(module: str) -> bool:
    """True when *module* matches :data:`RUN_SHELL_MODULE_ALLOWLIST`.

    An entry ending in ``*`` matches that name and everything below it
    (``pip.*`` covers ``pip`` and ``pip._internal``); a plain entry matches
    exactly.  The check is on the module ROOT, so ``venv`` being allowed does
    not silently open ``venv.something``.
    """
    root = module.split(".", 1)[0]
    for entry in RUN_SHELL_MODULE_ALLOWLIST:
        if entry.endswith(".*"):
            base = entry[:-2]
            if root == base:
                return True
        elif root == entry:
            return True
    return False


def _module_subcommand_blocked(module: str, sub: str) -> bool:
    """True when *sub* is a dangerous verb under an allow-listed *module*.

    Matched by prefix: ``config`` catches ``--config``-style spellings,
    ``install`` catches ``install-something``, and a plain flag (``-q``) can
    never match a verb.
    """
    root = module.split(".", 1)[0]
    blocked = RUN_SHELL_MODULE_BLOCKED_SUBCOMMANDS.get(root)
    if not blocked:
        return False
    # Exact match, or `--verb` / `--verb=…` (pip accepts `--install`,
    # `--index-url`, …); a bare `-q` can never match a verb.
    return any(
        sub == verb or sub == "--" + verb or sub.startswith("--" + verb + "=")
        for verb in blocked
    )


def is_run_shell_allowed(binary_name: str) -> bool:
    """Check *binary_name* against :data:`RUN_SHELL_COMMANDS` (normalized)."""
    return _normalize(binary_name) in RUN_SHELL_COMMANDS


def _strip_quoted_segments(cmd_str: str) -> str:
    """Blank out single/double-quoted spans so their contents are not scanned.

    ``python -c "import sys; sys.exit(3)"`` is a legitimate command: the ``;``
    is data inside a quoted argument, not a shell operator.  Quoted spans are
    replaced with an equal-length run of spaces so reported offsets stay valid
    and the surrounding text is still checked.
    """
    out: List[str] = []
    quote: str | None = None
    escaped = False
    for ch in cmd_str:
        if escaped:
            out.append(" " if quote else ch)
            escaped = False
            continue
        if ch == "\\" and quote:
            out.append(" ")
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            out.append(" ")
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


def find_destructive_shell_pattern(cmd_str: str) -> str | None:
    """Return the first destructive pattern description in *cmd_str*, or None.

    This is the TIER-1 policy: the destructive block-list that every execution
    path shares (plan #16).  It is the only gate the permissive NLP ``run``
    dev-shell applies — that path deliberately still executes pipelines and
    chaining (see :func:`scan_command` for why), so it must NOT be given the
    structural scan.
    """
    if not cmd_str:
        return None
    for pattern, description in DESTRUCTIVE_SHELL_PATTERNS:
        if pattern.search(cmd_str):
            return description
    return None


def scan_command(cmd_str: str) -> str | None:
    """Return a description of the first policy violation in *cmd_str*.

    This is the STRICT policy (TIER-1 + TIER-2), for callers that execute
    through the binary allow-list (``tool_router``, ``tools/shell_ops``).  It
    applies, in order:

    1. destructive-pattern matching (:data:`DESTRUCTIVE_SHELL_PATTERNS`);
    2. the structural metacharacter scan (:func:`find_unsafe_shell_pattern`)
       on the command with quoted spans blanked out.

    Returns ``None`` when the command is acceptable — the caller then decides
    whether to apply the binary allow-list or its own fallback.

    Note the two-tier split: the NLP ``run`` tool intentionally does NOT use
    this function.  It keeps a real shell and must still execute pipes
    (``... 2>&1 | ...``), which on cmd.exe fail silently and are surfaced to
    the model as a wrong-shell Hint — a documented feature pinned by
    ``tests/test_tool_loop_nlp.py::TestRunToolShellAwareness``.  Rejecting
    metacharacters there would delete that diagnostic.  The ``run`` tool's
    strict-mode policy is :func:`find_run_shell_pattern`, which allow-lists
    every shell segment instead of banning the pipe itself.
    """
    destructive = find_destructive_shell_pattern(cmd_str)
    if destructive is not None:
        return destructive
    return find_unsafe_shell_pattern(_strip_quoted_segments(cmd_str))


def is_command_allowed(binary_name: str) -> bool:
    """Check if a shell command binary is in the allow-list.

    The check is case-insensitive and strips common Windows executable suffixes
    (``.exe``, ``.bat``, ``.cmd``) before comparison so that commands like
    ``python.exe`` are accepted alongside ``python``.

    Parameters
    ----------
    binary_name : str
        The raw command string as provided by the caller (e.g. ``"Git"``,
        ``"cat.exe"``).

    Returns
    -------
    bool
        ``True`` when *binary_name* corresponds to an allowed command,
        ``False`` otherwise.
    """
    normalized = _normalize(binary_name)
    allowed = normalized in SAFE_COMMANDS or normalized in _dynamic_allowlist

    source: str = (
        "static" if normalized in SAFE_COMMANDS else ("dynamic" if allowed else "none")
    )
    _logger.info(
        "Command allowlist check",
        extra={
            "event": "command_allowlist_check",
            "binary": binary_name,
            "normalized": normalized,
            "allowed": allowed,
            "source": source,
        },
    )

    return allowed


def register_command(binary_name: str) -> None:
    """Dynamically add a command to the allow-list.

    Parameters
    ----------
    binary_name : str
        The command name to allow (suffixes will be stripped during normalization).
    """
    normalized = _normalize(binary_name)
    if normalized not in SAFE_COMMANDS and normalized not in _dynamic_allowlist:
        _dynamic_allowlist.add(normalized)
        _logger.info(
            "Command registered",
            extra={
                "event": "command_allowlist_register",
                "binary": binary_name,
                "normalized": normalized,
            },
        )


def unregister_command(binary_name: str) -> bool:
    """Dynamically remove a command from the allow-list.

    Parameters
    ----------
    binary_name : str
        The command name to disallow.

    Returns
    -------
    bool
        ``True`` if the command was removed, ``False`` otherwise.
    """
    normalized = _normalize(binary_name)
    if normalized in SAFE_COMMANDS:
        _logger.warning(
            "Cannot unregister static command",
            extra={
                "event": "command_allowlist_unregister_failed",
                "binary": binary_name,
                "reason": "static_command",
            },
        )
        return False

    if normalized in _dynamic_allowlist:
        _dynamic_allowlist.discard(normalized)
        _logger.info(
            "Command unregistered",
            extra={
                "event": "command_allowlist_unregister",
                "binary": binary_name,
                "normalized": normalized,
            },
        )
        return True

    return False


def get_allowed_commands() -> Set[str]:
    """Return the current union of static and dynamic allowed commands."""
    return SAFE_COMMANDS | _dynamic_allowlist
