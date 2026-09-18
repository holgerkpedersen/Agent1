"""Detect loaded modules whose files changed on disk since the REPL started.

Read-only freshness checks for the interactive loop. Fixes applied on disk
during a session (e.g. by a ``paste`` agent run) do NOT affect the running
process — Python keeps imported modules in memory. These helpers let the
REPL warn the user instead of silently executing stale code (2026-08-19
incident: a paste-session fix to ``workflow_cmd.py`` never took effect in
the running REPL).

Staleness is judged by CONTENT (a SHA-1 fingerprint), never by mtime alone.
An mtime bump is not a code change: the harnessfix repairs rewrite
``agent_core/llm/tool_loop.py`` with identical bytes during an apply/revert
cycle, editors touch files on save, and ``git`` operations can refresh
timestamps. Warning on those was a false positive — the running code was
byte-identical, so the REPL nagged the user to restart for nothing.
"""

from __future__ import annotations

import hashlib
import os
import sys

#: Package prefixes whose loaded module files are watched. Only modules
#: ALREADY imported matter — those are the code the process executes.
_WATCHED_PREFIXES = ("agent_core", "harnessfix")

#: Read size when hashing, so a large module is never slurped in one go.
_HASH_CHUNK = 1 << 20


def fingerprint_file(path: str) -> str:
    """SHA-1 hex digest of *path*'s bytes — the content authority for staleness.

    Deliberately ignores size and mtime: the same bytes rewritten (repair
    apply/revert, no-op save) yield the same fingerprint and are NOT stale.
    """
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_watched(path: str) -> bool:
    """True for files under the watched package directories."""
    norm = path.replace("\\", "/")
    return any(norm.startswith(f"{pkg}/") or f"/{pkg}/" in norm for pkg in _WATCHED_PREFIXES)


def loaded_module_fingerprints(entry_script: str | None = None) -> dict[str, str]:
    """Path -> content fingerprint for every loaded watched module file.

    ``entry_script`` (the agent.py path) is included explicitly because the
    main script lives in ``__main__``, not under a watched package.
    """
    paths: list[str] = []
    if entry_script:
        paths.append(entry_script)
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        src = f[:-4] + ".py" if f.endswith(".pyc") else f
        if _is_watched(src):
            paths.append(src)
    fingerprints: dict[str, str] = {}
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            fingerprints[os.path.abspath(p)] = fingerprint_file(p)
        except OSError:
            continue
    return fingerprints


def diff_snapshots(snapshot: dict[str, str]) -> list[str]:
    """Paths whose CONTENT differs from ``snapshot`` or whose file is gone.

    An mtime-only touch (identical bytes) is intentionally NOT reported.
    """
    stale: list[str] = []
    for path, digest in snapshot.items():
        try:
            current = fingerprint_file(path)
        except OSError:
            stale.append(path)
            continue
        if current != digest:
            stale.append(path)
    return sorted(stale)


def format_stale_warning(paths: list[str], limit: int = 5) -> str:
    """Warning text for stale loaded module files, capped at ``limit`` paths."""
    shown = paths[:limit]
    more = len(paths) - len(shown)
    lines = [
        f"WARNING: {len(paths)} loaded module file(s) changed on disk since this REPL started —",
        "the running code is STALE. Restart the REPL to load the changes.",
    ]
    lines.extend(f"  - {p}" for p in shown)
    if more > 0:
        lines.append(f"  - ... and {more} more")
    return "\n".join(lines)
