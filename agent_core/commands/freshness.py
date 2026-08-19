"""Detect loaded modules whose files changed on disk since the REPL started.

Read-only freshness checks for the interactive loop. Fixes applied on disk
during a session (e.g. by a ``paste`` agent run) do NOT affect the running
process — Python keeps imported modules in memory. These helpers let the
REPL warn the user instead of silently executing stale code (2026-08-19
incident: a paste-session fix to ``workflow_cmd.py`` never took effect in
the running REPL).
"""
import os
import sys

#: Package prefixes whose loaded module files are watched. Only modules
#: ALREADY imported matter — those are the code the process executes.
_WATCHED_PREFIXES = ("agent_core", "harnessfix")


def _is_watched(path: str) -> bool:
    """True for files under the watched package directories."""
    norm = path.replace("\\", "/")
    return any(norm.startswith(f"{pkg}/") or f"/{pkg}/" in norm for pkg in _WATCHED_PREFIXES)


def loaded_module_mtimes(entry_script: str | None = None) -> dict[str, float]:
    """Path -> mtime for every loaded watched module file.

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
    mtimes: dict[str, float] = {}
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            mtimes[os.path.abspath(p)] = os.path.getmtime(p)
        except OSError:
            continue
    return mtimes


def diff_snapshots(snapshot: dict[str, float]) -> list[str]:
    """Paths in ``snapshot`` whose on-disk mtime differs or whose file is gone."""
    stale: list[str] = []
    for path in snapshot:
        try:
            current = os.path.getmtime(path)
        except OSError:
            stale.append(path)
            continue
        if current != snapshot[path]:
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
