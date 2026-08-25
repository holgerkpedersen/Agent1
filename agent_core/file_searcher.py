"""File search operations for agent.

A deterministic pure-Python walker (no findstr/grep subprocess) that
excludes git-ignored runtime state and binary/cache files, and reports
matches as ``path:lineno: content`` lines so the LLM can act on them.
"""

from __future__ import annotations
import os

from agent_core.path_utils import safe_path

#: Directories never searched (mirrors .gitignore for session state/caches).
_IGNORED_DIRS = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".coverage", ".opencode", ".poolside", ".vscode", ".tox", ".nox",
    "venv", ".venv", "env", ".env", "node_modules", "backups",
    "htmlcov", ".eggs", "dist", "build", ".idea",
    # .docs holds timestamped workflow docs (spec/analysis/plan/tasks),
    # not source — must not surface as code matches.
    ".docs",
}

#: Individual files never searched (runtime/session state).
_IGNORED_FILES = {
    "chat_history.json", "model.json", "project_memory.json",
    ".env", ".env.example", ".coverage", "model",
}


def _is_ignored_file(name: str) -> bool:
    """True for runtime state files and temporary workflow artifacts."""
    if name in _IGNORED_FILES:
        return True
    # project_*.md are transient outputs of the workflow command (spec ->
    # analysis -> plan -> tasks -> entities), not source of truth — they must
    # not surface as code matches.
    return name.startswith("project_") and name.endswith(".md")

#: Extensions never searched (binary/cache/asset files).
_IGNORED_EXTENSIONS = {
    ".db", ".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".pdf", ".zip", ".gz", ".tar", ".exe", ".dll", ".so", ".class", ".woff",
    ".woff2", ".ttf", ".mp3", ".mp4", ".wav", ".pkl", ".pickle", ".h5",
    ".npy", ".npz", ".jsonl",
}

_MAX_RESULTS = 50
_MAX_LINE_LEN = 200


class FileSearcher:
    """Handles file searching with platform-appropriate commands."""

    def __init__(self, workspace: str | None = None):
        self.workspace = workspace

    async def search(self, query: str, path: str) -> str:
        """Search for text pattern in files.

        Returns one ``path:lineno: content`` line per match (at most
        ``_MAX_RESULTS``), or ``"No matches found"``.
        """
        local_path = self._safe_path(path)
        matches = self._walk_search(query, local_path)
        if not matches:
            return "No matches found"
        lines = []
        for p, lineno, line in matches:
            if len(line) > _MAX_LINE_LEN:
                line = line[:_MAX_LINE_LEN] + "..."
            lines.append(f"{p}:{lineno}: {line}")
        if len(lines) > _MAX_RESULTS:
            lines = lines[:_MAX_RESULTS]
            lines.append(f"... and {len(matches) - _MAX_RESULTS} more matches")
        return "\n".join(lines)

    def _safe_path(self, path: str) -> str:
        """Normalize path for search."""
        return safe_path(path)

    def _walk_search(self, query: str, local_path: str) -> list[tuple[str, int, str]]:
        """Pure-Python search honoring ignore rules; returns (path, line, text)."""
        matches: list[tuple[str, int, str]] = []
        if not os.path.isdir(local_path):
            return matches
        for root, dirs, files in os.walk(local_path):
            dirs[:] = sorted(
                d for d in dirs if d not in _IGNORED_DIRS and not d.endswith(".egg-info")
            )
            for file in sorted(files):
                if _is_ignored_file(file) or file.endswith((".pyc", ".pyo")):
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in _IGNORED_EXTENSIONS:
                    continue
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        for lineno, line in enumerate(f, 1):
                            if query in line:
                                matches.append(
                                    (os.path.normpath(filepath), lineno, line.rstrip())
                                )
                                if len(matches) >= _MAX_RESULTS * 2:
                                    return matches
                except OSError:
                    continue
        return matches
