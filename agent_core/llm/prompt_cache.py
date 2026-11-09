"""Versioned prompt template storage backed by db_io."""

from __future__ import annotations

from typing import Optional, Any

from .db_io import (
    get_latest_version,
    list_templates,
    load_template,
    save_template,
)


# Module-level "current command" context so template-cache lookups can be
# attributed to the REPL command that triggered them (surfaced in the perf view).
# Set/cleared by the command execution path; None means unattributed.
_current_command: str | None = None


def set_current_command(command: str | None) -> str | None:
    """Set the command context for per-command prompt-cache attribution.

    Returns the previous value so callers can restore it.
    """
    global _current_command
    previous = _current_command
    _current_command = command
    return previous


def get_current_command() -> str | None:
    """Return the active command context (or None)."""
    return _current_command


def _stats_dict(hits: int, misses: int) -> dict[str, float | int]:
    total = hits + misses
    hit_rate = round(hits / max(total, 1) * 100.0, 2)
    return {
        "hits": hits,
        "misses": misses,
        "total_lookups": total,
        "hit_rate_pct": hit_rate,
    }


class PromptCache:
    """In-memory + SQLite-backed cache of versioned prompt templates.

    Templates are keyed by ``(task_type, profile_type)`` and stored in the
    ``prompt_templates`` table via :mod:`db_io`.  The cache keeps a local dict
    so repeated reads for the same key avoid further DB hits until explicitly
    evicted.

    Hit/miss counters track template-cache performance (plan CACHE item 1).
    When a command context is active (see :func:`set_current_command`), lookups
    are also attributed per command so the perf view can report hit % per
    command.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], tuple[int, str]] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._by_command: dict[str, dict[str, int]] = {}

    def _record(self, *, hit: bool) -> None:
        if hit:
            self._hits += 1
        else:
            self._misses += 1
        cmd = _current_command
        if cmd:
            bucket = self._by_command.setdefault(cmd, {"hits": 0, "misses": 0})
            bucket["hits" if hit else "misses"] += 1

    def get_template(self, task_type: str, profile_type: str) -> str:
        """Return the latest template for *task_type*+*profile_type*.

        Falls back to a generic placeholder when no custom template exists.
        """
        key = (task_type, profile_type)
        cached = self._store.get(key)
        if cached is not None:
            self._record(hit=True)
            return cached[1]

        self._record(hit=False)
        version = get_latest_version(task_type, profile_type)
        if version is not None:
            text = load_template(task_type, profile_type, version)
            if text is not None:
                self._store[key] = (version, text)
                return text

        fallback = _DEFAULT_TEMPLATES.get(
            key, "Please analyze and respond to the following task: {task}"
        )
        self._store[key] = (-1, fallback)
        return fallback

    def get_stats(self) -> dict[str, float | int]:
        """Return template-cache hit/miss statistics.

        Returns a dict with keys ``hits``, ``misses``, ``total_lookups``,
        and ``hit_rate_pct`` (percentage).  When no lookups have been made,
        *hit_rate_pct* is reported as ``0.0``.
        """
        return _stats_dict(self._hits, self._misses)

    def get_command_stats(self) -> dict[str, dict[str, float | int]]:
        """Return per-command hit/miss statistics (sorted by lookups desc).

        Only commands that triggered at least one template lookup while the
        command context was set are included.
        """
        items = [
            (cmd, _stats_dict(b["hits"], b["misses"]))
            for cmd, b in self._by_command.items()
        ]
        items.sort(key=lambda kv: -int(kv[1]["total_lookups"]))
        return dict(items)

    def reset_stats(self) -> None:
        """Clear cumulative hit/miss counters (keeps cached templates)."""
        self._hits = 0
        self._misses = 0
        self._by_command.clear()

    def put_template(
        self,
        task_type: str,
        profile_type: str,
        template_text: str,
        *,
        version: int = 1,
    ) -> None:
        """Persist a new (or updated) template and refresh the local cache."""
        save_template(task_type, profile_type, version, template_text)
        self._store[(task_type, profile_type)] = (version, template_text)

    def evict(
        self,
        task_type: Optional[str] = None,
        profile_type: Optional[str] = None,
    ) -> int:
        """Remove cached entries.  Omit both args to clear everything."""
        before = len(self._store)
        keys_to_remove = [
            key
            for key in self._store
            if (task_type is None or key[0] == task_type)
            and (profile_type is None or key[1] == profile_type)
        ]
        for key in keys_to_remove:
            del self._store[key]
        return before - len(self._store)

    def list_all(self) -> list[dict[str, Any]]:
        """Return all stored templates (bypasses the local cache)."""
        return list_templates()


_DEFAULT_TEMPLATES: dict[tuple[str, str], str] = {
    ("implement", "fast_codegen"): (
        "Generate a minimal implementation of the following task: {task}"
    ),
    ("implement", "deep_analysis"): (
        "Thoroughly analyze and implement the following task "
        "with full error handling: {task}"
    ),
    ("implement", "precise"): (
        "Implement only what is strictly required for this task, nothing more: {task}"
    ),
    ("fix", "fast_codegen"): (
        "Fix the bug described below. Output only the corrected code: {task}"
    ),
    ("fix", "deep_analysis"): (
        "Diagnose and fix the root cause of this issue. Explain your reasoning: {task}"
    ),
    ("fix", "precise"): (
        "Apply the minimal change needed to resolve this issue: {task}"
    ),
}


# Process-wide singleton: the observable/wired prompt cache surfaced by the
# perf view and the monitoring dashboard.  Production code that resolves a
# prompt template should prefer this instance so cache effectiveness is
# measurable; fresh PromptCache() instances remain available for isolated use.
_prompt_cache: PromptCache | None = None


def get_prompt_cache() -> PromptCache:
    """Return the process-wide PromptCache singleton."""
    global _prompt_cache
    if _prompt_cache is None:
        _prompt_cache = PromptCache()
    return _prompt_cache


def reset_prompt_cache() -> None:
    """Replace the singleton (test/helper use)."""
    global _prompt_cache
    _prompt_cache = None


def get_prompt_cache_stats() -> dict[str, Any]:
    """Return the singleton's overall + per-command stats for dashboards."""
    pc = get_prompt_cache()
    return {
        "overall": pc.get_stats(),
        "by_command": pc.get_command_stats(),
    }
