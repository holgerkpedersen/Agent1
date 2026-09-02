from __future__ import annotations

"""Versioned prompt template storage backed by db_io."""


from typing import Optional, Any

from .db_io import (
    get_latest_version,
    list_templates,
    load_template,
    save_template,
)


class PromptCache:
    """In-memory + SQLite-backed cache of versioned prompt templates.

    Templates are keyed by ``(task_type, profile_type)`` and stored in the
    ``prompt_templates`` table via :mod:`db_io`.  The cache keeps a local dict
    so repeated reads for the same key avoid further DB hits until explicitly
    evicted.

    Hit/miss counters track template-cache performance (plan CACHE item 1).
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], tuple[int, str]] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get_template(self, task_type: str, profile_type: str) -> str:
        """Return the latest template for *task_type*+*profile_type*.

        Falls back to a generic placeholder when no custom template exists.
        """
        key = (task_type, profile_type)
        cached = self._store.get(key)
        if cached is not None:
            self._hits += 1
            return cached[1]

        self._misses += 1
        version = get_latest_version(task_type, profile_type)
        if version is not None:
            text = load_template(task_type, profile_type, version)
            if text is not None:
                self._store[key] = (version, text)
                return text

        fallback = _DEFAULT_TEMPLATES.get(key, "Please analyze and respond to the following task: {task}")
        self._store[key] = (-1, fallback)
        return fallback

    def get_stats(self) -> dict[str, float | int]:
        """Return template-cache hit/miss statistics.

        Returns a dict with keys ``hits``, ``misses``, ``total_lookups``,
        and ``hit_rate_pct`` (percentage).  When no lookups have been made,
        *hit_rate_pct* is reported as ``0.0``.
        """
        total = self._hits + self._misses
        hit_rate = round(self._hits / max(total, 1) * 100.0, 2)
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_lookups": total,
            "hit_rate_pct": hit_rate,
        }

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

    def evict(self, task_type: Optional[str] = None, profile_type: Optional[str] = None) -> int:
        """Remove cached entries.  Omit both args to clear everything."""
        before = len(self._store)
        keys_to_remove = [
            key for key in self._store
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
    ("implement", "fast_codegen"): "Generate a minimal implementation of the following task: {task}",
    ("implement", "deep_analysis"): "Thoroughly analyze and implement the following task with full error handling: {task}",
    ("implement", "precise"): "Implement only what is strictly required for this task, nothing more: {task}",
    ("fix", "fast_codegen"): "Fix the bug described below. Output only the corrected code: {task}",
    ("fix", "deep_analysis"): "Diagnose and fix the root cause of this issue. Explain your reasoning: {task}",
    ("fix", "precise"): "Apply the minimal change needed to resolve this issue: {task}",
}