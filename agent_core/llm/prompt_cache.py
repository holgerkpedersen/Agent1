"""Versioned prompt template storage backed by db_io."""

from __future__ import annotations

from typing import Optional

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
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], tuple[int, str]] = {}

    def get_template(self, task_type: str, profile_type: str) -> str:
        """Return the latest template for *task_type*+*profile_type*.

        Falls back to a generic placeholder when no custom template exists.
        """
        key = (task_type, profile_type)
        if key in self._store:
            return self._store[key][1]

        version = get_latest_version(task_type, profile_type)
        if version is not None:
            text = load_template(task_type, profile_type, version)
            if text is not None:
                self._store[key] = (version, text)
                return text

        fallback = _DEFAULT_TEMPLATES.get(key, "Please analyze and respond to the following task: {task}")
        self._store[key] = (-1, fallback)
        return fallback

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
        keys_to_remove = []
        for key in self._store:
            if task_type is not None and key[0] != task_type:
                continue
            if profile_type is not None and key[1] != profile_type:
                continue
            keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._store[key]
        return before - len(self._store)

    def list_all(self) -> list[dict]:
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
