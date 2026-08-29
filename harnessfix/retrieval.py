"""Phase 6 - episodic retrieval: index successful runs, surface top-k.

Builds an in-memory ``VectorDatabase`` over the successful-run corpus using a
dependency-free embedder (harnessfix.embed) and answers two query modes:

  - "fix": the query is an error message + failing file + symbols.  We FIRST
    filter episodes to those sharing a file stem or error class with the
    query (high-precision signal the path-keyed history already exploits),
    THEN rank by embedding similarity within that filter.  If the filter is
    empty we fall back to pure similarity so we never return nothing when a
    vaguely-similar prior run exists.
  - "implement": the query is the task description; pure semantic similarity.

The feature is opt-in via ``AGENT_RAG_EPISODES`` (default off) so the default
prompt is byte-identical to before.  When disabled, or when the corpus has no
successful episodes, ``format_episodic_notes`` returns ``""``.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional

from agent_core.memory.types import VectorDatabase

from .embed import DEFAULT_DIM, Embedder, build_embedder
from .episodes import Episode, clear_episode_cache, successful_episodes

_KW_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FILE_RE = re.compile(r"([\w./\\-]+?)\.py\b")
_ERROR_RE = re.compile(
    r"(ImportError|ModuleNotFoundError|NameError|AttributeError|TypeError|"
    r"ValueError|KeyError|IndexError|SyntaxError|FileNotFoundError|"
    r"PermissionError|RuntimeError|AssertionError|ZeroDivisionError|"
    r"RecursionError|UnicodeDecodeError|OverflowError|NotImplementedError)"
)


def _enabled() -> bool:
    val = os.environ.get("AGENT_RAG_EPISODES", "")
    return val.strip().lower() in ("1", "true", "yes", "on")


def _embedder() -> Embedder:
    mode = os.environ.get("AGENT_RAG_EMBEDDER") or None
    return build_embedder(mode)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _KW_RE.findall(text or "")}


def _query_signals(query: str) -> tuple[set[str], set[str]]:
    """Extract (file_stems, error_classes) from a free-text query.

    File stems are pulled from any ``*.py`` reference (``foo.py`` or
    ``agent_core/foo.py`` -> ``foo``); error classes from the known exception
    vocabulary.  These give the high-precision filter that the path-keyed
    history layer already exploits.
    """
    stems: set[str] = set()
    for m in _FILE_RE.findall(query or ""):
        base = m.replace("\\", "/").rsplit("/", 1)[-1]
        if "." in base:
            stem = base.rsplit(".", 1)[0]
        else:
            stem = base
        if stem:
            stems.add(stem.lower())
    errors = {e for e in _ERROR_RE.findall(query or "")}
    return stems, errors


class EpisodeIndex:
    """In-memory semantic index over successful episodes (per process)."""

    def __init__(self, episodes: Iterable[Episode], embedder: Optional[Embedder] = None) -> None:
        self._embedder = embedder or _embedder()
        self._episodes: List[Episode] = list(episodes)
        self._db = VectorDatabase(dimension=self._embedder.dim)
        self._dim = self._embedder.dim
        for i, ep in enumerate(self._episodes):
            vec = self._embedder.from_string(ep.index_text)
            self._db.add_vector(vec, {"episode_index": i})

    @classmethod
    def for_workspace(cls, workspace: str, embedder: Optional[Embedder] = None) -> "EpisodeIndex":
        eps = successful_episodes(workspace)
        return cls(eps, embedder=embedder)

    def retrieve(self, query: str, mode: str = "implement", k: int = 3) -> List[Episode]:
        if not self._episodes:
            return []
        q_vec = self._embedder.from_string(query)
        hits = self._db.search_similar(q_vec, k=len(self._episodes))
        scored: List[tuple[float, Episode]] = []
        for h in hits:
            ep = self._episodes[h["metadata"]["episode_index"]]
            scored.append((float(h["similarity_score"]), ep))

        if mode == "fix":
            q_stems, q_errors = _query_signals(query)
            if q_stems or q_errors:
                filtered = [
                    (sim, ep)
                    for sim, ep in scored
                    if (q_stems & set(ep.file_stems)) or (q_errors & set(ep.error_classes))
                ]
                if filtered:
                    filtered.sort(key=lambda x: x[0], reverse=True)
                    return [ep for _, ep in filtered[:k]]

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]


_INDEX_CACHE: dict[str, EpisodeIndex] = {}


def get_index(workspace: str, embedder: Optional[Embedder] = None) -> Optional[EpisodeIndex]:
    """Cached per-workspace index (rebuilt when the episode cache is cleared)."""
    key = os.path.abspath(workspace)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    if embedder is not None:
        idx = EpisodeIndex(list(successful_episodes(workspace)), embedder=embedder)
    else:
        idx = EpisodeIndex.for_workspace(workspace)
    _INDEX_CACHE[key] = idx
    return idx


def clear_index_cache() -> None:
    _INDEX_CACHE.clear()
    clear_episode_cache()


def format_episodic_notes(
    query: str,
    mode: str,
    workspace: str,
    k: int = 3,
    embedder: Optional[Embedder] = None,
) -> str:
    """Return a ``## SUCCESSFUL RUN EXAMPLES`` block, or ``""`` when off/empty."""
    if not _enabled():
        return ""
    try:
        idx = get_index(workspace, embedder=embedder)
    except Exception:
        return ""
    if idx is None:
        return ""
    eps = idx.retrieve(query, mode=mode, k=k)
    if not eps:
        return ""
    lines = [f"\n## SUCCESSFUL RUN EXAMPLES (episodic memory, {len(eps)} episode(s))"]
    for ep in eps:
        lines.append(ep.render(cap=300))
    return "\n".join(lines)


__all__ = [
    "EpisodeIndex",
    "get_index",
    "clear_index_cache",
    "format_episodic_notes",
    "DEFAULT_DIM",
]
