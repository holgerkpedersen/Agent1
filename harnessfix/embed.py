"""Dependency-free text embedding for episodic-memory retrieval.

The agent's ``EmbeddingService`` (agent_core/memory/types.py) is a zero-vector
stub, so it cannot drive semantic search.  For episodic RAG over the trace
corpus we need a *real*, deterministic embedder that works with no optional
dependencies installed.

Strategy (default ``HashEmbedder``):
  - tokenise into lowercased word n-grams (1-3) + character 4-grams,
  - hash each feature into a fixed 384-dim space (FNV-1a),
  - accumulate signed counts and L2-normalise.

This is a cheap, reproducible bag-of-features model.  It captures lexical /
near-duplicate similarity well (which is what FIX retrieval needs: matching
error text, file stems, symbol names) and is fine for top-k over a small
corpus.  An optional ``SentenceTransformerEmbedder`` upgrades to a real
semantic model when ``sentence_transformers`` (and the model weights) are
available, selected via ``AGENT_RAG_EMBEDDER=st``.

All embedders implement ``from_string(text) -> np.ndarray`` of shape
``(dim,)`` and are L2-normalised so ``VectorDatabase`` cosine search works.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Protocol

import numpy as np

DEFAULT_DIM = 384

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Embedder(Protocol):
    dim: int

    def from_string(self, text: str) -> np.ndarray: ...


def _fnv1a_32(blob: bytes) -> int:
    h = 0x811C9DC5
    for b in blob:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class HashEmbedder:
    """Deterministic, dependency-free bag-of-features embedder (384-dim)."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim

    def _features(self, text: str) -> List[str]:
        if not text:
            return []
        low = text.lower()
        feats: List[str] = []
        words = _WORD_RE.findall(low)
        # word unigrams + bigrams + trigrams
        for w in words:
            feats.append("w:" + w)
        for i in range(len(words) - 1):
            feats.append("w2:" + words[i] + " " + words[i + 1])
        for i in range(len(words) - 2):
            feats.append("w3:" + words[i] + " " + words[i + 1] + " " + words[i + 2])
        # character 4-grams (captures subword/identifier similarity)
        letters = re.sub(r"\s+", " ", low)
        for i in range(len(letters) - 3):
            feats.append("c:" + letters[i : i + 4])
        return feats

    def from_string(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for feat in self._features(text):
            h = _fnv1a_32(feat.encode("utf-8", "ignore"))
            idx = h % self.dim
            # sign from a second hash bit so collisions don't always add
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class SentenceTransformerEmbedder:
    """Optional real semantic embedder (lazy import, opt-in only)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = DEFAULT_DIM) -> None:
        self._model_name = model_name
        self.dim = dim
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                f"AGENT_RAG_EMBEDDER=st requires sentence_transformers "
                f"(and model '{model_name}'): {exc}"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def from_string(self, text: str) -> np.ndarray:
        vec = np.asarray(self._model.encode([text])[0], dtype=np.float64)
        if vec.shape[0] != self.dim:
            # Project / pad to the configured dim so VectorDatabase accepts it.
            out = np.zeros(self.dim, dtype=np.float64)
            n = min(self.dim, vec.shape[0])
            out[:n] = vec[:n]
            vec = out
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def build_embedder(mode: str | None = None) -> Embedder:
    """Return an embedder for the given mode string.

    ``mode`` is normally read from ``AGENT_RAG_EMBEDDER``:
      - "st" / "sentence-transformers" -> SentenceTransformerEmbedder
      - anything else (incl. None)      -> HashEmbedder (default, no deps)
    """
    if mode in ("st", "sentence-transformers", "sentence_transformers"):
        return SentenceTransformerEmbedder()
    return HashEmbedder()


__all__ = [
    "Embedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "build_embedder",
    "DEFAULT_DIM",
]
