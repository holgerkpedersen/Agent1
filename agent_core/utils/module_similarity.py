"""Semantic (geometric) module-duplication detection.

Precision-first: planned modules are compared against existing modules using
TF-IDF cosine similarity over docstrings and task descriptions, complementing
the name-token gates in :mod:`agent_core.patterns`.  An optional LM Studio
embeddings backend can be enabled via ``AGENT_EMBEDDING_MODEL``; it is only
used when the probe succeeds, and never silently replaces TF-IDF — the gate
reports exactly which signal fired and with what evidence.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import cast

import numpy as np

from agent_core.patterns import _GENERIC_NAME_TOKENS

#: Minimum top-1 cosine for TF-IDF before a pair is considered a duplicate.
TFIDF_THRESHOLD = float(os.environ.get("AGENT_DUPLICATE_TFIDF_THRESHOLD", "0.55"))
#: Minimum top-1 cosine for the embeddings backend.
EMBED_THRESHOLD = float(os.environ.get("AGENT_DUPLICATE_EMBED_THRESHOLD", "0.85"))
#: Required gap between the top-1 match and the runner-up — a duplicate is only
#: claimed when the match is unambiguous (precision over recall).  Zero: with
#: self-exclusion and a production-only corpus, the threshold is the guard and
#: near-ties between equally-related modules (path_utils exists twice) must not
#: suppress a true positive.
MARGIN = 0.0

#: Paths that must never appear as "existing modules" (test/benchmark noise).
_NON_PRODUCTION_PREFIXES = ("tests/", "benchmarks/", "performance_dashboard/")


def _is_production_module(rel: str) -> bool:
    """True for non-test, non-benchmark modules (precision: tests pollute the
    similarity corpus with near-duplicate docstrings)."""
    if rel.startswith(_NON_PRODUCTION_PREFIXES):
        return False
    name = rel.rsplit("/", 1)[-1]
    if name.startswith("test_") or name.endswith("_test.py"):
        return False
    return True


def _walk_production_py(ws: str) -> list[tuple[str, str]]:
    """(rel_path, abs_path) pairs for every production .py file."""
    out: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in (
            ".git", "__pycache__", ".pytest_cache", "backups", ".docs",
        )]
        for f in files:
            if not f.endswith(".py") or f == "__init__.py":
                continue
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, ws).replace("\\", "/")
            if _is_production_module(rel):
                out.append((rel, fp))
    return out


#: Corpus cache keyed by workspace; valid while the max mtime of the corpus
#: files is unchanged — deterministic reuse, identical precision.
_CORPUS_CACHE: dict[str, tuple[float, "_Corpus"]] = {}

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
    "as", "by", "at", "from", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "their", "there", "here",
    "which", "who", "whom", "what", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "will", "would", "can", "could", "should", "may", "might", "must", "shall",
    "do", "does", "did", "has", "have", "had", "into", "over", "under",
    "between", "through", "during", "before", "after", "above", "below",
    "use", "used", "using", "provide", "provides", "make", "makes", "make",
    "via", "per", "etc", "e.g", "i.e", "one", "two", "new", "file", "files",
    "code", "module", "modules", "class", "function", "return", "returns",
})

_PY_KEYWORDS = frozenset({
    "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
    "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
    "raise", "return", "try", "while", "with", "yield", "true", "false", "none",
})

_GENERIC_TOKENS = _GENERIC_NAME_TOKENS | _STOPWORDS | _PY_KEYWORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DOCSTRING_RE = re.compile(r'^\s*(?:"""|\'\'\')\s*(.*?)\s*(?:"""|\'\'\')?\s*$')


def _meaningful_tokens(text: str) -> list[str]:
    """Lowercased, stopword/generic-filtered tokens for TF-IDF features."""
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 3]
    return [t for t in tokens if t not in _GENERIC_TOKENS]


def _first_docstring(path: str) -> str:
    """First module docstring line (up to 200 chars), or empty string."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    m = _DOCSTRING_RE.match(stripped)
                    if m and m.group(1):
                        return m.group(1)[:200]
                    return ""
                if stripped.startswith("#"):
                    continue
                break
    except OSError:
        pass
    return ""


@dataclass
class PlannedModule:
    """A module the task plan wants to CREATE (no content exists yet)."""

    path: str
    description: str = ""


@dataclass
class SimilarityFinding:
    """A semantic duplicate claim: planned module vs existing module."""

    file: str
    existing: str
    score: float
    kind: str  # "tfidf" | "embedding"
    evidence: str


class _Corpus:
    """TF-IDF index over existing modules (deterministic, numpy-based)."""

    def __init__(self, docs: list[tuple[str, str]]) -> None:
        # docs: (rel_path, feature_text)
        self.paths = [p for p, _ in docs]
        self.tokenized = [_meaningful_tokens(t) for _, t in docs]
        vocab: set[str] = set()
        for toks in self.tokenized:
            vocab.update(toks)
        self.terms: list[str] = sorted(vocab)
        self.term_to_idx = {t: i for i, t in enumerate(self.terms)}
        self.idf = self._build_idf()
        self.matrix = np.vstack([self._tfidf(t) for t in self.tokenized]) if self.terms else np.zeros((0, 0))

    def _build_idf(self) -> np.ndarray:
        n = len(self.tokenized)
        df = np.zeros(len(self.terms))
        for toks in self.tokenized:
            for t in set(toks):
                if t in self.term_to_idx:
                    df[self.term_to_idx[t]] += 1
        return np.log(1 + n / (1 + df))

    def _tfidf(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.terms))
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total = max(sum(counts.values()), 1)
        for t, c in counts.items():
            idx = self.term_to_idx.get(t)
            if idx is not None:
                vec[idx] = (c / total) * self.idf[idx]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def cosine(self, tokens: list[str]) -> np.ndarray:
        """Cosine similarity of *tokens* against every corpus module."""
        q = self._tfidf(tokens)
        if self.matrix.size == 0 or not q.any():
            return np.zeros(len(self.paths))
        return cast(np.ndarray, self.matrix @ q)

    def top(self, tokens: list[str], k: int = 3) -> list[tuple[str, float]]:
        scores = self.cosine(tokens)
        order = np.argsort(scores)[::-1][:k]
        return [(self.paths[i], float(scores[i])) for i in order if scores[i] > 0]


class _EmbeddingBackend:
    """Optional LM Studio /v1/embeddings client (precision-boost plug-in)."""

    def __init__(self) -> None:
        self.model = os.environ.get("AGENT_EMBEDDING_MODEL", "")
        self.base_url = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
        self._available: bool | None = None

    def _probe(self) -> bool:
        if not self.model:
            self._available = False
            return False
        try:
            req = urllib.request.Request(f"{self.base_url}/models", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            loaded = any(m.get("id") == self.model for m in data.get("data", []))
            self._available = loaded
        except Exception:
            self._available = False
        return self._available

    @property
    def available(self) -> bool:
        if self._available is None:
            return self._probe()
        return self._available

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self.available:
            raise RuntimeError("embedding backend unavailable")
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        vecs = [d["embedding"] for d in data.get("data", [])]
        return np.asarray(vecs, dtype=float)


class ModuleSimilarity:
    """Precision-first semantic duplicate detector for planned modules."""

    def __init__(self, ws: str) -> None:
        self.ws = ws
        files = _walk_production_py(ws)
        stamp = max((os.path.getmtime(fp) for _, fp in files), default=0.0)
        cached = _CORPUS_CACHE.get(ws)
        if cached is not None and cached[0] == stamp:
            self.corpus = cached[1]
        else:
            docs = [
                (rel, f"{rel.rsplit('/', 1)[-1][:-3]} {_first_docstring(fp)}")
                for rel, fp in files
            ]
            self.corpus = _Corpus(docs)
            _CORPUS_CACHE[ws] = (stamp, self.corpus)
        self.embedding = _EmbeddingBackend()
        self.embedding_used = False

    def _planned_feature(self, planned: PlannedModule) -> str:
        stem = planned.path.rsplit("/", 1)[-1][:-3]
        return f"{stem} {planned.description}"

    def _tfidf_findings(self, planned: PlannedModule) -> list[SimilarityFinding]:
        tokens = _meaningful_tokens(self._planned_feature(planned))
        top = self.corpus.top(tokens, k=3)
        findings: list[SimilarityFinding] = []
        if not top:
            return findings
        # Never flag a module against ITSELF (defensive — the gates only
        # process planned files that do not exist yet, but be precise anyway).
        top = [(p, s) for p, s in top if p != planned.path]
        if not top:
            return findings
        best, best_score = top[0]
        second = top[1][1] if len(top) > 1 else 0.0
        if best_score >= TFIDF_THRESHOLD and (best_score - second) >= MARGIN:
            findings.append(SimilarityFinding(
                file=planned.path,
                existing=best,
                score=best_score,
                kind="tfidf",
                evidence=f"TF-IDF top-1 {best_score:.3f} (runner-up {second:.3f}, "
                         f"threshold {TFIDF_THRESHOLD})",
            ))
        return findings

    def _embedding_findings(self, planned: PlannedModule) -> list[SimilarityFinding]:
        if not self.embedding.available:
            return []
        try:
            texts = [self._planned_feature(planned)] + [
                f"{p.rsplit('/', 1)[-1][:-3]} {_first_docstring(os.path.join(self.ws, p))}"
                for p in self.corpus.paths
            ]
            vecs = self.embedding.embed(texts)
        except Exception:
            return []
        query = vecs[0]
        query = query / (np.linalg.norm(query) + 1e-8)
        rest = vecs[1:]
        norms = np.linalg.norm(rest, axis=1) + 1e-8
        scores = (rest / norms[:, None]) @ query
        order = np.argsort(scores)[::-1]
        best_idx = order[0]
        best_score = float(scores[best_idx])
        second = float(scores[order[1]]) if len(order) > 1 else 0.0
        self.embedding_used = True
        findings: list[SimilarityFinding] = []
        if best_score >= EMBED_THRESHOLD and (best_score - second) >= MARGIN:
            findings.append(SimilarityFinding(
                file=planned.path,
                existing=self.corpus.paths[best_idx],
                score=best_score,
                kind="embedding",
                evidence=f"embedding top-1 {best_score:.3f} (runner-up {second:.3f}, "
                         f"threshold {EMBED_THRESHOLD})",
            ))
        return findings

    def find_duplicates(self, planned: list[PlannedModule]) -> list[SimilarityFinding]:
        """All semantic duplicate findings for the planned modules.

        Name-token overlap is NOT re-checked here — the existing gates in
        :mod:`agent_core.patterns` cover that; this layer adds the geometric
        signal (TF-IDF always, embeddings when available).
        """
        findings: list[SimilarityFinding] = []
        for pm in planned:
            findings.extend(self._tfidf_findings(pm))
            findings.extend(self._embedding_findings(pm))
        return findings
