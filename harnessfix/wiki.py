"""WikiSkill layer 2 - persistent wiki of consolidated harness knowledge.

WikiSkill (arXiv 2608.27454) separates agent experience into three layers:
immutable execution traces, a persistent wiki (accumulated knowledge), and
executable skills.  Agent1 already captures traces (``harnessfix/tracing``)
and successful episodes (``harnessfix/episodes``); this module is the missing
middle layer — a consolidated, deduplicated, mergeable knowledge base that
continuously absorbs lessons from traces and serves them as prompt context.

The wiki has two kinds of pages:

- **failure pages** keyed ``"{layer}:{mechanism}"``: distilled from failed
  trace diagnoses (see :func:`harnessfix.diagnose.diagnose_graph`).  Each page
  consolidates the repair proposal, evidence traces and error classes across
  every run that hit the same failure mode.

- **success pages** keyed ``"success:{file_hash}"``: distilled from successful
  episodes (see :class:`harnessfix.episodes.Episode`).  Each page captures a
  working action pattern for a file stem so later fixes can reuse it.

Pages are persisted as JSONL under ``reports/wiki/`` (``wiki.jsonl``).  Writes
are append-only per consolidation run; merges happen in memory on load and the
whole file is rewritten atomically, mirroring :mod:`harnessfix.progress`.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .corpus import _is_failed_trace, collect_traces
from .diagnose import diagnose_graph
from .htir import compile_trace
from .episodes import Episode, extract_episode, successful_episodes
from .reader import TraceValidationError
from .tracing import LAYERS

#: Where wiki artifacts live (gitignored — decision #033).
REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO_ROOT / "reports" / "wiki"
WIKI_PATH = WIKI_DIR / "wiki.jsonl"

#: How many top pages to surface in stats / dashboard.
TOP_PAGES_LIMIT = 10


@dataclass
class WikiPage:
    """One consolidated wiki page (a failure mode or a success pattern)."""

    key: str                 # dedup key, e.g. "tool_interface:schema validation rejected tool args"
    title: str              # human-readable short label
    lesson: str             # consolidated knowledge (repair proposal / action pattern)
    evidence: list[str] = field(default_factory=list)  # contributing task_ids
    layer: str = ""          # harness layer facet ("(success)" for success pages)
    error_classes: tuple[str, ...] = ()
    file_stems: tuple[str, ...] = ()
    created_at: float = 0.0
    updated_at: float = 0.0
    hit_count: int = 0       # how many times this page was retrieved


def _file_hash(file_stems: Tuple[str, ...]) -> str:
    """Stable hash of a file-stem tuple for success-page keys."""
    h = hashlib.sha1("|".join(sorted(file_stems)).encode("utf-8")).hexdigest()[:12]
    return h


def load_wiki(path: Path | None = None) -> List[WikiPage]:
    """Load all wiki pages from disk (fail-open on corrupt/missing)."""
    p = path or WIKI_PATH
    if not p.is_file():
        return []
    pages: list[WikiPage] = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                pages.append(_dict_to_page(obj))
    except OSError:
        return []
    return pages


def _dict_to_page(obj: Dict[str, Any]) -> WikiPage:
    """Reconstruct a :class:`WikiPage` from its JSON dict (schema-tolerant)."""
    evidence = list(obj.get("evidence", []) or [])
    error_classes = tuple(obj.get("error_classes", []) or ())
    file_stems = tuple(obj.get("file_stems", []) or ())
    return WikiPage(
        key=str(obj.get("key", "")),
        title=str(obj.get("title", obj.get("key", ""))),
        lesson=str(obj.get("lesson", "")),
        evidence=evidence,
        layer=str(obj.get("layer", "")),
        error_classes=error_classes,
        file_stems=file_stems,
        created_at=float(obj.get("created_at", 0.0) or 0.0),
        updated_at=float(obj.get("updated_at", 0.0) or 0.0),
        hit_count=int(obj.get("hit_count", 0) or 0),
    )


def save_wiki(pages: List[WikiPage], path: Path | None = None) -> None:
    """Persist wiki pages atomically (temp + os.replace)."""
    p = path or WIKI_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    lines = [json.dumps(asdict(pg), ensure_ascii=False) for pg in pages]
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(tmp, p)


def _merge_page(pages: List[WikiPage], new: WikiPage) -> None:
    """Merge *new* into an existing page by key (in-place list mutation).

    Evidence task_ids are unioned; error_classes and file_stems are merged;
    hit_count is preserved from the existing page.  ``lesson`` is replaced
    with the new one only if it is non-empty and differs — we keep the first
    substantive lesson as the canonical distillation.
    """
    for i, pg in enumerate(pages):
        if pg.key == new.key:
            merged_ev = list(dict.fromkeys(pg.evidence + new.evidence))
            merged_ec = tuple(dict.fromkeys(list(pg.error_classes) + list(new.error_classes)))
            merged_fs = tuple(dict.fromkeys(list(pg.file_stems) + list(new.file_stems)))
            if new.lesson and (not pg.lesson or new.lesson != pg.lesson):
                # Keep the most recent non-empty lesson.
                pg.lesson = new.lesson
            pg.evidence = merged_ev[:50]  # cap evidence growth
            pg.error_classes = merged_ec
            pg.file_stems = merged_fs
            pg.updated_at = max(pg.updated_at, new.updated_at)
            return
    pages.append(new)


def _page_from_failed_trace(graph: Any, diag: Any) -> WikiPage:
    """Build a failure wiki page from a diagnosed failed trace."""
    key = f"{diag.root_layer}:{diag.mechanism}"
    title = f"[{diag.root_layer}] {diag.mechanism}"
    lesson = diag.repair_proposal or ""
    # Gather error classes + file stems from the graph for retrieval signals.
    ecs: list[str] = []
    fs: list[str] = []
    text_blobs: list[str] = []
    for s in getattr(graph, "steps", []) or []:
        payload = getattr(s, "payload", {}) or {}
        if isinstance(payload, dict):
            msg = str(payload.get("message", "") + payload.get("exception", ""))
            ecs.extend(_extract_error_classes(msg))
            text_blobs.append(msg)
            text_blobs.append(str(payload.get("result", "")))
            for f in _paths_from_payload(payload):
                fs.append(f)
    # Also scan all free-text fields (messages, results) for file references —
    # a trace message like "boom in foo.py" carries the failing file even when
    # no explicit path key was set.  This mirrors how episodes extract stems
    # from tool_result affected_files + args paths.
    combined_text = "\n".join(text_blobs)
    fs.extend(_stems_from(_FILE_RE.findall(combined_text)))
    now = time.time()
    return WikiPage(
        key=key,
        title=title,
        lesson=lesson,
        evidence=[graph.task_id],
        layer=diag.root_layer,
        error_classes=tuple(dict.fromkeys(ecs)),
        file_stems=tuple(_stems_from(fs)),
        created_at=now,
        updated_at=now,
    )


def _page_from_success_episode(ep: Episode) -> WikiPage:
    """Build a success wiki page from an episode."""
    key = f"success:{_file_hash(ep.file_stems)}" if ep.file_stems else "success:no_files"
    title = f"[success] {', '.join(ep.file_stems[:3]) or 'no files'}"
    lesson = ep.actions_summary or ""
    now = time.time()
    return WikiPage(
        key=key,
        title=title,
        lesson=lesson,
        evidence=[ep.task_id],
        layer="(success)",
        error_classes=(),
        file_stems=ep.file_stems,
        created_at=now,
        updated_at=now,
    )


def consolidate(
    trace_dir: Path | str,
    output_dir: Path | str | None = None,
) -> List[WikiPage]:
    """Absorb traces under *trace_dir* into the wiki (the WikiSkill step).

    Compiles each trace; failed traces become failure pages (via diagnosis),
    successful traces become success pages (via episode extraction).  Pages are
    merged by key and persisted to ``reports/wiki/wiki.jsonl`` (or a custom
    output dir for tests).  Returns the full page list after consolidation.

    Fail-open: any single trace that cannot compile or diagnose is skipped —
    wiki failure never halts the loop.
    """
    tdir = Path(trace_dir)
    odir = Path(output_dir) if output_dir else WIKI_DIR
    odir.mkdir(parents=True, exist_ok=True)
    wpath = odir / "wiki.jsonl"

    pages: List[WikiPage] = load_wiki(wpath)

    traces = collect_traces(tdir)
    for path in sorted(traces):
        try:
            graph = compile_trace(path)
        except TraceValidationError:
            continue
        if _is_failed_trace(graph):
            try:
                diag = diagnose_graph(graph)
            except Exception:
                continue
            new_page = _page_from_failed_trace(graph, diag)
            _merge_page(pages, new_page)
        else:
            ep = extract_episode(graph)
            if ep is not None and (ep.file_stems or ep.actions_summary):
                new_page = _page_from_success_episode(ep)
                _merge_page(pages, new_page)

    save_wiki(pages, wpath)
    return pages


def retrieve(
    query: str,
    k: int = 5,
    path: Path | None = None,
) -> List[WikiPage]:
    """Retrieve top-k wiki pages relevant to *query*.

    Two-stage retrieval (mirrors :mod:`harnessfix.retrieval` EpisodeIndex):

    1. **Fix-mode filter**: extract error classes + file stems from the query;
       rank failure pages whose ``error_classes`` or ``file_stems`` overlap,
       then success pages whose ``file_stems`` overlap.  This is high-precision.
    2. **Fallback similarity**: if no page matches the signals, fall back to a
       token-overlap score against the lesson text so retrieval still returns
       something for vague queries.

    Retrieved pages have their ``hit_count`` incremented (persisted on next
    save).  Returns at most *k* pages ordered by relevance.
    """
    pages = load_wiki(path)
    if not pages:
        return []

    stems, errors = _query_signals(query)
    scored: list[tuple[float, WikiPage]] = []

    for pg in pages:
        score = 0.0
        # Error-class overlap (strong signal for failure pages).
        ec_overlap = len(set(pg.error_classes) & errors) if errors else 0
        fs_overlap = len(set(s.lower() for s in pg.file_stems) & stems) if stems else 0
        if ec_overlap:
            score += 3.0 * ec_overlap
        if fs_overlap:
            score += 2.0 * fs_overlap
        # Token overlap on lesson text (weak signal, always available).
        tok_score = _token_overlap(query, pg.lesson)
        score += 0.5 * tok_score

        if score > 0:
            scored.append((score, pg))

    if not scored and pages:
        # Fallback: rank all pages by lesson token overlap alone.
        for pg in pages:
            tok = _token_overlap(query, pg.lesson)
            if tok > 0:
                scored.append((tok * 0.5, pg))

    scored.sort(key=lambda x: (-x[0], -x[1].hit_count, x[1].key))
    result = [pg for _, pg in scored[:k]]

    # Increment hit counts and persist (best-effort).
    if result:
        _increment_hits(pages, result)
        try:
            save_wiki(pages, path or WIKI_PATH)
        except OSError:
            pass  # hit-count persistence is best-effort; never fail retrieval.

    return result


def _increment_hits(all_pages: List[WikiPage], retrieved: List[WikiPage]) -> None:
    """Increment ``hit_count`` on the pages that were just retrieved."""
    for pg in all_pages:
        if any(pg.key == r.key for r in retrieved):
            pg.hit_count += 1


def _query_signals(query: str) -> Tuple[set[str], set[str]]:
    """Extract (file_stems, error_classes) from a free-text query.

    Reuses the same regex vocabulary as :mod:`harnessfix.retrieval` so wiki
    retrieval and episodic retrieval share signal extraction semantics.
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


def format_wiki_notes(
    query: str,
    k: int = 3,
    path: Path | None = None,
) -> str:
    """Render a ``## COMPILED KNOWLEDGE (wiki)`` block for prompt injection.

    Returns an empty string when the wiki is absent or no pages match — the
    same contract as :func:`harnessfix.retrieval.format_episodic_notes` so
    callers can append unconditionally without guarding for emptiness.  The
    block is purely additive context and never affects write/cascade invariants.
    """
    pages = retrieve(query, k=k, path=path)
    if not pages:
        return ""

    lines = [f"\n## COMPILED KNOWLEDGE (wiki) — {len(pages)} page(s)"]
    for pg in pages:
        ev = ", ".join(e[:8] for e in pg.evidence[:3]) or "—"
        fs = ", ".join(pg.file_stems[:3]) if pg.file_stems else ""
        ec = ", ".join(pg.error_classes[:3]) if pg.error_classes else ""
        meta_parts: list[str] = []
        if pg.layer and pg.layer != "(success)":
            meta_parts.append(f"layer={pg.layer}")
        if fs:
            meta_parts.append(f"files=[{fs}]")
        if ec:
            meta_parts.append(f"errors=[{ec}]")
        lines.append(
            f"- [{pg.key[:48]}] hits={pg.hit_count} evidence=[{ev}] "
            + (" ".join(meta_parts) if meta_parts else "")
            + "\n  lesson: " + pg.lesson.replace("\n", " ")[:300]
        )
    return "\n".join(lines)


def wiki_stats(path: Path | None = None) -> Dict[str, Any]:
    """Aggregate stats for the dashboard / CLI.

    Returns a dict with page_count, total_evidence, coverage (fraction of
    diagnosed layers that have at least one failure page), avg_hit_count,
    top_pages (by hit count), and last_consolidated timestamp.  Fail-open:
    returns zeros when the wiki is absent/corrupt.
    """
    pages = load_wiki(path)
    if not pages:
        return {
            "page_count": 0,
            "total_evidence": 0,
            "coverage": 0.0,
            "avg_hit_count": 0.0,
            "top_pages": [],
            "last_consolidated": None,
        }

    total_ev = sum(len(pg.evidence) for pg in pages)
    failure_layers = {pg.layer for pg in pages if pg.layer and pg.layer != "(success)"}
    coverage = len(failure_layers & set(LAYERS)) / max(1, len(LAYERS))
    avg_hits = total_ev > 0 and (sum(pg.hit_count for pg in pages) / len(pages)) or 0.0

    top = sorted(pages, key=lambda p: (-p.hit_count, -len(p.evidence), p.key))[:TOP_PAGES_LIMIT]
    top_pages = [
        {
            "key": pg.key,
            "title": pg.title,
            "hits": pg.hit_count,
            "evidence": len(pg.evidence),
            "layer": pg.layer,
            "lesson_preview": pg.lesson[:80],
        }
        for pg in top
    ]

    last_ts = max((pg.updated_at for pg in pages if pg.updated_at > 0), default=None)

    return {
        "page_count": len(pages),
        "total_evidence": total_ev,
        "coverage": round(coverage * 100, 1),
        "avg_hit_count": round(avg_hits, 2),
        "top_pages": top_pages,
        "last_consolidated": last_ts,
    }


def clear_wiki_cache() -> None:
    """No-op retained for API symmetry with :mod:`harnessfix.retrieval`.

    Wiki pages are loaded fresh from disk on every call (they may change
    between iterations); there is no process-level cache to clear.
    """
    pass


# --- internal helpers -------------------------------------------------------

import re as _re

_FILE_RE = _re.compile(r"[\w./\\\-]+\.\w+")
_ERROR_RE = _re.compile(
    r"(ImportError|ModuleNotFoundError|NameError|AttributeError|TypeError|"
    r"ValueError|KeyError|IndexError|SyntaxError|FileNotFoundError|"
    r"PermissionError|RuntimeError|AssertionError|ZeroDivisionError|"
    r"RecursionError|UnicodeDecodeError|OverflowError|NotImplementedError)"
)


def _extract_error_classes(text: str) -> list[str]:
    if not text:
        return []
    found = _ERROR_RE.findall(text)
    out: list[str] = []
    for e in found:
        if e not in out:
            out.append(e)
    return out


def _paths_from_payload(payload: dict[str, Any]) -> list[str]:
    """Extract file paths from a trace event payload (same keys as episodes)."""
    out: list[str] = []
    for key in ("path", "file", "filename", "target", "fpath"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value)
        elif isinstance(value, list):
            out.extend(v for v in value if isinstance(v, str) and v.strip())
    return out


def _stems_from(paths: list[str]) -> list[str]:
    stems: list[str] = []
    for p in paths:
        name = os.path.basename(str(p).replace("\\", "/").lower())
        if "." in name:
            name = name.rsplit(".", 1)[0]
        if name and name not in stems:
            stems.append(name)
    return stems


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two strings (for fallback retrieval)."""
    ta = set(t.lower() for t in a.split())
    tb = set(t.lower() for t in b.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


__all__ = [
    "WikiPage",
    "WIKI_DIR",
    "WIKI_PATH",
    "load_wiki",
    "save_wiki",
    "consolidate",
    "retrieve",
    "format_wiki_notes",
    "wiki_stats",
    "clear_wiki_cache",
]
