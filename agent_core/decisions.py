"""Decision tracking — record, search, and resolve design decisions.

Stores decisions in ``.decisions.json`` at the workspace root.

Integration points:
- optimize: load decisions before suggesting fixes, check for contradictions
- workflow: auto-extract decision candidates from analysis after pipeline runs
- decide: manual command to record/search/check/resolve decisions
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent import Agent

logger = logging.getLogger(__name__)

_DECISIONS_FILE = ".decisions.json"

#: Negative-existence phrasings ("no test coverage for X", "untested", ...) —
#: used by :func:`annotate_candidates` to mechanically check coverage claims.
_NEGATIVE_CLAIM_RE = re.compile(
    r"(?i)no\s+(?:real\s+|actual\s+|unit\s+)?(?:tests?|coverage|test\s+coverage)"
    r"|(?:untested|lacks?\s+(?:tests?|coverage|test\s+coverage)|missing\s+(?:tests?|coverage))"
)
_MODULE_REF_RE = re.compile(r"(?<![\w/])([\w./-]+\.py)(?![A-Za-z0-9_])")
_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "backups"}


# ── Storage ────────────────────────────────────────────────────────────────


def _decision_path(workspace: str | Path) -> Path:
    return Path(workspace) / _DECISIONS_FILE


def _canonical_rel(workspace: str | Path, path: str) -> str | None:
    """Workspace-relative canonical form (forward slashes), or None when the
    path does not resolve under the workspace.

    A relative path is interpreted against the WORKSPACE (not the process
    CWD) — the absolute form is always derivable via ``join(workspace, rel)``.
    """
    ws = Path(workspace).resolve()
    # Canonical input: treat backslashes as separators regardless of host OS
    # so decision records compare equal across platforms (on POSIX a raw
    # ``a\b`` would otherwise stay one literal component).
    p = Path(str(path).replace("\\", "/"))
    if not p.is_absolute():
        p = ws / p
    try:
        rel = p.resolve().relative_to(ws)
    except (ValueError, OSError):
        return None
    return rel.as_posix()


def normalize_affected_files(workspace: str | Path, files: list[str] | None) -> list[str]:
    """Canonical workspace-relative forms for *files*.

    - relative entries are resolved against the workspace
    - absolute entries are relativized when under the workspace
    - entries that do not resolve under the workspace (e.g. ``../ReactAgent``
      escapes) or do not exist are dropped
    - doc basenames (e.g. ``project_analysis.md``) fall back to the newest
      ``.docs/<ts>/`` run folder via ``find_doc``
    """
    ws = Path(workspace).resolve()
    out: list[str] = []
    for raw in files or []:
        f = str(raw).strip().replace("\\", "/")
        if not f:
            continue
        rel = _canonical_rel(ws, f)
        if rel is None:
            continue
        if not (ws / rel).exists():
            from agent_core.commands.doc_paths import find_doc
            found = find_doc(str(ws), f.rsplit("/", 1)[-1])
            if found:
                rel = _canonical_rel(ws, found)
            if rel is None or not (ws / rel).exists():
                continue
        if rel not in out:
            out.append(rel)
    return out


def load_decisions(workspace: str | Path) -> list[dict[str, Any]]:
    fp = _decision_path(workspace)
    if not fp.exists():
        return []
    try:
        parsed = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(parsed, list):
            decisions = [d for d in parsed if isinstance(d, dict)]
        else:
            return []
    except (json.JSONDecodeError, OSError):
        return []
    # Normalize legacy affected_files so matching is on ONE canonical form.
    for d in decisions:
        d["affected_files"] = normalize_affected_files(workspace, d.get("affected_files"))
    return decisions


def save_decisions(workspace: str | Path, decisions: list[dict[str, Any]]) -> None:
    _decision_path(workspace).write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def find_stale_decisions(
    workspace: str | Path, decisions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Decisions whose recorded affected_files no longer exist on disk.

    A stale reference means the decision's subject was deleted or renamed —
    a candidate for verification or supersession, not a broken ledger.  The
    decision ledger is a living document; the human gate decides what to do
    with it (decision #054).
    """
    stale: list[dict[str, Any]] = []
    for d in decisions:
        missing = [
            f for f in d.get("affected_files", [])
            if not (Path(str(workspace)) / f).exists()
        ]
        if missing:
            d = dict(d)
            d["_missing_files"] = missing
            stale.append(d)
    return stale


def _next_id(decisions: list[dict[str, Any]]) -> str:
    max_id = 0
    for d in decisions:
        try:
            max_id = max(max_id, int(d.get("id", "0")))
        except (ValueError, TypeError):
            logger.debug("Skipping non-numeric decision id: %r", d.get("id"))
    return str(max_id + 1).zfill(3)


def add_decision(
    workspace: str | Path,
    title: str,
    context: str = "",
    decision: str = "",
    rationale: str = "",
    affected_files: list[str] | None = None,
    tags: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    decisions = load_decisions(workspace)
    record = {
        "id": _next_id(decisions),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": title,
        "context": context,
        "decision": decision,
        "rationale": rationale,
        "affected_files": normalize_affected_files(workspace, affected_files or []),
        "tags": tags or [],
        "contradictions": [],
        "resolved_by": None,
    }
    if warnings:
        record["meta_warnings"] = warnings
    decisions.append(record)
    save_decisions(workspace, decisions)
    return record


# ── Search ──────────────────────────────────────────────────────────────────


def find_decisions(
    workspace: str | Path,
    *,
    tags: list[str] | None = None,
    files: list[str] | None = None,
    keyword: str = "",
) -> list[dict[str, Any]]:
    decisions = load_decisions(workspace)
    result = []
    for d in decisions:
        if tags and not any(t in d.get("tags", []) for t in tags):
            continue
        if files:
            canon = {_canonical_rel(workspace, f) for f in files}
            if not any(c in d.get("affected_files", []) for c in canon if c):
                continue
        if keyword:
            text = json.dumps(d).lower()
            if keyword.lower() not in text:
                continue
        result.append(d)
    return result


# ── Simple overlap check (instant, no LLM) ──────────────────────────────────


def find_overlaps(
    new_decision: dict[str, Any],
    existing: list[dict[str, Any]],
    workspace: str | Path,
) -> list[dict[str, Any]]:
    new_tags = set(new_decision.get("tags", []))
    new_files = {
        c for c in (_canonical_rel(workspace, f) for f in new_decision.get("affected_files", [])) if c
    }
    overlaps = []
    for d in existing:
        old_tags = set(d.get("tags", []))
        old_files = set(d.get("affected_files", []))  # already canonical (add/load normalized)
        tag_overlap = bool(new_tags & old_tags)
        file_overlap = bool(new_files & old_files)
        if tag_overlap or file_overlap:
            overlaps.append(d)
    return overlaps


def format_for_prompt(decisions: list[dict[str, Any]]) -> str:
    if not decisions:
        return ""
    lines = ["## Past Decisions (do NOT contradict these)"]
    for d in decisions:
        body = d.get("decision", "") or d.get("title", "")
        why = d.get("rationale", "")
        files = ", ".join(d.get("affected_files", [])[:5])
        lines.append(
            f"Decision #{d['id']}: {body}\n"
            f"  Rationale: {why}\n"
            f"  Affected files: {files}\n"
        )
    return "\n".join(lines)


# ── LLM-powered contradiction detection ─────────────────────────────────────


async def check_contradictions(
    agent: "Agent",
    decisions: list[dict[str, Any]],
    new_decision_text: str,
) -> str:
    if not decisions:
        return "No existing decisions to check against."
    sys_msg = (
        "You are a design reviewer. Compare the new decision against all past "
        "decisions. Flag ONLY genuine contradictions — different decisions about "
        "different code paths at different times are NOT contradictions.\n\n"
        "If a contradiction exists, list the conflicting decision IDs and explain "
        "the exact nature of the conflict. If no contradiction, say so clearly "
        "and explain why they are compatible."
    )
    past = []
    for d in decisions:
        past.append(
            f"#{d['id']}: {d['title']}\n"
            f"  Decision: {d.get('decision', '')}\n"
            f"  Rationale: {d.get('rationale', '')}\n"
            f"  Files: {', '.join(d.get('affected_files', []))}\n"
            f"  Tags: {', '.join(d.get('tags', []))}"
        )
    user_msg = (
        "## Past decisions\n\n" + "\n\n".join(past) + "\n\n"
        f"## New decision to evaluate\n\n{new_decision_text}\n\n"
        "Does the new decision contradict any past decision? "
        "Be specific — cite decision IDs."
    )
    response = await agent.llm.chat([
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ])
    return response if response else "LLM returned no response."


async def resolve_contradictions(
    agent: "Agent",
    d1: dict[str, Any],
    d2: dict[str, Any],
) -> str:
    sys_msg = (
        "You are a senior architect resolving a design contradiction. "
        "Examine the broader context of both decisions (affected files, "
        "rationale, tags) and propose a resolution that satisfies the "
        "intent of both. If one decision must take priority, explain why."
    )
    user_msg = (
        f"## Decision #{d1['id']}\n"
        f"Title: {d1['title']}\n"
        f"Decision: {d1.get('decision', '')}\n"
        f"Rationale: {d1.get('rationale', '')}\n"
        f"Affected files: {', '.join(d1.get('affected_files', []))}\n"
        f"Tags: {', '.join(d1.get('tags', []))}\n\n"
        f"## Decision #{d2['id']}\n"
        f"Title: {d2['title']}\n"
        f"Decision: {d2.get('decision', '')}\n"
        f"Rationale: {d2.get('rationale', '')}\n"
        f"Affected files: {', '.join(d2.get('affected_files', []))}\n"
        f"Tags: {', '.join(d2.get('tags', []))}\n\n"
        "Analyze both decisions holistically. Propose a resolution."
    )
    response = await agent.llm.chat([
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ])
    return response if response else "LLM returned no response."


# ── Extract decisions from analysis ─────────────────────────────────────────


def _parse_json_array(response: str) -> list[dict[str, Any]]:
    """Best-effort parse of a JSON array from an LLM response."""
    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return [d for d in parsed if isinstance(d, dict)]
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return [d for d in parsed if isinstance(d, dict)]
            except json.JSONDecodeError:
                logger.debug(
                    "Could not parse a JSON array from the LLM response "
                    "(contradiction/detection extraction) — response was: %.200s",
                    response,
                )
    return []


# ── Candidate warning annotation ───────────────────────────────────────────


def _test_files_for(workspace: str | Path, module_path: str) -> list[str]:
    """Test files anywhere under *workspace* whose name mentions *module_path*.

    Root-located ``test_x.py`` files count as tests even outside a ``tests/``
    directory (some repos keep them at the root).
    """
    base = Path(module_path).stem.lower()
    if not base:
        return []
    hits: list[str] = []
    ws = Path(workspace)
    for root, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            low = fn.lower()
            if "test" not in low:
                continue
            if base not in low:
                continue
            rel = os.path.relpath(os.path.join(root, fn), ws).replace("\\", "/")
            hits.append(rel)
    return hits


def annotate_candidates(
    candidates: list[dict[str, Any]],
    workspace: str | Path,
    verification_report: str = "",
) -> list[dict[str, Any]]:
    """Mechanical fact-check of extracted candidates, in place.

    Adds a ``warnings`` key to any candidate whose claims are contradicted or
    unverifiable against the workspace:

    - negative coverage claims ("no tests for X.py") that a matching test file
      contradicts (probed under the workspace)
    - negative existence claims without a file reference (cannot be checked)
    - ``affected_files`` entries that do not exist under the workspace
    - backticked symbols/paths repeated from a verification report of
      unverifiable claims

    Callers should refuse (or require explicit confirmation) to record warned
    candidates: a recorded decision is injected into every future LLM call.
    """
    ws = Path(workspace)
    flagged_tokens: set[str] = set()
    if verification_report:
        for tok in re.findall(r"`([^`]+)`", verification_report):
            tok = tok.strip()
            if len(tok) >= 4 and " " not in tok:
                flagged_tokens.add(tok)

    for c in candidates:
        text = " ".join(filter(None, (
            c.get("title", ""), c.get("context", ""), c.get("decision", ""),
        )))
        warnings: list[str] = []

        if _NEGATIVE_CLAIM_RE.search(text):
            refs = _MODULE_REF_RE.findall(text)
            if not refs:
                warnings.append(
                    "Negative existence claim without a file reference — "
                    "cannot be verified against the workspace."
                )
            for ref in refs:
                hits = _test_files_for(ws, ref)
                if hits:
                    warnings.append(
                        f"Contradicted by workspace: tests found for {ref} — "
                        f"{', '.join(hits[:3])}"
                    )

        for f in c.get("affected_files", []):
            if not (ws / f).exists():
                warnings.append(f"Affected file does not exist in workspace: {f}")

        for tok in sorted(flagged_tokens):
            if tok in text:
                warnings.append(f"Repeats unverified claim reference: {tok}")

        if warnings:
            seen: set[str] = set()
            deduped: list[str] = []
            for w in warnings:
                if w not in seen:
                    seen.add(w)
                    deduped.append(w)
            c["warnings"] = deduped

    return candidates


async def extract_from_analysis(
    agent: "Agent",
    analysis_text: str,
    *,
    inventory: str = "",
    verification_report: str = "",
) -> list[dict[str, Any]]:
    sys_msg = (
        "Extract design decisions from the project analysis.\n"
        "Return a JSON array of decision candidates. Each object must have:\n"
        '  "title": short description of the decision\n'
        '  "context": why this decision is needed (from the analysis text)\n'
        '  "decision": what was decided or needs to be decided\n'
        '  "rationale": why this is the right choice\n'
        '  "affected_files": list of file paths mentioned\n'
        '  "tags": list of keywords (e.g. "security", "import", "architecture")\n\n'
        "Accuracy rules — the analysis may contain UNVERIFIED claims:\n"
        "- Only repeat facts that are consistent with the workspace listing.\n"
        "  Cite file paths/symbols exactly as they appear in the listing.\n"
        "- NEVER state that something does not exist (no tests, no coverage,\n"
        "  no docs, missing module, etc.) unless the workspace listing shows\n"
        "  no such file. Absence in the listing is insufficient for an\n"
        "  existence claim — phrase it as a question instead.\n"
        "- Do NOT base candidates on claims listed in the verification report.\n"
        "Output ONLY valid JSON array, no other text."
    )
    user_msg = f"Extract decisions from:\n\n{analysis_text}"
    if inventory:
        user_msg += (
            "\n\n## Workspace listing (ground truth — only these modules exist):\n"
            f"{inventory}\n"
        )
    if verification_report:
        user_msg += (
            "\n\n## Verification report (claims in this report may be fabricated — "
            "do NOT trust or repeat them):\n"
            f"{verification_report}\n"
        )
    response = await agent.llm.chat([
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ], disable_thinking=True)
    return _parse_json_array(response)


async def extract_from_changes(
    agent: "Agent",
    files: list[str],
    context: str = "",
) -> list[dict[str, Any]]:
    """Extract design decisions from a set of file changes.

    Args:
        agent: Agent instance for LLM calls.
        files: List of relative file paths that were created/modified.
        context: Description of what was done (task/issue description).
    """
    sys_msg = (
        "Extract deliberate design decisions from these file changes.\n"
        "Focus on tradeoffs, architectural choices, API designs, naming\n"
        "conventions, patterns adopted, patterns rejected, and security\n"
        "decisions. NOT every change — only choices where a clear\n"
        "alternative existed and was either chosen or rejected.\n\n"
        "Return a JSON array. Each object must have:\n"
        '  "title": short description of the design decision\n'
        '  "context": what was being implemented and what alternatives existed\n'
        '  "decision": what was chosen\n'
        '  "rationale": why this choice was made\n'
        '  "affected_files": list of file paths affected\n'
        '  "tags": list of keywords\n\n'
        "Output ONLY valid JSON array, no other text."
    )
    user_msg = (
        "## Changed files\n" + "\n".join(f"- {f}" for f in files) + "\n"
    )
    if context:
        user_msg += f"\n## Task context\n{context}\n"
    user_msg += "\nExtract the design decisions embodied in these changes."

    response = await agent.llm.chat([
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ], disable_thinking=True)
    result = _parse_json_array(response)
    ws = str(Path(agent.workspace).resolve())
    return _filter_candidates(result, ws)


def decisions_as_system_prompt(workspace: str | Path, files: list[str]) -> str:
    """Build a constraints block for LLM prompts from past decisions.

    Returns empty string if no relevant decisions exist.
    """
    decisions = find_decisions(workspace=str(workspace), files=files) if files else load_decisions(workspace)
    if not decisions:
        return ""
    lines = [
        "\n\nCRITICAL DESIGN CONSTRAINTS — past decisions that MUST be respected:",
        "These are design choices already documented for this codebase.",
        "Your implementation MUST NOT contradict any of these decisions.",
        ""
    ]
    for d in decisions:
        warnings = d.get("meta_warnings")
        if warnings:
            lines.append(
                f"  Decision #{d['id']} ({d['title']}) — ⚠ RECORDED WITH UNVERIFIED "
                f"CLAIMS ({'; '.join(warnings)}):\n"
                f"    Chose: {d.get('decision', '')}\n"
                f"    Why: {d.get('rationale', '')}\n"
            )
        else:
            lines.append(
                f"  Decision #{d['id']} ({d['title']}):\n"
                f"    Chose: {d.get('decision', '')}\n"
                f"    Why: {d.get('rationale', '')}\n"
            )
    return "\n".join(lines)


def _filter_candidates(
    candidates: list[dict[str, Any]],
    workspace: str | Path,
) -> list[dict[str, Any]]:
    """Remove candidates that duplicate existing decisions or are already done."""
    existing = load_decisions(workspace)
    existing_titles_lower = {d.get("title", "").lower() for d in existing}
    existing_files = set()
    for d in existing:
        existing_files.update(d.get("affected_files", []))

    filtered = []
    for c in candidates:
        title_lower = c.get("title", "").lower()
        tags = [t.lower() for t in c.get("tags", [])]
        files = c.get("affected_files", [])

        # Skip meta-decisions about the decision system itself
        if any(t in tags for t in ("decision-tracking", "decide")):
            continue
        if any(t in ("meta", "tooling") for t in tags) and any(
            f in ["agent_core/decisions.py", "agent_core/commands/decide_cmd.py",
                  ".decisions.json"] for f in files
        ):
            continue

        # Skip if title is too similar to existing decision
        if title_lower in existing_titles_lower:
            continue
        if any(title_lower in et for et in existing_titles_lower):
            continue
        if any(et in title_lower for et in existing_titles_lower if len(et) > 15):
            continue

        # Skip if all affected files already exist AND are in existing decisions
        ws_path = Path(workspace) if isinstance(workspace, str) else workspace
        all_exist = files and all(
            (ws_path / f).exists() for f in files
        ) if files else False
        if all_exist and any(f in existing_files for f in files):
            continue

        filtered.append(c)
    return filtered
