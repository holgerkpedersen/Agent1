"""Decision tracking — record, search, and resolve design decisions.

Stores decisions in ``.decisions.json`` at the workspace root.

Integration points:
- optimize: load decisions before suggesting fixes, check for contradictions
- workflow: auto-extract decision candidates from analysis after pipeline runs
- decide: manual command to record/search/check/resolve decisions
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent import Agent

logger = logging.getLogger(__name__)

_DECISIONS_FILE = ".decisions.json"


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
    p = Path(path)
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


async def extract_from_analysis(
    agent: "Agent",
    analysis_text: str,
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
        "Output ONLY valid JSON array, no other text."
    )
    response = await agent.llm.chat([
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Extract decisions from:\n\n{analysis_text}"},
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
