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

#: Append-only JSONL ledger of autonomous-decision discrepancies (Phase 2 of
#: the autonomous-decision plan).  Kept beside ``.decisions.json`` so a review
#: can pair every flagged action with the decision it diverged from.
DISCREPANCIES_FILE = ".discrepancies.jsonl"

#: Category for decisions the agent took on its own during execution (Phase 1).
AUTONOMOUS_CATEGORY = "autonomous"

#: Tag applied to every recorded autonomous decision.
AUTONOMOUS_TAG = "autonomous"

#: Marker that a ledger decision came from execution rather than a human.
SOURCE_AUTONOMOUS = "agent"

#: A file path has at most this many name components, and each component is at
#: most this long (255 is the on-disk limit on every filesystem in use).  Used
#: by :func:`looks_like_path` to reject prose.
_MAX_PATH_PARTS = 8
_MAX_PATH_PART_LEN = 255

#: Characters that never appear inside a real path token: shell operators,
#: quotes, brackets, globbing and punctuation tails.  A candidate carrying one
#: of these is prose (``(idempotent),``) or a command, not a file.
_NON_PATH_CHARS = frozenset("&|;<>$`'\"()[]{}*?!^~,%#")

#: Words that read exactly like a path and are never one: the words shell
#: operators are described with in prose (``git status && echo done`` -> the
#: segment ``echo done``).  A real identifier is not an English word, so
#: rejecting this list costs nothing.
_PROSE_PATH_WORDS = frozenset(
    {
        "and",
        "then",
        "done",
        "ok",
        "the",
        "a",
        "an",
        "to",
        "for",
        "with",
        "once",
        "again",
        "cards",
        "test",
        "tests",
        "true",
        "false",
        "none",
        "null",
        "yes",
        "no",
    }
)

#: Append-only JSONL log of *observed* autonomous tool calls (Phase 1).
#: One line per detected call; :func:`record_autonomous_decisions` turns the
#: findings into ledger entries and records which ones it stored.  The log is
#: the source of truth for "was this already reviewed?", so promotion is
#: idempotent even if the ledger is rebuilt from scratch.
AUTONOMOUS_LOG_FILE = ".autonomous_decisions.jsonl"

#: Sequence rating: tool params that carry a work-target path.  Shared with
#: :mod:`agent_core.discrepancy` (which imports it) so "where does this call
#: point" has one definition.
_PATH_ARG_KEYS = (
    "path",
    "file",
    "filepath",
    "filename",
    "target",
    "files",
    "filenames",
    "source",
    "destination",
)

#: A path-like token in free prose (``agent_core/security/``).
_PATH_TOKEN_RE = re.compile(r"(?<![\w/`])((?:[\w.-]+/)+[\w.-]*)")

#: How many recent autonomous decisions are re-stated in a prompt block.
_RECENT_AUTONOMOUS_LIMIT = 5

#: ``Deleted <path>`` / ``Removed <path>`` in a tool result — the only evidence
#: a shell action leaves about WHICH file it destroyed.
_DELETED_PATH_RE = re.compile(r"(?i)\b(?:deleted|removed)\s+(\S+)")

#: ``git commit`` / ``git merge`` (possibly preceded by ``-C <path>`` or
#: global options) — repository state changes worth remembering.
_GIT_DECISION_RE = re.compile(
    r"(?<![\w-])git(?:\s+-\S+(?:\s+\S+)?)*\s+(commit|merge)\b"
)

# ── Status constants ────────────────────────────────────────────────────────

STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"
STATUS_ARCHIVED = "archived"
#: A reserved-but-never-written ID (a tombstone, see decision #043).  It is a
#: legitimate status: omitting it here made :func:`load_decisions` treat the
#: record as "unknown" and silently rewrite a tombstone into an ``active``
#: decision on the next save.
STATUS_VOID = "void"
_STATUSES = {STATUS_ACTIVE, STATUS_SUPERSEDED, STATUS_ARCHIVED, STATUS_VOID}

# ── Category set (curated from tag inventory) ──────────────────────────────

CATEGORIES = {
    "security",
    "architecture",
    "autonomous",
    "testing",
    "tracing",
    "workflow",
    "quality",
    "provider",
    "concurrency",
    "documentation",
    "planning",
    "git",
    "ops",
    "ux",
}

#: Negative-existence phrasings ("no test coverage for X", "untested", ...) —
#: used by :func:`annotate_candidates` to mechanically check coverage claims.
_NEGATIVE_CLAIM_RE = re.compile(
    r"(?i)no\s+(?:real\s+|actual\s+|unit\s+)?(?:tests?|coverage|test\s+coverage)"
    r"|(?:untested|lacks?\s+(?:tests?|coverage|test\s+coverage)|missing\s+(?:tests?|coverage))"
)
_MODULE_REF_RE = re.compile(r"(?<![\w/])([\w./-]+\.py)(?![A-Za-z0-9_])")
_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "backups",
    ".sessions",
    ".worktrees",
}


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


def normalize_affected_files(
    workspace: str | Path, files: list[str] | None
) -> list[str]:
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
    # Also normalize tags: stored records may carry null/non-list values,
    # which would otherwise crash consumers that slice them (e.g. decide list).
    # And date: some records store the timestamp under ``created_at`` instead
    # of ``date``; unify on ``date`` so list/show never KeyError on it.
    for d in decisions:
        d["affected_files"] = normalize_affected_files(
            workspace, d.get("affected_files")
        )
        tags = d.get("tags")
        d["tags"] = tags if isinstance(tags, list) else []
        if not d.get("date"):
            d["date"] = d.get("created_at") or ""
        # Normalize status: unknown or missing → active
        status = d.get("status", "")
        d["status"] = status if status in _STATUSES else STATUS_ACTIVE
        # Ensure category key exists for legacy records
        d.setdefault("category", "")
    return decisions


def save_decisions(workspace: str | Path, decisions: list[dict[str, Any]]) -> None:
    _decision_path(workspace).write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Discrepancy ledger (autonomous-decision plan, Phase 2) ─────────────────


def _discrepancies_path(workspace: str | Path) -> Path:
    return Path(workspace) / DISCREPANCIES_FILE


def record_discrepancy(
    workspace: str | Path, discrepancy: dict[str, Any] | Any
) -> None:
    """Append one discrepancy to :data:`DISCREPANCIES_FILE`.

    *discrepancy* is a :class:`agent_core.discrepancy.Discrepancy` (any object
    with ``__dict__``) or an already-built dict.  ``detected_at`` is stamped
    here when the record does not carry one, so ordering survives a review.

    A record is one line, so a partial write can never invalidate earlier
    entries.  I/O failures are logged and swallowed: bookkeeping must never
    break the tool call that triggered it.
    """
    record = (
        dict(discrepancy)
        if isinstance(discrepancy, dict)
        else dict(vars(discrepancy))
    )
    record.setdefault(
        "detected_at", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    try:
        with _discrepancies_path(workspace).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:  # pragma: no cover — disk/permission failure
        logger.warning("could not record discrepancy: %s", exc)


def load_discrepancies(
    workspace: str | Path,
    decision_id: str | None = None,
    severity: str | None = None,
) -> list[dict[str, Any]]:
    """Load discrepancy records, newest last.

    Unparsable lines (a truncated append, a hand edit) are skipped instead of
    failing the whole read.  *decision_id* / *severity* are exact-match
    filters; ``None`` means "no filter".
    """
    fp = _discrepancies_path(workspace)
    if not fp.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        lines = fp.read_text(encoding="utf-8").splitlines()
    except OSError as exc:  # pragma: no cover — disk/permission failure
        logger.warning("could not read discrepancies: %s", exc)
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if decision_id is not None and parsed.get("decision_id") != decision_id:
            continue
        if severity is not None and parsed.get("severity") != severity:
            continue
        out.append(parsed)
    return out


# ── Autonomous decisions (autonomous-decision plan, Phase 1) ──────────────

#: Tools whose calls decide *how* the work proceeds.  Read-only inspection is
#: excluded on purpose: every ``read``/``search`` call is a judgement of fact,
#: not a policy choice, and logging them would drown the real decisions.
_DECISION_TOOLS = frozenset(
    {
        "write",
        "edit",
        "run",
        "git",
        "tests",
        "fix",
        "analyze",
        "merge",
        "delegate",
        "delegate_batch",
        "create_subagent",
        "run_subagent_task",
        "plan_start",
        "plan_finish",
    }
)

#: Tools that spend another model — choosing to delegate is a decision.
_DELEGATION_TOOLS = frozenset(
    {
        "delegate",
        "delegate_batch",
        "create_subagent",
        "run_subagent_task",
    }
)

#: Plan lifecycle transitions: moving the plan forward is a decision.
_PLAN_TRANSITION_TOOLS = frozenset({"plan_start", "plan_finish"})

#: Result text that means "nothing happened" — the guard refused, the tool
#: found nothing, or it silently did nothing.  Such a call decided nothing, so
#: it is never recorded as a decision.
_NO_DECISION_RE = re.compile(
    r"(?i)\b(?:refus\w*|blocked|rejected|denied|not\s+allowed|forbidden"
    r"|unknown\s+tool|no\s+changes?|nothing\s+to)\b"
)

#: Plan-mode executor rejection prefix (a refused call, not a decision).
_PLAN_REJECTION_PREFIX = "[plan mode]"


def _outside_workspace(workspace: str | Path, value: str) -> bool:
    """True when *value* names a path that cannot live inside *workspace*.

    Used to keep foreign targets out of the decision ledger: a tool call that
    touches ``%TEMP%/...`` or another checkout is not a decision about THIS
    repository, and recording it leaves an affected_file that no checkout can
    ever contain (a permanently stale entry).
    """
    raw = str(value).strip().replace("\\", "/")
    if not raw:
        return False
    if not Path(raw).is_absolute():
        # A relative path is workspace-relative unless it climbs out.
        return raw.startswith("../") or "/../" in raw
    try:
        Path(raw).resolve().relative_to(Path(workspace).resolve())
    except (ValueError, OSError):
        return True
    return False


def looks_like_path(value: str) -> bool:
    """True when *value* is plausibly a file path and not prose.

    A decision record's ``affected_files`` is a claim about repository state,
    so every path-shaped string the agent typed must not become a target.  The
    detector collects paths from tool **arguments** (the keys of
    :data:`_PATH_ARG_KEYS`) and from result **prose** (:data:`_DELETED_PATH_RE`
    and friends), and both sources contain ordinary words:

    - ``run`` with ``command="git status && echo done"`` yields the shell
      segment ``echo done``, whose first token is a command, not a file.
    - A result sentence yields its nouns — "Delete once", "Change cards",
      ``(idempotent),`` — and a bare word resolves against the workspace
      perfectly well (``once`` -> ``<workspace>/once``), so the path checks in
      :func:`_outside_workspace` / :func:`_canonical_rel` cannot reject it.

    A token is a path candidate only when all of these hold:

    - one name, or a directory **chain** (the prose token regex
      :data:`_PATH_TOKEN_RE` matches the chain form);
    - every component is a plausible filesystem name — non-empty, at most
      :data:`_MAX_PATH_PART_LEN` characters, and free of
      :data:`_NON_PATH_CHARS` (shell operators, quotes, brackets, globbing,
      punctuation tails) and of whitespace;
    - it carries a path signal: a separator, an extension, a leading ``.``,
      or a drive-letter root;
    - it is not one of the prose words in :data:`_PROSE_PATH_WORDS` (which
      covers the single-word case with no extension, e.g. ``once``, ``cards``).

    Real targets pass: ``agent_core/decisions.py``, ``.decisions.json``,
    ``tests/test_decisions.py``, ``C:\\Dev\\Agent1\\x.py``, and a file that has
    just been deleted (existence is never required — a delete target is absent
    by definition).  A location reference such as ``decisions.py:738`` is **not**
    a target: it names a place inside a file, so the file it points at is not
    the thing that changed.
    """
    raw = str(value or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if lowered in _PROSE_PATH_WORDS:
        return False

    drive_root = bool(re.match(r"^[a-z]:[\\/]", lowered))
    # Strip a trailing separator (a directory marker, e.g. the ``agent_core/``
    # the prose regex matches) and split into components.
    stripped = raw.rstrip("/\\")
    parts = [p for p in re.split(r"[/\\]+", stripped) if p]
    has_chain = len(parts) > 1

    # ``decisions.py:738`` / ``src/mod.py:42`` is a line reference, not a path.
    # A colon is a Windows stream separator (``file:stream``) and never part of
    # a name on any filesystem in use, so a component carrying one is prose —
    # except the drive root, which the regex above already accepted.
    for index, part in enumerate(parts):
        if len(part) > _MAX_PATH_PART_LEN:
            return False
        if ":" in part and not (index == 0 and drive_root):
            return False
        if any(ch in _NON_PATH_CHARS for ch in part):
            return False
        if re.search(r"\s", part):
            return False

    # A single bare word with no extension is prose unless it starts with a
    # dot (``.decisions.json``).  ``once`` / ``cards`` die here; ``x.py`` and
    # ``README``-style names keep going (the word list above already caught
    # the English ones).
    if not has_chain:
        return "." in parts[0]

    # A chain needs at least one real name; a trailing separator
    # (``agent_core/security/``) is a directory marker the prose regex
    # deliberately matches.
    return True


def workspace_paths(workspace: str | Path, values: Any) -> list[str]:
    """Workspace-relative form of every **path-like** entry in *values*.

    *values* may be a single string, a list of strings, or arbitrary argument
    values (non-strings are skipped).  An entry is only a path candidate when
    :func:`looks_like_path` accepts it — the arguments of a tool call and the
    prose of a result text contain ordinary words, and an ordinary word resolves
    against the workspace just fine (``once`` -> ``C:\\Dev\\Agent1\\once``), so
    without that gate every ``run`` result turns its vocabulary into ledger
    entries.  See :func:`looks_like_path` for what counts as a path.

    Paths that do not exist on disk are kept: an action's target describes
    intent, and a delete target is absent by definition.  Paths outside
    *workspace* are dropped — they are not repository evidence (see
    :func:`_outside_workspace`).
    """
    items = values if isinstance(values, (list, tuple)) else [values]
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        if not looks_like_path(item):
            continue
        if _outside_workspace(workspace, item):
            continue
        rel = _canonical_rel(workspace, item)
        if rel is None:
            rel = str(item).strip().replace("\\", "/").lstrip("./") or None
        if rel and rel not in out:
            out.append(rel)
    return out


def _is_scratch_ref(path: str) -> bool:
    """True for throwaway probe/scratch paths (``_tmp_*``, ``tmp_*`` ...).

    A scratch file is not repository state.  Recording a decision about one
    leaves a stale ledger entry the moment the probe is deleted, which then
    fails the stale-reference audit (decision #054) for a file no checkout was
    ever meant to keep.  Scratch targets are therefore neither recorded (see
    :func:`detect_autonomous_decisions`) nor reported stale (see
    :func:`find_stale_decisions`).

    The predicate is imported from the pytest gate so "scratch" has exactly one
    definition in the codebase.
    """
    from agent_core.pytest_gate import is_scratch_path

    return is_scratch_path(path)


def _first_sentence(text: str, limit: int = 160) -> str:
    """Leading sentence of *text*, collapsed to one line and capped."""
    flat = " ".join(str(text or "").split())
    match = re.match(r"(.+?[.!?])(?:\s|$)", flat)
    out = match.group(1) if match else flat
    return out[:limit].rstrip()


def _git_decision_verb(name: str, args: dict[str, Any]) -> str:
    """``"commit"``/``"merge"`` when the call changes repo state, else ``""``.

    Two shapes must both count: a shell command (the ``run`` tool, where the
    verb sits inside the command text) and the ``git`` tool itself, which
    passes the verb in ``subcommand`` and the flags in ``args``.
    """
    text = " ".join(
        str(args.get(k) or "") for k in ("command", "subcommand", "args", "argv")
    )
    match = _GIT_DECISION_RE.search(text)
    if match:
        return match.group(1).lower()
    if name == "git":
        head = str(args.get("subcommand") or "").strip().strip("\"'")
        direct = re.match(r"(?i)(commit|merge)\b", head)
        if direct:
            return direct.group(1).lower()
    return ""


def detect_autonomous_decisions(
    workspace: str | Path,
    tool_name: str,
    args: dict[str, Any],
    result: str,
    *,
    user_input: str = "",
) -> list[dict[str, Any]]:
    """Detect the decisions an agent made autonomously in ONE tool call.

    A call counts as a decision when the agent picked an action the user never
    named:

    - **file targets** — the call writes/patches specific paths.  A path the
      user mentioned is theirs; every other path is the agent's choice.
    - **git commit / merge** — repository state changes.
    - **delegation and plan transitions** — the agent decided to spend a
      subagent or to move the plan forward.

    Only a call that succeeded is considered: refusals, ``unknown tool`` and
    error results decided nothing.  Read-only tools are never decisions.

    Returns candidate dicts (``title``, ``context``, ``decision``,
    ``rationale``, ``affected_files``, ``tags``, ``severity``), one per
    distinct target plus at most one repo-state candidate.  ``[]`` is the
    common case and means "nothing here to remember".
    """
    result_text = str(result or "")
    if result_text.lstrip().startswith(_PLAN_REJECTION_PREFIX):
        return []
    if _NO_DECISION_RE.search(result_text):
        return []
    name = str(tool_name or "").lower()
    if name not in _DECISION_TOOLS:
        return []

    removed = bool(re.search(r"(?i)\bdeleted\b", result_text))
    files = workspace_paths(
        workspace, [args.get(k) for k in _PATH_ARG_KEYS]
    )
    if not files:
        # A shell delete leaves no path argument, only the result text.  This
        # must come after the path args so a `del x.txt` on a path we already
        # know never produces a second candidate for the same target.
        files = workspace_paths(
            workspace, _DELETED_PATH_RE.findall(result_text)
        )
    # A path the user named is their instruction, not the agent's choice; the
    # decision prose recorded for the call is not evidence either way.
    user_text = str(user_input or "").replace("\\", "/")

    out: list[dict[str, Any]] = []
    for path in files:
        if user_text and path in user_text:
            continue
        # A scratch probe (`_tmp_*`, `tmp_*`) is not repository state: it is
        # deleted by design, so remembering it only creates a ledger entry that
        # goes stale.  See :func:`_is_scratch_ref`.
        if _is_scratch_ref(path):
            continue
        out.append(
            {
                "title": f"{'Delete' if removed else 'Change'} {path}",
                "context": _first_sentence(result_text, 200),
                "decision": (
                    f"{name} {'removed' if removed else 'modified'} {path} "
                    f"without that path being named in the request."
                ),
                "rationale": "Chosen during execution to advance the task.",
                "affected_files": [path],
                "tags": [AUTONOMOUS_TAG, name],
                "severity": "info",
            }
        )

    git_verb = _git_decision_verb(name, args)
    if git_verb:
        out.append(
            {
                "title": f"Ran git {git_verb} autonomously",
                "context": _first_sentence(result_text, 200),
                "decision": (
                    f"Committed repository state with git {git_verb} as part of "
                    f"the task, without a separate request to do so."
                ),
                "rationale": "Repository state change chosen during execution.",
                "affected_files": [],
                "tags": [AUTONOMOUS_TAG, "git"],
                "severity": "warning",
            }
        )

    if name in _DELEGATION_TOOLS:
        role = str(args.get("role") or "").strip()
        target = str(args.get("name") or "").strip()
        label = " ".join(part for part in (role, target) if part)
        out.append(
            {
                "title": f"Delegated work{' to ' + label if label else ''}",
                "context": _first_sentence(result_text, 200),
                "decision": (
                    f"Handed part of the task to a {role or 'sub'}agent"
                    f"{' (' + target + ')' if target else ''} instead of doing "
                    f"it inline."
                ),
                "rationale": "Delegation chosen during execution.",
                "affected_files": [],
                "tags": [AUTONOMOUS_TAG, name],
                "severity": "info",
            }
        )

    if name in _PLAN_TRANSITION_TOOLS:
        if name == "plan_start":
            title, why = "Started a plan autonomously", (
                "Started plan execution without being asked to begin."
            )
        else:
            title, why = "Finished a plan autonomously", (
                "Marked the plan executed once the remaining tasks passed."
            )
        out.append(
            {
                "title": title,
                "context": _first_sentence(result_text, 200),
                "decision": why,
                "rationale": "Plan transition chosen during execution.",
                "affected_files": [],
                "tags": [AUTONOMOUS_TAG, name],
                "severity": "warning",
            }
        )
    return out


def _autonomous_log_path(workspace: str | Path) -> Path:
    return Path(workspace) / AUTONOMOUS_LOG_FILE


def load_autonomous_log(workspace: str | Path) -> list[dict[str, Any]]:
    """Read the autonomous-decision observation log, newest last.

    Unparsable lines are skipped (a truncated append must not fail the read).
    """
    fp = _autonomous_log_path(workspace)
    if not fp.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        lines = fp.read_text(encoding="utf-8").splitlines()
    except OSError as exc:  # pragma: no cover — disk/permission failure
        logger.warning("could not read autonomous log: %s", exc)
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def record_autonomous_decisions(
    workspace: str | Path, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Persist detected *candidates*: log them, return the newly stored ones.

    Each candidate is first appended to :data:`AUTONOMOUS_LOG_FILE` with a
    ``decision_id`` (``None`` when not promoted).  A candidate is promoted to
    the ledger only when its ``title`` is not already recorded there, so a
    repeated detection is logged but never duplicated.  A ledger write
    collapses repeated paths — a 139-entry ledger must not gain a new entry
    because a file was touched three times in one turn.

    Returns the promoted decision records (possibly empty).
    """
    if not candidates:
        return []
    existing = {str(d.get("title") or "") for d in load_decisions(workspace)}
    stored: list[dict[str, Any]] = []
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_lines: list[str] = []
    for candidate in candidates:
        title = str(candidate.get("title") or "")
        promoted: dict[str, Any] | None = None
        if title and title not in existing:
            try:
                promoted = add_decision(
                    workspace,
                    title=title,
                    context=str(candidate.get("context") or ""),
                    decision=str(candidate.get("decision") or ""),
                    rationale=str(candidate.get("rationale") or ""),
                    affected_files=list(candidate.get("affected_files") or []),
                    tags=list(candidate.get("tags") or []),
                    category=AUTONOMOUS_CATEGORY,
                    source=SOURCE_AUTONOMOUS,
                )
            except OSError as exc:  # pragma: no cover — disk/permission failure
                logger.warning("could not record autonomous decision: %s", exc)
            else:
                existing.add(title)
                stored.append(promoted)
        log_lines.append(
            json.dumps(
                {
                    **candidate,
                    "detected_at": stamp,
                    "decision_id": promoted.get("id") if promoted else None,
                },
                ensure_ascii=False,
                default=str,
            )
        )
    try:
        with _autonomous_log_path(workspace).open("a", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n")
    except OSError as exc:  # pragma: no cover — disk/permission failure
        logger.warning("could not append autonomous log: %s", exc)
    return stored


def record_autonomous_decision(
    workspace: str | Path,
    title: str,
    context: str = "",
    decision: str = "",
    rationale: str = "",
    affected_files: list[str] | None = None,
    tags: list[str] | None = None,
    category: str = AUTONOMOUS_CATEGORY,
) -> dict[str, Any]:
    """Record a decision the agent made autonomously during execution.

    Unlike a manual decision (CLI, analysis, user request), this one was never
    asked for: the agent chose it while executing.  The record therefore
    carries the ``autonomous`` category *and* the ``autonomous`` tag — the
    category is a single slot, the tag survives re-categorisation — plus
    ``source="agent"`` so a review can tell the two origins apart.

    Raises ``ValueError`` when *title* is empty: an unnamed decision cannot be
    reviewed, contradicted or searched later.
    """
    if not str(title or "").strip():
        raise ValueError("an autonomous decision needs a title")
    wanted_tags: list[str] = []
    for tag in [AUTONOMOUS_TAG, *(tags or [])]:
        norm = str(tag).strip().lower()
        if norm and norm not in wanted_tags:
            wanted_tags.append(norm)
    return add_decision(
        workspace,
        title=str(title).strip(),
        context=context,
        decision=decision,
        rationale=rationale,
        affected_files=affected_files,
        tags=wanted_tags,
        category=category or AUTONOMOUS_CATEGORY,
        source=SOURCE_AUTONOMOUS,
    )


#: Top-level trees that are generated or machine-local (gitignored, see
#: .gitignore).  A decision referencing one of these — e.g. a test scratch
#: file under ``.pytest_tmp/`` — cannot be verified against the repository, so
#: the reference must NOT be reported stale just because another checkout (CI)
#: does not happen to have that directory.
_TRANSIENT_TOP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".pytest_tmp",
        ".mypy_cache",
        ".ruff_cache",
        ".docs",
        "backups",
        "reports",
        "checkpoints",
        "generated",
    }
)


def _is_transient_ref(path: str) -> bool:
    """True for a workspace-relative path under a generated/machine-local tree."""
    parts = Path(str(path).replace("\\", "/")).parts
    return bool(parts) and parts[0] in _TRANSIENT_TOP_DIRS


def find_stale_decisions(
    workspace: str | Path, decisions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Decisions whose recorded affected_files no longer exist on disk.

    A stale reference means the decision's subject was deleted or renamed —
    a candidate for verification or supersession, not a broken ledger.  The
    decision ledger is a living document; the human gate decides what to do
    with it (decision #054).

    References under a gitignored/transient tree (``.pytest_tmp``, ``reports``,
    ...) are ignored: they are machine-local artifacts, not repo files, so a
    checkout that lacks them must not be told the decision is stale.
    """
    stale: list[dict[str, Any]] = []
    for d in decisions:
        # Only active decisions can be "stale"; void/superseded/archived
        # records are historical and must not trip the audit.
        if str(d.get("status", "active")).lower() not in ("", "active"):
            continue
        missing = [
            f
            for f in d.get("affected_files", [])
            if not _is_transient_ref(f)
            and not _is_scratch_ref(f)
            and not (Path(str(workspace)) / f).exists()
        ]
        if missing:
            d = dict(d)
            d["_missing_files"] = missing
            stale.append(d)
    return stale


def find_open_contradictions(
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Decisions carrying unresolved contradictions.

    A contradiction entry is *open* when its ``status`` field is neither
    ``"resolved"`` nor ``"superseded"``.  Returns a shallow copy of each
    matching decision annotated with ``_open_contradiction_ids`` (the list
    of unresolved contradiction IDs).

    Decision #054: the human gate decides what to do with open
    contradictions.
    """
    open_list: list[dict[str, Any]] = []
    for d in decisions:
        contradictions = d.get("contradictions", [])
        if not contradictions:
            continue
        open_ids = [
            c["id"]
            for c in contradictions
            if isinstance(c, dict) and c.get("status") not in ("resolved", "superseded")
        ]
        if open_ids:
            d = dict(d)
            d["_open_contradiction_ids"] = open_ids
            open_list.append(d)
    return open_list


def find_meta_warnings(
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Decisions carrying meta_warnings (unverified claims, stale refs, etc.).

    Meta-warnings are rendered into LLM prompts (decisions_as_system_prompt)
    but are also surfaced here for audit-time visibility so they are not
    invisible to humans running ``decide review`` or ``audit_invariants``.

    Decision #080, #082, #084, #086, #087: unverified-claim warnings.
    """
    flagged: list[dict[str, Any]] = []
    for d in decisions:
        warnings = d.get("meta_warnings")
        if warnings:
            d = dict(d)
            d["_meta_warnings"] = list(warnings)
            flagged.append(d)
    return flagged


def ledger_health(
    workspace: str | Path,
    decisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Combined health report for the decision ledger.

    Returns a dict with keys ``stale``, ``open_contradictions``, and
    ``meta_warnings``, each a list of annotated decision dicts.
    """
    if decisions is None:
        decisions = load_decisions(workspace)
    return {
        "stale": find_stale_decisions(workspace, decisions),
        "open_contradictions": find_open_contradictions(decisions),
        "meta_warnings": find_meta_warnings(decisions),
    }


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
    category: str | None = None,
    source: str = "",
) -> dict[str, Any]:
    """Append one decision to the ledger and return the stored record.

    *source* records where the decision came from (``"agent"`` for one the
    agent took autonomously, see :func:`record_autonomous_decision`).  It is
    stored only when non-empty so hand-written and legacy records keep their
    existing shape.
    """
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
        "status": STATUS_ACTIVE,
        "category": category or "",
    }
    if warnings:
        record["meta_warnings"] = warnings
    if source:
        record["source"] = source
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
    status: str | None = None,
    category: str | None = None,
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
        if status and d.get("status", STATUS_ACTIVE) != status:
            continue
        if category and d.get("category") != category:
            continue
        result.append(d)
    return result


def count_by_status(decisions: list[dict[str, Any]]) -> dict[str, int]:
    """Count decisions grouped by status."""
    counts: dict[str, int] = {}
    for d in decisions:
        s = d.get("status", STATUS_ACTIVE)
        counts[s] = counts.get(s, 0) + 1
    return counts


def count_by_category(decisions: list[dict[str, Any]]) -> dict[str, int]:
    """Count decisions grouped by category (uncategorized last)."""
    counts: dict[str, int] = {}
    for d in decisions:
        c = d.get("category") or "(uncategorized)"
        counts[c] = counts.get(c, 0) + 1
    return dict(sorted(counts.items()))


# ── Simple overlap check (instant, no LLM) ──────────────────────────────────


def find_overlaps(
    new_decision: dict[str, Any],
    existing: list[dict[str, Any]],
    workspace: str | Path,
) -> list[dict[str, Any]]:
    new_tags = set(new_decision.get("tags", []))
    new_files = {
        c
        for c in (
            _canonical_rel(workspace, f) for f in new_decision.get("affected_files", [])
        )
        if c
    }
    overlaps = []
    for d in existing:
        old_tags = set(d.get("tags", []))
        # already canonical (add/load normalized)
        old_files = set(d.get("affected_files", []))
        tag_overlap = bool(new_tags & old_tags)
        file_overlap = bool(new_files & old_files)
        if tag_overlap or file_overlap:
            overlaps.append(d)
    return overlaps


def build_decision_graph(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a knowledge graph of decision-code relationships.

    Returns:
        Graph structure: {
            "decisions": {decision_id: {...}},
            "files": {file_path: [decision_ids]},
            "tags": {tag: [decision_ids]},
            "symbols": {symbol_name: [decision_ids]}
        }
    """
    graph: dict[str, Any] = {"decisions": {}, "files": {}, "tags": {}, "symbols": {}}

    for d in decisions:
        d_id = d["id"]
        graph["decisions"][d_id] = d

        # File relationships
        for f in d.get("affected_files", []):
            graph["files"].setdefault(f, []).append(d_id)

        # Tag relationships
        for tag in d.get("tags", []):
            graph["tags"].setdefault(tag, []).append(d_id)

    return graph


# ── Cached contradiction checks ────────────────────────────────────────────
# Note: a manual dict cache is used instead of functools.lru_cache because
# lru_cache would store coroutine objects, which cannot be reused.

_cached_results: dict[tuple[frozenset[str], str, str], str] = {}


def _contradiction_cache_key(
    decisions: list[dict[str, Any]],
    new_decision_text: str,
    workspace: str = "",
) -> tuple[frozenset[str], str, str]:
    """Cache key: the decision IDs involved + the new text + the workspace.

    Decision IDs are assigned per-workspace (001, 002, ...), so the workspace
    must be part of the key to avoid cross-workspace collisions.
    """
    ids = frozenset(str(d.get("id", "")) for d in decisions)
    return (ids, new_decision_text, workspace)


async def check_contradictions_cached(
    agent: "Agent",
    decisions: list[dict[str, Any]],
    new_decision_text: str,
    workspace: str = "",
) -> str:
    """Cached wrapper around check_contradictions.

    Cache key is the set of decision IDs + the new decision text (+ workspace
    when provided) — not the full decision objects.
    """
    if not decisions:
        return "No existing decisions to check against."
    cache_key = _contradiction_cache_key(decisions, new_decision_text, workspace)
    cached = _cached_results.get(cache_key)
    if cached is not None:
        return cached
    result = await check_contradictions(agent, decisions, new_decision_text)
    _cached_results[cache_key] = result
    return result


def clear_contradiction_cache() -> None:
    """Clear the contradiction check cache."""
    _cached_results.clear()


def build_category_index(decisions: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build inverted index from category to decision IDs.

    Returns:
        {category: [decision_ids]}
    """
    index: dict[str, list[str]] = {}
    for d in decisions:
        category = d.get("category", "uncategorized").lower()
        index.setdefault(category, []).append(d["id"])
    return index


class DecisionCache:
    """Cache for decisions with lazy loading."""

    def __init__(self, workspace: str | Path):
        self.workspace = workspace
        self._cached_decisions: list[dict[str, Any]] | None = None
        self._cached_graph: dict[str, Any] | None = None
        self._cached_category_index: dict[str, list[str]] | None = None

    def get_decisions(self) -> list[dict[str, Any]]:
        """Load decisions if not cached."""
        if self._cached_decisions is None:
            self._cached_decisions = load_decisions(self.workspace)
        return self._cached_decisions

    def get_graph(self) -> dict[str, Any]:
        """Build and cache decision graph."""
        if self._cached_graph is None:
            self._cached_graph = build_decision_graph(self.get_decisions())
        return self._cached_graph

    def get_category_index(self) -> dict[str, list[str]]:
        """Build and cache category index."""
        if self._cached_category_index is None:
            self._cached_category_index = build_category_index(self.get_decisions())
        return self._cached_category_index


def format_for_prompt(decisions: list[dict[str, Any]]) -> str:
    if not decisions:
        return ""
    lines = ["## Past Decisions (do NOT contradict these)"]
    lines += [
        f"Decision #{d['id']}: {d.get('decision', '') or d.get('title', '')}\n"
        f"  Rationale: {d.get('rationale', '')}\n"
        f"  Affected files: {', '.join(d.get('affected_files', [])[:5])}\n"
        for d in decisions
    ]
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
    past = [
        f"#{d['id']}: {d['title']}\n"
        f"  Decision: {d.get('decision', '')}\n"
        f"  Rationale: {d.get('rationale', '')}\n"
        f"  Files: {', '.join(d.get('affected_files', []))}\n"
        f"  Tags: {', '.join(d.get('tags', []))}"
        for d in decisions
    ]
    user_msg = (
        "## Past decisions\n\n" + "\n\n".join(past) + "\n\n"
        f"## New decision to evaluate\n\n{new_decision_text}\n\n"
        "Does the new decision contradict any past decision? "
        "Be specific — cite decision IDs."
    )
    response = await agent.llm.chat(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]
    )
    return response if response else "LLM returned no response."


async def check_contradictions_fast(
    agent: "Agent",
    new_decision: dict[str, Any],
    new_decision_text: str,
    workspace: str | Path,
) -> str:
    """Fast contradiction check with pre-filtering.

    Uses find_overlaps() to filter decisions before LLM call.
    Returns early if no overlaps found.
    """
    # Load all decisions once
    all_decisions = load_decisions(workspace)

    # Fast pre-filter: only check decisions that share files/tags
    candidate_decisions = find_overlaps(new_decision, all_decisions, workspace)

    if not candidate_decisions:
        return "No similar decisions found - no contradiction risk."

    # Check cache (await: check_contradictions_cached is async)
    return await check_contradictions_cached(
        agent,
        candidate_decisions,
        new_decision_text,
        workspace=str(workspace),
    )


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
    response = await agent.llm.chat(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]
    )
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
        text = " ".join(
            filter(
                None,
                (
                    c.get("title", ""),
                    c.get("context", ""),
                    c.get("decision", ""),
                ),
            )
        )
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
    response = await agent.llm.chat(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        disable_thinking=True,
    )
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
    user_msg = "## Changed files\n" + "\n".join(f"- {f}" for f in files) + "\n"
    if context:
        user_msg += f"\n## Task context\n{context}\n"
    user_msg += "\nExtract the design decisions embodied in these changes."

    response = await agent.llm.chat(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        disable_thinking=True,
    )
    result = _parse_json_array(response)
    ws = str(Path(agent.workspace).resolve())
    return _filter_candidates(result, ws)


def is_autonomous_decision(record: dict[str, Any]) -> bool:
    """True when *record* was taken by the agent without being asked.

    Detected by provenance (``source``) or by the tag, so a record survives
    re-categorisation; both are checked because the category slot can hold
    only one value while the tag is additive.
    """
    if str(record.get("source") or "") == SOURCE_AUTONOMOUS:
        return True
    return AUTONOMOUS_TAG in [str(t) for t in record.get("tags") or []]


def _recent_autonomous_block(workspace: str | Path) -> str:
    """Prompt lines listing the most recent autonomous decisions, if any.

    Phase 1 of the autonomous-decision plan: these decisions were taken
    without a request, so re-stating the newest ones keeps the agent
    consistent instead of silently reversing its own choices.  Only the last
    :data:`_RECENT_AUTONOMOUS_LIMIT` are shown — older ones are already
    superseded context, and a long list would bury the real constraints.
    """
    records = [
        d for d in load_decisions(workspace) if is_autonomous_decision(d)
    ]
    if not records:
        return ""
    lines = [
        "",
        "RECENT AUTONOMOUS DECISIONS — choices you made while executing, "
        "without being asked:",
        "Keep behaviour consistent with them; do not silently reverse one.",
        "",
    ]
    for d in records[-_RECENT_AUTONOMOUS_LIMIT:]:
        files = ", ".join(d.get("affected_files") or [])
        lines.append(
            f"  #{d['id']} ({d['title']}): {d.get('decision', '')}"
            + (f" [files: {files}]" if files else "")
        )
    return "\n".join(lines) + "\n"


def decisions_as_system_prompt(
    workspace: str | Path,
    files: list[str] | None = None,
    include_autonomous: bool = True,
) -> str:
    """Build a constraints block for LLM prompts from past decisions.

    *files* scopes the constraints to the paths in play; ``None``/empty means
    "every active decision".  *include_autonomous* adds the most recent
    autonomously-taken decisions (see :func:`_recent_autonomous_block`).

    Returns empty string if no relevant decisions exist.
    """
    decisions = (
        find_decisions(
            workspace=str(workspace), files=files, status=STATUS_ACTIVE
        )
        if files
        else [d for d in load_decisions(workspace) if d.get("status") == STATUS_ACTIVE]
    )
    # Autonomous records are re-stated in their own (milder) block below: they
    # were never agreed with the user, so listing them as CRITICAL constraints
    # the implementation "MUST respect" would overstate their authority.
    decisions = [d for d in decisions if not is_autonomous_decision(d)]
    autonomous = _recent_autonomous_block(workspace) if include_autonomous else ""
    if not decisions:
        return autonomous
    lines = [
        "\n\nCRITICAL DESIGN CONSTRAINTS — past decisions that MUST be respected:",
        "These are design choices already documented for this codebase.",
        "Your implementation MUST NOT contradict any of these decisions.",
        "",
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
    return "\n".join(lines) + autonomous


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
            f
            in [
                "agent_core/decisions.py",
                "agent_core/commands/decide_cmd.py",
                ".decisions.json",
            ]
            for f in files
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
        all_exist = (
            files and all((ws_path / f).exists() for f in files) if files else False
        )
        if all_exist and any(f in existing_files for f in files):
            continue

        filtered.append(c)
    return filtered
