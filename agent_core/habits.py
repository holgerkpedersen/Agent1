"""Workspace habit learning: recurring user preferences mined from history.

A *habit* is a short line describing something the user repeatedly asks for
("keep answers short", "often mentions agent_core/commands/perf_cmd.py",
"avoid editing tests to make them pass").  Only the compact BLOCK — the
``HABITS_MARKER`` header plus at most 8 ``- <habit> (seen N times)`` lines —
is injected into the chat system prompt, so a fresh prompt with no habits
stays **byte-identical** to the pre-habits baseline (the same contract
``Agent._decision_constraints_block`` follows: empty means the empty string,
never a stray header).

Design notes
------------

* **Pure stdlib** (mirrors ``agent_core/skills.py`` structurally but without
  its Pydantic dependency): JSON persistence, regex mining, no I/O beyond the
  one workspace-local ledger.
* **Fail-open**: :func:`load_habits` NEVER raises — a corrupt ledger is
  quarantined to ``.habits.json.bad-<timestamp>`` (byte-compatible with
  ``agent._read_json_quarantining``, agent.py:2957) and mining returns ``[]``
  on any internal error, so a broken ``.habits.json`` must not kill a chat
  turn.
* **Never loop noise**: mining skips loop-injected notes (messages tagged
  ``LOOP_NOTE_TAG_KEY``) and plan-mode turn notes (``"[PLAN MODE]"`` prefix,
  see :func:`agent_core.modes.plan_mode_turn_note`) — those describe the
  harness, not the user.
* **Recency decay**: occurrence scores use exponential decay over the
  trailing ``window`` inputs, so a habit the user stopped repeating fades out
  instead of being remembered forever.
* **Prompt budget**: each of the five categories is capped and the rendered
  block is hard-capped at 8 lines (~600 chars) — habits are a hint, not a
  database dump.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: Exact prefix of the injected block — also the marker
#: ``agent._strip_dynamic_system_blocks`` uses to rebuild it every turn
#: (without it a long session would accumulate one stale block per turn).
HABITS_MARKER = "\n\nUSER HABITS"

#: Workspace-local ledger file name (same convention as ``.decisions.json``).
HABITS_FILENAME = ".habits.json"

#: Hard cap on rendered lines — the block is a prompt, not a database.
MAX_BLOCK_LINES = 8

#: Per-category cap so one loud category cannot crowd out the others
#: (5 categories x 3 = 15 candidates; the block then picks the top 8).
MAX_PER_CATEGORY = 3

#: Habit text is one prompt line, not an essay.
MAX_HABIT_CHARS = 120

#: Mining defaults (see :func:`mine_habits`).
DEFAULT_MIN_COUNT = 3
DEFAULT_WINDOW = 40

#: Per-occurrence exponential recency weight: an occurrence ``k`` steps back
#: from the newest input scores ``DECAY ** k`` (0.9 => a 10-step-old repeat is
#: worth ~35% of a fresh one).
DECAY = 0.9

#: Prefix of plan-mode turn notes (agent_core.modes.plan_mode_turn_note) —
#: harness text, not a user preference.
_PLAN_NOTE_PREFIX = "[PLAN MODE]"

#: Loop-injected note tag (agent_core.constants.LOOP_NOTE_TAG_KEY); imported
#: by value to keep this module dependency-light (stdlib-only contract).
_LOOP_NOTE_KEY = "_loop_note"

# ---------------------------------------------------------------------------
# Candidate extraction rules (the five categories)
# ---------------------------------------------------------------------------

#: (category, compiled pattern, canonical habit text) for typed commands.
_COMMAND_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("command", re.compile(r"\b(?:run|execute)\b[^.!?\n]*\btests?\b|\bpytest\b",
                           re.I),
     "run the test suite (pytest)"),
    ("command", re.compile(r"\b(?:lint|ruff|flake8|pylint)\b", re.I),
     "lint the changed files"),
    ("command", re.compile(r"\bgit\s+(?:commit|push)\b", re.I),
     "commit and push with git"),
)

#: Output-style preferences: how the user wants answers rendered.
_STYLE_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("style", re.compile(r"\b(?:short|brief|concise|terse|summar\w*)\b", re.I),
     "keep answers short"),
    ("style", re.compile(r"\b(?:verbose|detailed|in detail|thorough|step by step)\b",
                         re.I),
     "give detailed step-by-step answers"),
)

#: A path-like token: dirs with slashes or a file name with an extension.
_PATH_RE = re.compile(
    r"(?:[\w.@-]+/)+[\w.@-]+|[\w.@-]+\.(?:py|md|json|toml|ya?ml|txt|cfg|ini)\b"
)

#: Workflow markers: plan-ish then apply-ish within adjacent inputs.
_PLAN_RE = re.compile(r"\b(?:plan|propose|design|outline)\b", re.I)
_APPLY_RE = re.compile(
    r"\b(?:apply|implement|execute|go ahead|proceed|ship it|do it|build it)\b", re.I
)

#: Corrections: the user rephrasing/contradicting after an answer.
_CORRECTION_RE = re.compile(
    r"^\s*(?:no[,.!?]?\s|nope\b|wrong\b|not (?:that|what|quite)\b|instead\b|"
    r"i (?:meant|said)\b|don'?t\b|stop\b|avoid\b)",
    re.I,
)


def _user_texts(history: Sequence[Any]) -> list[str]:
    """Extract real user texts from chat history, skipping harness notes.

    Only ``role == "user"`` messages count; loop-injected notes (tagged
    ``_loop_note``) and plan-mode turn notes (``[PLAN MODE]`` prefix) are
    harness text, never a preference.  String content only — image turns keep
    their content blocks out of mining.
    """
    texts: list[str] = []
    for message in history:
        if not isinstance(message, Mapping):
            continue
        if message.get("role") != "user":
            continue
        if _LOOP_NOTE_KEY in message:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        text = content.strip()
        if not text or text.startswith(_PLAN_NOTE_PREFIX):
            continue
        texts.append(text)
    return texts


def _trace_texts(traces: Iterable[Any]) -> list[str]:
    """Extract ``task_begin.user_input`` texts from parsed trace records."""
    texts: list[str] = []
    for record in traces:
        if not isinstance(record, Mapping):
            continue
        # Flat {"kind": "task_begin", "user_input": ...} or nested
        # {"task_begin": {"user_input": ...}} — accept both shapes.
        user_input = record.get("user_input")
        if user_input is None and isinstance(record.get("task_begin"), Mapping):
            user_input = record["task_begin"].get("user_input")
        if isinstance(user_input, str) and user_input.strip():
            texts.append(user_input.strip())
    return texts


def _classify(text: str) -> list[tuple[str, str]]:
    """Return ``(category, habit_text)`` candidates for one user input."""
    found: list[tuple[str, str]] = []
    for category, pattern, canonical in _COMMAND_RULES:
        if pattern.search(text):
            found.append((category, canonical))
            break  # one command habit per input
    for category, pattern, canonical in _STYLE_RULES:
        if pattern.search(text):
            found.append((category, canonical))
            break
    for match in _PATH_RE.findall(text):
        if "/" in match or "." in match:
            found.append(("files", f"often mentions {match.rstrip('/')}"))
    if _CORRECTION_RE.search(text):
        tail = _CORRECTION_RE.sub("", text, count=1).strip() or text
        tail = re.sub(r"\s+", " ", tail)[:MAX_HABIT_CHARS]
        if tail:
            found.append(("corrections", f"avoid {tail}"))
    return found


def mine_habits(
    history: Sequence[Any],
    traces: Sequence[Any],
    *,
    min_count: int = DEFAULT_MIN_COUNT,
    window: int = DEFAULT_WINDOW,
) -> list[dict[str, Any]]:
    """Mine recurring user preferences into habit entries.

    Sources: chat-history ``role == "user"`` messages (loop/plan notes
    excluded) and trace records carrying ``task_begin.user_input``.  Inputs
    beyond the trailing *window* are ignored; every occurrence scores
    ``DECAY ** age`` (age = steps back from the newest input), candidates
    need **raw count >= min_count** to be promoted, and each of the five
    categories (command / style / files / workflow / corrections) is capped at
    :data:`MAX_PER_CATEGORY` by decayed score.  The repeated-workflow habit
    ("plan, then apply") is counted from plan-ish inputs immediately followed
    by apply-ish ones.

    Returns a list of ``{"text", "count", "category", "score"}`` dicts,
    highest score first.  Never raises: any internal failure logs and returns
    ``[]`` (a broken miner must not kill a chat turn).
    """
    try:
        inputs = _user_texts(history) + _trace_texts(traces)
        inputs = inputs[-max(1, int(window)):]
        if not inputs:
            return []

        # Raw counts + decayed scores per (category, text).
        raw: dict[tuple[str, str], int] = {}
        score: dict[tuple[str, str], float] = {}
        newest = len(inputs)
        for age, text in enumerate(inputs):
            weight = DECAY ** (newest - 1 - age)
            for category, habit_text in _classify(text):
                key = (category, habit_text)
                raw[key] = raw.get(key, 0) + 1
                score[key] = score.get(key, 0.0) + weight
            if _PLAN_RE.search(text):
                # remember a plan-ish input awaiting an apply-ish follow-up
                pass

        # Repeated workflow: plan-ish input immediately followed by apply-ish.
        for before, after in zip(inputs, inputs[1:]):
            if _PLAN_RE.search(before) and _APPLY_RE.search(after):
                key = ("workflow", "plan first, then apply the changes")
                raw[key] = raw.get(key, 0) + 1
                score[key] = score.get(key, 0.0) + 1.0

        promoted = [
            {"text": text, "count": count, "category": category,
             "score": round(score[(category, text)], 4)}
            for (category, text), count in raw.items()
            if count >= max(1, int(min_count))
        ]
        if not promoted:
            return []

        # Per-category cap, then global score order.
        capped: list[dict[str, Any]] = []
        for category in ("command", "style", "files", "workflow", "corrections"):
            bucket = sorted(
                (h for h in promoted if h["category"] == category),
                key=lambda h: h["score"], reverse=True,
            )
            capped.extend(bucket[:MAX_PER_CATEGORY])
        capped.sort(key=lambda h: h["score"], reverse=True)
        return capped
    except Exception:  # noqa: BLE001 - contract: mining never raises
        logger.exception("Habit mining unavailable:\n")
        return []


# ---------------------------------------------------------------------------
# Rendering (system-prompt block)
# ---------------------------------------------------------------------------

def _line(habit: Any) -> str | None:
    """Render one ``- <habit> (seen N times)`` line (``None`` to skip)."""
    if isinstance(habit, Mapping):
        text = str(habit.get("text") or "").strip()
        try:
            count = int(habit.get("count", 1))
        except (TypeError, ValueError):
            count = 1
    else:
        text = str(habit).strip()
        count = 1
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)[:MAX_HABIT_CHARS]
    return f"- {text} (seen {count} times)"


def habits_block(habits: Sequence[Any] | str | None) -> str:
    """The system-prompt block: ``""`` when empty, else marker + <= 8 lines.

    Empty input (``[]``, ``""``, ``None``) returns the empty string so a
    fresh workspace prompt stays byte-identical to the pre-habits baseline.
    Accepts dicts (``{"text", "count"}``) or plain strings.
    """
    if not habits:
        return ""
    if isinstance(habits, str):
        habits = [habits]
    lines: list[str] = []
    for habit in habits:
        rendered = _line(habit)
        if rendered is not None:
            lines.append(rendered)
        if len(lines) >= MAX_BLOCK_LINES:
            break
    if not lines:
        return ""
    block = HABITS_MARKER + "\n" + "\n".join(lines) + "\n"
    assert block.startswith(HABITS_MARKER)  # marker contract (see module doc)
    return block


# ---------------------------------------------------------------------------
# Persistence (workspace-local .habits.json)
# ---------------------------------------------------------------------------

def _habits_path(workspace: str | Path) -> Path:
    """The workspace's ``.habits.json`` ledger path."""
    return Path(workspace) / HABITS_FILENAME


def load_habits(workspace: str | Path) -> list[dict[str, Any]]:
    """Load the workspace ledger — NEVER raises (``[]`` on any failure).

    A corrupt file is quarantined to ``.habits.json.bad-<timestamp>`` so the
    bytes stay inspectable (same rule as ``_read_json_quarantining``,
    agent.py:2957).  Returns ``[]`` when the ``habits_enabled`` workspace
    preference is off (``habits off``), which also suppresses the prompt
    block.
    """
    try:
        from agent_core.llm.workspace_prefs import get_pref

        if get_pref(Path(workspace), "habits_enabled") is False:
            return []
    except Exception:  # pref file broken — fall through to the ledger read
        logger.debug("habits_enabled pref unreadable, continuing", exc_info=True)

    path = _habits_path(workspace)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine = f"{path}.bad-{stamp}"
        try:
            os.replace(path, quarantine)
        except OSError:
            logger.warning("Failed to quarantine corrupt habits at %s", path)
        else:
            logger.warning("Corrupt habits file moved to %s", quarantine)
        return []
    except Exception:  # noqa: BLE001 - contract: never raise
        logger.exception("Habit load unavailable:\n")
        return []

    if not isinstance(data, list):
        return []
    habits: list[dict[str, Any]] = []
    for entry in data:
        if isinstance(entry, Mapping) and str(entry.get("text") or "").strip():
            habits.append(dict(entry))
        elif isinstance(entry, str) and entry.strip():
            habits.append({"text": entry.strip(), "count": 1, "category": "pinned"})
    return habits


def save_habits(workspace: str | Path, habits: Sequence[Any]) -> None:
    """Persist the ledger atomically (tmp + ``os.replace``); never raises.

    Entries are normalized to ``{"text", "count", "category"}`` dicts so a
    round-trip through :func:`load_habits` is lossless.
    """
    try:
        normalized: list[dict[str, Any]] = []
        for habit in habits:
            if isinstance(habit, Mapping) and str(habit.get("text") or "").strip():
                entry = dict(habit)
                entry.setdefault("count", 1)
                entry.setdefault("category", "pinned")
                normalized.append(entry)
            elif isinstance(habit, str) and habit.strip():
                normalized.append(
                    {"text": habit.strip(), "count": 1, "category": "pinned"}
                )
        path = _habits_path(workspace)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(normalized, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:  # noqa: BLE001 - a failed save must not kill a turn
        logger.warning("Could not save habits ledger", exc_info=True)


__all__ = [
    "DECAY",
    "DEFAULT_MIN_COUNT",
    "DEFAULT_WINDOW",
    "HABITS_FILENAME",
    "HABITS_MARKER",
    "MAX_BLOCK_LINES",
    "MAX_PER_CATEGORY",
    "habits_block",
    "load_habits",
    "mine_habits",
    "save_habits",
]
