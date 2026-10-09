"""Bidirectional Kanban ↔ Agent1 sync — disk-backed JSONL enqueueing.

When a change occurs in Agent1 (issue created, updated, resolved), it is
immediately appended to the target Kanban queue on disk. A background reader
processes queued messages when Agent1 runs.  This makes the integration
tolerant of restarts and partial outages — no message is lost as long as
the filesystem survives.

Usage::

    KANBAN_SYNC_ENABLED=1 python agent.py        # auto-process on boot

``agent.py`` calls :func:`start_inbound_processor` at startup when
``KANBAN_SYNC_ENABLED=1``, so the REPL process both enqueues outbound
messages and drains the inbound queue.

or manually::

    python -m harnessfix.kanban_bridge --process-in         # one-shot poll

Queue layout (one directory SHARED with the Kanban app, so both processes see
the same files)::

    <KANBAN_WORKING_DIR>/
        queue-kanban-to-agent1/   ← Kanban writes here (Agent1 reads)
        │   ├── messages.jsonl
        │   └── offset.txt        ← reader cursor only (see enqueue())
        queue-agent1-to-kanban/   ← Agent1 writes here (Kanban reads)
            ├── messages.jsonl
            └── offset.txt
        .card_id_map.json          ← bidirectional issue↔card ID mapping

``KANBAN_WORKING_DIR`` defaults to the sibling Kanban checkout's ``data/``
directory and must resolve to the same path on both sides.

Message format (one JSON object per line)::

    {seq, ts, op, source_id, target_ref, payload, retry_count, processed}
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Repo-root ``.env`` consulted for the sync master switch.  Module-level (not
#: a constant inlined into the functions) so tests can redirect it and stay
#: independent of the developer's real ``.env``.
ENV_FILE_PATH = REPO_ROOT / ".env"


def _read_env_file_value(key: str) -> str | None:
    """Value of *key* in the repo ``.env``, or ``None`` if absent/unreadable.

    Same dialect as the rest of the repo (``agent_core.config._load_env_file``,
    ``agent._read_env_value``, ``conftest._load_env``): ``#`` comments skipped,
    surrounding quotes and whitespace stripped.
    """
    try:
        text = ENV_FILE_PATH.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        env_key, _, value = stripped.partition("=")
        if env_key.strip() == key:
            return value.strip().strip("\"'")
    return None


def sync_enabled() -> bool:
    """True when Kanban sync is switched on — process env, else the repo ``.env``.

    ``.env.example`` documents ``KANBAN_SYNC_ENABLED`` as an ``.env`` setting
    and tells the operator to set it to ``1``, but nothing in this repo bridges
    ``.env`` into ``os.environ`` (``agent_core.config._load_env_file`` returns a
    dict consumed only inside ``load_agent_settings()``; no module calls
    ``load_dotenv``/``putenv``/``os.environ.update``).  Reading ``os.environ``
    alone therefore made the documented master switch silently dead: editing
    ``.env`` did nothing unless the variable was also exported in the shell.

    Resolved the way every other ``.env`` value in this repo is: an explicit
    process-env export wins, the file is the fallback.
    """
    value = os.environ.get("KANBAN_SYNC_ENABLED")
    if value is None:
        value = _read_env_file_value("KANBAN_SYNC_ENABLED")
    return (value or "").strip() == "1"


def active_board_allows_agent1(working_dir: str | Path | None = None) -> bool:
    """Fail-closed check: does the Kanban app's ACTIVE board allow agent1 sync?

    Reads ``<data>/active.json`` (shape ``{"active": "<catalog_id>"}``), then
    that board's working copy ``board-<catalog_id>.json`` and its top-level
    ``"sync"`` map. The Kanban app persists the same map from its UI toggle,
    so both sides of the sync link agree without a restart or shared process
    state — this function just re-reads what the user last saved.

    Missing files, malformed JSON, non-dict payloads, path-like catalog ids,
    or an absent/falsy ``agent1`` entry all count as NOT allowed: per-board
    opt-in is fail-closed by design (sync stays off until explicitly enabled).
    """
    root = Path(working_dir) if working_dir else DATA_DIR
    try:
        registry = json.loads((root / "active.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    active_id = registry.get("active") if isinstance(registry, dict) else None
    if not isinstance(active_id, str) or not active_id:
        return False
    # Catalog ids are generated tokens / safe slugs; refuse anything that could
    # escape the data directory (fail closed).
    if "/" in active_id or "\\" in active_id or ".." in Path(active_id).parts:
        return False
    try:
        board = json.loads((root / f"board-{active_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    sync_map = board.get("sync") if isinstance(board, dict) else None
    if not isinstance(sync_map, dict):
        return False
    return bool(sync_map.get("agent1"))


# The queue root is SHARED with the Kanban app.  Both sides must resolve the
# same physical directory or messages pile up in two disjoint trees that never
# meet.  ``KANBAN_WORKING_DIR`` is the single source of truth; the default is
# the sibling Kanban checkout's data/ directory (the two repos live side by
# side under a common parent).
KANBAN_WORKING_DIR = os.environ.get(
    "KANBAN_WORKING_DIR", str(REPO_ROOT.parent / "Kanban" / "data")
)
DATA_DIR = Path(KANBAN_WORKING_DIR)
CARD_ID_MAP_PATH = DATA_DIR / ".card_id_map.json"

# --- Queue paths ---------------------------------------------------------------
# Names describe the DIRECTION of travel, not the side that owns the file.
# The original "in"/"out" naming was a defect in itself: "queue-kanban-out"
# meant Agent1→Kanban on this side but Kanban→Agent1 on the other, so even with
# a shared root the two sides read and wrote different queues.

QUEUE_AGENT1_TO_KANBAN = DATA_DIR / "queue-agent1-to-kanban"   # Agent1 writes, Kanban reads
QUEUE_KANBAN_TO_AGENT1 = DATA_DIR / "queue-kanban-to-agent1"   # Kanban writes, Agent1 reads

# A message that fails this many times is moved to <queue_dir>/dead_letters/.
# Patchable at module level so tests can redirect the dead-letter directory.
MAX_RETRIES = 3
DEAD_LETTER_DIR: Path | None = None

# --- Locking -------------------------------------------------------------------

# Per-queue locks for offset updates and file writes.  A single global lock
# caused hangs when enqueue() and read_queue()/process_inbound() ran
# concurrently: enqueue() held SEQ_LOCK while reading the offset, and
# process_inbound() (running in the QueueProcessor thread) also read the
# offset file.  Per-queue locks avoid cross-queue contention and eliminate
# the deadlock because each queue now has exactly one lock governing both
# its write (enqueue) and its read (read_queue/advance_offset).
_QUEUE_LOCKS: Dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _read_int(path: Path) -> int:
    """Read an integer from a counter file, tolerating a missing/garbled file."""
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return 0


def _lock_for(queue_dir: Path) -> threading.Lock:
    """Get the per-queue lock, creating it lazily."""
    key = str(queue_dir.resolve())
    with _locks_lock:
        if key not in _QUEUE_LOCKS:
            _QUEUE_LOCKS[key] = threading.Lock()
        return _QUEUE_LOCKS[key]


def enqueue(message: dict[str, Any], queue_dir: Path | None = None) -> int:
    """Append a single message to the target queue's messages.jsonl.

    Returns the sequence number assigned to this message.  The append is
    atomic at the line level (open in append mode).
    """
    if queue_dir is None:
        queue_dir = QUEUE_AGENT1_TO_KANBAN
    elif isinstance(queue_dir, dict):
         # This should not happen with correct usage but handles erroneous patching
         raise TypeError(f"enqueue() argument 'queue_dir' must be Path, not dict")

    if not isinstance(queue_dir, Path):
        queue_dir = Path(queue_dir)

    queue_dir.mkdir(parents=True, exist_ok=True)
    msg_file = queue_dir / "messages.jsonl"

    message["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _lock_for(queue_dir):
        # The writer's sequence counter lives in its OWN file (seq.txt).  It
        # must NOT be offset.txt: that file is the *reader's* cursor, and the
        # reader is a different process.  Sharing one file between "next seq"
        # and "lines consumed" meant every enqueue advanced the counterpart's
        # cursor past the line it had just written, so no message was ever
        # delivered.  Read and update both happen under the per-queue lock, so
        # concurrent enqueue() calls stay serialized without deadlock.
        seq = _read_int(queue_dir / "seq.txt") + 1
        message["seq"] = seq
        message.setdefault("retry_count", 0)
        message.setdefault("processed", False)
        line = json.dumps(message, ensure_ascii=False) + "\n"
        with open(msg_file, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        (queue_dir / "seq.txt").write_text(str(seq), encoding="utf-8")

    logger.info("Enqueued [%s] seq=%d to %s", message.get("op"), seq, msg_file)
    return seq


# --- ID mapping ----------------------------------------------------------------

def _load_id_map() -> dict[str, str]:
    """Load the issue↔card bidirectional mapping."""
    if CARD_ID_MAP_PATH.exists():
        try:
            data = json.loads(CARD_ID_MAP_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_id_map(mapping: dict[str, str]) -> None:
    """Atomically save the issue↔card bidirectional mapping."""
    tmp = CARD_ID_MAP_PATH.with_suffix(CARD_ID_MAP_PATH.suffix + ".tmp")
    tmp.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        tmp.replace(CARD_ID_MAP_PATH)
    except OSError:
        shutil.copy2(str(tmp), str(CARD_ID_MAP_PATH))


def link_ids(source_id: str, target_id: str, mapping: dict[str, str] | None = None) -> dict[str, str]:
    """Record a bidirectional ID pair and save it."""
    if mapping is None:
        mapping = _load_id_map()
    mapping[source_id] = target_id
    mapping[target_id] = source_id  # reverse lookup
    _save_id_map(mapping)
    return mapping


def resolve_id(source_id: str, mapping: dict[str, str] | None = None) -> str | None:
    """Look up the counterpart ID for a known source ID."""
    if mapping is None:
        mapping = _load_id_map()
    return mapping.get(source_id)


# --- Message conversion --------------------------------------------------------

def issue_to_card_data(issue: dict[str, Any]) -> dict[str, Any]:
    """Convert an Agent1 issue dict into Kanban card payload data."""
    # Split evidence and suggested_approach using a delimiter that won't
    # appear in normal text.  The card's `text` field holds both.
    parts: list[str] = []
    if issue.get("evidence"):
        parts.append(issue["evidence"])
    if issue.get("suggested_approach"):
        parts.append(f"Approach:\n{issue['suggested_approach']}")

    text = "\n\n---\n".join(parts)

    # Build tags from category and severity
    tags: list[str] = [f"[{issue.get('category', 'general')}]", f"[severity:{issue.get('severity', 'low')}]", f"[status:{issue.get('status', 'open')}]"]

    return {
        "title": issue["title"],
        "text": text,
        "tags": tags,
        "system": "harnessfix",
    }


def card_to_issue_data(card: dict[str, Any]) -> dict[str, Any]:
    """Convert a Kanban card dict into Agent1 issue payload data."""
    # Split card text back into evidence and suggested_approach
    text = card.get("text", "")
    evidence = ""
    suggested_approach = ""
    if "\n\n---\n" in text:
        evidence, _, suggested_approach = text.partition("\n\n---\n")
        suggested_approach = suggested_approach.lstrip()

    # Parse tags for category and severity
    category = "general"
    severity = "low"
    status = "open"
    for tag in card.get("tags", []):
        if tag.startswith("[severity:"):
            severity = tag[len("[severity:"):-1]
        elif tag.startswith("[status:"):
            status = tag[len("[status:"):-1]
        elif tag.startswith("[") and tag.endswith("]") and ":" not in tag:
            category = tag[1:-1]

    return {
        "title": card["title"],
        "category": category,
        "severity": severity,
        "evidence": evidence,
        "suggested_approach": suggested_approach,
        "status": status,
        "tags": card.get("tags", []),
    }


# --- Frame → Issue status mapping ----------------------------------------------

FRAME_TO_STATUS = {
    "Issues": "open",
    "Planned": "planned",
    "On-going": "in-progress",
    "Quality & Assurance": "reviewing",
    "Finished": "resolved",
}


# --- Apply functions (Agent1 processes Kanban inbound messages) ----------------

def _apply_card_create(payload: dict[str, Any]) -> bool:
    """Create an Agent1 issue from a Kanban card CREATE message."""
    from . import issues as issue_store  # lazy import to avoid cycles

    data = card_to_issue_data(payload)
    locations = [f"kanban:{payload.get('id', 'unknown')}"]
    issue = issue_store.make_issue(
        category=data["category"],
        title=data["title"],
        locations=locations,
        severity=data["severity"],
        evidence=data["evidence"],
        suggested_approach=data["suggested_approach"],
        autonomy_level=1,  # Kanban cards default to auto-safe
    )

    # Persist the issue before recording the mapping.  ``make_issue`` only
    # *builds* the dict — it does not write the ledger (the repo-scan collector
    # does the upsert).  Linking without saving left a phantom pair in the id
    # map: later card_update/card_move messages resolved to an issue id that
    # existed nowhere, so every follow-up logged "no matching issue" and the
    # card could never be updated or resolved.
    issues = issue_store.load_issues()
    if issue_store.upsert(issues, issue):
        issue_store.save_issues(issues)

    # Record the ID mapping
    card_id = payload.get("id", "")
    if card_id:
        link_ids(issue["id"], card_id)

    return True


def _apply_card_update(payload: dict[str, Any]) -> bool:
    """Update an existing Agent1 issue from a Kanban card UPDATE message."""
    from . import issues as issue_store  # lazy import

    source_id = payload.get("source_id", "")
    mapping = _load_id_map()

    if not source_id:
        return False

    # Resolve the issue ID via the card's source_id (Kanban card id)
    issue_id = resolve_id(source_id, mapping) or source_id
    issues = issue_store.load_issues()
    existing = issue_store.find_by_id(issues, issue_id)

    if not existing:
        logger.warning("card_update: no matching issue for %s (source=%s)", issue_id, source_id)
        return False

    data = card_to_issue_data(payload.get("payload", payload))
    updated = False

    if "title" in data and data["title"]:
        existing["title"] = data["title"]
        updated = True
    if "evidence" in data and data["evidence"]:
        existing["evidence"] = f"{existing.get('evidence', '')}\n[note] {data['evidence']}".strip()
        updated = True
    if "suggested_approach" in data and data["suggested_approach"]:
        existing["suggested_approach"] = data["suggested_approach"]
        updated = True

    # Update tags as status indicator
    new_status = data.get("status", "")
    if new_status:
        existing["status"] = new_status
        updated = True

    if updated:
        issue_store.save_issues(issues)
        logger.info("Applied card_update to issue %s", issue_id)

    return True


def _resolve_frame_status(payload: dict[str, Any]) -> str:
    """Map a card_move payload to an issue status.

    The Kanban side sends the column *title* ("Finished") alongside the opaque
    frame id.  ``FRAME_TO_STATUS`` is keyed by title, so resolving the id alone
    could never hit: every move fell back to "open", silently rewriting a
    resolved issue back to open.  The id is still accepted for messages that
    were enqueued before the title was added.
    """
    inner = payload.get("payload", payload) or {}
    for key in ("frame_title", "frame_id"):
        value = inner.get(key, "")
        if value and value in FRAME_TO_STATUS:
            return FRAME_TO_STATUS[value]
    return "open"


def _apply_card_move(payload: dict[str, Any]) -> bool:
    """Move a Kanban card → update Agent1 issue status."""
    from . import issues as issue_store  # lazy import

    source_id = payload.get("source_id", "")
    mapping = _load_id_map()

    if not source_id:
        return False

    issue_id = resolve_id(source_id, mapping) or source_id
    status = _resolve_frame_status(payload)

    issues = issue_store.load_issues()
    existing = issue_store.find_by_id(issues, issue_id)

    if not existing:
        logger.warning("card_move: no matching issue for %s (source=%s)", issue_id, source_id)
        return False

    old_status = existing["status"]
    existing["status"] = status
    if status in ("resolved", "wontfix") and old_status not in ("resolved", "wontfix"):
        existing["resolved_at"] = issue_store._now()

    issue_store.save_issues(issues)
    logger.info("Applied card_move: %s → %s (was %s)", issue_id, status, old_status)
    return True


def _apply_card_delete(payload: dict[str, Any]) -> bool:
    """Delete/wontfix an Agent1 issue from a Kanban card DELETE message."""
    from . import issues as issue_store  # lazy import

    source_id = payload.get("source_id", "")
    mapping = _load_id_map()

    if not source_id:
        return False

    issue_id = resolve_id(source_id, mapping) or source_id
    issues = issue_store.load_issues()
    existing = issue_store.find_by_id(issues, issue_id)

    if not existing:
        logger.warning("card_delete: no matching issue for %s (source=%s)", issue_id, source_id)
        return False

    # Mark as wontfix rather than hard-delete to preserve history
    issue_store.resolve(issues, issue_id, disposition="wontfix", note="Card deleted from Kanban")
    logger.info("Applied card_delete: marked %s as wontfix", issue_id)
    return True


# --- Inbound processor ---------------------------------------------------------

def _process_single_message(msg: dict[str, Any]) -> bool:
    """Apply a single inbound message. Returns True on success."""
    op = msg.get("op", "")
    payload = msg.get("payload", {})

    if op == "card_create":
        return _apply_card_create(payload)
    elif op == "card_update":
        return _apply_card_update(msg)
    elif op == "card_move":
        return _apply_card_move(msg)
    elif op == "card_delete":
        return _apply_card_delete(msg)
    else:
        logger.warning("Unknown inbound op: %s", op)
        return False


def _quarantine_malformed(
    queue_dir: Path, idx: int, line: str, exc: Exception
) -> None:
    """Preserve an unparseable queue line instead of losing it to the log.

    A line that is not JSON can never be retried into success, so re-reading it
    forever is pointless — but silently dropping it destroys the only copy of
    whatever the sender wrote.  Quarantining satisfies both: the offset moves
    past the bad line and the raw bytes survive for inspection.
    """
    try:
        dl_dir = _dead_letter_dir(queue_dir)
        dl_dir.mkdir(parents=True, exist_ok=True)
        dl_file = dl_dir / f"malformed_line_{idx}.jsonl"
        dl_file.write_text(
            json.dumps(
                {"seq": idx + 1, "error": str(exc), "raw": line},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.error(
            "Quarantined malformed JSON from queue line %d to %s: %s",
            idx, dl_file, exc,
        )
    except OSError as write_exc:  # noqa: BLE001 — never break the read loop
        logger.error(
            "Could not quarantine malformed queue line %d (%s); line dropped",
            idx, write_exc,
        )


def read_queue(queue_dir: Path) -> tuple[list[dict[str, Any]], int]:
    """Read new lines from the queue's messages.jsonl starting at offset.txt.

    Returns (new_messages, next_offset).  A line with invalid JSON can never
    be retried into success, so it is quarantined to the dead-letter
    directory and the offset advances past it.  Re-reading it forever was
    the old behaviour: one bad byte re-logged an error on every poll for
    the life of the queue, and the bytes were lost to nothing but the log.
    """
    queue_dir = queue_dir.resolve()
    msg_file = queue_dir / "messages.jsonl"
    if not msg_file.exists():
        return [], 0

    try:
        current_offset = int(
            (queue_dir / "offset.txt").read_text(encoding="utf-8").strip()
        )
    except (ValueError, OSError):
        current_offset = 0

    lines = []
    try:
        content = msg_file.read_text(encoding="utf-8")
        lines = content.splitlines()
    except OSError as exc:
        logger.error("Failed to read queue file %s: %s", msg_file, exc)
        return [], current_offset

    new_messages: list[dict[str, Any]] = []
    next_offset = current_offset

    for idx, line in enumerate(lines):
        if idx < current_offset:
            continue  # already processed
        next_offset = idx + 1

        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
            new_messages.append(msg)
        except json.JSONDecodeError as exc:
            _quarantine_malformed(queue_dir, idx, line, exc)

    return new_messages, next_offset


def advance_offset(queue_dir: Path, offset: int) -> None:
    """Update the queue's offset.txt to mark messages as processed."""
    queue_dir = queue_dir.resolve()
    with _lock_for(queue_dir):
        (queue_dir / "offset.txt").write_text(str(offset), encoding="utf-8")


def _dead_letter_dir(queue_dir: Path) -> Path:
    """Directory that quarantines permanently failing messages."""
    return DEAD_LETTER_DIR if DEAD_LETTER_DIR is not None else queue_dir / "dead_letters"


def _record_failure(queue_dir: Path, msg: dict[str, Any]) -> None:
    """Persist an incremented retry_count so retries can actually accumulate.

    The queue is an append-only JSONL log, so a failed message is rewritten at
    the tail with ``retry_count + 1``; that keeps the failure durable across
    process restarts, which is what makes dead-lettering reachable at all.
    (Indexing ``messages.jsonl`` by line number instead would be wrong: ``seq``
    is ``offset + 1``, so ``idx == seq`` never matches the right line.)
    """
    retry_count = int(msg.get("retry_count", 0)) + 1
    msg["retry_count"] = retry_count

    if retry_count >= MAX_RETRIES:
        dl_dir = _dead_letter_dir(queue_dir)
        dl_dir.mkdir(parents=True, exist_ok=True)
        dl_file = dl_dir / f"seq_{msg.get('seq', '?')}.jsonl"
        dl_file.write_text(
            json.dumps(msg, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logger.error(
            "Dead-lettered seq=%s after %d attempts to %s",
            msg.get("seq"), retry_count, dl_file,
        )
        return

    msg["processed"] = False
    with open(queue_dir / "messages.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def process_inbound(queue_dir: Path | None = None) -> int:
    """Read and apply all pending inbound messages. Returns count applied.

    Every message consumed advances the offset — including failures, which are
    re-queued at the tail with a higher ``retry_count``. Without that, a single
    poison message would pin the offset and stall the whole queue forever.
    """
    if queue_dir is None:
        queue_dir = QUEUE_KANBAN_TO_AGENT1

    queue_dir = queue_dir.resolve()
    current_offset = _read_int(queue_dir / "offset.txt")
    new_messages, next_offset = read_queue(queue_dir)
    applied = 0

    for msg in new_messages:
        op = msg.get("op", "")

        # Skip already-processed messages (defensive)
        if msg.get("processed"):
            continue

        try:
            success = _process_single_message(msg)
            if not success:
                raise RuntimeError(f"_process_{op} returned False")
            applied += 1
        except Exception:
            logger.exception("Failed processing op=%s: %r", op, msg)
            _record_failure(queue_dir, msg)

    # Advance whenever the cursor actually needs to move — not merely when a
    # *parsed* message was seen.  A queue holding only a malformed line returns
    # no messages but a higher next_offset; guarding on `new_messages` left the
    # cursor at 0, so that unparseable line was re-read (and re-quarantined) on
    # every poll for the life of the queue.
    if next_offset > current_offset:
        advance_offset(queue_dir, next_offset)
        logger.info(
            "Processed %d/%d messages from %s (offset now %d)",
            applied, len(new_messages), queue_dir, next_offset,
        )

    return applied


# --- Background processor thread -----------------------------------------------

class QueueProcessor:
    """Background thread that polls an inbound queue and processes messages."""

    def __init__(self, queue_dir: Path | None = None, poll_interval: float = 5.0):
        if queue_dir is None:
            queue_dir = QUEUE_KANBAN_TO_AGENT1
        self.queue_dir = queue_dir
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()

    def run_once(self) -> int:
        """Process one batch of pending messages. Returns count applied."""
        try:
            return process_inbound(self.queue_dir)
        except Exception as exc:
            logger.error("QueueProcessor error: %s", exc, exc_info=True)
            return 0

    def run_forever(self) -> None:
        """Poll the queue continuously until stopped."""
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                logger.error("QueueProcessor fatal error: %s", exc, exc_info=True)

            # Sleep in small increments so we can stop quickly
            for _ in range(int(self.poll_interval * 10)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)

    def stop(self) -> None:
        """Signal the processor thread to exit."""
        self._stop_event.set()


# --- Boot integration ----------------------------------------------------------

# One processor per process, mirroring the Kanban side's module-level
# ``_SYNC_HANDLES`` sentinel.  Every extra poller on the same queue directory
# widens the window where an interpreter exit lands ``messages.jsonl``
# mid-append, and the reader then advances past the torn line.  The sentinel
# starts as ``[]`` ("no processor yet") and is falsy, so a truthiness test —
# not ``is None`` — is what lets the first sync-enabled boot win.
_BOOT_HANDLES: list["QueueProcessor"] = []


def start_inbound_processor(
    queue_dir: Path | None = None, poll_interval: float = 5.0,
) -> "QueueProcessor | None":
    """Start the inbound poller on a daemon thread; return it (None if reused).

    This is the production boot path for Kanban → Agent1 messages.  Without a
    caller, ``QueueProcessor`` was constructed nowhere outside tests: Agent1
    enqueued ``issue_create``/``issue_resolve`` outbound while every inbound
    card move/update/delete sat unread in ``queue-kanban-to-agent1`` forever.
    That is the half-wired state this function closes.

    Gated on ``KANBAN_SYNC_ENABLED=1`` and idempotent per process.  Sync
    failure is swallowed (logged, not raised): a broken queue must never
    prevent the agent from booting.
    """
    if not sync_enabled():
        return None
    if _BOOT_HANDLES:
        # Reuse the process's first processor — the poller is queue-dir based
        # and stateless between polls, so a second one adds nothing but a
        # wider mid-append kill window on the same files.
        logger.debug("Reusing the process's inbound Kanban processor")
        return None
    try:
        proc = QueueProcessor(queue_dir=queue_dir, poll_interval=poll_interval)
        # Registered before start() so a failure to spawn the thread still
        # leaves stop() reachable; stop() is idempotent and only sets a flag.
        atexit.register(proc.stop)
        threading.Thread(
            target=proc.run_forever, name="kanban-inbound", daemon=True,
        ).start()
        _BOOT_HANDLES.append(proc)
        logger.info("Kanban inbound processor started (queue=%s)", proc.queue_dir)
        return proc
    except Exception as exc:  # noqa: BLE001 — never crash boot on sync failure
        logger.error("Failed to start Kanban inbound processor: %s", exc)
        return None


def inbound_processor_running() -> bool:
    """True when this process already owns a started inbound processor."""
    return bool(_BOOT_HANDLES)


# --- CLI entry point -----------------------------------------------------------

def main() -> int:
    """CLI entry point for manual queue processing."""
    import argparse

    parser = argparse.ArgumentParser(description="Process Kanban sync queues")
    parser.add_argument("--process-in", action="store_true", help="Process inbound (Kanban -> Agent1)")
    # Documented as ``--process`` in this module's docstring for a long time;
    # kept as a hidden alias so the old invocation in operator notes still
    # works instead of exiting with "unrecognized arguments".
    parser.add_argument("--process", dest="process_in", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--queue-dir", type=str, help="Queue directory to process")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.process_in:
        qdir = Path(args.queue_dir) if args.queue_dir else QUEUE_KANBAN_TO_AGENT1
        count = process_inbound(qdir)
        print(f"Processed {count} messages from {qdir}")
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
