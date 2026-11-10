# Kanban Card Work-Visibility Plan (child cards with parent_id)

Status: **draft — awaiting confirmation before implementation.**

## Goal

Make work performed by Agent1 on a synchronized Kanban card visible on the Kanban
board as child cards attached to the parent card via `parent_id`, and capture the work
so Agent1 can understand what was done on the card later. The Kanban side already
supports `parent_id` end-to-end (card create, move, tree grouping, and the card-create
form all handle it); the gap is on the Agent1 side of the bidirectional bridge —
attributing Agent1 issue operations to a card and emitting child-card payloads — plus
capturing the work for later understanding.

Governing rule: **facts are never guessed** — a missing fact stays `None`, it is never
fabricated as `0`; every emission is best-effort — child-card creation must never break
the existing bidirectional bridge or the Kanban card move path.

## What I verified (grounding)

- Kanban side supports `parent_id` fully: `C:\Dev\Kanban\src\agent1\models.py`
  (`Card.parent_id` field, serialized in `to_dict`), `store.py` `add_card` reads
  `parent_id`, `routes_cards.py` create/move set `parent_id`, `routes_frames.py`
  tree view groups parent→child, `static/main.js` card-create form reads a parent
  dropdown. → No change needed on the Kanban side for child-card support itself.
- Agent1 side bridge maps issues↔cards via `.card_id_map.json` (bidirectional
  issue↔card ID mapping) and `kanban:<card_id>` locations in issue payloads
  (`_find_issue_by_card_location`, `_lookup_issue_for`). Agent1 issue create/update/
  resolve enqueue to the Kanban side via `harnessfix/issues.py::_enqueue_sync`.
- Message format: `{seq, ts, op, source_id, target_ref, payload, retry_count,
  processed}` written to the Kanban queue's `messages.jsonl` (both sides share
  `KANBAN_WORKING_DIR`).

## Scope

**In scope:**

1. Agent1-side bridge emits child Kanban cards (with `parent_id`) when Agent1 performs
   work on a synchronized card (issue create/update/resolve attributable to a card).
2. The work content is carried on the child-card payload so the work is visible on the
   board (and thus understood from the board alone).
3. Capture the work so Agent1 can understand the initial prompt / what was done on the
   card later (the work already lives in the Agent1 issue ledger; the child-card
   payloads carry it for board visibility).

**Out of scope (decide):**

- Re-rendering the Kanban UI — the tree view already renders children generically; no
  UI change expected.
- Changing the Kanban card create/move flow itself — already supports `parent_id`.
- Auto-summarizing work into card titles — decide whether child-card titles are
  auto-generated or left to the user.
- Adding a separate Agent1-side per-card work ledger (only if the user wants Agent1 to
  *retrieve* the work later, not just see it on the board).

## Components

### Component 1 — Bridge child-card emission (Agent1 side)

Module: `harnessfix/kanban_bridge.py` (+ the `harnessfix/issues.py` enqueue point).

- Reuse the existing card↔issue resolution (`_find_issue_by_card_location` /
  `.card_id_map.json`) to attribute an Agent1 issue op to a Kanban card.
- When an issue op (`issue_create`/`issue_update`/`issue_resolve`) is enqueued to the
  Kanban side AND it is attributable to a card, emit a child-card payload with:
  - `parent_id` = the card's id,
  - `title` = work summary (issue id + status + evidence/approach),
  - `text`/`description` = the work record (evidence/approach),
  - work metadata: issue id, status, op, ts.
- Only emit for work-on-card ops; other (non-card) issues stay as-is.
- Best-effort: any failure to emit a child card degrades to a no-op (the parent op
  still syncs); never raises into the bridge.

### Component 2 — Kanban reader creates child cards under the parent

Repo: `C:\Dev\Kanban`. Modules: `src\agent1\kanban_sync.py` (reader), `routes_cards.py`
/ `store.py` / `models.py` (already support `parent_id`).

- Verify the reader's card-create path already accepts and persists `parent_id` and
  routes the new card to the parent's frame (tree grouping). Based on the verified
  support above, this is very likely already supported — confirm via the Kanban test
  suite before changing code.
- If the reader needs a child-card branch (e.g., to create the card in the parent's
  frame explicitly), add a minimal branch that creates the card in the parent's frame
  with `parent_id` set, reusing the existing `parent_id` handling in
  `store.add_card`/`routes_cards`.

### Component 3 — Work capture for understanding the initial prompt

Module: `harnessfix/kanban_bridge.py` + `harnessfix/issues.py`.

- The work performed on a card is already recorded in the Agent1 issue ledger
  (`issue_create`/`issue_update`/`issue_resolve` write to `.issues.json` with
  evidence, suggested_approach, status, resolved_at). This is the Agent1-side record
  of the work.
- The child-card payloads (Component 1) carry the work summary + evidence, so the work
  is visible on the board — which is the "understand the initial prompt" surface (read
  the card's work children). No new Agent1-side work ledger is required for this
  scope.
- (Optional, only if requested): a per-card work ledger on the Agent1 side for Agent1 to
  *retrieve* the work later — separate decision, not in this plan.

## Decision points to confirm with you before implementation

1. **Child-card granularity:** one child card per work-op (issue create/update/resolve),
   each with `parent_id`, or one summarizing child card per card? I lean one-per-op
   (the user said "adding children Kanban cards" — plural).
2. **Child-card content:** auto-generate the title/body from the work summary (issue id
   + status + evidence), or leave the title to the user? Auto is less work and keeps the
   board self-explanatory; flag your preference.
3. **Work capture scope:** is "understand the initial prompt" satisfied by the work being
   visible on the board (child cards + issue ledger), or do you also want an
   Agent1-side per-card work ledger for later retrieval?
4. **Scope boundary:** only work on *synchronized* cards (cards linked to an Agent1 issue
   via the bridge) — matches "Agent1 synchronized Cards". I'll scope to that; confirm.

## Verification

- Regression tests for Component 1: child-card payload carries `parent_id` + work
  metadata; only work-on-card ops emit; non-card issues unchanged; best-effort
  degradation (emission failure → parent op still syncs, no crash).
- Verify Component 2: the Kanban reader already persists `parent_id` and groups child
  cards under the parent (tree view) — confirm via `C:\Dev\Kanban` tests
  (`tests/test_kanban_bridge.py`, `tests/test_kanban_roundtrip.py`, plus the repo's
  store/routes tests).
- Run the Agent1 + Kanban test suites (`tests/test_kanban_bridge.py` 40,
  `tests/test_kanban_roundtrip.py` 10, `tests/test_llm_decision_attribution.py`
  referenced) before claiming the move path is healthy.
- Full suite on master before committing (commit-after-every-session invariant).

## Data locations

- Child-card payloads: emitted via the existing Kanban sync queue
  (`kanban_sync.py` messages.jsonl) on the Kanban side; the persisted work record lives
  in the Kanban queue + board state.
- Agent1-side work record: `.issues.json` (already exists; unchanged in this scope).

## Known gaps / follow-ups (not started)

- Original-user-prompt capture: if "understand the initial prompt" means storing the
  original request that motivated the work on the card, that is a separate capture not in
  this plan.
- No UI change expected (tree view renders children); verify in the Kanban repo.
