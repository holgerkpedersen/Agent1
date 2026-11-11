# Kanban Card Work Visibility — Reconsidered Plan

Status: **draft — awaiting confirmation before implementation.**
Supersedes the narrower child-card-only plan in
`docs/plans/2026-11-10-kanban-card-work-visibility-plan.md`.

## 1. What the user actually needs

> "Is it visible in Kanban for Agent1 synchronized Cards, that these measures
> have been done? … be able to see what work has been done on a Kanban Card,
> by adding children Kanban cards … The user needs a way to understand what
> Agent1 session did to solve a particular Kanban Card!"

Two surfaces, not one:

1. **On the board** — the work Agent1 performed on a synchronized card must
   appear as distinct, inspectable artifacts attached to that card (children
   via `parent_id`), so a human reading the board sees *what was done*, not just
   a mutated parent card.
2. **For Agent1 later** — an Agent1 session must be able to retrieve, for a
   given card, the record of what was done (the initial prompt + the work
   performed), not just see it on the board.

The existing implementation (commits ef461cd, 2e1e1e0) covers surface 1
partially: it emits a child card on `issue_create` (intake) and on `issue_resolve`
(resolution transition), gated by `emit_child_card=True`. It does **not** cover
intermediate work, does not dedupe on retry, and provides no Agent1-side
retrieval.

## 2. Gaps in the current implementation (verified against code)

- **`harnessfix/issues.py`** — only `make_issue` and `resolve` enqueue. There is
  no `update`/`note` path, so intermediate work (evidence/approach changes,
  notes added during a session) is never surfaced as a child card.
- **`harnessfix/kanban_bridge.py::_emit_child_card_for`** — builds one
  `child_card_create` per call. No idempotency key; a retried or re-run op can
  spawn duplicate child cards. No work-log retrieval helper exists.
- **`harnessfix/kanban_inspect.py`** — `kanban card <id>` composes the parent's
  description but does **not** list child work records, so Agent1 cannot
  "understand what was done" from its own side.
- **Kanban reader** (`src/agent1/kanban_sync.py::_apply_child_card_create`) —
  creates the child in the parent's frame with `parent_id` set and tags
  `[status:…] [op:…] [issue:…]`. Good. But children are only visible in the
  tree view; in the flat board view they are indistinguishable siblings.
- **Child card title** — `[work] <issue id> (<status>)` is terse and not
  human-friendly; the work body lives in `text` (evidence/approach), which is
  fine but the title should signal "Agent1 work".

## 3. Reconsidered approach

### 3.1 Capture work at every milestone, not just create/resolve

Add an `issue_update`/`issue_note` path in `harnessfix/issues.py` that:
- mutates evidence/suggested_approach/status on the issue,
- enqueues `issue_update` to Kanban (existing op),
- when `emit_child_card=True` and the issue is card-attributable, emits a
  `child_card_create` describing the *delta* (what changed this step).

Milestones that emit: intake (create), each recorded note/update, resolution.
This gives the board a chronological trail of work, matching "children … plural".

### 3.2 Idempotency on the emission side

Give each child card a deterministic-ish identity derived from
`(parent_card_id, issue_id, op, ts-bucket)` and have the Kanban reader skip
creation when a child with the same `(parent_id, issue_id, op)` already exists.
Best-effort: a missing parent or a duplicate is a quiet no-op, never a dead
letter. This prevents duplicate children on retries/echoes.

### 3.3 Make children visible in the flat board view

- Prefix child titles with `[Agent1 work]` so they read as work records in any
  view (flat or tree).
- Carry a `[card:<parent_id>]` tag on the child (in addition to `[issue:…]`)
  so a flat view can badge "child of X" without relying on tree grouping.
- Keep `system="harnessfix"` so the existing styling distinguishes Agent1 work.

### 3.4 Agent1-side retrieval ("understand what was done")

Extend `kanban_inspect`:
- `kanban card <id>` prints, after the composed description, a **Work log**
  section listing child work records for that card (from `.issues.json` issues
  whose `parent_card_id` matches, ordered by `created_at`/`resolved_at`), each
  with status, evidence, suggested_approach, and resolved_at.
- Optional: `kanban work <card-id>` dedicated view.

This makes Agent1 able to reconstruct "what the session did" from its own
ledger, which already holds evidence/suggested_approach/status/resolved_at.

### 3.5 Work capture scope (answer to open question 3)

Board visibility + issue ledger is sufficient **if** the ledger is queried by
card. No separate per-card work ledger is needed — the issue ledger already is
the work ledger; we just need the card→issues query (3.4). This keeps the
change small and avoids a second source of truth.

### 3.6 Scope boundary (answer to open question 4)

Only synchronized cards (card↔issue linked via the bridge). Confirmed: a card
created directly on the board with no issue link has nothing to surface, and
emitting for it would invent work. Matches "Agent1 synchronized Cards".

## 4. Files to touch (smallest coherent change)

Agent1 side:
- `harnessfix/issues.py` — add `update_issue(..., emit_child_card=...)` (or
  reuse `resolve`-style) that enqueues `issue_update` + conditional child card;
  thread `parent_card_id` through.
- `harnessfix/issues.py::_emit_child_card_for` — add op in payload, idempotency
  key, `[Agent1 work]` title prefix, `[card:<parent>]` tag.
- `harnessfix/kanban_bridge.py` — `_work_summary_for_child_card` title prefix;
  carry `op`/`ts` already (verify).
- `harnessfix/kanban_inspect.py` — Work log section in `issue_view`/`card`.
- `agent_core/commands/issue_cmd.py` — expose `issue note <id> "..."` (or
  reuse `issue add` location) so a session can record intermediate work with
  `emit_child_card=True`.

Kanban side:
- `src/agent1/kanban_sync.py::_apply_child_card_create` — dedupe by
  `(parent_id, issue_id, op)` before creating; keep best-effort no-op semantics.
- (Optional) flat-view badge in `static/main.js` / templates — only if tree view
  is insufficient; decide after testing.

## 5. Decision points (need your answer)

1. **Intermediate work:** emit a child card per note/update, or only at
   create + resolve + one optional "session summary" child? I lean per-note
   (matches "children plural" and gives a trail), but it can be noisy.
2. **Idempotency:** skip duplicates by `(parent, issue, op)` even if the text
   changed, or allow re-emission with new text? I lean skip-if-same-op (a retry
   should not duplicate; a genuinely new note gets a new op/timestamp).
3. **Flat-view badge:** add `[Agent1 work]` prefix + tag only (no JS change),
   or also modify the board renderer? I lean prefix+tag only first.
4. **Agent1 retrieval:** add Work log to `kanban card` (my lean) — confirm.

## 6. Verification plan

- Agent1: extend `test_kanban_child_card_emission.py` with update/note path,
  dedup, and `kanban card` Work log; keep `test_issue_resolve_child_card_emission.py`.
- Kanban: extend `test_kanban_sync.py` with dedup and tag assertions.
- Run both suites + full Agent1 suite before commit.
- Manual smoke: sync a card, resolve with a note, confirm child card appears
  under parent in tree view and Work log prints on `kanban card <id>`.

## 7. What I will NOT do without your "yes"

No code changes. I will implement only after you reply **yes** (optionally with
preferences on the 4 decision points). Until then this plan is documentation only.
